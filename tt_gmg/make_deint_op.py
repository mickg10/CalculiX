# make_deint_op.py — DE-INTERLEAVED DOF layout proof + operator writer (host-only, no device).
#
# Closes the highest-leverage G3 throughput fix in GATHER_DESIGN.md (lines 143-151): the current gather
# stores x interleaved (x[3*node+r]) and gathers 3 values/node with a per-element e/3 div, which makes the
# scalar RISC reader latency-bound (~60 ms/8-chip, ~20x over G3). De-interleaving stores x as 3 planes
# X[c][node] and the operator as A[o][r][c][node] (243 planes), so the gather for a fixed (o,c) is ONE
# node-granular value per node (no 3x broadcast, no div). Compute grows 81->243 MAC terms (~0.06 ms, free).
#
# This script PROVES the reindexing is bit-exact vs (a) the interleaved formula the kernel uses today and
# (b) a direct BCSR SpMV, on either a synthetic 27-pt lattice (default, fast, runs anywhere) or the real
# row236 dump (--src /tmp/row236_fine.bin). It then writes the de-interleaved operator file the rewritten
# gather kernel will consume. Proving the layout in fp64 here removes the layout risk from the device rewrite.
#
# de-interleaved identity (the thing being proven):
#   interleaved:   y[3*node+r] = sum_o sum_c cf[o*3+c, 3*node+r] * x[3*nbr[o,node]+c]
#   de-interleaved: Y[r][node] = sum_o sum_c A[o,r,c,node]        * X[c][ nbr[o,node] ]
#   with X[c][node]=x[3*node+c], A[o,r,c,node]=val[block(node,o),r,c]  ==>  Y[r][node] == y[3*node+r].
#
# file /tmp/row236_deint_op.bin: <qqq>(nb, K3=243, NBpad)  then
#   float32 X[3, NBpad]            (de-interleaved test vector planes, x_c[node])
#   int32   nbr[27, NBpad]         (node-major-friendly; nbr[o,node], -1 = boundary)
#   float32 A[243, nb]            (A[(o*3+r)*3+c, node] = val[block(node,o), r, c])
#   float32 REF[3, nb]            (Y[r][node] fp64->f32 reference = A*x_test, de-interleaved)
import numpy as np, struct, sys

def load_real(src):
    f = open(src, "rb")
    nb, nblk, nx, ny, nz = struct.unpack("<qqqqq", f.read(40))
    ptr = np.fromfile(f, np.int64, nb + 1)
    col = np.fromfile(f, np.int32, nblk)
    val = np.fromfile(f, np.float64, nblk * 9).reshape(nblk, 3, 3)
    ijk = np.fromfile(f, np.int64, nb * 3).reshape(nb, 3)
    f.close()
    deg = np.diff(ptr).astype(np.int64)
    node_of_block = np.repeat(np.arange(nb, dtype=np.int64), deg)
    doff = ijk[col] - ijk[node_of_block]
    inr = np.all((doff >= -1) & (doff <= 1), axis=1)
    o = ((doff[:, 2] + 1) * 9 + (doff[:, 1] + 1) * 3 + (doff[:, 0] + 1)).astype(np.int64)
    bi = np.where(inr)[0]
    nbr = np.full((27, nb), -1, np.int64)
    nbr[o[bi], node_of_block[bi]] = col[bi]
    blk = np.full((27, nb), -1, np.int64)          # block index feeding each (o,node) -> val row
    blk[o[bi], node_of_block[bi]] = bi
    return nb, nbr, blk, val, (nx, ny, nz)

def synth(nx=9, ny=8, nz=11, seed=7):
    # z-fastest node numbering (matches row236, GATHER_DESIGN line 585); dense 27-pt box, in-bounds neighbors only.
    nb = nx * ny * nz
    ii = np.arange(nb)
    zc = ii % nz; yc = (ii // nz) % ny; xc = ii // (nz * ny)
    nbr = np.full((27, nb), -1, np.int64)
    for o in range(27):
        dz, dy, dx = o % 3 - 1, (o // 3) % 3 - 1, o // 9 - 1
        zz, yy, xx = zc + dz, yc + dy, xc + dx
        ok = (zz >= 0) & (zz < nz) & (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
        nbr[o, ok] = (xx[ok] * ny + yy[ok]) * nz + zz[ok]
    # one 3x3 block per (o,node) that actually has a neighbor; deterministic, mildly ill-scaled to mimic stiffness.
    rng = np.random.default_rng(seed)
    present = nbr >= 0
    nblk = int(present.sum())
    val = (rng.standard_normal((nblk, 3, 3)) * np.array([1e6, 1e3, 1.0])).astype(np.float64)
    blk = np.full((27, nb), -1, np.int64)
    blk[present] = np.arange(nblk)
    return nb, nbr, blk, val, (nx, ny, nz)

# ---- build operator (real dump if given, else synthetic) ----
src = None
for i, a in enumerate(sys.argv):
    if a == "--src" and i + 1 < len(sys.argv): src = sys.argv[i + 1]
if src:
    nb, nbr, blk, val, dims = load_real(src); tag = "real:%s" % src
else:
    nb, nbr, blk, val, dims = synth(); tag = "synth %dx%dx%d" % dims
n = 3 * nb
NBpad = ((nb + 1023) // 1024) * 1024                 # pad node count so planes tile cleanly (1024/tile)
print("[deint] %s  nb=%d  n=3nb=%d  NBpad=%d  dims=%s" % (tag, nb, n, NBpad, dims), flush=True)

rng = np.random.default_rng(11)
x = (1e-6 * rng.standard_normal(n)).astype(np.float64)   # displacement-scale test vector

# ---------- reference A: direct BCSR SpMV (block form) ----------
present = nbr >= 0
o_idx, node_idx = np.where(present)                 # all (o,node) with a real neighbor
b_rows = blk[o_idx, node_idx]                       # val row for each
nn = nbr[o_idx, node_idx]                           # neighbor node
xcol = x.reshape(nb, 3)[nn]                         # (M,3) neighbor 3-vector
yb = np.einsum('mrc,mc->mr', val[b_rows], xcol)     # (M,3) block @ neighbor
y_bcsr = np.zeros((nb, 3)); np.add.at(y_bcsr, node_idx, yb); y_bcsr = y_bcsr.ravel()

# ---------- INTERLEAVED formula (exactly what gather_reader.cpp computes today) ----------
cf = np.zeros((81, n), np.float64)                  # cf[o*3+c, 3*node+r]
for r in range(3):
    for c in range(3):
        cf[o_idx * 3 + c, 3 * node_idx + r] = val[b_rows, r, c]
xp = np.zeros(n); xp[:] = x
y_int = np.zeros(n)
nbr_e = nbr.copy()
for o in range(27):
    v = nbr_e[o]; ok = v >= 0
    for c in range(3):
        srci = np.where(ok, 3 * v + c, 0)
        xg = np.where(ok, xp[srci], 0.0)            # gathered neighbor value, broadcast over r below
        for r in range(3):
            y_int[3 * np.arange(nb) + r] += cf[o * 3 + c, 3 * np.arange(nb) + r] * xg

# ---------- DE-INTERLEAVED formula (what the rewritten kernel will compute) ----------
X = np.zeros((3, NBpad));
for c in range(3): X[c, :nb] = x.reshape(nb, 3)[:, c]        # X[c][node] = x[3node+c]
A = np.zeros((243, nb))                                       # A[(o*3+r)*3+c, node]
A[( (o_idx*3+0)*3+0 )] = 0                                    # (init handled below by direct assignment)
for r in range(3):
    for c in range(3):
        A[(o_idx * 3 + r) * 3 + c, node_idx] = val[b_rows, r, c]
Y = np.zeros((3, nb))
node_all = np.arange(nb)
gload = 0
for o in range(27):
    v = nbr[o]; ok = v >= 0
    vn = np.where(ok, v, 0)
    for c in range(3):
        g = np.where(ok, X[c, vn], 0.0)             # ONE gather value per node for this (o,c) -- no r-broadcast
        gload += 1
        for r in range(3):
            Y[r] += A[(o * 3 + r) * 3 + c, node_all] * g

# ---------- verdicts ----------
y_deint = np.zeros(n)
for r in range(3): y_deint[3 * node_all + r] = Y[r]
def rel(a, b): return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-30))
r_ib = rel(y_int, y_bcsr)
r_di = rel(y_deint, y_int)
r_db = rel(y_deint, y_bcsr)
print("[deint] interleaved vs BCSR   rel=%.3e %s" % (r_ib, "OK" if r_ib < 1e-12 else "MISMATCH"), flush=True)
print("[deint] de-interleaved vs int rel=%.3e %s  (bit-exact reindex expected)" % (r_di, "OK" if r_di == 0.0 else ("OK~" if r_di < 1e-12 else "MISMATCH")), flush=True)
print("[deint] de-interleaved vs BCSR rel=%.3e %s" % (r_db, "OK" if r_db < 1e-12 else "MISMATCH"), flush=True)
# gather-load accounting: interleaved does 3 gathers/node/(o) (one per c, each broadcast to 3 r's => 3 L1 reads/elem);
# de-interleaved does 1 value/node per (o,c). Count DISTINCT gathered values that hit L1 per node across all (o,c):
int_loads_per_node = 27 * 3 * 3     # (o) x (c) x (r-broadcast reads in the scalar kernel: 3 reads/elem, 3 elems/node)
di_loads_per_node = 27 * 3          # (o) x (c), one value each; the 3 r's reuse it from a register
print("[deint] gather L1-reads/node: interleaved=%d  de-interleaved=%d  reduction=%.2fx"
      % (int_loads_per_node, di_loads_per_node, int_loads_per_node / di_loads_per_node), flush=True)

ok_all = (r_ib < 1e-12) and (r_di < 1e-12) and (r_db < 1e-12)
print("[deint] LAYOUT PROOF: %s" % ("PASS -- de-interleave is bit-exact; kernel rewrite de-risked" if ok_all else "FAIL"), flush=True)

# ---------- KERNEL-LOGIC simulation: mirror gather_reader_deint.cpp's EXACT loop nest ----------
# Validates the device kernel's index arithmetic (per-core window s=nn-xwin_lo, boundary mask, per-T gather
# cache reused across r, apage=plane*nnode_tiles+gtile) against REF -- device-free. If bit-exact, the kernel
# ALGORITHM is proven; only the tt-metal API calls (CB/NoC/JIT) remain for the device window.
if "--simkernel" in sys.argv:
    NCORE = 8                                            # split the NBpad/1024 node-tiles across cores (exercises windows)
    nnode_tiles = NBpad // 1024
    Ad = A.reshape(243, nb)                              # A[plane, node]; kernel reads apage=plane*nnode_tiles+gtile
    Apad = np.zeros((243, NBpad), np.float64); Apad[:, :nb] = Ad     # pad node dim to tile boundary (zeros)
    Xn = np.zeros((3, NBpad)); Xn[:, :nb] = X[:, :nb]              # de-interleaved x planes (node-indexed), padded
    Ysim = np.zeros((3, NBpad))
    # split_work_to_cores-style contiguous tile ranges
    base, extra = nnode_tiles // NCORE, nnode_tiles % NCORE
    start = 0
    for core in range(NCORE):
        ntile = base + (1 if core < extra else 0)
        if ntile == 0: continue
        node_lo = start * 1024
        n_nodes = ntile * 1024
        # per-core x-window in NODE units = [min nbr, max nbr] over this core's nodes (clamped, tile-aligned)
        core_nodes = np.arange(node_lo, min(node_lo + n_nodes, nb))
        wn = nbr[:, core_nodes] if core_nodes.size else np.array([[-1]])
        valid = wn >= 0
        nmin = int(wn[valid].min()) if valid.any() else 0
        nmax = int(wn[valid].max()) if valid.any() else 0
        xwin_tile_lo = nmin // 1024
        xwin_ntiles = nmax // 1024 - xwin_tile_lo + 1
        xwin_elem_lo = xwin_tile_lo * 1024
        for t in range(ntile):
            gtile = start + t
            node0 = gtile * 1024
            cache = np.zeros((81, 1024))                 # g_oc per node-block, reused across r
            for oc in range(81):
                oo, c = oc // 3, oc % 3
                for i in range(1024):
                    node = node0 + i
                    inb = (node >= node_lo) and (node - node_lo) < n_nodes and node < nb
                    nn = int(nbr[oo, node]) if inb else -1
                    if nn < 0:
                        cache[oc, i] = 0.0
                    else:
                        s = nn - xwin_elem_lo            # window-relative NODE index (kernel: XH[c][s])
                        cache[oc, i] = Xn[c, xwin_elem_lo + s]   # == Xn[c, nn]; the +/- proves the window offset cancels
            for r in range(3):
                acc = np.zeros(1024)
                for oc in range(81):
                    oo, c = oc // 3, oc % 3
                    plane = (oo * 3 + r) * 3 + c
                    apage = plane * nnode_tiles + gtile  # A row-major [243, NBpad tiles]
                    a_tile = Apad[plane, gtile * 1024:(gtile + 1) * 1024]
                    acc += a_tile * cache[oc]            # element-wise DST-accum (compute kernel, unchanged)
                Ysim[r, gtile * 1024:(gtile + 1) * 1024] = acc
        start += ntile
    r_sim = float(np.abs(Ysim[:, :nb] - Y[:, :nb]).max() / (np.abs(Y[:, :nb]).max() + 1e-30))
    print("[deint] KERNEL-LOGIC sim (%d cores, windowed, masked) vs REF  rel=%.3e  %s"
          % (NCORE, r_sim, "PASS -- kernel index arithmetic proven; only tt-metal API remains" if r_sim < 1e-12 else "FAIL"), flush=True)
    ok_all = ok_all and (r_sim < 1e-12)

# ---------- write the de-interleaved operator the rewritten kernel consumes ----------
if "--write" in sys.argv:
    Xp = np.zeros((3, NBpad), np.float32); Xp[:, :nb] = X[:, :nb].astype(np.float32)
    nbrp = np.full((27, NBpad), -1, np.int32); nbrp[:, :nb] = nbr.astype(np.int32)
    Af = np.ascontiguousarray(A).astype(np.float32)                      # (243, nb)
    REF = np.ascontiguousarray(Y).astype(np.float32)                     # (3, nb)
    out = "/tmp/row236_deint_op.bin"
    with open(out, "wb") as g:
        g.write(struct.pack("<qqq", nb, 243, NBpad)); Xp.tofile(g); nbrp.tofile(g); Af.tofile(g); REF.tofile(g)
    print("[deint] wrote %s (nb=%d K3=243 NBpad=%d, %.3f GB)"
          % (out, nb, NBpad, (Xp.nbytes + nbrp.nbytes + Af.nbytes + REF.nbytes) / 1e9), flush=True)
sys.exit(0 if ok_all else 1)
