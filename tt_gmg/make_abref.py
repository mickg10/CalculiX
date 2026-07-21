# make_abref.py — build a REAL a/b/ref operator file for the spmv_mac on-device MAC correctness check.
# Unlike make_dia.py (coeff-only), this also stores b=gather(x_test) and ref=A*x_test in spmv_mac's exact
# tile layout so the on-device bf16x3 DIA MAC can be validated (real rel_err, not the trivial b=0 case).
#   real_op.bin: <qq>(n_out,K=81), float32 a[n_out,K,1024], then b[n_out,K,1024], then ref[n_out,1024]
#   a[t,k,i]=cf[k,t*1024+i];  b[t,k,i]=bcf[k,t*1024+i];  ref[t,i]=sum_k cf*bcf (= A*x_test element)
# x_test ~ 1e-6 (displacement scale) so spmv_mac's VSCALE=1e6 brings b to O(1) for the bf16 tile engine.
import numpy as np, struct, sys
src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/row236_fine.bin"
f = open(src, "rb")
nb, nblk, nx, ny, nz = struct.unpack("<qqqqq", f.read(40))
ptr = np.fromfile(f, np.int64, nb + 1)
col = np.fromfile(f, np.int32, nblk)
val = np.fromfile(f, np.float64, nblk * 9).reshape(nblk, 3, 3)
ijk = np.fromfile(f, np.int64, nb * 3).reshape(nb, 3)
f.close()
print("[abref] nb=%d nblk=%d lattice=%dx%dx%d" % (nb, nblk, nx, ny, nz), flush=True)
deg = np.diff(ptr).astype(np.int64)
node_of_block = np.repeat(np.arange(nb, dtype=np.int64), deg)
doff = ijk[col] - ijk[node_of_block]
inrange = np.all((doff >= -1) & (doff <= 1), axis=1)
o = ((doff[:, 2] + 1) * 9 + (doff[:, 1] + 1) * 3 + (doff[:, 0] + 1)).astype(np.int64)
bi = np.where(inrange)[0]
ob = o[bi]; nob = node_of_block[bi]
nbr = np.full((27, nb), -1, np.int32); nbr[ob, nob] = col[bi]
n = 3 * nb; n_pad = ((n + 1023) // 1024) * 1024; K = 81; n_out = n_pad // 1024
cf = np.zeros((K, n_pad), np.float32)
for r in range(3):
    for c in range(3):
        np.add.at(cf, (ob * 3 + c, 3 * nob + r), val[bi, r, c].astype(np.float32))
# test vector at displacement scale. --cancel => a rigid x-translation (near-null of the elasticity
# operator) so |Ax|<<|terms| — the EXTREME-CANCELLATION case that broke matmul-diagonal/ttnn (strategy
# lines 163-252). This is the decisive precision test the compensated bf16x3 MAC must pass, NOT a random vector.
CANCEL = ("--cancel" in sys.argv)
rng = np.random.default_rng(7)
xp = np.zeros(n_pad)
if CANCEL:
    xp[0:n:3] = 1e-6                                   # uniform x-displacement on every node = rigid translation
    x = xp[:n].copy()
else:
    x = (1e-6 * rng.standard_normal(n)).astype(np.float64); xp[:n] = x
# gather: bcf[o*3+c, 3*node+r] = xp[3*nbr[o,node]+c]  (broadcast over r; nbr<0 -> 0)
bcf = np.zeros((K, n_pad), np.float32)
idx = np.arange(nb)
for oo in range(27):
    nn = nbr[oo]; valid = nn >= 0
    for c in range(3):
        srcidx = np.where(valid, 3 * nn + c, 0); xg = np.where(valid, xp[srcidx], 0.0).astype(np.float32)
        for r in range(3):
            bcf[oo * 3 + c, 3 * idx + r] = xg
# reference y = sum_k cf ⊙ bcf  (fp64) = A*x_test
ref_cf = (cf.astype(np.float64) * bcf.astype(np.float64)).sum(0)  # (n_pad,)
# cross-check vs a direct BCSR SpMV
xcol = x.reshape(nb, 3)[col]; yb = np.einsum('brc,bc->br', val, xcol)
ybcsr = np.zeros((nb, 3)); np.add.at(ybcsr, node_of_block, yb); ybcsr = ybcsr.ravel()
rel = np.abs(ref_cf[:n] - ybcsr).max() / (np.abs(ybcsr).max() + 1e-30)
print("[abref] ref-vs-BCSR rel=%.3e %s  |ref|max=%.3e" % (rel, "OK" if rel < 1e-10 else "MISMATCH", np.abs(ref_cf).max()), flush=True)
# cancellation depth: the per-element MAC is y[e]=sum_k cf*bcf; term scale vs |y| tells how much cancellation.
# The on-device MAC error must be judged as abs_err / term_scale (product precision), NOT rel-to-|ref|.
term_max = float(np.abs(cf.astype(np.float64) * bcf.astype(np.float64)).max())
ax_max = float(np.abs(ref_cf[:n]).max())
VSCALE = 1e6   # spmv_mac scales b+ref by this; scaled term seen by the MAC = VSCALE*|cf*bcf|
print("[abref] CANCEL=%s  |Ax|max=%.3e  term|cf*bcf|max=%.3e  cancel_ratio|Ax|/term=%.3e  scaled_term(x1e6)=%.3e"
      % (CANCEL, ax_max, term_max, ax_max/(term_max+1e-30), term_max*VSCALE), flush=True)
print("[abref] PASS if on-device abs_err/scaled_term <~1e-4 (fp32 products); FAIL(diverge) if >~1e-2 (bf16 products)", flush=True)
ad = np.ascontiguousarray(cf.reshape(K, n_out, 1024).transpose(1, 0, 2)).astype(np.float32)      # (n_out,K,1024)
bd = np.ascontiguousarray(bcf.reshape(K, n_out, 1024).transpose(1, 0, 2)).astype(np.float32)     # (n_out,K,1024)
refd = np.ascontiguousarray(ref_cf.reshape(n_out, 1024)).astype(np.float32)                       # (n_out,1024)
with open("/tmp/row236_real_op.bin", "wb") as g:
    g.write(struct.pack("<qq", n_out, K)); ad.tofile(g); bd.tofile(g); refd.tofile(g)
print("[abref] wrote a/b/ref real_op.bin: n_out=%d K=%d (%.2f GB)" % (n_out, K, (ad.nbytes * 2 + refd.nbytes) / 1e9), flush=True)
# x_test (the pre-gather vector) for the on-device gather path (spmv_mac SPMV_GATHER): n_pad float32, and the
# padded nbr as int32[27,n_pad_nodes] so a core can index nbr[o*NB + node] directly (NB = n_pad/3 rounded).
xpf = np.ascontiguousarray(xp).astype(np.float32)                       # (n_pad,)  x[3*node+r]
with open("/tmp/row236_x.bin", "wb") as g:
    g.write(struct.pack("<qq", n_pad, 1)); xpf.tofile(g)
NBn = n_pad // 3                                                        # node capacity of the padded vector
nbr_pad = np.full((27, NBn), -1, np.int32); nbr_pad[:, :nb] = nbr
with open("/tmp/row236_nbrpad.bin", "wb") as g:
    g.write(struct.pack("<qq", 27, NBn)); nbr_pad.tofile(g)
print("[abref] wrote x.bin (n_pad=%d) + nbrpad.bin (27 x %d)" % (n_pad, NBn), flush=True)
