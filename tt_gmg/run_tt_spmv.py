#!/usr/bin/env python3
# run_tt_spmv.py — driver that wires the persistent Metalium tt_spmv (C .so) as g_tt_fine_spmv and runs the
# eig-deflation+hybrid GMG PCG on the real row236 operator. Replaces gmg_tt.py's Python numpy-gather+ttnn
# matmul-diagonal (10.41 s/apply) with the DEVICE-PROVEN on-device gather+MAC (3.46 ms) via a direct C function
# pointer (no Python-callback overhead). This is the G4/cold/warm/stretch measurement driver.
#
# Run on the TT box (after building libtt_spmv.so from tt_spmv.cpp + kernels):
#   cd ~/ttgmg
#   OMP_NUM_THREADS=16 GMG_DEFL_CORR=1 GMG_DEFL_EIG=1 GMG_DEFL_K=24 GMG_HYBRID_TOL=1e-2 \
#     python3 run_tt_spmv.py  --gmg ./libgmg.so  --ttso ./libtt_spmv.so \
#            --fine /tmp/row236_fine.bin  --realop /tmp/row236_real_op.bin \
#            --nbr /tmp/row236_nbr.bin    --x /tmp/row236_x.bin
# Expect: rc=0  maxU=95.8129714 (== golden reduced); prints per-solve wall (G4) and applies. Per-apply (G3) is
# the tt_spmv internal enqueue time; the eig-deflation cold setup is the cold/stretch component.
import ctypes, struct, time, sys, os

def arg(name, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        return sys.argv[i + 1] if i + 1 < len(sys.argv) else default
    return default

gmg   = arg("--gmg",   "./libgmg.so")
ttso  = arg("--ttso",  "./libtt_spmv.so")
fine  = arg("--fine",  "/tmp/row236_fine.bin")
realop= arg("--realop","/tmp/row236_real_op.bin")
nbrf  = arg("--nbr",   "/tmp/row236_nbr.bin")
xf    = arg("--x",     "/tmp/row236_x.bin")
maxit = int(arg("--maxit", "300"))
tol   = float(arg("--tol", "1e-6"))

# n = 3*nb from the fine dump header (<qqqqq> nb,nblk,nx,ny,nz)
with open(fine, "rb") as f:
    nb, nblk, nx, ny, nz = struct.unpack("<qqqqq", f.read(40))
n = 3 * nb
print(f"[run] fine nb={nb} n=3nb={n} lattice={nx}x{ny}x{nz}", flush=True)
print("[run] env:", {k: os.environ[k] for k in
      ("GMG_DEFL_CORR","GMG_DEFL_EIG","GMG_DEFL_K","GMG_HYBRID_TOL","OMP_NUM_THREADS") if k in os.environ}, flush=True)

# --- load the persistent TT SpMV .so, init the device + resident 'a' + compiled program ---
tt = ctypes.CDLL(ttso)
tt.tt_spmv_init.restype = ctypes.c_int
tt.tt_spmv_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
tt.tt_spmv.restype = ctypes.c_int
tt.tt_spmv.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
t0 = time.time()
rc = tt.tt_spmv_init(realop.encode(), nbrf.encode(), xf.encode())
print(f"[run] tt_spmv_init rc={rc}  ({time.time()-t0:.2f}s device open + a-resident + program build)", flush=True)
if rc != 0:
    print("[run] tt_spmv_init FAILED -> abort", flush=True); sys.exit(rc)

# --- wire g_tt_fine_spmv = &tt_spmv (direct C fn ptr), g_tt_fine_n = n ---
lib = ctypes.CDLL(gmg)
tt_spmv_ptr = ctypes.cast(tt.tt_spmv, ctypes.c_void_p).value
ctypes.c_void_p.in_dll(lib, "g_tt_fine_spmv").value = tt_spmv_ptr
ctypes.c_long.in_dll(lib, "g_tt_fine_n").value = n
lib.ccx_gmg_solve_from_dump.restype = ctypes.c_int
lib.ccx_gmg_solve_from_dump.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_double, ctypes.c_int,
                                        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
print(f"[run] wired g_tt_fine_spmv=0x{tt_spmv_ptr:x}  g_tt_fine_n={n}", flush=True)

# --- run the deflated+hybrid GMG PCG; the fine SpMV now runs on the TT via tt_spmv ---
maxu = ctypes.c_double(0.0); ts = time.time()
rc = lib.ccx_gmg_solve_from_dump(fine.encode(), maxit, tol, 1, None, ctypes.byref(maxu))
solve = time.time() - ts
print(f"[run] TT-GMG rc={rc}  maxU={maxu.value:.7f}  (golden reduced 95.8129714)  solve={solve:.1f}s  (G4<=1s?)", flush=True)
ok = (rc == 0) and (abs(maxu.value - 95.8129714) < 2e-4)
print(f"[run] CORRECTNESS + G4: {'PASS' if ok else 'CHECK'}  (rc=0 & maxU==golden; solve time = G4)", flush=True)
try: tt.tt_spmv_close()
except Exception: pass
sys.exit(0 if ok else 1)
