#!/usr/bin/env python3
# cuda_gmg_bridge.py — NVIDIA C-COR golden run: the CUDA analogue of gmg_tt.py. ctypes-loads libgmg.so and sets
# g_tt_fine_spmv -> a cupy fused-DIA-SpMV callback, so the shared fp64 outer PCG + eig-deflation + hybrid + true-
# residual gate (all in gmg_solve.cpp, unchanged) run the fine-level A0*x on the RTX GPU. The fine operator is the
# 27-pt stencil (make_dia self-check: DIA == BCSR to float32), so the DIA SpMV IS A0*x. CUDA fp32 (rel_err 2e-7) is
# far better than TT's bf16x3 5e-4 floor, so it converges with the SAME or LESS deflation, and the fp64 host gate
# guarantees a wrong GPU result can never be accepted (falls back to CPU GMG).
#   run:  CUDA_PATH=/tmp/ch GMG_DEFL_CORR=1 GMG_DEFL_EIG=1 GMG_DEFL_K=24 GMG_HYBRID_TOL=1e-2 \
#         python3 cuda_gmg_bridge.py /tmp/row236_fine.bin /path/to/libgmg.so
# expect: rc=0  maxU=95.8129714 (reduced golden).  This is the turn-key golden run once operator+libgmg.so are staged.
import ctypes, numpy as np, struct, time, sys
import cupy as cp

dump = sys.argv[1] if len(sys.argv) > 1 else "/tmp/row236_fine.bin"
libp = sys.argv[2] if len(sys.argv) > 2 else "/tmp/libgmg.so"
# ---- build component-major DIA (A[243,nb]+nbr[27,nb]) from the BCSR dump (same as cuda_real_spmv.py) ----
f = open(dump, "rb"); nb, nblk, nx, ny, nz = struct.unpack("<qqqqq", f.read(40))
ptr = np.fromfile(f, np.int64, nb+1); col = np.fromfile(f, np.int32, nblk)
val = np.fromfile(f, np.float64, nblk*9).reshape(nblk, 3, 3)
ijk = np.fromfile(f, np.int64, nb*3).reshape(nb, 3); f.close()
deg = np.diff(ptr).astype(np.int64); nofb = np.repeat(np.arange(nb, dtype=np.int64), deg)
doff = ijk[col] - ijk[nofb]; inr = np.all((doff >= -1) & (doff <= 1), axis=1)
o = ((doff[:,2]+1)*9 + (doff[:,1]+1)*3 + (doff[:,0]+1)).astype(np.int64)
bi = np.where(inr)[0]; ob = o[bi]; nob = nofb[bi]
nbr = np.full((27, nb), -1, np.int32); nbr[ob, nob] = col[bi]
A = np.zeros((243, nb), np.float32)
for r in range(3):
    for c in range(3): np.add.at(A, (ob*9+r*3+c, nob), val[bi, r, c].astype(np.float32))
n = 3*nb
print("[cuda-gmg] nb=%d n=%d  DIA A=%.2fGB uploaded" % (nb, n, A.nbytes/1e9), flush=True)
dA = cp.asarray(A); dnbr = cp.asarray(nbr); dx = cp.empty(n, cp.float32); dy = cp.empty(n, cp.float32)

src = r'''
extern "C" __global__ void dia_spmv(const float* __restrict__ A, const int* __restrict__ nbr,
                                    const float* __restrict__ x, float* __restrict__ y, const int nb){
  int m=blockIdx.x*blockDim.x+threadIdx.x; if(m>=nb) return; float y0=0,y1=0,y2=0;
  for(int o=0;o<27;++o){ int nn=nbr[(long)o*nb+m]; if(nn<0) continue;
    float x0=__ldg(&x[3*nn]),x1=__ldg(&x[3*nn+1]),x2=__ldg(&x[3*nn+2]); long b=(long)(o*9)*nb+m;
    y0+=A[b+0*nb]*x0+A[b+1*nb]*x1+A[b+2*nb]*x2; y1+=A[b+3*nb]*x0+A[b+4*nb]*x1+A[b+5*nb]*x2;
    y2+=A[b+6*nb]*x0+A[b+7*nb]*x1+A[b+8*nb]*x2; }
  y[3*m]=y0; y[3*m+1]=y1; y[3*m+2]=y2; }'''
ker = cp.RawKernel(src, "dia_spmv"); thr = 256; blk = (nb+thr-1)//thr
nap = 0
def spmv(xp, yp):
    global nap; nap += 1
    x = np.ctypeslib.as_array(xp, (n,)).astype(np.float32)     # fp64 (GMG) -> fp32 (GPU)
    dx.set(x); ker((blk,), (thr,), (dA, dnbr, dx, dy, np.int32(nb)))
    np.ctypeslib.as_array(yp, (n,))[:] = cp.asnumpy(dy).astype(np.float64)   # fp32 -> fp64 back to the host PCG
    return 0
CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)); cb = CB(spmv)
lib = ctypes.CDLL(libp)
ctypes.c_void_p.in_dll(lib, "g_tt_fine_spmv").value = ctypes.cast(cb, ctypes.c_void_p).value
ctypes.c_long.in_dll(lib, "g_tt_fine_n").value = n
lib.ccx_gmg_solve_from_dump.restype = ctypes.c_int
lib.ccx_gmg_solve_from_dump.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_double, ctypes.c_int,
                                        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
maxu = ctypes.c_double(0.0); t0 = time.time()
rc = lib.ccx_gmg_solve_from_dump(dump.encode(), 500, 1e-6, 1, None, ctypes.byref(maxu))
print("[cuda-gmg] rc=%d maxU=%.7f  (golden reduced 95.8129714)  fine-applies=%d  wall=%.2fs" % (rc, maxu.value, nap, time.time()-t0), flush=True)
