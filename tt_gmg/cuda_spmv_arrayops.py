#!/usr/bin/env python3
# cuda_spmv_arrayops.py — CUDA DIA SpMV via CuPy ARRAY OPS (no RawKernel/nvrtc, so no CUDA-header dependency).
# Proves C-G3 CORRECTNESS on the GPU + a timing baseline while the header setup for the fused RawKernel is sorted.
# Gather = cupy fancy-indexing (xr[idx]); MAC = broadcast multiply-add. These are precompiled cupy kernels.
# Same synthetic 27-pt 3x3-block operator as cuda_spmv_probe.py.
import sys, time
import numpy as np
try:
    import cupy as cp
except Exception as e:
    print("NEED_CUPY:", e); sys.exit(2)

nb = int(sys.argv[1]) if len(sys.argv) > 1 else 1290738
rng = np.random.default_rng(7)
nx = int(round(nb ** (1/3))); ny = nx; nz = (nb + nx*ny - 1)//(nx*ny)
ii = np.arange(nb); zx = ii//(nx*ny); yx = (ii % (nx*ny))//nx; xx = ii % nx
nbr = np.full((27, nb), -1, np.int32)
for o in range(27):
    dz, dy, dx = o//9-1, (o//3)%3-1, o%3-1
    zz, yy, xw = zx+dz, yx+dy, xx+dx
    ok = (zz>=0)&(zz<nz)&(yy>=0)&(yy<ny)&(xw>=0)&(xw<nx); nn = zz*nx*ny+yy*nx+xw
    ok &= (nn>=0)&(nn<nb); nbr[o, ok] = nn[ok]
A = (rng.standard_normal((243, nb)).astype(np.float32) * 1e6)
x = (1e-6 * rng.standard_normal(3*nb)).astype(np.float32)
print("[cuda-ao] nb=%d  A=%.2fGB" % (nb, A.nbytes/1e9), flush=True)

# numpy fp64 reference
xr = x.astype(np.float64).reshape(nb,3); yref = np.zeros((nb,3))
for o in range(27):
    nn=nbr[o]; v=nn>=0; xn=np.zeros((nb,3)); xn[v]=xr[nn[v]]
    for r in range(3):
        for c in range(3): yref[:,r]+=A[o*9+r*3+c].astype(np.float64)*xn[:,c]
yref=yref.ravel()

# ---- GPU SpMV via cupy array ops ----
dA=cp.asarray(A); dnbr=cp.asarray(nbr); dxr=cp.asarray(x).reshape(nb,3)
def spmv():
    y=cp.zeros((nb,3),cp.float32)
    for o in range(27):
        nn=dnbr[o]; valid=nn>=0; idx=cp.where(valid,nn,0); xn=dxr[idx]
        xn=cp.where(valid[:,None],xn,0.)
        b=o*9
        for r in range(3):
            y[:,r]+= dA[b+r*3+0]*xn[:,0]+dA[b+r*3+1]*xn[:,1]+dA[b+r*3+2]*xn[:,2]
    return y.ravel()
ycu=cp.asnumpy(spmv()); cp.cuda.Stream.null.synchronize()
abs_err=float(np.abs(ycu.astype(np.float64)-yref).max()); scale=float(np.abs(yref).max())
term=float(np.abs(A).max()*np.abs(x).max())
print("[cuda-ao] correctness: rel_err=%.3e  abs_err/term=%.3e (fp32-class ~1e-6 expected -> kernel math correct)"%(abs_err/(scale+1e-30),abs_err/term),flush=True)
N=20; cp.cuda.Stream.null.synchronize(); t=time.time()
for _ in range(N): spmv()
cp.cuda.Stream.null.synchronize(); ms=(time.time()-t)/N*1e3
print("[cuda-ao] SpMV(array-ops) = %.3f ms/apply  (launch-bound, NOT the fused-kernel number; RawKernel target <=3ms BW-bound)"%ms,flush=True)
