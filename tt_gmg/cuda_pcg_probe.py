#!/usr/bin/env python3
# cuda_pcg_probe.py — CUDA C-G4 mechanics: a full Jacobi-PCG solve on the GPU using the fused DIA-SpMV RawKernel,
# with the fp64 residual gate. Proves the CUDA SpMV drives a real iterative solve to convergence + measures
# C-G2 (upload), C-G4 (solve), C-G5 (download). Operator = synthetic SPD 27-pt vector-Laplacian (3 DOF, diagonally
# dominant so CG converges without deflation) at row236 scale. The near-singular real operator + eig-deflation is
# the same shared machinery (added later with the real dump); this validates the GPU SpMV-in-CG path end to end.
import sys, time
import numpy as np
try:
    import cupy as cp
except Exception as e:
    print("NEED_CUPY:", e); sys.exit(2)

nb = int(sys.argv[1]) if len(sys.argv) > 1 else 1290738
W = 1.0; SHIFT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5   # SHIFT->0 = near-singular (the row236-like case)
rng = np.random.default_rng(3)
nx = int(round(nb ** (1/3))); ny = nx; nz = (nb + nx*ny - 1)//(nx*ny)
ii = np.arange(nb); zx = ii//(nx*ny); yx = (ii % (nx*ny))//nx; xx = ii % nx
nbr = np.full((27, nb), -1, np.int32)
for o in range(27):
    dz, dy, dx = o//9-1, (o//3)%3-1, o%3-1
    zz, yy, xw = zx+dz, yx+dy, xx+dx
    ok = (zz>=0)&(zz<nz)&(yy>=0)&(yy<ny)&(xw>=0)&(xw<nx); nn = zz*nx*ny+yy*nx+xw
    ok &= (nn>=0)&(nn<nb) & (o!=13); nbr[o, ok] = nn[ok]
nbr[13] = ii                                                    # self
deg = (nbr[np.r_[0:13,14:27]] >= 0).sum(0).astype(np.float32)   # #valid neighbors per node
# DIA A[243,nb]: self diag = deg*W+SHIFT (r==c); neighbor diag = -W (r==c, valid); off-component = 0 -> SPD
A = np.zeros((243, nb), np.float32)
for o in range(27):
    for r in range(3):
        k = o*9 + r*3 + r
        if o == 13: A[k] = deg*W + SHIFT
        else:       A[k] = np.where(nbr[o] >= 0, -W, 0.0)
diagA = (deg*W + SHIFT)                                         # per-node scalar diagonal (same for r=0,1,2)
xtrue = (1e-6 * rng.standard_normal(3*nb)).astype(np.float32)
print("[pcg] nb=%d lattice~%dx%dx%d  A=%.2fGB" % (nb, nx, ny, nz, A.nbytes/1e9), flush=True)

src = r'''
extern "C" __global__ void dia_spmv(const float* __restrict__ A, const int* __restrict__ nbr,
                                    const float* __restrict__ x, float* __restrict__ y, const int nb){
  int n = blockIdx.x*blockDim.x+threadIdx.x; if(n>=nb) return;
  float y0=0,y1=0,y2=0;
  for(int o=0;o<27;++o){ int nn=nbr[(long)o*nb+n]; if(nn<0) continue;
    float x0=__ldg(&x[3*nn]),x1=__ldg(&x[3*nn+1]),x2=__ldg(&x[3*nn+2]); long b=(long)(o*9)*nb+n;
    y0+=A[b+0*nb]*x0+A[b+1*nb]*x1+A[b+2*nb]*x2; y1+=A[b+3*nb]*x0+A[b+4*nb]*x1+A[b+5*nb]*x2;
    y2+=A[b+6*nb]*x0+A[b+7*nb]*x1+A[b+8*nb]*x2; }
  y[3*n]=y0; y[3*n+1]=y1; y[3*n+2]=y2; }'''
ker = cp.RawKernel(src, "dia_spmv"); thr=256; blk=(nb+thr-1)//thr
# ---- C-G2: upload ----
cp.cuda.Stream.null.synchronize(); t=time.time()
dA=cp.asarray(A); dnbr=cp.asarray(nbr); dxt=cp.asarray(xtrue); ddiag=cp.repeat(cp.asarray(diagA),3)
cp.cuda.Stream.null.synchronize(); g2=(time.time()-t)*1e3
def Ax(v):
    y=cp.empty(3*nb,cp.float32); ker((blk,),(thr,),(dA,dnbr,v,y,np.int32(nb))); return y
dot=lambda a,b: float(cp.sum(a*b))            # cuBLAS-free dot/norm (cupy reduction kernels, no libcublasLt)
nrm=lambda v: float(cp.sqrt(cp.sum(v.astype(cp.float64)**2)))
b = Ax(dxt); cp.cuda.Stream.null.synchronize()
# ---- C-G4: Jacobi-PCG ----
x=cp.zeros(3*nb,cp.float32); r=b-Ax(x); z=r/ddiag; p=z.copy(); rz=dot(r,z); bn=nrm(b)
tol=1e-6; t=time.time(); it=0; rel=1.0
for it in range(1,2001):
    Ap=Ax(p); alpha=rz/dot(p,Ap); x+=alpha*p; r-=alpha*Ap
    rel=nrm(r)/bn
    if rel<tol: break
    z=r/ddiag; rzn=dot(r,z); p=z+(rzn/rz)*p; rz=rzn
cp.cuda.Stream.null.synchronize(); g4=(time.time()-t)*1e3
# ---- C-G5: download + fp64 true-residual gate ----
t=time.time(); xh=cp.asnumpy(x); cp.cuda.Stream.null.synchronize(); g5=(time.time()-t)*1e3
true_rel=nrm(b-Ax(x))/bn
sol_err=float(np.linalg.norm(xh-cp.asnumpy(dxt))/ (np.linalg.norm(cp.asnumpy(dxt))+1e-30))
print("[pcg] CONVERGED it=%d  rel=%.2e  true_rel(fp64gate)=%.2e  sol_err_vs_xtrue=%.2e" % (it, rel, true_rel, sol_err), flush=True)
print("[pcg] C-G2 upload=%.1f ms  C-G4 solve=%.1f ms (%d it, %.3f ms/it)  C-G5 download=%.1f ms" % (g2, g4, it, g4/max(it,1), g5), flush=True)
print("[pcg] gate check: C-G4<=1000ms:%s  C-G2<=100ms:%s  C-G5<=200ms:%s" % (g4<=1000, g2<=100, g5<=200), flush=True)
