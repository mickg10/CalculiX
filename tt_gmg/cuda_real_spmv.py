#!/usr/bin/env python3
# cuda_real_spmv.py — run the CUDA DIA-SpMV on the REAL row236 operator (not synthetic) and report rel_err +
# ms/apply. Reads the BCSR dump (row236_fine.bin), builds the component-major DIA (A[243,nb] = a[o*9+r*3+c][node],
# nbr[27,nb]) exactly like make_abref.py, then runs the fused RawKernel. This is C-G3 on the real near-singular
# operator: the fp32 rel_err here (vs a fp64 BCSR reference) is the CUDA analogue of TT's on-device MAC accuracy,
# and it's the SpMV that the shared eig-deflation fp64 PCG will call for the golden maxU run.
import sys, time, struct
import numpy as np
try:
    import cupy as cp
except Exception as e:
    print("NEED_CUPY:", e); sys.exit(2)

src_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/row236_fine.bin"
f = open(src_path, "rb")
nb, nblk, nx, ny, nz = struct.unpack("<qqqqq", f.read(40))
ptr = np.fromfile(f, np.int64, nb + 1); col = np.fromfile(f, np.int32, nblk)
val = np.fromfile(f, np.float64, nblk * 9).reshape(nblk, 3, 3)
ijk = np.fromfile(f, np.int64, nb * 3).reshape(nb, 3); f.close()
print("[real] nb=%d nblk=%d lattice=%dx%dx%d" % (nb, nblk, nx, ny, nz), flush=True)
deg = np.diff(ptr).astype(np.int64); node_of_block = np.repeat(np.arange(nb, dtype=np.int64), deg)
doff = ijk[col] - ijk[node_of_block]; inrange = np.all((doff >= -1) & (doff <= 1), axis=1)
o = ((doff[:,2]+1)*9 + (doff[:,1]+1)*3 + (doff[:,0]+1)).astype(np.int64)
bi = np.where(inrange)[0]; ob = o[bi]; nob = node_of_block[bi]
nbr = np.full((27, nb), -1, np.int32); nbr[ob, nob] = col[bi]
A = np.zeros((243, nb), np.float32)                              # component-major DIA for the CUDA kernel
for r in range(3):
    for c in range(3):
        np.add.at(A, (ob*9 + r*3 + c, nob), val[bi, r, c].astype(np.float32))
x = (1e-6 * np.random.default_rng(7).standard_normal(3*nb)).astype(np.float32)
# fp64 BCSR reference y = A x_true
xcol = x.astype(np.float64).reshape(nb,3)[col]; yb = np.einsum('brc,bc->br', val, xcol)
yref = np.zeros((nb,3)); np.add.at(yref, node_of_block, yb); yref = yref.ravel()
print("[real] built DIA A=%.2fGB  |y|max=%.3e  (stiffness cancellation: |y|/(|A||x|)=%.2e)"
      % (A.nbytes/1e9, np.abs(yref).max(), np.abs(yref).max()/(np.abs(A).max()*np.abs(x).max())), flush=True)

src = r'''
extern "C" __global__ void dia_spmv(const float* __restrict__ A, const int* __restrict__ nbr,
                                    const float* __restrict__ x, float* __restrict__ y, const int nb){
  int n=blockIdx.x*blockDim.x+threadIdx.x; if(n>=nb) return; float y0=0,y1=0,y2=0;
  for(int o=0;o<27;++o){ int nn=nbr[(long)o*nb+n]; if(nn<0) continue;
    float x0=__ldg(&x[3*nn]),x1=__ldg(&x[3*nn+1]),x2=__ldg(&x[3*nn+2]); long b=(long)(o*9)*nb+n;
    y0+=A[b+0*nb]*x0+A[b+1*nb]*x1+A[b+2*nb]*x2; y1+=A[b+3*nb]*x0+A[b+4*nb]*x1+A[b+5*nb]*x2;
    y2+=A[b+6*nb]*x0+A[b+7*nb]*x1+A[b+8*nb]*x2; }
  y[3*n]=y0; y[3*n+1]=y1; y[3*n+2]=y2; }'''
ker=cp.RawKernel(src,"dia_spmv"); thr=256; blk=(nb+thr-1)//thr
dA=cp.asarray(A); dnbr=cp.asarray(nbr); dx=cp.asarray(x); dy=cp.empty(3*nb,cp.float32)
ker((blk,),(thr,),(dA,dnbr,dx,dy,np.int32(nb))); cp.cuda.Stream.null.synchronize()
ycu=cp.asnumpy(dy); abs_err=float(np.abs(ycu.astype(np.float64)-yref).max()); scale=float(np.abs(yref).max())
print("[real] CUDA SpMV on REAL row236: rel_err=%.3e (vs fp64 BCSR)  abs_err=%.3e" % (abs_err/(scale+1e-30), abs_err), flush=True)
N=30; cp.cuda.Stream.null.synchronize(); t=time.time()
for _ in range(N): ker((blk,),(thr,),(dA,dnbr,dx,dy,np.int32(nb)))
cp.cuda.Stream.null.synchronize(); ms=(time.time()-t)/N*1e3
gb=(A.nbytes+x.nbytes+nbr.nbytes+dy.nbytes)/1e9
print("[real] SpMV = %.3f ms/apply  %.0f GB/s  (C-G3<=3ms on the REAL operator)" % (ms, gb/(ms/1e3)), flush=True)
