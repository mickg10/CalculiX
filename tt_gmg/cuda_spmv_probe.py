#!/usr/bin/env python3
# cuda_spmv_probe.py — CUDA backend C-G3 core: the DIA fine-SpMV as a CuPy RawKernel (runtime-compiled via
# nvrtc, so NO nvcc/toolkit install needed — just `pip install cupy-cuda12x`). Validates correctness vs a
# numpy reference on a SYNTHETIC 27-point 3x3-block elasticity-like operator (no 2.6GB transfer needed for the
# kernel bring-up), then times it. This is the CUDA analogue of the TT gather+MAC — but native fp32 + native
# SIMT gather, so it should be BW-bound and simple (Lessons L1/L2: CUDA erases TT's two hard problems).
#
# Operator (matches make_dia.py's DIA): y[3n+r] = sum_o sum_c A[k=o*9+r*3+c][n] * x[3*nbr[o][n]+c], k in [0,243).
# Layout is PLANE-major A[k*nb + n] so consecutive threads (nodes) read consecutive addresses -> coalesced.
import sys, time
import numpy as np
try:
    import cupy as cp
except Exception as e:
    print("NEED_CUPY: pip install cupy-cuda12x  (%s)" % e); sys.exit(2)

# ---- synthetic row236-scale operator: nb nodes on a lattice, 27-pt stencil, random-but-symmetric 3x3 blocks ----
nb = int(sys.argv[1]) if len(sys.argv) > 1 else 1290738          # row236 fine block-rows
rng = np.random.default_rng(7)
nx = int(round(nb ** (1/3))); ny = nx; nz = (nb + nx*ny - 1)//(nx*ny)   # a box that covers nb
# nbr[o,n]: 27-pt lattice neighbors (node index = z*nx*ny + y*nx + x); -1 if out of box or >= nb
ii = np.arange(nb); zx = ii // (nx*ny); yx = (ii % (nx*ny)) // nx; xx = ii % nx
nbr = np.full((27, nb), -1, np.int32)
for o in range(27):
    dz, dy, dx = o//9 - 1, (o//3)%3 - 1, o%3 - 1
    zz, yy, xw = zx+dz, yx+dy, xx+dx
    ok = (zz>=0)&(zz<nz)&(yy>=0)&(yy<ny)&(xw>=0)&(xw<nx)
    nn = zz*nx*ny + yy*nx + xw
    ok &= (nn>=0)&(nn<nb); nbr[o, ok] = nn[ok]
A = (rng.standard_normal((243, nb)).astype(np.float32) * 1e6)    # stiffness-scale coeffs (~1e6, like real)
x = (1e-6 * rng.standard_normal(3*nb)).astype(np.float32)        # displacement-scale vector
print("[cuda] nb=%d lattice~%dx%dx%d  A=%.2fGB x=%.1fMB" % (nb, nx, ny, nz, A.nbytes/1e9, x.nbytes/1e6), flush=True)

# ---- numpy reference (fp64) ----
t=time.time()
xr = x.astype(np.float64).reshape(nb,3)
yref = np.zeros((nb,3))
for o in range(27):
    nn = nbr[o]; v = nn>=0; xn = np.zeros((nb,3)); xn[v] = xr[nn[v]]
    for r in range(3):
        for c in range(3):
            yref[:,r] += A[o*9+r*3+c].astype(np.float64) * xn[:,c]
yref = yref.ravel(); print("[cuda] numpy ref built in %.1fs" % (time.time()-t), flush=True)

# ---- CuPy RawKernel: one thread per node, gather x[3*nbr+c], 3x3 block MAC, write y[3n+r] ----
src = r'''
extern "C" __global__ void dia_spmv(const float* __restrict__ A, const int* __restrict__ nbr,
                                    const float* __restrict__ x, float* __restrict__ y, const int nb) {
  int n = blockIdx.x * blockDim.x + threadIdx.x; if (n >= nb) return;
  float y0=0.f, y1=0.f, y2=0.f;
  for (int o=0; o<27; ++o) {
    int nn = nbr[(long)o*nb + n]; if (nn < 0) continue;
    float x0 = __ldg(&x[3*nn+0]), x1 = __ldg(&x[3*nn+1]), x2 = __ldg(&x[3*nn+2]);   // native SIMT gather (L2)
    long b = (long)(o*9)*nb + n;
    y0 += A[b+0*nb]*x0 + A[b+1*nb]*x1 + A[b+2*nb]*x2;
    y1 += A[b+3*nb]*x0 + A[b+4*nb]*x1 + A[b+5*nb]*x2;
    y2 += A[b+6*nb]*x0 + A[b+7*nb]*x1 + A[b+8*nb]*x2;
  }
  y[3*n+0]=y0; y[3*n+1]=y1; y[3*n+2]=y2;
}'''
ker = cp.RawKernel(src, "dia_spmv")
dA = cp.asarray(A); dnbr = cp.asarray(nbr); dx = cp.asarray(x); dy = cp.zeros(3*nb, cp.float32)
thr = 256; blk = (nb + thr - 1)//thr
ker((blk,), (thr,), (dA, dnbr, dx, dy, np.int32(nb))); cp.cuda.Stream.null.synchronize()   # warmup
ycu = cp.asnumpy(dy)
abs_err = float(np.abs(ycu.astype(np.float64) - yref).max()); scale = float(np.abs(yref).max())
term = float(np.abs(A).max()*np.abs(x).max())
print("[cuda] correctness: rel_err=%.3e  abs_err/term=%.3e (fp32-class ~1e-6 expected)" % (abs_err/(scale+1e-30), abs_err/term), flush=True)
# ---- time (matrix-only DRAM: A=%.2fGB) ----
N=50; cp.cuda.Stream.null.synchronize(); t=time.time()
for _ in range(N): ker((blk,), (thr,), (dA, dnbr, dx, dy, np.int32(nb)))
cp.cuda.Stream.null.synchronize(); ms=(time.time()-t)/N*1e3
gb = (A.nbytes + x.nbytes + nbr.nbytes + dy.nbytes)/1e9
print("[cuda] SpMV = %.3f ms/apply   %.0f GB/s   (G3 target <=3ms; RTX3090 ~936 GB/s roofline)" % (ms, gb/(ms/1e3)), flush=True)
