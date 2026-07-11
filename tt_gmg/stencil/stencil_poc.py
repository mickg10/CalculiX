# PROOF: reorganize the operator into 27 geometric-offset diagonals (dict-compressed) and apply via dense-box
# SHIFTS (no gather). Must reproduce the exact bspmv. If it does, the fine SpMV is streamable at BW -> gates open.
import numpy as np, time
f=open('/home/ttuser/ttgmg/staged/row236_fine.bin','rb')
h=np.frombuffer(f.read(40),dtype=np.int64); nb,nblk,nx,ny,nz=[int(v) for v in h[:5]]
ptr=np.frombuffer(f.read(8*(nb+1)),dtype=np.int64).copy()
col=np.frombuffer(f.read(4*nblk),dtype=np.int32).copy()
val=np.frombuffer(f.read(8*nblk*9),dtype=np.float64).reshape(nblk,9).copy()
ijk=np.frombuffer(f.read(8*nb*3),dtype=np.int64).reshape(nb,3).copy()
S=ny*nz  # z-fastest lattice strides: lin = i*ny*nz + j*nz + k
lin = ijk[:,0]*S + ijk[:,1]*nz + ijk[:,2]          # box-linear index per active node
box2node = np.full(nx*ny*nz, -1, np.int64); box2node[lin]=np.arange(nb)
# 27 geometric offsets (di,dj,dk) -> box linear delta
offs=[(di,dj,dk) for di in(-1,0,1) for dj in(-1,0,1) for dk in(-1,0,1)]
odelta={o:(o[0]*S+o[1]*nz+o[2]) for o in offs}
# For each node's stored blocks, assign to the geometric-offset diagonal
# diag[o] : per active-node 3x3 block (0 if no neighbor at that offset)
diag={o:np.zeros((nb,9)) for o in offs}
for n in range(nb):
    i,j,k=ijk[n]
    for b in range(ptr[n],ptr[n+1]):
        J=col[b]; o=(int(ijk[J,0]-i),int(ijk[J,1]-j),int(ijk[J,2]-k))
        diag[o][n]=val[b]
# reference bspmv on random x
rng=np.random.default_rng(0); x=rng.standard_normal((nb,3))
yref=np.zeros((nb,3))
for n in range(nb):
    for b in range(ptr[n],ptr[n+1]):
        m=val[b].reshape(3,3); yref[n]+=m@x[col[b]]
# STENCIL-SHIFT apply on the dense box (contiguous shifts, no gather)
t0=time.time()
xbox=np.zeros((nx*ny*nz,3)); xbox[lin]=x
ybox=np.zeros((nx*ny*nz,3))
for o in offs:
    d=odelta[o]; A=diag[o]                        # A on active nodes -> scatter to box
    Abox=np.zeros((nx*ny*nz,9)); Abox[lin]=A
    xs=np.zeros_like(xbox)                        # shift x by -d (neighbor at node+offset)
    if d>=0: xs[:nx*ny*nz-d]=xbox[d:]
    else:    xs[-d:]=xbox[:nx*ny*nz+d]
    ybox += np.einsum("nj,nij->ni", xs, Abox.reshape(-1,3,3))
y=ybox[lin]; dt=time.time()-t0
err=np.abs(y-yref).max()/np.abs(yref).max()
print("stencil-shift vs bspmv rel_err = %.2e  (%s)  apply=%.2fs (numpy, not TT)"%(err,"MATCH -> streamable, gates open" if err<1e-12 else "MISMATCH",dt))
idxbytes=(nx*ny*nz)*27*2/1e9
print("dense-box index-stream = %.2f GB -> at 1.7TB/s ~ %.2f ms/SpMV (vs 45ms gather); dict=101 blocks resident"%(idxbytes, idxbytes/1.7))
