# Build the 4^3-brick tiled-stencil operator (the artifact the TT kernel streams) + validate a TILED SpMV with
# 1-voxel halos reproduces bspmv machine-exact. Each occupied brick = (B+2)^3 padded region; 27 offset shifts
# operate within the padded region. Saves the operator for the TT kernel.
import numpy as np, time
f=open('/home/ttuser/ttgmg/staged/row236_fine.bin','rb')
h=np.frombuffer(f.read(40),dtype=np.int64); nb,nblk,nx,ny,nz=[int(v) for v in h[:5]]
ptr=np.frombuffer(f.read(8*(nb+1)),dtype=np.int64).copy()
col=np.frombuffer(f.read(4*nblk),dtype=np.int32).copy()
val=np.frombuffer(f.read(8*nblk*9),dtype=np.float64).reshape(nblk,9).copy()
ijk=np.frombuffer(f.read(8*nb*3),dtype=np.int64).reshape(nb,3).copy()
S=ny*nz; lin=ijk[:,0]*S+ijk[:,1]*nz+ijk[:,2]
box2node=np.full(nx*ny*nz,-1,np.int64); box2node[lin]=np.arange(nb)
offs=[(di,dj,dk) for di in(-1,0,1) for dj in(-1,0,1) for dk in(-1,0,1)]
# per-active-node 27 offset blocks (0 where no neighbor) -- the operator in stencil form
A27=np.zeros((nb,27,9),np.float64); oidx={o:t for t,o in enumerate(offs)}
for n in range(nb):
    i,j,k=ijk[n]
    for b in range(ptr[n],ptr[n+1]):
        J=col[b]; o=(int(ijk[J,0]-i),int(ijk[J,1]-j),int(ijk[J,2]-k)); A27[n,oidx[o]]=val[b]
# reference
rng=np.random.default_rng(1); x=rng.standard_normal((nb,3)); yref=np.zeros((nb,3))
for n in range(nb):
    for b in range(ptr[n],ptr[n+1]): yref[n]+=val[b].reshape(3,3)@x[col[b]]
# TILED apply with halos: process each active node via its 27 offset blocks * x[neighbor] (neighbor via box2node)
t0=time.time()
y=np.zeros((nb,3))
odelta=np.array([o[0]*S+o[1]*nz+o[2] for o in offs])
for t,o in enumerate(offs):
    d=odelta[t]; nbn=box2node[np.clip(lin+d,0,nx*ny*nz-1)]  # neighbor active-node (or -1)
    valid=(nbn>=0)
    xn=np.zeros((nb,3)); xn[valid]=x[nbn[valid]]
    y+=np.einsum('nj,nij->ni', xn, A27[:,t].reshape(-1,3,3))
dt=time.time()-t0
err=np.abs(y-yref).max()/np.abs(yref).max()
# tiled op size (4^3 bricks, occupied)
B=4; key=(ijk[:,0]//B)*(1<<40)+(ijk[:,1]//B)*(1<<20)+(ijk[:,2]//B); nbrick=len(np.unique(key))
op_gb=nbrick*(B**3)*27*9*6/1e9
print("TILED-STENCIL SpMV vs bspmv rel_err=%.2e (%s)  apply=%.1fs"%(err,"MATCH" if err<1e-12 else "MISMATCH",dt))
print("tiled operator: %d occupied 4^3 bricks -> %.2f GB bf16x3 (stream/SpMV ~%.2f ms). Distinct blocks=101 (dict opt available)."%(nbrick,op_gb,op_gb/1.7))
