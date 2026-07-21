# make_dia.py — rebuild the DIA operator (row236_real_op.bin) + neighbor table (row236_nbr.bin)
# from the BCSR fine dump (row236_fine.bin), matching gmg_tt.py's layout:
#   real_op.bin: <qq> (n_out,K=81), then float32 ad[n_out,K,1024] with ad.transpose(1,0,2).reshape(K,n_pad)=cf,
#                cf[o*3+c, 3*node+r] = val_block(node->offset o)[r][c]
#   nbr.bin:     <qq> (0,nb), then int32 nbr[27,nb]; nbr[o,node] = neighbor block index at lattice offset o (or -1)
#   offset index o = (dz+1)*9 + (dy+1)*3 + (dx+1),  (dx,dy,dz) in {-1,0,1}^3
import numpy as np, struct, sys
src = sys.argv[1] if len(sys.argv)>1 else "/tmp/row236_fine.bin"
f=open(src,"rb")
nb,nblk,nx,ny,nz=struct.unpack("<qqqqq",f.read(40))
ptr=np.fromfile(f,np.int64,nb+1)
col=np.fromfile(f,np.int32,nblk)
val=np.fromfile(f,np.float64,nblk*9).reshape(nblk,3,3)
ijk=np.fromfile(f,np.int64,nb*3).reshape(nb,3)
f.close()
print("[dia] nb=%d nblk=%d lattice=%dx%dx%d"%(nb,nblk,nx,ny,nz),flush=True)
deg=np.diff(ptr).astype(np.int64)
node_of_block=np.repeat(np.arange(nb,dtype=np.int64),deg)
doff=ijk[col]-ijk[node_of_block]
inrange=np.all((doff>=-1)&(doff<=1),axis=1)
print("[dia] blocks in 27-stencil: %d / %d (dropped %d)"%(int(inrange.sum()),nblk,int(nblk-inrange.sum())),flush=True)
o=((doff[:,2]+1)*9+(doff[:,1]+1)*3+(doff[:,0]+1)).astype(np.int64)
bi=np.where(inrange)[0]
ob=o[bi]; nob=node_of_block[bi]
# neighbor table
nbr=np.full((27,nb),-1,np.int32)
nbr[ob,nob]=col[bi]
# coefficient planes cf[K,n_pad]
n=3*nb; n_pad=((n+1023)//1024)*1024; K=81; n_out=n_pad//1024
cf=np.zeros((K,n_pad),np.float32)
for r in range(3):
    for c in range(3):
        np.add.at(cf,(ob*3+c,3*nob+r),val[bi,r,c].astype(np.float32))
# --- fp64 self-check: DIA SpMV must equal BCSR SpMV on a random vector ---
dup=len(ob)-len(set(zip(ob.tolist(),nob.tolist())))
print("[dia] duplicate (offset,node) pairs: %d (must be 0)"%dup,flush=True)
rng=np.random.default_rng(0); x=rng.standard_normal(n).astype(np.float64)
xcol=x.reshape(nb,3)[col]; yb=np.einsum('brc,bc->br',val,xcol)
ybcsr=np.zeros((nb,3)); np.add.at(ybcsr,node_of_block,yb); ybcsr=ybcsr.ravel()
xp=np.zeros(n_pad); xp[:n]=x; ydia=np.zeros(n_pad); idx=np.arange(nb)
for oo in range(27):
    nn=nbr[oo]; valid=nn>=0
    for c in range(3):
        src=np.where(valid,3*nn+c,0); xg=np.where(valid,xp[src],0.0)
        for r in range(3): ydia[3*idx+r]+=cf[oo*3+c,3*idx+r]*xg
rel=np.abs(ydia[:n]-ybcsr).max()/(np.abs(ybcsr).max()+1e-30)
print("[dia] self-check DIA-vs-BCSR rel_err=%.3e  %s"%(rel,"OK" if rel<1e-10 else "MISMATCH!"),flush=True)
if rel>=1e-5: sys.exit(1)
ad=np.ascontiguousarray(cf.reshape(K,n_out,1024).transpose(1,0,2))   # (n_out,K,1024)
with open("/tmp/row236_real_op.bin","wb") as g:
    g.write(struct.pack("<qq",n_out,K)); ad.tofile(g)
with open("/tmp/row236_nbr.bin","wb") as g:
    g.write(struct.pack("<qq",0,nb)); nbr.tofile(g)
print("[dia] wrote real_op.bin (n_out=%d K=%d -> %.2f GB) + nbr.bin (nb=%d)"%(n_out,K,ad.nbytes/1e9,nb),flush=True)
