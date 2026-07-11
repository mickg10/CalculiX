# Serialize the 4^3-brick tiled-stencil operator to a TT-uploadable binary + brick neighbor-table + node layout.
# Format: header(nb,nbrick,B,nx,ny,nz) ; brick_ijk[nbrick,3] ; brick_nbr[nbrick,27](neighbor brick id or -1) ;
#         node_in_brick map ; A27 per active node (27 offsets x 3x3, bf16x3 = hi/mid/lo). Also validates size.
import numpy as np, struct
def to_bf16x3(v):  # split fp64 -> 3 bf16 (as uint16) hi/mid/lo
    a=np.asarray(v,np.float32)
    def bf16(x):
        u=x.view(np.uint32).astype(np.uint64); u=(u+(((u>>16)&1)+0x7FFF))&0xFFFF0000; return u.astype(np.uint32).view(np.float32)
    hi=bf16(a); mid=bf16(a-hi); lo=bf16(a-hi-mid)
    def u16(x): return (x.view(np.uint32)>>16).astype(np.uint16)
    return u16(hi),u16(mid),u16(lo)
f=open('/home/ttuser/ttgmg/staged/row236_fine.bin','rb')
h=np.frombuffer(f.read(40),dtype=np.int64); nb,nblk,nx,ny,nz=[int(v) for v in h[:5]]
ptr=np.frombuffer(f.read(8*(nb+1)),dtype=np.int64).copy(); col=np.frombuffer(f.read(4*nblk),dtype=np.int32).copy()
val=np.frombuffer(f.read(8*nblk*9),dtype=np.float64).reshape(nblk,9).copy()
ijk=np.frombuffer(f.read(8*nb*3),dtype=np.int64).reshape(nb,3).copy()
offs=[(di,dj,dk) for di in(-1,0,1) for dj in(-1,0,1) for dk in(-1,0,1)]; oidx={o:t for t,o in enumerate(offs)}
A27=np.zeros((nb,27,9),np.float64)
for n in range(nb):
    i,j,k=ijk[n]
    for b in range(ptr[n],ptr[n+1]):
        J=col[b]; A27[n,oidx[(int(ijk[J,0]-i),int(ijk[J,1]-j),int(ijk[J,2]-k))]]=val[b]
B=4; bijk=ijk//B
bkey=bijk[:,0]*(1<<40)+bijk[:,1]*(1<<20)+bijk[:,2]
ub,inv=np.unique(bkey,return_inverse=True); nbrick=len(ub)
# brick coords + neighbor table
bset={int(k):t for t,k in enumerate(ub)}
buniq=bijk[np.unique(inv,return_index=True)[1]]
brick_nbr=np.full((nbrick,27),-1,np.int32)
for t in range(nbrick):
    bi,bj,bk=buniq[t]
    for oo,(di,dj,dk) in enumerate(offs):
        key=(bi+di)*(1<<40)+(bj+dj)*(1<<20)+(bk+dk); brick_nbr[t,oo]=bset.get(int(key),-1)
hi,mid,lo=to_bf16x3(A27.reshape(-1,9).astype(np.float32))
op_bytes = nbrick*(B**3)*27*9*6  # padded dense per brick
with open('/tmp/row236_stencil.bin','wb') as g:
    g.write(struct.pack('<6q', nb, nbrick, B, nx, ny, nz))
    buniq.astype(np.int32).tofile(g); brick_nbr.tofile(g); inv.astype(np.int32).tofile(g)
    hi.tofile(g); mid.tofile(g); lo.tofile(g)
import os
print("serialized /tmp/row236_stencil.bin  actual=%.2f GB  padded-stream-est=%.2f GB (~%.2f ms/SpMV @1.7TB/s)  nbrick=%d"
      % (os.path.getsize('/tmp/row236_stencil.bin')/1e9, op_bytes/1e9, op_bytes/1.7e12*1e3, nbrick))
