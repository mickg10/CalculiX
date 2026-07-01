import numpy as np, struct, torch, ttnn
f=open("/tmp/row236_real_op.bin","rb"); n_out,K=struct.unpack("<qq",f.read(16)); TE=1024; nA=n_out*K*TE
ad=np.fromfile(f,np.float32,nA).reshape(n_out,K,TE); bd=np.fromfile(f,np.float32,nA).reshape(n_out,K,TE)
yc=np.fromfile(f,np.float32,n_out*TE).reshape(n_out,TE)
def bf16(x):
    u=x.astype(np.float32).copy().view(np.uint32); u=(u+0x8000)&0xFFFF0000; return u.view(np.float32)
def split3(v): h=bf16(v); m=bf16(v-h); l=bf16(v-h-m); return h,m,l
M=32; Kp=96
ah=np.zeros((M,Kp),np.float32);am=ah.copy();al=ah.copy(); bh=np.zeros((Kp,M),np.float32);bm=bh.copy();bl=bh.copy()
for k in range(K):
    a_h,a_m,a_l=split3(ad[0,k,:M]); b_h,b_m,b_l=split3(bd[0,k,:M])
    ah[:,k]=a_h;am[:,k]=a_m;al[:,k]=a_l; bh[k,:]=b_h;bm[k,:]=b_m;bl[k,:]=b_l
ref=yc[0,:M]
dev=ttnn.open_device(device_id=0)
cfg=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,fp32_dest_acc_en=True,packer_l1_acc=True)
def T(x): return ttnn.from_torch(torch.tensor(x),dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev)
tah,tam,tal,tbh,tbm,tbl=T(ah),T(am),T(al),T(bh),T(bm),T(bl)
def mm(A,B): return ttnn.to_torch(ttnn.matmul(A,B,dtype=ttnn.float32,compute_kernel_config=cfg)).numpy()
# 6 cross-terms level-sum<=2, summed in fp32 (numpy) then diagonal
C = mm(tah,tbh)+mm(tah,tbm)+mm(tam,tbh)+mm(tah,tbl)+mm(tam,tbm)+mm(tal,tbh)
diag=np.array([C[m,m] for m in range(M)]); sc=np.abs(ref).max()
print("bf16x3 matmul-diagonal: rel_err=%.3e  (host bf16x3=2.6e-7; eltwise TT=1.5)"%(np.abs(diag-ref).max()/(sc+1e-30)))
ttnn.close_device(dev)
