import numpy as np, struct, torch, ttnn
# Load the random-v operator (tile-laid ad=coeff[k][e], bd=b_k[e], yc=reference).
f=open("/tmp/row236_real_op.bin","rb"); n_out,K=struct.unpack("<qq",f.read(16)); TE=1024; nA=n_out*K*TE
ad=np.fromfile(f,np.float32,nA).reshape(n_out,K,TE); bd=np.fromfile(f,np.float32,nA).reshape(n_out,K,TE)
yc=np.fromfile(f,np.float32,n_out*TE).reshape(n_out,TE)
def bf16(x):
    u=x.astype(np.float32).copy().view(np.uint32); u=(u+0x8000)&0xFFFF0000; return u.view(np.float32)
# first 32 output elements of tile 0.  Ah[m,k]=coeff_k[m]  Bh[k,n]=b_k[n]  -> diag(Ah@Bh)[m]=sum_k coeff_k[m]*b_k[m]
M=32; Kp=96  # K=81 padded to 96 (3 tiles)
Ah=np.zeros((M,Kp),np.float32); Bh=np.zeros((Kp,M),np.float32)
for k in range(K):
    Ah[:,k]=bf16(ad[0,k,:M]); Bh[k,:]=bf16(bd[0,k,:M])
ref=yc[0,:M]
dev=ttnn.open_device(device_id=0)
tA=ttnn.from_torch(torch.tensor(Ah),dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev)
tB=ttnn.from_torch(torch.tensor(Bh),dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev)
cfg=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,fp32_dest_acc_en=True,packer_l1_acc=True)
tC=ttnn.matmul(tA,tB,dtype=ttnn.float32,compute_kernel_config=cfg)
C=ttnn.to_torch(tC).numpy()
diag=np.array([C[m,m] for m in range(M)])
sc=np.abs(ref).max()
print("matmul-diagonal fp32 accumulate: rel_err=%.3e  (eltwise bf16x1 was 0.38, host fp32 bf16x1=2.3e-3)"%(np.abs(diag-ref).max()/(sc+1e-30)))
print("diag[0:4]=",diag[:4]," ref[0:4]=",ref[:4])
ttnn.close_device(dev)
