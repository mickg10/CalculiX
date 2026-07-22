import ctypes, numpy as np, struct, torch, ttnn, time, sys
CHECK = "--check" in sys.argv
f=open("/tmp/row236_real_op.bin","rb"); n_out,K=struct.unpack("<qq",f.read(16)); TE=1024; nA=n_out*K*TE
ad=np.fromfile(f,np.float32,nA).reshape(n_out,K,TE)
fn=open("/tmp/row236_nbr.bin","rb"); _,nb=struct.unpack("<qq",fn.read(16)); nbr=np.fromfile(fn,np.int32,27*nb).reshape(27,nb)
n=3*nb; n_pad=n_out*TE; Kp=96; G=n_pad//32
def bf16(x): u=np.ascontiguousarray(x,np.float32).view(np.uint32); return ((u+np.uint32(0x8000))&np.uint32(0xFFFF0000)).view(np.float32)
def split3(v): h=bf16(v); m=bf16(v-h); l=bf16(v-h-m); return h,m,l
cf=np.ascontiguousarray(ad.transpose(1,0,2)).reshape(K,n_pad)
Aflat=np.zeros((Kp,n_pad),np.float32); Aflat[:K]=cf
A=np.ascontiguousarray(Aflat.T.reshape(G,32,Kp)); Ah,Am,Al=split3(A)
print("[tt] loaded n=%d G=%d"%(n,G),flush=True)
dev=ttnn.open_device(device_id=0)
cfg=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,fp32_dest_acc_en=True,packer_l1_acc=True)
CH=4096
def T(x): return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(x)),dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev)
tA=[(T(Ah[i:i+CH]),T(Am[i:i+CH]),T(Al[i:i+CH])) for i in range(0,G,CH)]
print("[tt] A resident",flush=True)
def gather(x):
    b=np.zeros((Kp,n_pad),np.float32)
    for o in range(27):
        nn=nbr[o]
        for c in range(3):
            src=np.where(nn>=0,3*nn+c,-1); xg=np.where(src>=0,x[np.clip(src,0,n-1)],0.0).astype(np.float32)
            k=o*3+c
            for r in range(3): b[k, r:3*nb:3]=xg
    return b
def spmv_core(x):
    b=gather(x); B=np.ascontiguousarray(b.reshape(Kp,G,32).transpose(1,0,2)); Bh,Bm,Bl=split3(B)
    y=np.zeros(n_pad,np.float32)
    for ci,i in enumerate(range(0,G,CH)):
        ah,am,al=tA[ci]; bh=T(Bh[i:i+CH]); bm=T(Bm[i:i+CH]); bl=T(Bl[i:i+CH])
        def mm(a,bb): return ttnn.matmul(a,bb,dtype=ttnn.float32,compute_kernel_config=cfg)
        C=mm(ah,bh)
        for p in (mm(ah,bm),mm(am,bh),mm(ah,bl),mm(am,bm),mm(al,bh)): C=ttnn.add(C,p)
        Cn=ttnn.to_torch(C).float().numpy(); m=Cn.shape[0]
        y[i*32:(i+m)*32]=Cn[:,np.arange(32),np.arange(32)].reshape(-1)
    return y[:n]
if CHECK:
    def probe(name,x):
        b=gather(x); yh=(Aflat[:,:n].astype(np.float64)*b[:,:n].astype(np.float64)).sum(0)
        yt=spmv_core(x).astype(np.float64); ae=np.abs(yt-yh)
        # relative to |A||x| scale (abs err vs input scale) AND vs |y| (output scale)
        axscale=np.abs(Aflat).sum(0)[:n].astype(np.float64); axscale=np.abs(x).max()*axscale.max()
        print("[tt] %-10s |y|max=%.3e  absErr max=%.3e  err/|y|max=%.3e  err/(|A||x|)=%.3e"%(
              name, np.abs(yh).max(), ae.max(), ae.max()/(np.abs(yh).max()+1e-30), ae.max()/(axscale+1e-30)),flush=True)
    probe("random", np.random.default_rng(2).standard_normal(n).astype(np.float32))
    node=np.arange(nb)
    tx=np.zeros(n,np.float32); tx[0::3]=1.0; probe("transl-x", tx)          # rigid-body translation (A*x ~ 0)
    ix,iy,iz=None,None,None
    sm=np.zeros(n,np.float32); sm[0::3]=np.sin(node*1e-4).astype(np.float32); probe("smooth", sm)  # low-freq smooth
    ttnn.close_device(dev); sys.exit(0)
napply=[0]; tapply=[0.0]
def spmv(xp,yp):
    t0=time.time(); x=np.ctypeslib.as_array(xp,(n,)).astype(np.float32)
    y=spmv_core(x); np.ctypeslib.as_array(yp,(n,))[:]=y
    if napply[0]<3:  # runtime accuracy on the ACTUAL solver vectors
        bb=gather(x); yh=(Aflat[:,:n].astype(np.float64)*bb[:,:n].astype(np.float64)).sum(0)
        ae=np.abs(y.astype(np.float64)-yh)
        print("[tt] apply%d ACTUAL-x rel_err=%.3e |x|max=%.3e |y|max=%.3e"%(napply[0],ae.max()/(np.abs(yh).max()+1e-30),np.abs(x).max(),np.abs(yh).max()),flush=True)
    napply[0]+=1; tapply[0]+=time.time()-t0
    if napply[0]<=2 or napply[0]%50==0: print("[tt] applies=%d avg=%.2fs"%(napply[0],tapply[0]/napply[0]),flush=True)
    return 0
CB=ctypes.CFUNCTYPE(ctypes.c_int,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double)); cb=CB(spmv)
lib=ctypes.CDLL("/home/ttuser/ttgmg/libgmg.so")
ctypes.c_void_p.in_dll(lib,"g_tt_fine_spmv").value=ctypes.cast(cb,ctypes.c_void_p).value
ctypes.c_long.in_dll(lib,"g_tt_fine_n").value=n
lib.ccx_gmg_solve_from_dump.restype=ctypes.c_int
lib.ccx_gmg_solve_from_dump.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_double,ctypes.c_int,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double)]
maxu=ctypes.c_double(0.0); t0=time.time()
rc=lib.ccx_gmg_solve_from_dump(b"/tmp/row236_fine.bin",300,1e-6,1,None,ctypes.byref(maxu))
print("[tt] TT-GMG rc=%d maxU=%.6f applies=%d solve=%.1fs avg=%.2fs"%(rc,maxu.value,napply[0],time.time()-t0,tapply[0]/max(1,napply[0])),flush=True)
ttnn.close_device(dev)
