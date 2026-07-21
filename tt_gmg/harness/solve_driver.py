import ctypes, numpy as np, struct, time
fn=open('/tmp/row236_nbr.bin','rb'); _,nb=struct.unpack('<qq',fn.read(16)); nbr=np.fromfile(fn,np.int32,27*nb).reshape(27,nb); fn.close()
n=3*nb; n_pad=((n+1023)//1024)*1024; K=81
f=open('/tmp/row236_real_op.bin','rb'); no,kk=struct.unpack('<qq',f.read(16)); ad=np.fromfile(f,np.float32,no*kk*1024).reshape(no,kk,1024); f.close()
cf=np.ascontiguousarray(ad.transpose(1,0,2).reshape(K,n_pad)); del ad
def cpu_gather(xd):
    xp=np.zeros(n_pad); xp[:n]=xd; yd=np.zeros(n_pad); idx=np.arange(nb)
    for oo in range(27):
        vv=nbr[oo]; ok=vv>=0
        for c in range(3):
            src=np.where(ok,3*vv+c,0); xg=np.where(ok,xp[src],0.0)
            for r in range(3): yd[3*idx+r]+=cf[oo*3+c,3*idx+r]*xg
    return yd[:n]
FINE=b'/home/ttuser/ttgmg/staged/row236_fine.bin'
tt=ctypes.CDLL('/tmp/libtt_spmv.so'); tt.tt_spmv_init.restype=ctypes.c_int; tt.tt_spmv_init.argtypes=[ctypes.c_char_p]*3
tt.tt_spmv.restype=ctypes.c_int; tt.tt_spmv.argtypes=[ctypes.POINTER(ctypes.c_double)]*2
rc=tt.tt_spmv_init(b'/tmp/row236_real_op.bin',b'/tmp/row236_nbr.bin',b'/tmp/row236_nbr.bin'); print('tt_init rc=',rc,flush=True)
if rc!=0: exit(rc)
lib=ctypes.CDLL('/home/ttuser/ttgmg/libgmg.so')
CB=ctypes.CFUNCTYPE(ctypes.c_int,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double)); napply=[0]; tap=[0.0]
def cb_fn(xp,yp):
    t0=time.time(); tt.tt_spmv(xp,yp); napply[0]+=1; tap[0]+=time.time()-t0
    if napply[0]<=4:
        xv=np.ctypeslib.as_array(xp,(n,)).astype(np.float64).copy(); yv=np.ctypeslib.as_array(yp,(n,)).astype(np.float64).copy()
        yh=cpu_gather(xv); rel=np.linalg.norm(yv-yh)/max(np.linalg.norm(yh),1e-30)
        print('[SpMV] apply%d rel_err=%.3e |Ax/x|=%.2e'%(napply[0],rel,np.abs(yh).max()/(np.abs(xv).max()+1e-30)),flush=True)
    if napply[0]%25==0: print('[tt] applies=%d avg=%.1fms'%(napply[0],tap[0]/napply[0]*1000),flush=True)
    return 0
cb=CB(cb_fn); ctypes.c_void_p.in_dll(lib,'g_tt_fine_spmv').value=ctypes.cast(cb,ctypes.c_void_p).value
ctypes.c_long.in_dll(lib,'g_tt_fine_n').value=n
lib.ccx_gmg_solve_from_dump.restype=ctypes.c_int
lib.ccx_gmg_solve_from_dump.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_double,ctypes.c_int,ctypes.c_void_p,ctypes.POINTER(ctypes.c_double)]
maxu=ctypes.c_double(0); t0=time.time()
rc=lib.ccx_gmg_solve_from_dump(FINE,300,1e-6,1,None,ctypes.byref(maxu))
print('[SOLVE] rc=%d maxU=%.7f solve=%.1fs applies=%d avg=%.1fms/apply GOLDEN=95.8129714'%(rc,maxu.value,time.time()-t0,napply[0],tap[0]/max(napply[0],1)*1000),flush=True)
print('DIAG3 DONE',flush=True)
