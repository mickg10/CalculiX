import ctypes, numpy as np, time
lib=ctypes.CDLL("/home/ttuser/ttgmg/libgmg.so")
lib.ccx_gmg_solve_from_dump.restype=ctypes.c_int
lib.ccx_gmg_solve_from_dump.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_double,ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double)]
maxu=ctypes.c_double(0.0)
t0=time.time()
rc=lib.ccx_gmg_solve_from_dump(b"/tmp/row236_fine.bin",500,1e-6,1,None,ctypes.byref(maxu))
print("ctypes bridge OK: rc=%d maxU=%.6f (%.1fs) [CPU hook=null -> converges]"%(rc,maxu.value,time.time()-t0))
