# CORRECT compensated-MAC emulator: fp32 dst accumulate over all 6*K (or 5*K) cross-terms in device order
# (k-outer, cross-inner). Tests bf16x2-b (b in 2 levels -> 5 terms, 1/3 less gather write) vs proven bf16x3.
import numpy as np
def to_bf16(x):
    a = np.atleast_1d(np.asarray(x, np.float32)); u = a.view(np.uint32).astype(np.uint64)
    u = (u + (((u >> np.uint64(16)) & np.uint64(1)) + np.uint64(0x7FFF))) & np.uint64(0xFFFF0000)
    return u.astype(np.uint32).view(np.float32).reshape(a.shape)
def split3(v):
    vh=to_bf16(v); vm=to_bf16(v-vh); vl=to_bf16(v-vh-vm); return vh,vm,vl
K=81
def mac(cf,b,terms):                              # terms: list of (a_level_idx,b_level_idx); 0=h,1=m,2=l
    ah,am,al=split3(cf); bh,bm,bl=split3(b)
    A=[ah,am,al]; B=[bh,bm,bl]; N=cf.shape[1]
    y=np.zeros(N,np.float32)
    for k in range(K):
        for (ai,bi) in terms: y=(y+(A[ai][k]*B[bi][k]).astype(np.float32)).astype(np.float32)
    return y
T6=[(0,0),(0,1),(1,0),(0,2),(1,1),(2,0)]          # bf16x3: 6 cross-terms
T5=[(0,0),(0,1),(1,0),(2,0),(1,1)]                # bf16x2-b: drop (0,2)=ah*bl (needs b's 3rd level)
for scale,RATIO in [(1e-6,1e-4),(1e-6,7e-5),(1.0,1e-4),(1.0,3e-5)]:
    rng=np.random.default_rng(11); N=120000
    cf=rng.standard_normal((K,N)).astype(np.float32)
    b0=(scale*rng.standard_normal((K,N))).astype(np.float32)
    cf64=cf.astype(np.float64); b64=b0.astype(np.float64); cfcf=(cf64*cf64).sum(0)
    b64=b64-((cf64*b64).sum(0)/cfcf)*cf64                       # orthogonalize (sum~0), b keeps ~scale
    tsc=np.abs(cf64*b64).max(0); target=(tsc*RATIO)*rng.standard_normal(N)
    b64=b64+(target/cfcf)*cf64; b=b64.astype(np.float32)
    ref=(cf.astype(np.float64)*b.astype(np.float64)).sum(0); absref=np.abs(ref)
    term=float(np.abs(cf.astype(np.float64)*b.astype(np.float64)).max())
    y3=mac(cf,b,T6); y5=mac(cf,b,T5)
    e3=np.abs(y3.astype(np.float64)-ref)/term; e2=np.abs(y5.astype(np.float64)-ref)/term
    print("scale=%.0e RATIO=%.1e | x3 err/term med=%.2e max=%.2e | x2b err/term med=%.2e max=%.2e  %s"%(
        scale,RATIO,np.median(e3),e3.max(),np.median(e2),e2.max(),
        "x2b OK" if e2.max()<RATIO*0.3 else "x2b LOSES CANCELLATION"))
