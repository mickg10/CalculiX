# emulate_compensated_mac.py — HOST (CPU-only, no TT device) verification of the DECISIVE precision gate:
# does the on-device bf16x3-COMPENSATED MAC survive the smoother's extreme-cancellation vectors (|Ax|<<|x|),
# the case that broke matmul-diagonal/ttnn (TT_GMG_STRATEGY.md 163-252)? The device already proved it
# implements this exact scheme faithfully (rel_err 6.4e-7 on a RANDOM vector); whether the SCHEME holds on
# CANCELLATION is a math property we can check on the host without touching the device (so no tt-fold disruption).
#
# We emulate mac_compute.cpp's exact arithmetic: split each fp32 into ah/am/al (3 bf16 levels), form the 6
# cross-terms with level-sum<=2 (each an EXACT bf16xbf16 product, representable in fp32), accumulate in fp32.
# Compare vs fp64 truth on a synthetic per-element cancellation operator at the real smoother depth (~1e-4),
# and contrast with a NAIVE bf16x1 product (the thing that diverges).
import numpy as np

def to_bf16(x):
    x = np.asarray(x, np.float32); u = x.view(np.uint32).astype(np.uint64)
    u = (u + 0x7fff + ((u >> 16) & 1)) & 0xFFFF0000                 # round-to-nearest-even to bf16
    return u.astype(np.uint32).view(np.float32)

def split3(v):
    ah = to_bf16(v); r = v.astype(np.float32) - ah
    am = to_bf16(r); r2 = (r - am).astype(np.float32)
    al = to_bf16(r2); return ah, am, al

def cprod_bf16x3(a, b):                                            # compensated product, 6 cross-terms, fp32
    ah, am, al = split3(a); bh, bm, bl = split3(b)
    p = (ah * bh).astype(np.float32)                              # level 0
    p = (p + (ah * bm).astype(np.float32) + (am * bh).astype(np.float32)).astype(np.float32)          # level 1
    p = (p + (ah * bl).astype(np.float32) + (am * bm).astype(np.float32) + (al * bh).astype(np.float32)).astype(np.float32)  # level 2
    return p

def prod_bf16x1(a, b):                                            # naive single bf16 product (diverges)
    return (to_bf16(a) * to_bf16(b)).astype(np.float32)

# ---- synthetic extreme-cancellation operator (same construction as make_cancel_op.py) ----
K, N, RATIO = 81, 200000, 1e-4                                    # 200k elements is plenty for the stat
rng = np.random.default_rng(11)
cf = rng.standard_normal((K, N)).astype(np.float32)
b  = (1e-6 * rng.standard_normal((K, N))).astype(np.float32)
cf64 = cf.astype(np.float64); b64 = b.astype(np.float64)
partial = (cf64[:K-1] * b64[:K-1]).sum(0)
term_scale = np.abs(cf64 * b64).max(0)
target = (term_scale * RATIO) * rng.standard_normal(N)
b64[K-1] = (target - partial) / cf64[K-1]; b = b64.astype(np.float32)
ref = (cf.astype(np.float64) * b.astype(np.float64)).sum(0)       # fp64 truth of the fp32-rounded operator
term_max = float(np.abs(cf.astype(np.float64) * b.astype(np.float64)).max())

def mac(prodfn):
    acc = np.zeros(N, np.float32)
    for k in range(K): acc = (acc + prodfn(cf[k], b[k])).astype(np.float32)   # fp32 accumulate
    return acc.astype(np.float64)

for name, fn in [("bf16x3 COMPENSATED (device scheme)", cprod_bf16x3), ("bf16x1 naive (diverges)", prod_bf16x1)]:
    y = mac(fn); abs_err = float(np.abs(y - ref).max()); pp = abs_err / term_max
    verdict = "PASS (fp32-class -> converges)" if pp < 1e-4 else ("FAIL (bf16-class -> diverges)" if pp > 1e-2 else "MARGINAL")
    print("[emu] %-38s abs_err/term = %.3e   %s" % (name, pp, verdict), flush=True)
print("[emu] cancellation depth: |ref|max/term = %.3e (smoother apply2 was ~4e-4)" % (float(np.abs(ref).max())/term_max), flush=True)
print("[emu] NOTE: proves the SCHEME's precision on cancellation; the device implements this scheme (proven 6.4e-7 on random). On-device cancellation confirm awaits a TT window (tt-fold holds the device now).", flush=True)
