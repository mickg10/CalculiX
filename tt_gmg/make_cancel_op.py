# make_cancel_op.py — SYNTHETIC extreme-cancellation operator in spmv_mac's real_op.bin layout, to test
# whether the on-device bf16x3-COMPENSATED MAC survives the smoother's |Ax|<<|x| regime (the case that
# broke matmul-diagonal/ttnn per TT_GMG_STRATEGY.md lines 163-252). This is the DECISIVE precision gate:
# a random vector (make_abref default) does NOT exercise cancellation and is misleading.
#
# Per element e: y[e] = sum_k cf[k,e]*b[k,e]. We draw cf~O(1), b~O(1e-6) (displacement scale), then set the
# LAST b so the sum is a tiny target (ratio ~1e-4 of the term scale = the apply2 |Ax|/|x| depth). ref is
# computed from the fp32-rounded a/b (exactly what the MAC consumes), so the only thing under test is the
# bf16-vs-fp32 product/accumulate error.
import numpy as np, struct
n_out, K = 3782, 81
RATIO = 1e-4                                    # |sum|/term ~ smoother apply2 depth (|Ax|/|x| ~ 1/2400)
N = n_out * 1024
rng = np.random.default_rng(11)
cf = rng.standard_normal((K, N)).astype(np.float32)
b  = (1e-6 * rng.standard_normal((K, N))).astype(np.float32)
# store->reload as fp32 (what the device sees), then force cancellation on the fp32 values
cf64 = cf.astype(np.float64); b64 = b.astype(np.float64)
partial = (cf64[:K-1] * b64[:K-1]).sum(0)                       # (N,)
term_scale = np.abs(cf64 * b64).max(0)                          # per-element max |term|
target = (term_scale * RATIO) * rng.standard_normal(N)         # tiny per-element target
b64[K-1] = (target - partial) / cf64[K-1]
b = b64.astype(np.float32)                                      # fp32-round the forced last term
# ref = EXACT fp64 sum of the fp32-rounded a/b (the answer the MAC must reproduce)
ref = (cf.astype(np.float64) * b.astype(np.float64)).sum(0)
term_max = float(np.abs(cf.astype(np.float64) * b.astype(np.float64)).max())
ax_max = float(np.abs(ref).max())
VSCALE = 1e6
print("[cancel] N=%d K=%d  |sum|max=%.3e  term|cf*b|max=%.3e  ratio=%.3e  scaled_term(x1e6)=%.3e"
      % (N, K, ax_max, term_max, ax_max/term_max, term_max*VSCALE), flush=True)
print("[cancel] DECISION via spmv_mac rel_err: <~0.1 => fp32-class products (compensated MAC HOLDS, converges);"
      "  >~1 => bf16-class products (diverges, like matmul-diagonal)", flush=True)
ad = np.ascontiguousarray(cf.reshape(K, n_out, 1024).transpose(1, 0, 2)).astype(np.float32)
bd = np.ascontiguousarray(b.reshape(K, n_out, 1024).transpose(1, 0, 2)).astype(np.float32)
refd = np.ascontiguousarray(ref.reshape(n_out, 1024)).astype(np.float32)
with open("/tmp/row236_real_op.bin", "wb") as g:
    g.write(struct.pack("<qq", n_out, K)); ad.tofile(g); bd.tofile(g); refd.tofile(g)
print("[cancel] wrote synthetic-cancellation real_op.bin (n_out=%d K=%d, %.2f GB)"
      % (n_out, K, (ad.nbytes * 2 + refd.nbytes) / 1e9), flush=True)
