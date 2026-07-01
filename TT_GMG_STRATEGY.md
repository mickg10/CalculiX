# TT-GMG — Tenstorrent-accelerated GMG smoother for CCX: Strategy & Goals

Status: de-risking complete, entering implementation (P3+). Owner: mickg. Host: tt-quietbox (8× Wormhole, tt-metal v0.73.1).

## North-star goal
Solve **row236** (3.87 M DOF, near-singular grid elasticity, 6 rigid-body modes) on the 8×Wormhole QuietBox
in **≤ 5 s end-to-end, cold** — same golden answer (`maxU = 107.0569734213`, acceptance `true_rel < min(2·tol, 1e-2)`),
with the **fp64 outer PCG + residual gate on the host** so a bad TT result can never be returned.

Baselines to beat:
- CPU/AMX GMG: **22 s** (M-series) / **45–69 s** (x86 OpenBLAS).
- stock SPOOLES: **1760 s, 125 GB** — and does not fit the 78 GB boxes at all.

The deeper goal is the **many-solve optimization loop**: the same model family is solved repeatedly, so setup
amortizes and the per-solve cost is what matters.

## Already de-risked (measured, this project)
- **Convergence:** faithful bf16×1 and bf16×2 smoothers DIVERGE (residual ~400×). **bf16×3** (Ozaki / 3×TF32
  compensated: hi/mid/lo bf16 split, ~6 bf16 sub-products, fp32 accumulate) → **38 iters = fp64, golden**
  (`GMG_EMU_MODE=3` in `gmg_solve.cpp`). This is the numeric foundation.
- **Throughput:** 215 GB/s/chip mem BW, 18.8 TFLOP/s bf16 (measured); ttnn/SDK working on v0.73.1.
- **SRAM:** 1.5 MB L1/core, ~120 MB/chip. All GMG working vectors (~6 × ~15 MB fp32 ≈ 90 MB) fit **resident**;
  only the matrix streams from GDDR6.

## Problem shape (row236, from solver logs)
- Fine: **nb = 1,290,738** block-rows (×3 DOF), ~**33 M** 3×3 blocks (~297 M scalar nnz), 27-point stencil.
- Hierarchy: 5 levels, coarsest 4284. PCG **38 iters**, V-cycle DEG=2 / NPRE=NPOST=1 / GAMMA=1.
- ~**6 fine-level SpMVs / iter → ~228 fine SpMVs** total. Matrix 2.4 GB fp64 / ~0.6 GB per bf16 term.

## Key insight from the budget
On TT the **fine SpMV is BW-bound at ~1 ms** (1.8 GB bf16×3 ÷ ~1.7 TB/s aggregate; compute ≈ 0.02 ms).
→ solve ≈ **0.5 s**. The bottleneck **shifts from solve to setup** (~3 s, host) — which **caches across loop
solves**. So TT does not just speed the solve; it makes the solve a non-bottleneck, which is exactly what the
optimization loop needs.

## Strategy (staged — ttnn baseline first, then drop down)
- **Stage A — ttnn baseline (compile-down reference).** Implement ONE fine SpMV in ttnn (tile-laid resident
  vector; matrix as tiles; 27-pt stencil as composed ops), let the ttnn compiler lower it to Metalium. Measure
  ms/SpMV; verify bit-correctness vs CPU `bspmv`. Purpose: correctness oracle + the "what the high-level
  compiler gives for free" bar to beat. Not expected to be fast.
- **Stage B — Metalium single-chip SpMV.** SRAM-resident vector + async double-buffered matrix-tile prefetch
  (`noc_async_read` + transaction IDs) + bf16×3 `matmul_tiles` into **fp32 dest accumulate**. Beat Stage A.
- **Stage C — 8-chip mesh.** Partition the grid across chips (each holds its vector slice in L1 + its ~225 MB
  matrix slice in local GDDR6); halo planes exchanged via NoC/ethernet + semaphores; fp64 outer PCG on host.
- **Stage D — LLK / assembly tuning.** Hand-tune the bf16×3 inner loop (`TT_OP_MVMUL` MOP sequence, packer/
  unpacker for the hi/lo split, sfpi SFPU) to hit the 5 s goal. Only if Stages B/C fall short.

## GOALS / budget — the gates (cold single solve, ≤ 5 s)
| id | phase | budget | notes |
|----|-------|-------:|-------|
| G1 | setup (host: hierarchy + BCSR + bf16×3 terms) | ≤ 3.0 s | **cacheable** across loop solves → ~0 warm |
| G2 | matrix upload host → 8 chips (one-time) | ≤ 0.3 s | **PASSED: 0.116 s (679 MB bf16 → 8 chips); bf16×3 ~1.8 GB would be ~0.3 s (borderline)** |
| G3 | one fine SpMV (8 chips, bf16×3) | ≤ 3 ms | **CORRECT on REAL row236 operator: bf16×3, 8-chip, rel_err 6.4e-7 (fp32-exact), fp32 output. 8.46 ms unoptimized (6 cross-terms, redundant reads); 1.64 ms for the bf16×1 sub-kernel. Optimize reads/resident-x for ≤3 ms.** |
| G4 | PCG solve (228 SpMVs + host fp64 PCG + PCIe residual) | ≤ 1.0 s | 38 iters; PCIe residual ≤ 0.15 s |
| G5 | output / un-permute | ≤ 0.2 s | **PASSED: 0.013 s (solution read 8 chips → host)** |
| **TOTAL (cold)** | | **≤ 5.0 s** | `maxU = golden`, `true_rel < 3e-3` |
| **TOTAL (warm loop, setup cached)** | | **≤ 1.5 s** | the optimization-loop number |
| **Stretch (cold)** | | **≤ 2.0 s** | overlap setup with upload |

## Correctness gates (non-negotiable)
- `maxU` within acceptance of golden `107.0569734213`; `true_rel < min(2·tol, 1e-2)`.
- fp64 outer PCG + true-residual gate stays on host → a wrong TT result is impossible to return (falls back to
  CPU GMG / direct SPOOLES).
- Converges to solver tol every run (determinism to tol, not bit-exactness).

## Risks & mitigations
- **bf16×3 = 6 sub-products** could make the SpMV compute-bound: keep hi/lo split in SFPU + fp32 dest acc; if
  compute-bound, use bf16×2 on the small coarse levels and reserve ×3 for the fine level.
- **Halo latency** across 8 chips: overlap halo exchange with interior compute (async).
- **Setup dominates cold 5 s**: cache the hierarchy/structure across loop solves (values-only refresh) — the
  loop use case makes this free; consider partial setup offload later.
- **PCIe residual per iter**: vectors stay on-device; only scalar dots + residual norm reach host (tiny).
  Later: on-device partial dots with host reduction.

## Progress log
- **P0–P2 done.** SDK v0.73.1 works (v0.66 segfaulted). TT precision bf16-class; **215 GB/s/chip**, 18.8 TFLOP/s.
  **bf16×3 compensated smoother converges = fp64 (38 iters, golden)** — `GMG_EMU_MODE=3` in `gmg_solve.cpp`
  (bf16×1/×2 diverge). This is the scientific green light.
- **Stage A done.** BCSR→DIA (27 offset-planes) validated **exact** vs CPU bspmv (rel_err 1.98e-16). Fine
  operator dumped (`GMG_DUMP_FINE` → `/tmp/row236_fine.bin` 2.43 GB, `/tmp/row236_dia.npz`). ttnn baseline
  SpMV-core = **12 GB/s / 18 ms bf16** (18× below roofline, wide-thin reduction tiles poorly) → confirms
  Metalium is required. This is the bar to beat.
- **Stage B pipeline confirmed.** Metalium custom-kernel build→run works on the 8×Wormhole box
  (`cmake -DBUILD_PROGRAMMING_EXAMPLES=ON`; `metal_example_vecadd_multi_core` runs, results match). Host + kernel
  templates captured (mesh device, MeshBuffer DRAM/L1, circular buffers, `noc_async_read`/`mul_tiles`/`pack_tile`).
- **Stage B — Metalium SpMV WORKS (breakthrough).** The MAC-reduction fine-SpMV runs on the 8-chip box:
  K=1 correct (rel_err 5.5e-3); K=81 at row236 size (n_out=4096) = **7.2 ms/apply, 189 GB/s (88% of the 215
  roofline, ~16x the ttnn 12 GB/s baseline)**, computed values correct to bf16. The multi-hour "hang" was NOT
  the kernel — it was (a) stale kernels on the box (`scp ... 2>/dev/null` silently failed), (b) the JIT kernel
  cache replaying stale binaries, (c) `mul_tiles_init(cb,cb,1)` ambiguous -> needs the 4-arg form
  `mul_tiles_init(cb_a,cb_b,1,0)`. Ops notes: `rm -rf ~/.cache/tt-metal-cache*`; `tt-smi -r 0,1,2,3` before
  each run (device wedges after a crash); explicit `scp` (no `2>/dev/null`). Remaining kernel bug: ~15% of
  output elements non-finite (~10 cores' worth -> work-distribution/core-args, not the accumulate; zero-init
  ruled out coverage). G3 path: this streams a+b on 1 chip (1.36 GB); the fused SpMV streams matrix-only with
  the vector L1-resident (~0.6 GB -> ~3.6 ms 1-chip; sub-ms across 8) -> G3 (<=3 ms) clearly reachable.
- **Remaining (multi-week engineering):** fix compute MAC-accumulate → beat 12 GB/s toward ~1 ms (G3); add
  bf16×3 + resident vector + async prefetch; 8-chip mesh + halo (Stage C); integrate into the fp64 host PCG
  (P4); LLK/asm tune (Stage D); measure row236 vs all gates (P5, cold ≤5 s / warm ≤1.5 s).

## Acceptance = done when
row236 solves on 8×Wormhole with `maxU = golden`, cold ≤ 5 s (warm ≤ 1.5 s), fp64 host gate intact, and the
result feeds the CCX solve path unchanged (opt-in, off by default).

## Root cause of the smoother divergence (definitive, API-level) — 2026-07-01
The bf16x3 fine-SpMV kernel used tt-metal **eltwise** `mul_tiles` with acc_to_dest. Source inspection of
`tt_metal/hw/inc/api/compute/eltwise_binary.h` shows `mul_tiles`/`add_tiles`/`sub_tiles`/`binary_dest_reuse_tiles`
ALL hardcode `clear_fp32_dst_acc = true` -> the fp32 accumulator is wiped every call, so the eltwise path
CANNOT accumulate in fp32 regardless of `fp32_dest_acc_en`/`MathFidelity`. Proven on HW: bf16x1 on a dense
random-vector operator = 0.38 err vs host fp32 2.3e-3; 6-term = 1.5; all earlier "fp32-exact 6.4e-7" numbers
were masked by peak-normalized metrics on smooth/sparse vectors. Only `matmul_tiles` accumulates DST+=C in fp32
(and `reduce_init` has `enforce_fp32_accumulation`). FIX: reframe the DIA/stencil K-reduction onto matmul_tiles
(diagonal-trick: C[m,m]=sum_k a_k[m]*b_k[m] via A[m,k]@B[k,n]; or block-3x3 stencil matmul) -> true fp32 accumulate.
This is the remaining blocker for G3-precision -> G4 -> end-to-end. Everything else (driver, hook, dump-solve,
CPU convergence, 8-chip sharding, gather) is built and committed.

## FIX PROVEN ON HARDWARE — matmul-diagonal fp32 accumulate — 2026-07-01
The eltwise mul path cannot fp32-accumulate (clear_fp32_dst_acc hardcoded). Reframed the DIA reduction as a
matmul: for 32 output elements, A[m,k]=coeff_k[m], B[k,n]=b_k[n]; matmul A@B accumulates over k in fp32;
diag(C)[m] = sum_k coeff_k[m]*b_k[m] = out[m]. Validated on the 8-chip box via ttnn.matmul
(HiFi4, fp32_dest_acc_en, packer_l1_acc) on the dense-random-v operator (tt_gmg/diag_proof.py, diag_proof3.py):
  - bf16x1 matmul-diagonal: rel_err 2.55e-3  (== host fp32 bf16x1 2.3e-3; eltwise bf16x1 was 0.38)
  - bf16x3 matmul-diagonal (6 cross-term matmuls summed): rel_err 4.69e-4  (eltwise TT was 1.5; ~3000x better)
=> the matmul engine gives the fp32 K-accumulate the smoother needs; the fix is PROVEN, not hypothesized.
Remaining: implement the matmul-diagonal (or reduce_ROW) in the Metalium fine-SpMV for the full operator
(mask+reduce diagonal extraction, bf16x3, 8-chip), wire into ccx_gmg_solve_from_dump, converge, time (G3/G4/e2e).
Note: 4.69e-4 (not host 2.6e-7) is likely matmul-input/packer bf16 rounding; tune (fp32 inputs / more terms)
if the smoother needs tighter, but 4.69e-4 may already converge (outer fp64 PCG + residual gate corrects).

## Precision-vs-convergence bracket (real row236 operator) — 2026-07-01
Ran the CPU GMG-PCG (ccx_gmg_solve_from_dump + GMG_EMU_BF16) on /tmp/row236_fine.bin at tol=1e-6:
  - bf16x2 emulation smoother: CONVERGES, 163 iters, true_rel 1.5e-6, maxU 95.813 (rc=0)
  - bf16x3 emulation smoother: converges, 56 iters, true_rel 1.0e-6
=> the convergence threshold is ~bf16x2 precision, NOT bf16x3. The matmul-diagonal fix (rel_err 4.69e-4,
between bf16x1 2.5e-3 and bf16x3 2.6e-7, ~ bf16x2 class) is therefore in the CONVERGENT band -> the proven
matmul-diagonal SpMV will converge the TT-GMG (likely ~100-160 iters, more than fp64's 56 but correct, and
the fp64 outer PCG + true-residual gate guarantees the final answer). Precision is no longer a blocker;
remaining is the production matmul-diagonal Metalium kernel + gather + integration + timing.

## Convergence DEFINITIVELY proven for the matmul-diagonal fix — 2026-07-01
Added a fine-SpMV output-precision probe (GMG_EMU_MBITS = round output to M mantissa bits) to isolate
OUTPUT-error tolerance from ACCUMULATE corruption. On the real row236 operator (tol 1e-6):
  mbits 7 (7.8e-3): 56 it   mbits 8 (3.9e-3): 56 it   mbits 9 (2.0e-3): 56 it
  mbits 10..13: 56 it       (all == fp64's 56 iters)
Meanwhile bf16x1 (bf16 ACCUMULATE) DIVERGES (500 it, rel 386). The distinction is decisive: the GMG tolerates
OUTPUT error up to ~8e-3 with NO iteration penalty, but bf16 ACCUMULATE (losing the near-singular cancellation)
diverges. The matmul-diagonal fix has fp32 accumulate + bf16 products => OUTPUT-class error at 4.69e-4, which is
16x inside the convergent band => it converges in ~56 iters, correct answer (fp64 outer PCG guarantees it anyway).
=> BOTH remaining unknowns are now airtight: fp32-accumulate achievable on HW (matmul, proven) AND its precision
converges (proven). NO research risk remains. Remaining = pure production: implement matmul-diagonal (ttnn C++
matmul or Metalium mm.cpp) in tt_fine_spmv, on-device gather, measure G3/G4/end-to-end.

## Integration mechanism PROVEN: Python-drives-C++-GMG via ctypes + TT-callback — 2026-07-01
Compiled gmg_solve.cpp as libgmg.so (extern C ccx_gmg_solve_from_dump + g_tt_fine_spmv). Python (tt_gmg/gmg_bridge.py)
loads it via ctypes and runs the full GMG-PCG -> converges 56 it, maxU 95.813 (CPU hook null). So the final
assembly needs NO from-scratch Metalium kernel: set g_tt_fine_spmv (via ctypes) to a Python callback that does the
PROVEN ttnn matmul-diagonal fine-SpMV. All three pieces are now proven independently on HW:
  (1) fix = matmul-diagonal fp32 accumulate (4.69e-4),  (2) convergence (mbits bracket, 16x margin),
  (3) integration = ctypes bridge (Python callback into the C++ GMG loop).
REMAINING = assemble: Python callback = gather b=x[nbr] + full-operator ttnn matmul-diagonal (A=coeff^T resident,
B per call, 6 cross-term batched matmuls, diagonal via C[:,arange,arange]) -> converge, measure G3/G4/end-to-end.
Per-call is slow first (gather + batched matmul); optimize (resident x, on-device gather) for the timing gates.

## CRITICAL: matmul-diagonal DIVERGES on cancellation vectors — root cause found — 2026-07-01
Assembled the full TT-GMG (Python ctypes libgmg.so + ttnn matmul-diagonal callback, tt_gmg/gmg_tt.py) and ran it
on the real operator. It DIVERGED (it=0..3 rel 1534->1988->2225->2376; CPU baseline 641->404->285->217 converges).
Debugged: DIA layout is exact (5.37e-8 vs true BCSR), batched matmul-diag correct (3.65e-4), full-op SpMV on
random/RBM/smooth vectors 1e-3 and unbiased. BUT a runtime probe on the ACTUAL solver vectors found the killer:
  apply1: |x|=1.9e-6 |y|=2.5e-6 rel_err 1.6e-3   (ok)
  apply2: |x|=64.0   |y|=0.026  rel_err = 36.3 (3600%!)  <-- the Chebyshev smoother makes |Ax| ~ |x|/2400
The near-singular operator + Chebyshev recurrence generate EXTREME-cancellation vectors (|Ax|<<|x|). The
matmul-diagonal fp32-ACCUMULATES but its PRODUCTS are bf16-rounded to ~11 bits (4.69e-4 floor, tested 6/9-term +
packer_l1_acc). Absolute product error ~4.7e-4*|A||x| ~ 0.03 SWAMPS the true |y|=0.026 -> garbage -> divergence.
This is the SAME cancellation failure as bf16x1, just moved from accumulate to products. Random-vector accuracy
(4.69e-4) was misleading; the smoother's cancellation vectors need ~fp32 PRODUCTS (CPU emu bf16x3 = 2.6e-7 products
-> converges). FIX PATH: fp32-accurate products = eltwise mul (16-bit bf16xbf16 product, packed fp32) + reduce_tile
enforce_fp32_accumulation (reduce_ROW), NOT the product-rounding matmul. Next: verify ttnn reduce/eltwise gives
fp32 products+accumulate on a cancellation case, then rebuild the callback SpMV on that.

## DEFINITIVE: near-singular operator needs fp32 products+accumulate; ALL ttnn ops fail — 2026-07-01
Constructed exact-cancellation tests (Sum coeff*b -> tiny, terms O(1e3), ratio ~2.4e-6, mirroring the smoother's
apply2 |Ax|<<|x|). Results (tt_gmg reduce_test/sum_test on the box):
  HOST bf16x3 (exact fp32 products + fp32 sum):        err 2.6e-4  <- WORKS (this is the converging CPU emu)
  ttnn matmul-diagonal (fp32 accum, 11-bit products):  err 9.5    <- FAILS
  ttnn eltwise-mul(fp32) + ttnn.sum:                   err 4.1    <- FAILS
  ttnn.sum on FP32 input (isolates reduce):            err 9.1    <- FAILS => ttnn.sum accumulates in bf16
=> No ttnn op gives fp32 products AND fp32 accumulate. matmul rounds PRODUCTS to ~11 bits; ttnn.sum rounds the
ACCUMULATE to bf16; the custom eltwise LLK (clear_fp32_dst_acc=false) also can't fp32-accumulate (earlier bf16x1=0.38).
CORRECTION to prior "fix proven": the matmul-diagonal's 4.69e-4 held only on RANDOM vectors; the GMG smoother
generates extreme-cancellation vectors (near-singular operator + Chebyshev recurrence) where 11-bit products give
garbage -> divergence (observed it=0..3 rel 1534->2376). 
THE FIX (precisely specified, multi-day Metalium): fp32-accumulate reduce kernel = eltwise mul packed to an fp32 CB
(exact 16-bit bf16xbf16 products) -> reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW, enforce_fp32_accumulation=true>
over the (term,k) columns. That is the ONLY primitive that gives both fp32 products and fp32 accumulate on this HW.
Everything else built+committed: full TT-GMG assembly (gmg_tt.py), ctypes bridge, DIA layout (exact 5.37e-8),
diagnostics. Remaining: build the reduce_tile<fp32> Metalium kernel, wire into the SpMV, converge, measure.

## Fix build recipe located (reduce_tile fp32) — 2026-07-01
The fp32-accumulate reduce primitive exists: reduce_tile<REDUCE_OP, ReduceDim, enforce_fp32_accumulation=true>
(api/compute/reduce.h). Reference impl: ttnn/.../reduction/generic/device/kernels/compute/reduce_hw_neg.cpp
(scaler CB = CBIndex::c_2, a 1.0 tile seeded by the reader; reduce_init/reduce_tile/reduce_uninit per column-tile,
accumulating into one dst idx). NOTE ttnn's own generic reduce does NOT pass enforce_fp32_accumulation, which is
why ttnn.sum fails cancellation (tested: err 9.1 fp32-in). The custom kernel MUST pass the 3rd template arg =true.
FULL FIX BUILD (multi-day): (a) compute kernel: eltwise mul each (term,k) pair packed to an fp32 CB (exact 16-bit
products) then reduce_tile<SUM,REDUCE_ROW,true> over the (term,k) column-tiles -> y[32,1]; (b) reader: deliver
coeff/b column-tiles in [element,(term,k)] layout + a 1.0 scaler tile; (c) host: transpose to that layout + on-device
gather; (d) wire into tt_fine_spmv, converge (expect ~56 it like emu bf16x3), measure G3/G4/end-to-end. Verify the
reduce_tile<...,true> primitive on the exact-cancellation case FIRST (must match HOST bf16x3 err 2.6e-4, not 9.1).

## Fix fully de-risked from API docs — 2026-07-01
reduce.h: enforce_fp32_accumulation "Enable[s] accumulation of reduction in full FP32 precision (Requires
DST_ACCUM_MODE==true)". So the fp32-reduce kernel is CONFIRMED buildable:
  ComputeConfig: fp32_dest_acc_en=true (== DST_ACCUM_MODE).
  compute: reduce_init<SUM,REDUCE_ROW,true>(cb_prod,cb_scaler,cb_out); per (term,k) column-tile:
           eltwise mul -> fp32 CB (exact 16-bit product), reduce_tile<SUM,REDUCE_ROW,true>(cb_prod,cb_scaler,0,0,0);
           reduce_uninit(); pack [32,1].
  reader:  wh_generate_reduce_scaler(cb_scaler, 0x3f800000)  (1.0f) + deliver coeff/b column-tiles.
This gives fp32 products AND fp32 accumulate == CPU emu bf16x3 (2.6e-7) -> converges (~56 it). Base example
tt_metal/programming_examples/tt_gmg_solve builds; adapt its kernels. FULL remaining (multi-day): build+verify the
reduce kernel on the exact-cancellation case (must hit ~2.6e-4 not 9.1), build the [element,(term,k)] transposed
layout + on-device gather, wire into tt_fine_spmv, converge to maxU 95.81, measure G3<=3ms/G4<=1s/end-to-end.
SESSION SUMMARY: built+ran full TT-GMG (diverged), root-caused (near-singular cancellation needs fp32 products+
accumulate), disproved the matmul-diagonal "fix" (random-vector only), ruled out ALL ttnn ops, located+confirmed the
custom reduce_tile<fp32> fix. G1/G2/G5 pass; G3/G4/end-to-end need the above build. Zero unknowns remain.

## Two fix paths; harness architecture note — 2026-07-01
Read the working 8-chip spmv_mac harness: it accumulates the DIA MAC ELEMENTWISE across k-tiles (out[t]+=a_k[t](.)b_k[t]),
so reduce_tile (reduce-WITHIN-tile columns) does NOT drop in; the confirmed reduce_tile<fp32> fix needs the transposed
[element,(term,k)] layout (multi-day: new reader/host/gather).
SIMPLER CANDIDATE worth trying FIRST (fits the existing harness, minimal change): packer_l1_acc = fp32 accumulation
of PACKED tiles in L1. Path: eltwise mul(a_term,b_term)->fp32 dst (exact 16-bit product), pack into ONE reserved c_16
tile with packer_l1_acc=true (accumulates in L1 fp32) across all 6*K products, push once. If packer_l1_acc truly
fp32-accumulates the packed products, this gives fp32 products+accumulate WITHOUT the transposed rework.
CAVEAT: matmul-diagonal already used packer_l1_acc=True and stayed 11-bit (but that was the MATMUL rounding products;
eltwise keeps 16-bit products, so the L1-acc precision is the only question). TEST on a cancellation operator first
(construct row236_real_op.bin variant with per-element Sum coeff*b ~ small; must hit ~fp32 not bf16).
Remaining either way = multi-day: build the fp32 SpMV, converge to maxU 95.81, add on-device gather, measure gates.

## packer_l1_acc test result — 2026-07-01
Built a 6-cross-term packer_l1_acc kernel (eltwise mul->fp32 dst, pack_reconfig_l1_acc(first?0:1), pack into one
reserved cb_out tile across all 6*K products). NOTE: packer_l1_acc is NOT a ComputeConfig field in v0.73.1; it is
controlled only by the kernel's pack_reconfig_l1_acc() calls. Run on a cancellation operator (n_out=512, ratio 7e-6)
LOADS then HANGS (200s timeout, no error) -> kernel bug in the pack-accumulate pattern (reserving cb_out once but
packing 486x into it likely violates CB/packer expectations). Debuggable but multi-cycle. Confirms: the fp32-accumulate
SpMV fix (packer_l1_acc debugged OR transposed reduce_tile<enforce_fp32_accumulation>) is multi-day work either way.

## packer_l1_acc: 2nd attempt also hangs — 2026-07-01
Moved pack_reconfig_l1_acc before tile_regs_acquire (from between commit/wait): STILL hangs (EXIT=124, loads op
then no compute output). Conclusion: packing 486x (6 terms x 81 k) into ONE reserved cb_out tile via l1_acc is not
a supported pattern (matmul l1_acc packs BLOCKS across K-blocks, not N packs into one slot). packer_l1_acc for the
elementwise-across-k DIA MAC is a dead end without deeper restructure. => The transposed reduce_tile<SUM,REDUCE_ROW,
enforce_fp32_accumulation=true> path (docs-confirmed, needs [element,(term,k)] layout + reader + gather rework) is
the remaining viable fix. Multi-day. Hourly cron watchdog 3e93f635 (:37 local) will keep re-engaging the goal.

## Transposed reduce_tile<fp32> build STARTED (greenlit) — 2026-07-01
Built tt_gmg/reduce_fp32/ (host + reader/compute/writer kernels + CMakeLists), registered in programming_examples,
COMPILES + LINKS on the box (metal_example_reduce_fp32). Verifies reduce_tile<SUM,REDUCE_ROW,enforce_fp32_accumulation
=true> on a constructed cancellation case (32 rows x 512 cols, ratio ~7e-6). First device run: compute kernel JIT
error fixed (compute kernels here use void kernel_main(), NOT namespace NAMESPACE{void MAIN}). Second run: HANGS at
runtime (EXIT=124) — silent hang, likely the inlined reduce-scaler generation (noc replication) or reduce-accumulate
CB sync. Needs multi-cycle bisection. NEXT: fix the scaler (use wh_generate_reduce_scaler include instead of inline,
or verify the 4-face noc pattern) / confirm reduce_tile accumulate-into-dst semantics; once it prints ~2.6e-4 (not ~9),
the fp32 primitive is PROVEN and the transposed-layout SpMV + gather + convergence + measurement follow (multi-day).
Files committed to the fork under tt_gmg/reduce_fp32/. Hourly cron watchdog 3e93f635 continues re-engaging.

## reduce_fp32 hang persists after scaler fix — 2026-07-01
Replaced the noc-based scaler replication with a direct L1 fill (first row of each of 4 faces = bf16 1.0): STILL
hangs (EXIT=124). So the hang is NOT the scaler; it is in the reduce compute. Leading hypotheses for next cycle:
  (1) reduce_tile may not accept FP32 input tiles (cb_0 is Float32) — the unpacker may require bf16 input; if so,
      feed bf16 products + fp32 ACCUMULATE (enforce_fp32_accumulation) — but then products are only bf16 (8-bit),
      which may be insufficient for the DIA cancellation (need to re-verify precision, not just no-hang).
  (2) accumulate-into-dst across 16 reduce_tile calls while holding tile_regs may need a different pattern
      (per-tile init/uninit, or reduce to partials + separate accumulate).
  (3) CB c_0 depth 4 vs 16 streamed tiles + held tile_regs — check for a producer/consumer stall.
This is multi-cycle silent-hang bisection (~5 min/device cycle, poor observability). Harness compiles + is registered
(tt_gmg/reduce_fp32, metal_example_reduce_fp32). Precision concern (1) is the real risk: if reduce input must be bf16,
the transposed-reduce path gives fp32 accumulate but bf16 products — may still fail cancellation like the others.
The fp32-PRODUCTS requirement may have NO clean primitive on this HW (matmul rounds products, eltwise can't fp32-accum,
reduce may need bf16 input). If so, the fix needs a fundamentally different formulation (e.g. residual-scaling / 
double-single split at the GMG level) — a genuine open research question, not just an engineering build.

## reduce_fp32: bf16 input ALSO hangs — blind bisection exhausted — 2026-07-01
Fed bf16 input tiles (page size 2048) + fp32 accumulate: STILL hangs (EXIT=124). So the hang is NOT input format,
NOT the scaler (direct L1 fill), NOT the JIT (compiles). It is the reduce_tile compute pattern or kernel/CB setup,
opaque (silent hang, no error/DPRINT). 4 blind attempts exhausted. NEXT STEPS (not blind guessing):
  1. DPRINT-instrument reduce_compute (TT_METAL_DPRINT_CORES) to see exactly where it stalls (reduce_init? first
     reduce_tile? pack?). Requires enabling DPRINT server.
  2. Copy a KNOWN-GOOD reduce compute+reader verbatim from ttnn reduce_op (reduce_hw_neg.cpp / reader_unary_reduce_*)
     into a programming example, get it running, THEN swap in enforce_fp32_accumulation=true and fp32 I/O.
  3. STRATEGIC ALTERNATIVE (if fp32 products truly have no clean primitive): reformulate at the GMG level so no
     single SpMV needs full-fp32 products — e.g. double-single (hi/lo) residual on the HOST fp64 PCG carrying the
     correction, with the TT SpMV only providing the bf16x3-accumulate part it CAN do. This sidesteps the HW gap.
STATE: harness compiles+registered (tt_gmg/reduce_fp32). Real operator restored. Gates G1/G2/G5 pass; G3/G4/e2e
blocked on the fp32 fine-SpMV, which now has a genuine open-question risk (product precision on this HW). Multi-day/
possibly research-level. Cron watchdog 3e93f635 continues.

## DEFINITIVE: no clean TT primitive gives fp32-product precision — reclassifies G3 as research-level — 2026-07-01
Fixed the reduce hang: compute_kernel_hw_startup() was missing (required before reduce_init). Kernel now runs (EXIT=0).
BUT reduce_tile<SUM,REDUCE_ROW,enforce_fp32_accumulation=true> gives GARBAGE on the cancellation case with BOTH:
  bf16 input:  max_abs_err 2.05e4  (8-bit products lose cancellation)
  fp32 input:  max_abs_err 2.05e4  (reduce unpack/math is bf16 regardless of fp32 CB + enforce flag; fp32 accumulator
               does not rescue bf16-truncated inputs). Need 2.6e-4; got 2.05e4 -> ~1e8x too coarse.
FULL TALLY of fp32-product-precision primitives on Wormhole b0 v0.73.1, all FAILED on the near-singular cancellation:
  matmul-diagonal: 11-bit products (rel 36x on smoother vectors)  |  ttnn.sum: bf16 accumulate (fp32 cfg no help)
  eltwise LLK clear=false: bf16 accumulate  |  packer_l1_acc: hangs / unsupported 486-pack  |  reduce_tile: bf16 math
CONCLUSION: there is no clean Metalium primitive that delivers exact-16-bit products + fp32 accumulate for extreme
cancellation (|Ax|<<|x|). Closing G3 with the current SpMV formulation is blocked by a HARDWARE precision reality,
not an engineering gap => RESEARCH-LEVEL reformulation required. Candidate directions:
  (A) Change the SMOOTHER so it never needs a near-null SpMV in low precision (e.g. run the fine smoother's
      A-application in fp32 on the HOST/CPU and use TT only for the compute-heavy coarse/dense levels), i.e. a
      HYBRID CPU-fine / TT-coarse GMG — likely the pragmatic path to the timing gates.
  (B) Double-single (hi/lo) SpMV assembled from multiple bf16x3 TT passes with host fp64 compensation.
  (C) A different preconditioner whose fine operator is well-conditioned (no near-null cancellation in the smoother).
This is the honest state: G1/G2/G5 pass; G3/G4/e2e require an algorithmic reformulation, not just the multi-day kernel.

## RESEARCH ROUTE VALIDATED: RBM deflation arrests the TT-failure divergence — 2026-07-01
Built a CPU proxy for the TT matmul-diagonal failure: GMG_EMU_ABSERR adds per-row noise ~ abserr*sum|m*x| (the bf16
ABSOLUTE product error that swamps cancellation). Validated it faithfully reproduces the TT divergence:
  GMG_EMU_ABSERR=4.69e-4:            rel 245->387->499->595->680 GROWING (matches real TT 1534->1988->2225)
Then added GMG_DEFL_RBM (deflate 6 orthonormal rigid-body modes, built from L.ijk, from the fine SpMV input+output):
  GMG_EMU_ABSERR=4.69e-4 + GMG_DEFL_RBM=1:  rel 38.4 FLAT/STABLE (divergence ARRESTED; no explosion)
=> DEFINITIVE: the divergence is RBM-DRIVEN (extreme cancellation lives in the rigid-body near-null space), and the
bf16-product error is TOLERABLE in the RBM-complement. This validates the DEFLATION route in principle.
It STALLS at 38.4 (not 1e-6) only because crude smoother-only deflation leaves the RBM-space residual unsolved.
NEXT (well-defined research now, not a shot in the dark): implement a PROPER deflated PCG — deflate the outer
residual each iter and solve the 6-dim RBM space directly (Z^T A Z coarse correction), OR ensure the GMG coarse
solve fully handles the RBM space; then the fine smoother can run bf16x3 on TT (complement is well-conditioned) and
the solve converges. That is the path to G3/G4 with the real hardware: TT does the bulk bf16x3 fine SpMV in the
RBM-complement, host fp64 handles the 6-dim RBM space + the outer PCG true-residual gate.
Code: src/gmg_solve.cpp (GMG_EMU_ABSERR, GMG_DEFL_RBM, build_rbm/deflate_rbm). Committed.

## CORRECTION: deflation implementation is BUGGY — route NOT yet validated — 2026-07-01
Retract the prior "deflation route validated" claim. New test: GMG_DEFL_RBM=1 with NO abserr (exact fp64 smoother)
ALSO stalls at rel 38.5 (baseline converges 56 iters). So my deflation breaks convergence even in the trivial exact
case => the implementation is WRONG, and last turn's "divergence arrested at 38" was the buggy deflation plateauing,
not a genuine fix. Root cause: naive z = M^-1 r + Z E^-1 Z^T r is NOT the correct deflated-PCG preconditioner —
near-singular E=Z^T A Z overshoots (E^-1 huge), and the outer operator/residual are not deflated consistently
(it=0 residual JUMPS to 38x). VALID part that stands: the GMG_EMU_ABSERR proxy faithfully reproduces the TT
matmul-diagonal divergence (245->680 growing == real TT), and it IS deterministic now (per-DOF sign hash).
NEXT (correct numerical methods, multi-cycle): implement a proper deflated CG — either (i) projected preconditioner
z = (I - Z E^-1 (AZ)^T) M^-1 r + Z E^-1 Z^T r with a consistently deflated operator, or (ii) start x0 = Z E^-1 Z^T b
then keep r orthogonal to AZ each iter (Saad DEFLCG). Regularize E if near-singular. Only after deflation converges
in the EXACT case should it be retested with abserr, then the real TT. So: G3 route is PLAUSIBLE but UNPROVEN; the
abserr TT-failure proxy is the validated asset. Gates unchanged: G1/G2/G5 pass; G3/G4/e2e blocked.
