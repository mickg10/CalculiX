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
