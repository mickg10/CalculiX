# TT-GMG timing-gate closure — executable implementation plan

Status correction (2026-07-15): `../GATE_CONTRACT.md` is authoritative. Only G5 is currently performance-green;
reduced-dump correctness is separately green. G1 and G2 exceed their exact cold/full-payload budgets, and the old
serializer described below was active-node-order rather than brick-major. `brick_format.py` plus the new
`serialize_tiled.py`/`verify_tiled.py` now define the versioned brick-major source and host oracle. Device work must
consume that format and emit contract-qualified evidence.

## Current device frontier — 2026-07-15

- The versioned brick-major operator, device layout, four-vector host oracle, and resident-x device program now
  exist. The best qualifying device result is run52: full bf16×3, reverse term order, all eight Wormhole chips,
  persistent program, resident x, no host x/y round trip, and correct output. Its default-off selective b-low path
  retains the Manhattan-distance-≤2 terms plus the opposite-corner pair (`63/81`). Samples are
  `3.379803/3.377273/3.256412 ms`; median `3.377273 ms`, so G3 remains red by `0.377273 ms`.
- Lowering the three degree-2 products to HiFi3 or HiFi2 per term is numerically valid but slower on device:
  run44 median `3.534256 ms`; run47 median `3.550619 ms`.
- The exact three-term HiFi2 grouping passes the random/smooth/boundary/constant host oracle, but it is not
  device-safe. Runs45, 46, 48, and the observed run49 all corrupt chip 2 / local group 162 / component 0 /
  lane 204 by about `-32`, while the remaining output stays near the normal error floor. Direct MOP switching,
  full LLK reinitialization, a full Tensix pipeline drain, and an explicit FP32 DEST clear do not change it.
  The strict host reconstruction for that lane equals the reference exactly. Do not spend another cold boot on
  this cross-term batching family.
- Fixed HiFi2 with an exact leading product passes all host vectors and is device-correct, but run50 measured a
  `18.312919 ms` median. Full bf16×2 fails the smooth-vector host tolerance and was not sent to hardware.
- The vectorized/precomputed affine A-low path did reach the device, but run54 deadlocked and runs55–57 repeated the
  same two output errors under fixed, zero, and dummy-NoC cadence. It is closed device-incorrect. Runs58/58b and
  run59 were infrastructure-only negatives; the correct split-A retry run59b was slower (`3.505936 ms`).
- A full raw-operator audit proves a narrower exact structure than the historical global dictionary claim: all
  corner entries use three BF16x3 triples with a shared presence mask. The lossless compressed-buffer run60 was
  correct but slower (`3.532192 ms`), and the original-buffer representative-reuse run61 was also correct but slower
  (`3.405163 ms`). Exact corner-coefficient transport is closed as the G3 wall.
- A full host-only joint inversion-symmetric A-low/B-low search found eligible four-vector-green masks, but no
  schedule can omit more than `36/162` low products in the tested action model. Run52 already omits `18/162`, and
  its measured gain was only `0.024918 ms`; another 18 omissions cannot credibly close the `0.377273 ms` gap. No
  hardware run is justified for that mask family.

## Next executable route

1. Keep run52 as the G3 evidence frontier and preserve the hardened runner: portal jobs empty, no conflicting
   workload, no D-state task, no device holder after stopping `tt-fold`, one TT workload per boot ID, runner-owned
   timeout, and unconditional `tt-fold` restoration. Never use `tt-smi -r`.
2. Before another hardware run, design a structural reduction of the remaining arithmetic or B-materialization path
   (for example, an exact algebraic factorization or compute-layout rewrite). Require all four host vectors, every
   offline BRISC/TRISC role, and a traffic/instruction budget capable of removing at least `0.377273 ms`. Affine
   folding, split-A redistribution, exact corner transport, cross-term fidelity regrouping, and another small
   low-product mask are closed lever families.
3. Only after a correct median at or below `3 ms`, implement persistent device-vector PCG and measure G4 with the
   fp64 true-residual decision inside the timed boundary. Then measure cold, proven-cache-hit warm, and explicit
   setup/upload overlap as separate contract workloads.

The older Step 1–3 text below is retained as historical design rationale. Its projected “close G3” language is
superseded by the measurements above and by `../GATE_CONTRACT.md`.

## Why this closes the gates (measured facts, not assumptions)
- The fine SpMV apply is 44ms because the reader **scalar-materializes** b (1024-elem loop/k). Two clean on-device
  diagnostics proved it: contiguous x-reads = 40ms (gather isn't the wall); 1 MAC term vs 6 = 44ms (MAC isn't the wall).
- `mac_reader.cpp` (DMA tile-reads of pre-stored a+b) measured **3.46ms** — the efficient structure that avoids the
  scalar materialize. bf16×2 (5 streams, host-validated precision) → ~2.9ms ≤ 3ms.
- Operator is a **perfect 27-pt stencil**; stencil-shift SpMV = bspmv **machine-exact (rel_err 3.83e-16)**.
  The old “101 blocks” text was a literal print, not a measured dictionary cardinality. Raw coefficients contain
  many noise-distinct blocks; only the eight corners currently have a proved three-triple exact compression. So
  b = shifted x → b-tiles are x-tiles at shifted addresses → DMA-readable, no scalar loop.
- Active lattice is spatially coherent: 4^3-brick tiling covers 1.3× active → operator 1.89–2.36GB → ~1.4ms stream.

## Historical Step 1 — G3-timing measurement (superseded projection)
Goal: measure one SpMV of the efficient MAC ≤3ms (the strategy's own G3 methodology).
Edits to `tt_gmg/tt_spmv.cpp` (harness):
  1. Add sharded buffers `bh,bm` (bf16×2) alongside `ah,am,al` (same shape `n_out_pad*K`, `MakeBuf(...,NCHIP)`).
  2. In `tt_spmv()`: materialize b on host once — `b[e,k]=x[nbr[e,k]]`, split to bf16×2 — and `WriteShard` per chip
     (same per-shard loop as `a`). (For the BENCHMARK, upload once; time only the workload, not the upload.)
  3. Swap reader `gather_reader.cpp` → `mac_reader.cpp`; runtime args = `{ah,am,al,bh,bm,bl addresses, npc, K, start}`
     + `TensorAccessorArgs` for all 6 buffers (mac_reader reads 3 a + 2 b tiles/k when bf16×2 → drop the al/bl reads).
  4. `mac_compute.cpp` → 5 cross-terms (drop `ah·bl`): keep `ah·bh, ah·bm, am·bh, al·bh, am·bm` (the T5 set,
     host-validated fp32-class at apply2 depth in emu_bf16x2b_precision.py).
Run: `whrun_solve.sh` with SPMV_TIMING; read PHASE workload. Expect ~2.9ms → **G3-timing ≤3ms**.

## Step 2 — real stencil SpMV (produce b on-device via shift, no pre-store)
  1. Lay out x on the dense brick-box (from `serialize_tiled.py`'s brick map). x resident (SRAM, ~30MB across 8 chips).
  2. Reader: for each of 27 offsets, DMA-read x at the shifted brick address (in-brick contiguous + 1-voxel halo via
     the serialized `brick_nbr` table) → b-tile. Element-level misalignment (delta mod tile): read 2 adjacent pages +
     realign with the unpacker/`tilize`. No scalar loop.
  3. a = the serialized 27-diagonal stencil operator (`row236_stencil.bin`, bf16×2/×3). Stream from DRAM.
  4. MAC = Step-1 compute. Validate maxU=golden vs the machine-exact host reference (stencil_poc.py).

## Step 3 — x-resident PCG (CLOSE G4/cold/warm)
  Keep x/y resident on device across PCG iters; ship only the scalar dot-products over PCIe (device reduction, or a
  tiny read). Removes per-apply xwrite=5.1 + readback=9.6 + host round-trip. Then 89–228 applies × ~1.4ms + scalars
  ≤ 1s (G4); + G1 setup(cached) → warm ≤1.5s / cold ≤5s.

## Artifacts (this session, on accel-gmg-backend)
- `stencil_poc.py`, `build_tiled_stencil.py`, `serialize_tiled.py` (representation + tiled operator, validated).
- `emu_bf16x2b_precision.py` (bf16×2 precision host-validated).
- `row236_stencil.bin` (serialized operator; regenerate via serialize_tiled.py).
- Efficient kernels present on box: `~/src/tt-metal/tt_metal/programming_examples/spmv_mac/kernels/mac_reader.cpp`.
- GATHER_DESIGN.md has the full measured forensic trail.
