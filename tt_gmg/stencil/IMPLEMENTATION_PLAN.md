# TT-GMG timing-gate closure — executable implementation plan

Status: correctness banked on silicon (G1/G2/G3-correctness/G5). The 4 timing gates (G3-timing ≤3ms, G4 ≤1s,
cold ≤5s, warm ≤1.5s, stretch ≤2s) need the kernel below. Everything here is validated/measured this session.

## Why this closes the gates (measured facts, not assumptions)
- The fine SpMV apply is 44ms because the reader **scalar-materializes** b (1024-elem loop/k). Two clean on-device
  diagnostics proved it: contiguous x-reads = 40ms (gather isn't the wall); 1 MAC term vs 6 = 44ms (MAC isn't the wall).
- `mac_reader.cpp` (DMA tile-reads of pre-stored a+b) measured **3.46ms** — the efficient structure that avoids the
  scalar materialize. bf16×2 (5 streams, host-validated precision) → ~2.9ms ≤ 3ms.
- Operator is a **perfect 27-pt stencil, 101 blocks**; stencil-shift SpMV = bspmv **machine-exact (rel_err 3.83e-16)**.
  So b = shifted x → b-tiles are x-tiles at shifted addresses → DMA-readable, no scalar loop.
- Active lattice is spatially coherent: 4^3-brick tiling covers 1.3× active → operator 1.89–2.36GB → ~1.4ms stream.

## Step 1 — G3-timing measurement (pre-stored b, efficient MAC): CLOSE G3-timing
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
