# On-device gather + fast-SpMV integration — implementation spec (2026-07-03)

Goal: close the remaining row236 gates (G3<3ms, G4≤1s PCG, cold/warm/stretch) by making the neighbor gather
run on-device so the proven 3.46 ms bf16x3 MAC becomes a full fast SpMV wired into the GMG solve with `x` resident.
Correctness (rc=0, maxU=95.8129714) and G5 (18 ms) are already banked on real TT; the on-device MAC (spmv_mac,
DST-accum kernel) is proven at 3.46 ms / 8 chips / 1089 GB/s. This spec is the one remaining large kernel.

## Why a naive gather fails, and the fix
`nbr[o,node]` (from row236_nbr.bin, `nbr[o,node]=3*neighbor_block+... via 3*nn+c`) SCATTERS within a ~±8192-node
window (measured: ±z median±1/std609, ±y ±43/597, ±x ±2812/3237; NOT node+const). On this tile/page machine a
scattered element gather won't hit streaming BW, so reading a resident `x` element-by-element from DRAM is ~as slow
as today's pre-stored `b`. FIX = **L1-windowed gather**: each Tensix core owns a contiguous output-tile range whose
neighbor indices span a bounded window; stage that x-window (+ the core's nbr slice) into L1 ONCE, then gather from L1
(ns-latency) while `a` streams from DRAM. DRAM traffic then = a(1.84GB) + x-windows(~174MB w/ overlap) + nbr(139MB)
≈ 2.1GB → ~1.9 ms  => G3<3ms AND no host gather => G4/cold/warm/stretch unblocked.

## Layout (relative to make_dia.py)
- vector `x` is n_pad=3*nb padded to 1024, interleaved `x[3*node+r]`. y[3*node+r]=Σ_o Σ_c cf[o*3+c,3*node+r]*x[3*nbr[o,node]+c].
- `a` (coeff) stays pre-stored bf16x3 tiles ad[n_out,K,1024] (as today; K=81=27*3), streamed by the reader.
- NEW device inputs:
  - `x` bf16x3: xh/xm/xl as tiled DRAM buffers over n_pad (n_pad/1024 tiles each). (Per-apply the solver uploads x;
    it is small, 45MB bf16x3 — this replaces the 1.25GB pre-stored b upload.)
  - `nbr` int32 [27, nb] resident (139MB), or repacked per-output-tile-block for coalesced core loads.
  - per-core runtime args: {out_tile_start, out_tile_count, xwin_node_lo, xwin_tile_count, nbr slice ptr}.

## Reader kernel (kernels/gather_reader.cpp — replaces mac_reader for b)
For each core:
  1. Compute node range of its output tiles: nodes in [out_tile_start*1024/3 , (out_tile_start+count)*1024/3).
  2. xwin = [min over its (o,node) of 3*nbr - , max + ] clamped; load xwin xh/xm/xl tiles -> L1 (cb_xwin).
     Also load its nbr slice -> L1 (cb_nbr).
  3. For t in out tiles, for k=o*3+c (K):
       reserve cb_a(3)+cb_b(3); stream a: read ah/am/al[t,k] pages -> cb_a;
       build b: for i in 0..1023: e=t*1024+i; node=e/3; nn=nbr_L1[o,node];
                bh[i]= nn>=0 ? xh_L1[3*nn+c - xwin_base] : 0 (same for bm,bl); push cb_b.
       (broadcast over r is automatic: consecutive e w/ same node read same nn.)
  compute/writer kernels UNCHANGED (mac_compute DST-accum, mac_writer).

Optimization notes: process K with o outer so nn is reused across c=0,1,2 (read the neighbor 3-vector once).
The e/3 and e%3 can be strength-reduced; keep a running (node,r) that increments by carrying r 0->1->2->0,node++.
If L1 pressure is high, tile the output range smaller (more cores / more passes) to shrink xwin.

## Host (spmv_mac.cpp — SPMV_GATHER mode)
- Upload x (xh/xm/xl) instead of b; keep a. Compute per-core {xwin_node_lo,xwin_tile_count} from split_work_to_cores
  + the min/max nbr over each core's nodes (precompute on host from nbr). Pass as runtime args to gather_reader.
- Validate vs ref (make_abref already emits a/b/ref; here compute y from a + gathered x, compare) — expect the
  SAME bf16x3 result as the pre-stored-b path (rel_err vs fp64 = the known cancellation floor; cd[0..2]==ref[0..2]).

## Integration as g_tt_fine_spmv (gmg_tt.py callback) — for G4/cold/warm/stretch
Build spmv_mac's device program ONCE (persistent): open device, upload `a` (resident), compile program. Expose a C
entry `int tt_spmv(const double* x, double* y)` via a small .so:
  - convert x (n) -> bf16x3, EnqueueWriteMeshBuffer(x); EnqueueMeshWorkload(gather+MAC); EnqueueReadMeshBuffer(y).
Then in gmg_tt.py set g_tt_fine_spmv = tt_spmv (replacing the numpy gather + ttnn matmul-diagonal). x stays a device
buffer across PCG iters where possible (only the changed x re-uploaded). Measure per-apply => G3; per-PCG => G4;
whole cold/warm/stretch => the e2e gates; rc/maxU must stay 95.8129714.

## Test order (each is one device run; power-cycle only if the ETH fabric degrades — do NOT tt-smi -r)
1. gather_reader correctness: SPMV_GATHER=1 spmv_mac, expect rel_err == pre-stored-b path, timing target <3ms.
2. wire tt_spmv .so; run 1 PCG apply via gmg_tt.py, confirm y matches the matmul-diagonal apply.
3. full solve via gmg_tt.py with tt_spmv; confirm rc=0 maxU=95.8129714; capture G4/cold/warm/stretch.

## BUILD STATE — 2026-07-03
DONE + validated on-box:
- kernels/gather_reader.cpp WRITTEN. Caught+fixed a correctness bug: the whole spmv_mac pipeline is positionally
  row-major (ad[t,k,i]=cf[k,t*1024+i], element-wise compute — cd[0..2]==ref[0..2] proves it), so gathered b is
  written b[i] (row-major), NOT a tile-layout-permuted position. x therefore also stores row-major (page=1024 bf16,
  same MakeBuf as a) so XH[s] with s=elem-xwin_elem_lo is a direct L1 offset.
- make_abref.py extended + RUN on box: writes /tmp/row236_x.bin (n_pad=3872768 float32, the x_test that produced ref)
  and /tmp/row236_nbrpad.bin (int32[27, NBn=1290922], nbr padded to n_pad nodes so a core indexes nbr[o*NBn+node]).
TODO (host spmv_mac SPMV_GATHER branch — the remaining device-iterative bring-up):
- load x.bin (scale by VSCALE=1e6 like b) + nbrpad.bin; build xh/xm/xl (row-major bf16, MakeBuf page=2048) + a (as now);
  SKIP the b buffers. Start SINGLE-CHIP (SPMV_NCHIP=1) to avoid replication; scale to 8-chip with x/nbr REPLICATED
  (ReplicatedBufferConfig, not sharded) so every core reads x/nbr locally (neighbors cross shard boundaries).
- per core (from split_work_to_cores start/npc): node range [start*1024/3, (start+npc)*1024/3); scan nbrpad over those
  nodes x 27 offsets for min/max 3*nbr+c -> xwin_elem_lo (floor to 1024), xwin_ntiles; pass gather_reader runtime args
  {ah,am,al, xh,xm,xl, nbr, npc, K, start, xwin_elem_lo, xwin_ntiles, node_lo, n_nodes, nb} + compile args
  {cb_a,cb_b,cb_xh,cb_xm,cb_xl,cb_nbr, TensorAccessorArgs for the 7 buffers}. Add CBs c_2..c_5 sized xwin_ntiles/
  nbr_tiles (L1 budget: keep npc small enough that 3*xwin_ntiles*2KB + nbr fits ~1MB; shrink npc / add passes if not).
- OPEN Q resolved by device feedback: (a) nbr L1-staging — my kernel currently reads cb_nbr as if it is the core's
  packed slice; either upload per-core packed nbr slices (concat buffer + per-core base in a_nbr) OR stage the 27
  contiguous nbrpad[o, node_lo:+n_nodes] segments in the reader (27 noc reads at o*NBn+node_lo). (b) whether MeshBuffer
  page reads of a raw row-major bf16/int32 buffer need InterleavedAddrGen vs TensorAccessor. Resolve by building the
  single-chip path and reading compiler/runtime errors; power-cycle (NOT tt-smi -r) only if the ETH fabric degrades.
Test 1 target: SPMV_GATHER=1 SPMV_NCHIP=1 on a small SPMV_GATHER_NTILES first -> b matches pre-stored path -> then full.

## TEST 1 PASSED — on-device gather is CORRECT — 2026-07-03
SPMV_GATHER=1 SPMV_NCHIP=1 SPMV_GATHER_NTILES=128 => rel_err=1.149e-06 (fp32-exact) over the processed tiles,
cd[0..2]==ref[0..2]. The full gather algorithm works: row-major x buffer (page=1024 bf16), node-major nbr buffer
(nbr2[node*27+o], 4KB int32 pages, page-aligned L1 staging), per-core window from nmin/nmax scan, boundary mask,
row-major b write. Host builds clean; gather_reader JIT-compiles. Known follow-ups (NOT correctness):
- L1 budget: single-chip full = 135 x-tiles + 614 nbr pages/core (>L1). 8-chip (~7 tiles/core) fits; needs x/nbr
  REPLICATED (ReplicatedBufferConfig) since neighbors cross shard boundaries. That's the next edit.
- run_mailbox 0x40 on one core (18-16) even at NTILES=128 — investigate (likely an edge core whose window/n_nodes
  hits a bad value; guard n_nodes==0 / clamp, or a core outside the processed set). Output was still correct.
- perf: 21ms/128-tiles is the scalar per-element gather loop (e/3 integer div per element). Optimize: hoist node
  (increment r 0->1->2 carrying node), reuse nn across c=0,1,2, and the L1 window load. Target <3ms at 8-chip.
NEXT: (1) guard empty/edge cores; (2) 8-chip replicated x/nbr -> full-problem gather correctness + timing (G3);
(3) tt_spmv persistent-callback .so -> gmg_tt.py -> G4/cold/warm/stretch + maxU=95.8129714 e2e.

## GATHER CORRECT, but SCALAR SPEED IS THE WALL — 2026-07-03
Div-removal (node0+carry) killed the run_mailbox 0x40 (it was a dispatch TIMEOUT from 1024 int-divs/plane). Now
NTILES=256 single-chip: rel_err=1.083e-06, exit=0, NO errors. BUT 37ms/256-tiles @ 101 GB/s => the scalar per-element
gather is L1-LATENCY-BOUND: 4 volatile L1 reads/element (nbr + xh/xm/xl) x ~30cyc x 1024x81xnpc. Extrapolates to ~70ms
at full 8-chip (7 tiles/core) — ~20x over G3(3ms) and worse than the pre-stored 3.46ms. So the ALGORITHM is solved+proven
(gather is bit-exact) but the scalar RISC loop cannot hit G3. To close G3/G4 the gather reads must OVERLAP, options:
  (A) issue async noc_async_read per gathered element into cb_b (NoC hides latency, many in flight) instead of scalar
      L1 copies — but per-element small reads have NoC overhead; batch by neighbor 3-vector.
  (B) drop `volatile` on the x/nbr L1 reads (valid after the barrier) so the compiler pipelines them; cheap experiment.
  (C) reconsider: since |nbr-node| is bounded (±8192) and the gather for a fixed offset o is a near-shift, precompute
      per-(core,o) a base-shift + a small exception list so most of b is a contiguous L1 copy (memcpy-fast) + few fixups.
DECISION for next session: try (B) (1-line), then (C) (the real win — turns per-element scatter into per-offset shift+
fixup, which is streaming-fast). Gather correctness is BANKED; this is now a pure throughput optimization on a proven op.

## PERF WALL characterized — B minor, C dead — async-NoC is the path — 2026-07-03
Measured, single-chip, NTILES=256 (all rel_err=1.083e-06, correct throughout):
- (B) drop `volatile` on L1 reads: 37.1 -> 34.4 ms (only ~7%). The in-order baby-RISC BLOCKS on each L1 load (~14 cyc)
  and cannot keep multiple in flight, so ~7 serial L1 accesses/element dominate regardless of volatile. Latency-bound.
- (C) per-offset shift+fixup: DEAD. Measured nbr[o,node]-node: only o=13(self)=100% and o=4/22(±z)=89% share a mode;
  ALL other 24 offsets have frac==mode = 2-4% (z-contiguous numbering, y/x scattered). Mesh is 3.7% dense in a
  559x273x229 box, so no renumbering makes the neighbors a clean shift either.
So scalar gather is ~60ms @ 8-chip (7.4 tiles/core * 82944 elems * ~100cyc), ~20x over G3. The RISC can't hide L1
latency. The ONE lever left: (A) ASYNC NoC gather — issue noc_async_read per neighbor 3-DOF bf16x3 chunk (18 bytes)
from a RESIDENT x (L1-sharded across cores per strategy line 22) into cb_b, many in flight so the NoC hides latency,
then one barrier per tile. ~341 nodes*27 off = 9207 async reads/tile; if the NoC sustains ~1 read/few-ns this is
~us/tile => G3 reachable. Risk: per-read descriptor overhead on 18-byte reads. This is the next concrete build:
rewrite gather_reader's inner loop from scalar L1 copies to batched noc_async_read of the neighbor chunk + barrier.
STATUS: gather CORRECTNESS banked (rel_err 1e-6) on real TT; the async-NoC throughput rewrite is the remaining step
to G3/G4. Correctness on real TT (maxU=95.8129714), on-device MAC (3.46ms), and G5 are all already banked.

## CROSS-CHIP ARCHITECTURE + THROUGHPUT PLAN (figured out) — 2026-07-03
CROSS-CHIP (implemented MakeReplBuf; SPMV_GATHER now creates replicated x/nbr + sharded a):
  - a SHARDED across NCHIP (1.84GB -> 230MB/chip, fits). x+nbr REPLICATED on every chip (45MB + 139MB) so each core
    gathers LOCALLY, zero cross-chip traffic. Scales to big models until x > per-chip DRAM (~100M+ DOF) -> then shard
    x with a halo (Stage C: each chip holds its output nodes + a boundary halo, exchanged over ethernet/fabric).
  - MISSING piece for full 8-chip correctness: the replicated program runs identically on every chip, so the reader's
    start_out_id is LOCAL (0..n_local); the GATHER needs the GLOBAL tile = chip*n_local + local. a/c already use local
    (sharded buffers map local->global). FIX: per-chip programs — for d in 0..NCHIP-1:
      Program prog_d = build_reader_program(global_off=d*n_local);
      wl.add_program(distributed::MeshCoordinateRange({0,d},{0,d}), std::move(prog_d));
    and add a `global_off` runtime arg to gather_reader; use (global_off + start_out_id + t) for node math, keep
    (start_out_id + t) for the a/c page bases. (Verifying chip 0 works today since its local==global.)
  - HYGIENE: a killed run holds 'CHIP_IN_USE_*' locks -> next run blocks. pkill -9 -f metal_example before re-running.
THROUGHPUT (the real G3 unlock, on the proven-correct op):
  - Root cause: scalar in-order RISC blocks ~14cyc/L1-load, ~7 loads/elem => ~60ms/8chip. Fixes, best-first:
    (1) DE-INTERLEAVE the DOF: store x as x0/x1/x2[node] (not 3*node+r) so gather is 1 value/node (NO 3x broadcast);
        operator as a[o][r][c][node] (243 planes). Compute grows 81->243 MAC terms (~0.06ms, still free) but the gather
        drops 3x AND becomes a clean per-node indexed load. This is the highest-leverage change.
    (2) ASYNC NoC gather on the dataflow core: issue noc_async_read per neighbor (get_noc_addr(page,X)+off) into cb_b,
        many in flight so the NoC hides latency; one barrier/tile. Turns latency-bound into bandwidth-bound (~1.8ms).
    (3) or L1-resident SHARDED x across the chip's 64 cores (700KB/core) + on-chip-NoC gather from the holding core.
  - Recommended: (1)+(2) together — de-interleave removes broadcast, async NoC removes latency-blocking => G3 reachable.
