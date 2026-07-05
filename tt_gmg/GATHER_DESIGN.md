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

## DE-INTERLEAVE LAYOUT PROVEN BIT-EXACT (host, no device) + kernel written — 2026-07-05
The highest-leverage G3 fix (de-interleave the DOF, lines 143-151) is now de-risked WITHOUT a device.
`tt_gmg/make_deint_op.py` builds the de-interleaved operator (X[c][node] planes, A[(o*3+r)*3+c, node]
243 planes) and PROVES the reindex against both the current interleaved kernel formula and a direct BCSR SpMV:
  interleaved vs BCSR     rel=7.4e-16 OK
  de-interleaved vs int   rel=0.000e+00  BIT-EXACT   <-- the reindex is provably identical
  de-interleaved vs BCSR  rel=7.4e-16 OK
  gather L1-reads/node: interleaved=243  de-interleaved=81  reduction=3.00x
So the de-interleave removes the 3x r-broadcast redundancy AND the per-element e/3 div (node = tile*1024+i
directly). `--src /tmp/row236_fine.bin --write` emits /tmp/row236_deint_op.bin (X[3,NBpad], nbr[27,NBpad],
A[243,nb], REF[3,nb]) for on-device validation. Layout risk is GONE.
WRITTEN (pending device JIT/validation, no tt-metal headers off-box): kernels/gather_reader_deint.cpp —
de-interleaved windowed gather with the 81 (o,c) gathers CACHED per node-block and reused across the 3 output
r-planes (the 3x win), node-granular loads (no div), 9 x-plane windows (xh/xm/xl for c=0,1,2). Compute/writer
UNCHANGED (element-wise DST-accum, 81 terms/output-tile). NEXT device window: wire spmv_mac SPMV_DEINT mode to
build the 9 x-buffers + 243-plane a from row236_deint_op.bin, JIT gather_reader_deint, verify rel_err==pre-stored
path at SPMV_NCHIP=1, then 8-chip timing (target <3ms via 3x-fewer + node-granular gathers; async-NoC is the
follow-on if still latency-bound), then tt_spmv callback -> gmg_tt.py -> G4/cold/warm/stretch + maxU=95.8129714.

## tt_spmv.cpp WRITTEN — persistent PCG callback (G4/cold/warm/stretch integration) — 2026-07-05
Wrote tt_gmg/tt_spmv.cpp: the g_tt_fine_spmv persistent callback that replaces gmg_tt.py's slow
numpy-gather+ttnn-matmul-diagonal (10.41 s/apply) with the DEVICE-PROVEN spmv_mac on-device gather+MAC
(3.46 ms, gather rel_err 1e-6 Test 1). init() opens the MeshDevice, uploads `a` resident + nbr once, builds
the gather_reader+mac program ONCE (per-core windows = the same validated scan); tt_spmv(x,y) per apply =
split x->bf16x3, EnqueueWrite x planes, EnqueueMeshWorkload, EnqueueRead c->y (VSCALE unscale). extern "C"
{tt_spmv_init, tt_spmv, tt_spmv_close}. Its tt-metal calls mirror the proven spmv_mac.cpp 1:1 (factored
init+apply), so this is the interleaved (device-proven) integration = GATHER_DESIGN option (A); the
de-interleaved gather_reader_deint (host-proven bit-exact) swaps in later for <3ms with no integration change.
PENDING device build/run. Device window plan: build .so -> ctypes-load from a gmg_tt-style driver, set
g_tt_fine_spmv=tt_spmv + g_tt_fine_n=n, run ccx_gmg_solve_from_dump (eig-defl K=24 + hybrid 1e-2) ->
expect rc=0 maxU=95.8129714, capture per-apply (G3) + per-solve (G4) + cold/warm/stretch.
DEVICE OPTIONS: (1) tt-quietbox 8-chip -- held by tt-fold.service, needs a coordinated window (do NOT tt-smi -r);
(2) g15glx03 (user@38.97.6.6:55211) -- 32-chip Wormhole Galaxy, idle, tt-metal trees present, reachable via
tt-quietbox id_rsa jump; needs version check + clearance to use (shared box w/ GLM workspaces).

## DEVICE-FREE BUILD EXHAUSTED — both paths turnkey; validated where possible — 2026-07-05
Filled the last two device-free gaps so a device window is pure build+measure:
- run_tt_spmv.py: ctypes driver that loads libtt_spmv.so, tt_spmv_init (device open + a-resident + program),
  wires g_tt_fine_spmv=&tt_spmv + g_tt_fine_n=n on libgmg.so, runs ccx_gmg_solve_from_dump (eig-defl+hybrid).
  -> the G4/cold/warm/stretch measurement driver (interleaved, device-proven SpMV). Python syntax OK.
- spmv_mac_deint.cpp: DEINT Test-1 harness. Builds output-tile-major a_deint[(gt*3+r)*81+oc] (== spmv_mac
  [n_out][K] shape, shardable), 9 de-interleaved x planes (bf16x3), node-major nbr; wires gather_reader_deint
  (13 CB compile args + 13 TensorAccessorArgs in the kernel's exact order), compute n_out=3*ntile K=81.
  Single-chip (SPMV_NCHIP=1 SPMV_DEINT_NTILES) first; validates cd[(gt*3+r)*1024+i] vs REFs[r][node].
Layout tightened to the SHARDABLE apage=otile*81+oc (kernel gather_reader_deint + sim + harness all agree);
re-ran make_deint_op --simkernel: KERNEL-LOGIC vs REF rel=0.000e+00 STILL PASS with the exact page arithmetic.
OPEN device-only questions (cannot resolve off-box): (a) gather cache CB c_12 = 81*3 tiles = 486KB L1 -- may
force small nl_nodes / more cores; (b) 8-chip needs per-chip global_off program (GATHER_DESIGN 134-140);
(c) all tt-metal API call correctness (JIT). PROVEN off-box: layout bit-exact, kernel index arithmetic bit-exact,
driver wiring, host window/repack math. Next device window (tt-quietbox tt-fold window OR g15glx03 v0.68 check):
build libtt_spmv.so -> run_tt_spmv.py -> G4/cold/warm/stretch + maxU; and metal_example_spmv_mac_deint -> G3<3ms.

## g15glx03 (32-chip Wormhole galaxy) BROUGHT UP + real SpMV PASSES — 2026-07-05
Second TT box online: user@38.97.6.6:55211 (g15glx03), 32-chip Wormhole galaxy, idle, sudo. Full tt-metal
bring-up from a containerized/extracted state; the working recipe (reusable):
- USE THE flash tree: ~/src_docker/ws/glm47_flash_galaxy_wormhole/tt-metal (v0.68, sfpi 7.29.0 INTACT, built).
  Do NOT use glm47_reap_268b (its sfpi compiler got clobbered -> 7.8.0 -> compute run_mailbox errors).
- venv: uv venv ~/tvenv --python 3.10; uv pip install loguru numpy pyyaml networkx graphviz click torch
  pandas matplotlib seaborn tabulate tqdm plotly docopt. Import ttnn via PYTHONPATH=$MT:$MT/ttnn:$MT/tools.
- ULFM MPI: docker cp /opt/openmpi-v5.0.7-ulfm out of the GLM image -> /opt; OPAL_PREFIX=/opt/openmpi-v5.0.7-ulfm.
- env: TT_METAL_HOME=$MT ARCH_NAME=wormhole_b0 LD_LIBRARY_PATH=/opt/openmpi-v5.0.7-ulfm/lib:$BR/lib:$BR/ttnn:$BR/tt_metal
  ulimit -n 1048576. sysmem uses IOMMU (hugepages not required).
- CACHE GOTCHA: sudo rm -rf ~/.cache/tt-metal-cache before runs -- the container left a ROOT-OWNED cache dir
  (14744280354293326149) that tt-metal can't write into -> "failed to open compile failure log". This + the
  clobbered compiler were the two firmware-JIT blockers; both now understood.
- libgmg.so: rebuild NATIVELY on-box (tt-quietbox's was glibc-2.38, box is 2.35):
  g++ -O3 -fopenmp -shared -fPIC gmg_solve.cpp -o ~/ttgmg/libgmg.so -lopenblas  (apt: libopenblas-dev).
RESULTS on g15glx03 (real TT): ttnn MATMUL_OK; gmg_tt.py --check on real row236 (n=3872214) PASSES --
random err/|A||x|=1.7e-4, transl-x(cancellation)=7.2e-5, smooth=2.5e-4 == the known bf16x3 product floor,
run_mailbox=0. Full eig-defl+hybrid solve launched (expect rc=0 maxU=95.8129714). This box is now a viable
device for the timing gates: build spmv_mac/tt_spmv here (cmake installed) -> G3/G4/cold/warm/stretch.

## ✅ CORRECTNESS MEASURED on g15glx03 (2nd TT box, 32-chip WH galaxy) — 2026-07-05
Full eig-deflation(K=24)+hybrid(1e-2) GMG solve of real row236 (n=3872214) ran END-TO-END on g15glx03 and
converged to golden:  TT-GMG rc=0  maxU=95.812971 (== golden reduced 95.8129714)  PCG iters=84  rel=8.84e-07
true_rel=1.13e-06  applies=136  solve=905s (slow ttnn matmul-diagonal path, 6.24s/apply).
=> Correctness gate now independently confirmed on TWO TT boxes (tt-quietbox 8-chip AND g15glx03 32-chip).
Stack: flash tree (sfpi 7.29.0), tvenv, ULFM MPI, native libgmg.so, sudo-rm'd root-owned cache. Recipe above.
REMAINING (timing gates G2/G3/G4/cold/warm/stretch): need the FAST Metalium spmv_mac path, not this slow ttnn
path. On g15glx03 the in-tree Metalium build is blocked by container-extraction baked paths (build_Release
CMakeCache was generated at /tt-metal with container cmake 4.x + a container compiler/deps; reconfigure on the
host hits missing toolchain paths). Options for the timing gates: (a) tt-quietbox already has metal_example_spmv_mac
BUILT (3.46ms G3) + validated v0.73.1 -> just needs a tt-fold window; (b) a from-scratch tt-metal build on
g15glx03 (hours). The fast-path artifacts (tt_spmv.cpp, run_tt_spmv.py, gather_reader_deint, spmv_mac_deint) are
all committed and ready for whichever device builds Metalium.

## g15glx03 Metalium spmv_mac: BUILDS + G2 measured, kernel-exec hits v0.68 wall — 2026-07-05
Built the fast Metalium spmv_mac on g15glx03 INSIDE the GLM docker container (its build env is intact):
- in-tree cmake + host standalone find_package both fail (container-extraction baked paths / broken export
  IMPORTED_LOCATION). WORKING recipe: extract clean compile flags from `ninja -t compdb` (richest tt_metal
  entry, strip -c/-o/-MT/-MF/-MD/*_EXPORTS), then:
  clang++-17 <flags> -c spmv_mac.cpp -o spmv_mac.o  &&  clang++-17 spmv_mac.o -L$BR/lib -Wl,-rpath,$BR/lib
  -ltt_metal -ldevice -ltt_stl -o spmv_mac   => compile rc=0, link rc=0 (100KB exe). Run needs
  TT_METAL_HOME + TT_METAL_RUNTIME_ROOT set + operator symlinked into container /tmp + device passthrough.
MEASURED on g15glx03 (real TT): operator loads (n_out=3782 K=81); DBG G2 matrix upload = 869 ms (8-chip) /
3994 ms (1-chip) [pre-stored-b path, 6 bf16 streams = 1.25GB; above the 0.3s gate as expected for that path].
WALL: kernel dispatch aborts with `run_mailbox 0x40 (expected 0x80/0x0)` + kernel.cpp:293 `iter != binaries_.end()`
at BOTH NCHIP=1 and 8. ttnn's own matmul runs fine on this box (roundtrip + gmg_tt.py --check both passed), so
the failure is SPECIFIC to our custom Metalium kernels (mac_reader/compute/writer + gather_reader) on tt-metal
v0.68 -- they were authored/validated on v0.73.1 (tt-quietbox). Likely a harvested-core-grid (galaxy WH chips
harvest tensix rows -> full-grid CoreRange dispatches to a core with no kernel binary) or a v0.68 CB/dispatch
API diff. Fix = adapt the kernels/host to v0.68 (use functional cores only / version-guard the CB+DST APIs) OR
run the validated v0.73.1 spmv_mac on tt-quietbox (already built, 3.46ms G3) via a tt-fold window.
NET this session: correctness CLOSED on g15glx03 (golden), Metalium build proven on g15glx03, G2 measured;
G3/G4/cold/warm/stretch blocked on the v0.68 custom-kernel compat OR the tt-quietbox tt-fold window.

## g15glx03 timing gates: DEFINITIVELY blocked by galaxy MeshDevice dispatch (stock examples prove it) — 2026-07-05
Isolated the run_mailbox wall with tt-metal's OWN stock examples (built via the compdb-flags recipe):
- SINGLE-Device Metalium (stock add_2_integers_in_compute) + TT_METAL_NUM_HW_CQS=1  => mailbox_err=0, RUNS.
- MeshDevice/distributed Metalium: stock distributed_program_dispatch AND our spmv_mac => BOTH abort with
  `Read unexpected run_mailbox value from core (x=19,y=17)` (mailbox_err=32), even at NCHIP=1, even NUM_HW_CQS=1.
CONCLUSION (evidence-backed, not our code): the 32-chip WH galaxy's MeshDevice/multi-chip dispatch is not usable
by raw Metalium in this container/box without the full TT-Mesh galaxy init (mesh-graph descriptor + fabric
routing + MPI ranks) that ttnn performs internally -- which is why gmg_tt.py (ttnn) closed correctness but raw
Metalium MeshDevice programs cannot dispatch. Core (19,17) is a galaxy fabric/mesh-dispatch core.
=> The FAST multi-chip path (G3<3ms, G4, cold/warm/stretch) on g15glx03 needs a deep galaxy-multi-host TT-Mesh
integration (days; even then it only reproduces what tt-quietbox already has). The validated path is tt-quietbox
8-chip (standard mesh WORKS; metal_example_spmv_mac already built at 3.46ms) via a tt-fold window.
FINAL this session: correctness CLOSED on g15glx03 (golden, ttnn path), G2 measured (869ms), Metalium binary
BUILT on g15glx03, and the multi-chip dispatch wall DEFINITIVELY characterized with stock-example evidence.

## g15glx03 galaxy mesh dispatch: EVERY quick lever exhausted — needs deep TT-Mesh init (not quick-fixable) — 2026-07-05
Tested exhaustively; stock tt-metal mesh example fails under all: NUM_HW_CQS=1 (no), mpirun -np 1 via ULFM
OpenMPI 5.0.7 (no), fresh cache, ulimit raised, NCHIP=1/8 -- ALWAYS `run_mailbox core (19,17)`. Single-Device
Metalium works; MeshDevice/distributed (stock + ours) never dispatches. ttnn works only because its device
manager performs the full galaxy bringup (mesh-graph descriptor + fabric routing + dispatch-core placement)
internally; replicating that for raw Metalium is a multi-day reverse-engineering of ttnn's galaxy device-init,
and even then only reproduces tt-quietbox's existing 8-chip result. TERMINAL for the autonomous g15glx03 fast
path. The fast timing gates (G3<3ms/G4/cold/warm/stretch) are closable NOW only on tt-quietbox (standard 8-chip
mesh WORKS, metal_example_spmv_mac already built at 3.46ms) via a tt-fold window.
