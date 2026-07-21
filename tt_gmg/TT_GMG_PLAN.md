# TT-GMG — Execution Plan to the Goal State (row236, 8×Wormhole, eight performance gates + correctness)

> Accounting note (2026-07-15): `GATE_CONTRACT.md` supersedes status/count wording in this historical plan.
> Correctness is an acceptance invariant, not a ninth performance gate; projections and partial workloads do not
> close gates. The implementation phases remain useful, but each exit must now emit evidence accepted by the
> machine-readable contract.
>
> Device-frontier note (2026-07-15): run52 is the best qualifying correct persistent resident-x apply. Its
> default-off selective b-low policy measured `3.379803/3.377273/3.256412 ms`, median `3.377273 ms`, so G3 remains
> red by `0.377273 ms`. Fixed exact-leading HiFi2 is correct but much slower; cross-term mixed-fidelity grouping is
> closed device-incorrect. Runs54–57 also close affine A-low as device-incorrect. Correct split-A run59b and
> lossless exact-corner runs60/61 are slower (`3.505936/3.532192/3.405163 ms` medians). The host-only joint-mask
> search tops out at 36 omitted low products versus run52's 18, insufficient at the measured slope. The next G3
> work must structurally reduce remaining arithmetic or B-materialization and pass host/offline proof before a boot.
> Persistent PCG/G4 still follows only after a correct ≤3 ms apply exists.

Owner: mickg. Target: `calculix-fork/TT_GMG_STRATEGY.md` acceptance — row236 solves on 8×Wormhole,
cold ≤ 5 s / warm ≤ 1.5 s / stretch ≤ 2 s, `maxU = golden`, fp64 host PCG + residual gate intact.
This document is the step-by-step route from **where we are** to **that state**. Every phase ends in a
**measured** exit gate, not a description. Do not advance a phase until its gate is green on the real box.

--------------------------------------------------------------------------------
## 0. Ground truth — where we are TODAY (all measured on the real 8-Wormhole box)

BANKED (proven, do not re-litigate):
- **Correctness on real TT**: full GMG solve `rc=0`, `maxU = 95.8129714` == golden (reduced problem). The
  eig-deflation + fp64-hybrid algorithm converges on real silicon.
- **On-device DIA MAC** (SpMV with b pre-gathered), 8-chip: **3.46 ms/apply, 1089 GB/s** — fp32 output,
  bf16×3 compensated, validated. This is ~4200× the host matmul-diagonal path.
- **On-device gather** (fold x→b via nbr): **bit-exact, rel_err 1.08e-6**, but **scalar-slow**
  (~34–37 ms / 256 output tiles, single chip; ~60 ms projected at 8-chip). Correctness banked; throughput is the wall.
- **G2 upload** 0.62 s (bf16×3, borderline vs 0.3 s gate). **G5 output read** 18 ms (passes ≤0.2 s).
- **Cross-chip scaffolding**: `MakeReplBuf` added — x + nbr **replicated** per chip, a **sharded**. Chip-0 gather
  is correct today; full 8-chip needs a per-chip global output offset (Phase 2).

CURRENT END-TO-END (honest): the only *complete* solve today is the **slow host path** ≈ **25 min**
(host numpy gather + 32×-wasteful matmul-diagonal). Correct, but slower than the CPU baseline (~22 s).
The fast SpMV kernel exists and is proven, but is **not yet integrated** into the solve loop.

THE GAP, in one sentence: we have a *correct* solve and a *proven-fast* SpMV kernel, but not a *fast
end-to-end solve* — because (a) the gather is latency-bound, and (b) the fast kernel isn't wired as the
PCG SpMV callback. This historical plan closes those two gaps, then measures all eight performance gates plus the
separate correctness invariant.

--------------------------------------------------------------------------------
## 1. The goal state — eight performance gates plus correctness

| id | phase | budget | today | closes in |
|----|-------|-------:|------:|-----------|
| G1 | host setup (hierarchy+BCSR+bf16×3), **cacheable** | ≤ 3.0 s | ~ (host, cached warm) | Phase 5 (measure) |
| G2 | matrix upload host→8 chips (one-time) | ≤ 0.3 s | 0.62 s | Phase 5 (optimize) |
| G3 | one fine SpMV (8 chips, bf16×3) | ≤ 3 ms | MAC 3.46 / gather ~60 | **Phase 1–3** |
| G4 | PCG solve (≈228 SpMVs + host fp64 PCG + PCIe residual) | ≤ 1.0 s | — | **Phase 4** |
| G5 | output / un-permute | ≤ 0.2 s | 0.018 s ✅ | done |
| — | **cold total** | ≤ 5.0 s | ~1500 s | Phase 5 |
| — | **warm total** (setup cached) | ≤ 1.5 s | — | Phase 5 |
| — | **stretch** (overlap setup+upload) | ≤ 2.0 s | — | Phase 6 |

Correctness gates (non-negotiable, must hold at every phase that touches numerics):
`maxU` within acceptance of golden; **fp64 outer PCG + true-residual gate stays on host** so a bad TT
result can never be returned (falls back to CPU GMG / SPOOLES); converges to solver tol every run.

--------------------------------------------------------------------------------
## 2. Critical path (the shortest route to the first FAST end-to-end number)

```
Phase 1 (gather throughput)  ──►  Phase 3 (fuse gather+MAC, resident x)  ──►  Phase 4 (PCG callback)  ──►  Phase 5 (measure gates)
        │                                                                            ▲
        └──►  Phase 2 (8-chip global-offset gather correctness) ─────────────────────┘
```
Phase 1 and Phase 2 are independent and can proceed in parallel; both are required before Phase 3.
Phase 6 (scale-up) is after acceptance. **First milestone that matters: end Phase 4 with a correct,
fast end-to-end solve number — even if it misses a gate — because that converts the 25-min baseline
into a real optimization target.**

--------------------------------------------------------------------------------
## PHASE 0 — Baseline lock & harness hygiene  (0.5 day)

**Objective:** make every subsequent measurement reproducible and never lose the box to a stale lock.

Steps:
1. Use only the hardened `run_brick_hw.sh`: it requires empty portal jobs, a fresh boot ID, a settled active
   `tt-fold`, no conflicting workload, no D-state task, and no device holder after its controlled service stop; it
   marks the boot before mesh open, owns the timeout, and restores `tt-fold` in an EXIT trap. Never use
   `pkill -9 -f`, never manually interrupt a TT workload, and never run a second workload on the same boot.
2. Bank the three current numbers into `GATHER_DESIGN.md` as the "before" row: MAC 3.46 ms, gather
   34 ms/256-tiles, e2e 25 min. (Already recorded — verify still present.)
3. Confirm the fp64 host residual gate path in `gmg_tt.py` is active and un-bypassed (grep for the
   fallback-to-CPU branch); this must never be removed by any optimization.

**Exit gate:** one command runs the single-chip gather subset and prints `rel_err` + `ms` with no manual
lock cleanup; the fp64 gate is confirmed present.

--------------------------------------------------------------------------------
## PHASE 1 — Gather throughput: the G3 unlock  (the hard core; 1–2 weeks)

**Objective:** bring the on-device gather from ~60 ms (8-chip) to the point where **gather+MAC fused ≤ 3 ms**
(G3). This is the single highest-value piece of work in the project.

**Root cause (measured):** the scalar, in-order baby-RISC blocks ~14 cyc on each L1 load; ~7 L1 accesses
per output element (1 nbr + 3 xh/xm/xl, plus the ÷3/broadcast bookkeeping) ⇒ latency-bound. Two "easy"
escapes were measured and rejected: dropping `volatile` (7% only — RISC can't pipeline); per-offset
shift+fixup (dead — only ±z shares a dominant delta; the mesh is 3.7%-dense so numbering is scattered).

### 1a. De-interleave the DOF layout  (banked 3× win + simpler index — DO THIS FIRST)
Today x is `x[3·node+r]`, operator `cf[o·3+c, 3·node+r]` (81 planes), and the gather value
`x[3·nbr[o,node]+c]` is **broadcast over r=0,1,2** ⇒ every unique gather is done 3×.
Change to **component-major**: store `x0[node], x1[node], x2[node]`; operator `a[o][r][c][node]`
(243 planes); output `y_r[node] = Σ_o Σ_c a[o,r,c,node]·x_c[nbr[o,node]]`.
- Gather becomes **one indexed load per (node,o,c) = 81·nb**, i.e. **3× fewer gathers**, and the index is
  `x_c[nbr[o,node]]` with **no ÷3 and no broadcast**.
- The r-broadcast moves to the **compute** side (same gathered b tile feeds 3 a-planes) — cheap, in DST/L1.
- Compute grows 81→243 MAC terms ≈ +0.04 ms — free (BW-bound, MAC is 3.46 ms with headroom).
- Work: edit `make_abref.py` / `make_dia.py` to emit component-major x + 243-plane a + node-major nbr;
  edit `gather_reader.cpp` inner loop (gather 1 value/node, drop the rr-carry); edit `mac_compute.cpp`
  to iterate 243 planes with the r-broadcast. Re-validate `rel_err ≤ 1e-6` on the subset.

**Exit gate 1a:** single-chip subset gather still `rel_err ≤ 1e-6`, wall drops ≈ 3× (target ≤ ~13 ms/256-tiles).

### 1b. Diagnose the residual bottleneck  (measure before optimizing further)
After 1a, re-measure and attribute the remaining time: instrument the kernel to time (i) the one-time
x-window + nbr staging vs (ii) the per-tile gather loop. Decide latency-bound vs bandwidth-bound with a
2-point test (halve the gathered values, see if time halves). This picks 1c.

### 1c. Latency-hiding — pick per 1b result (the real ≤3 ms push)
- **Option A — DRAM-resident x + async NoC gather** (best if latency-bound): keep x in DRAM (not L1),
  issue `noc_async_read(get_noc_addr(page,X)+off)` per gathered value into `cb_b`, **many in flight**,
  one `noc_async_read_barrier()` per tile. NoC hides per-read latency ⇒ bandwidth-bound. Risk: descriptor
  overhead on tiny (2–4 B) reads — batch by node (read the 3 components / a small nbr run as one
  transaction where contiguous).
- **Option B — multi-core L1-sharded x + on-chip NoC** (best if L1-bandwidth-bound): shard x across the
  chip's ~64 cores (~700 KB/core), gather from the holding core over the on-chip NoC.
- **Option C — SFPU/vectorized gather** (if the scalar loop itself dominates): use the vector engine's
  gather/permute to do 32 lanes at once.
Prefer **A**; it matches strategy line 31 ("compute ≈ 0.02 ms; SpMV BW-bound ~1 ms").

**Exit gate 1c (== G3 precursor):** single-chip fused gather+MAC ≤ ~8 ms ⇒ projects to ≤ ~1 ms at 8-chip
(work ÷ 8). `rel_err ≤ 1e-6` throughout. If it can't get under ~8 ms single-chip, fall back: keep the
host gather only for the *coarse* levels (tiny) and device-gather the fine level, or accept bf16×2 on
coarse levels (strategy risk-mitigation). Do NOT proceed to Phase 3 until this gate is green.

--------------------------------------------------------------------------------
## PHASE 2 — Full 8-chip gather correctness (per-chip global offset)  (2–3 days)

**Objective:** make the gather correct on **all 8 chips**, not just chip 0. (Independent of Phase 1;
can run in parallel.)

**Why it's needed:** x + nbr are replicated (Phase-0 scaffolding), but the program is replicated too, so
each chip's reader sees a **local** `start_out_id` (0..n_local). The a/c buffers are *sharded* so local
indices map correctly, but the **gather** needs the **global** output node = `chip·n_local + local` to
pick the right neighbors from the replicated x/nbr.

Steps:
1. Add a `global_off` runtime arg to `gather_reader.cpp`; use `(global_off + start_out_id + t)` for the
   node math, keep `(start_out_id + t)` for the a/c page bases.
2. Host: build **per-chip programs** and place each on its chip:
   `for d in 0..NCHIP-1: prog_d = build_reader_program(global_off = d·n_local);
    wl.add_program(MeshCoordinateRange({0,d},{0,d}), std::move(prog_d));`
   (a/c stay sharded; x/nbr stay replicated.)
3. Validate: full-problem 8-chip gather vs the fp64 ref over **all** n_out tiles (not just the chip-0
   subset). Reuse `make_abref.py`'s `ref`.

**Exit gate 2:** 8-chip full gather `rel_err ≤ 1e-6` over all 3782 output tiles.

--------------------------------------------------------------------------------
## PHASE 3 — Fuse gather+MAC into one persistent SpMV op  (3–5 days)

**Objective:** one device program per apply that takes x (device-resident), gathers b, does the bf16×3
MAC, and writes y — with x kept resident across applies so only the changed vector is refreshed.

Steps:
1. Merge `gather_reader.cpp` + `mac_reader.cpp` paths: the reader streams a from DRAM and produces the
   gathered b in the same CB the MAC compute consumes (already the shape today — verify no double-staging).
2. Keep the x-window / DRAM-x resident between applies (don't re-stage nbr; nbr is constant across the
   whole solve). Only x changes per apply.
3. Wrap as a **persistent callable**: open device once, build the mesh workload + programs once, expose
   `apply(x_device) -> y_device` that just updates x and enqueues the workload (no per-apply program
   build — that would blow G4).

**Exit gate 3:** standalone `apply()` runs at **G3 ≤ 3 ms on 8-chip**, `rel_err ≤ 1e-6`, and repeated
calls (no rebuild) hold the time. This is the moment **G3 is green**.

--------------------------------------------------------------------------------
## PHASE 4 — Integrate as the PCG SpMV callback in gmg_tt.py  (1 week)

**Objective:** replace the ~25-min host path with the Phase-3 `apply()` in the fp64 outer PCG, producing
the first **fast, correct, end-to-end** row236 solve.

Design decision (the G4 crux — vector residency vs PCIe round-trip):
- **Option A — simplest first**: host fp64 PCG; per apply upload x (~23 MB bf16×3) + download y. Estimate:
  228 applies × (upload + apply + download). Transfers ~2–4 ms each at PCIe ~10 GB/s ⇒ ~228 × ~6 ms ≈
  **1.4 s** — slightly over G4=1 s, but a correct baseline. Build this FIRST.
- **Option B — vectors resident** (strategy line 72): keep PCG vectors in device DRAM, do the axpy/dot on
  device (or keep y on device and feed straight into next apply), send only scalar dots + residual norm to
  host. This is what makes G4 ≤ 1 s. Build after A works.

Steps:
1. Build the persistent-callback `.so` (from Phase 3) with a C ABI: `tt_spmv_init(...)`, `tt_spmv_apply(x,y)`,
   `tt_spmv_shutdown()`.
2. Wire `g_tt_fine_spmv` in `gmg_tt.py` to call `tt_spmv_apply`; keep the coarse levels on host (tiny) or
   device per Phase-1c fallback.
3. **Keep the fp64 host residual gate** — after the TT-accelerated PCG, verify true residual; on failure
   fall back to CPU GMG / SPOOLES. Never return an unverified TT result.
4. Run the full solve. Confirm `maxU = 95.8129714` (reduced) / `107.0569734213` (full), `true_rel < 3e-3`.

**Exit gate 4:** end-to-end row236 solve is **correct** and produces a real wall-clock number (Option A).
This is the milestone that converts 25 min into an optimization target.

--------------------------------------------------------------------------------
## PHASE 5 — Measure & close all eight performance gates  (1 week)

**Objective:** hit cold ≤ 5 s, warm ≤ 1.5 s with every sub-gate green.

Steps:
1. Instrument G1..G5 with the timing contract (`analysis/gate_timing.py` style: start/end ISO, duration_s,
   tool vs thinking). Emit a gate table per run.
2. **G4**: if Option A missed (~1.4 s), implement Option B (device-resident vectors + partial dots). Target
   ≤ 1.0 s.
3. **G2**: 0.62 s → ≤ 0.3 s — overlap the 3 bf16×3 streams, or upload while G1 finishes (borderline; the
   strategy already flags it). Warm caching can improve WARM but cannot turn the separate G2 workload green.
4. **G1**: confirm cacheable across loop solves (hierarchy/BCSR/bf16×3 terms) ⇒ ~0 warm.
5. Full cold + warm runs, 3× each, report medians against the table.

**Exit gate 5 (== ACCEPTANCE):** row236 on 8×Wormhole, cold ≤ 5 s, warm ≤ 1.5 s, `maxU = golden`,
`true_rel < 3e-3`, fp64 gate intact, all of G1–G5 green. Update TT_GMG_STRATEGY.md "Acceptance = done".

--------------------------------------------------------------------------------
## PHASE 6 — Scale to a bigger model (cross-chip Stage C)  (post-acceptance)

**Objective:** make the design hold for models larger than row236 (the "many-solve optimization loop" and
bigger meshes).

- Replication (Phase 0/2) holds until x exceeds per-chip DRAM (~100M+ DOF). Past that: **shard x with a
  halo** — each chip holds its output nodes + a boundary halo; exchange halo planes over
  ethernet/fabric with semaphores, overlap halo exchange with interior compute (strategy Stage C / risks).
- Add a model-size preflight: given nb, compute per-chip a-shard + x-shard + halo footprint; pick
  replicated-x (small) vs sharded-x+halo (large) automatically.
- Re-run acceptance on a 2× and 4× model to prove scaling.

**Exit gate 6:** a larger model solves within the same gate structure; halo path validated.

--------------------------------------------------------------------------------
## Risk register (carry-forward, most-likely-to-bite first)
1. **Gather ≤3 ms may need Option A *and* de-interleave** — if 1a+1c together still miss, fall back to
   fine-level-only device gather + bf16×2 coarse (strategy-sanctioned). Do not let this block a *correct*
   Phase-4 number.
2. **G4 PCIe round-trip** (~1.4 s Option A) — plan already has Option B; don't over-engineer before A works.
3. **Hardware wedge / stale lock** — Phase 0 hygiene script; BMC cold power-cycle recovers (documented in
   TT_GMG_STRATEGY.md), and `tt-smi -r` re-wedges healthy cards so **avoid it** — run directly on
   freshly-cycled cards.
4. **Multithreaded Spooles non-reproducibility** (seen earlier) — for any host-side modal/parity check use
   single-thread audit; not on the TT critical path but don't trust raw 16-thread modal output.
5. **Wrong tt-metal version** — always `~/src/tt-metal-073` (v0.73.1); v0.66 segfaults in TopologyMapper.

## Definition of done (TT backend)
`row236` solves on the 8×Wormhole QuietBox: **cold ≤ 5 s, warm ≤ 1.5 s**, `maxU = 107.0569734213`
(full) / `95.8129714` (reduced), `true_rel < min(2·tol, 1e-2)`, fp64 host PCG + residual gate intact,
result feeds CCX unchanged (opt-in, off by default), and the same path scales to a larger model (Phase 6).

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
# PART II — CROSS-PLATFORM (NVIDIA CUDA) + CCX END-TO-END

Phases 0–6 above take the **TT** backend to acceptance. Phases 7–12 add the **CUDA** backend behind the
same interface, then drive the user-facing **warm ccx ≤ 0.5 s** headline. Gate IDs here match
`GMG_ACCEL_GOAL.md` (C-*, X-*, E2E-*). The ordering is deliberate: extract the backend-agnostic
interface (Phase 7) **before** writing CUDA, so CUDA drops into a slot that already has the shared fp64
gate — that is what makes the project transferable rather than two forks.

--------------------------------------------------------------------------------
## PHASE 7 — Extract the backend-agnostic SpMV interface + shared fp64 gate  (X-1, X-2)  (3–5 days)

**Objective:** one interface, one shared correctness gate, TT behind it — with **zero change** to the
already-green TT numbers. This is a refactor, not new physics; do it while TT Phase 3/4 is fresh.

**Why now:** if CUDA is written before the interface exists, we get two divergent solvers and the shared
fp64 gate (the portability anchor, Lesson L6) drifts. Lock the seam first.

Steps:
1. Define the C ABI (stable, backend-neutral):
   `spmv_h spmv_init(const op_desc* op);  void spmv_apply(spmv_h, const dev_vec x, dev_vec y);
    void spmv_shutdown(spmv_h);` where `op_desc` carries the DIA operator (offsets, coeffs, nbr, nb, layout
   tag) and `dev_vec` is an opaque device-resident handle. Header `spmv_backend.h`.
2. Move the existing TT persistent-callback (Phase 3 `.so`) behind this ABI unchanged → `TtSpmv`.
3. Move the fp64 outer PCG + true-residual gate in `gmg_tt.py` into one backend-agnostic path that only
   ever calls `spmv_apply` + host fp64 dots/gate. No backend name appears in the PCG/gate code.
4. Runtime select: `GMG_BACKEND=tt` (default) picks `TtSpmv`; unknown/unset → CPU GMG fallback.

**Exit gate 7 (== X-1, X-2):** TT solve reproduces the **exact** Phase-4/5 numbers through the new
interface (correctness + timings unchanged); the PCG/gate source contains no backend-specific code.

--------------------------------------------------------------------------------
## PHASE 8 — CUDA bring-up on desktop-ivlvav4  (C-G0)  (1–2 days)

**Objective:** a usable CUDA dev box with a recorded GPU/toolkit baseline. **Blocked today: SSH/22 to
desktop-ivlvav4 (tailnet 100.77.233.42) is not answering** — this phase clears that first.

Steps:
1. Reach the box: it's a live tailnet peer but sshd isn't responding. Options in order — (a) wake/enable
   sshd via the `glkvm4090` KVM peer (KVM console), (b) confirm the SSH key/user, (c) Wake-on-LAN if
   asleep. Do not assume; verify `ssh mickg@desktop-ivlvav4` returns a shell.
2. Probe and record the CUDA baseline into `GMG_ACCEL_GOAL.md`: `nvidia-smi` (GPU name, VRAM, driver),
   `nvcc --version` (CUDA toolkit), cuSPARSE/cuBLAS presence, `nproc`/RAM. (glkvm4090 peer ⇒ expect a
   4090-class, 24 GB — row236's 2.4 GB fp64 operator fits with room, so single-GPU, no sharding for row236.)
3. Clone the fork on the box; confirm the CCX host build compiles there (the host/PCG/gate code is shared,
   only the SpMV backend differs).
4. Stage row236 inputs on the box (`row236_fine.bin` / dia / nbr / x) — same artifacts `make_abref.py`
   emits, reused as the CUDA correctness oracle.

**Exit gate 8 (== C-G0):** `ssh mickg@desktop-ivlvav4` works; GPU + CUDA versions recorded; fork builds;
row236 artifacts staged. Hardware-hygiene note captured (Xid/ECC reset, driver pin) per Lesson L7-CUDA.

--------------------------------------------------------------------------------
## PHASE 9 — CUDA fine SpMV  (C-G2, C-G3, C-COR-partial)  (1 week)

**Objective:** the DIA/BCSR fine SpMV on the GPU, bit-faithful to CPU `bspmv`, at G3 speed. CUDA removes
TT's two hardest problems (Lessons L1, L2): native fp32 (no bf16×3 Ozaki scheme) and SIMT-coalesced gather
(no scalar-RISC latency wall), so this is expected to be **markedly simpler** than TT Phases 1–3.

Steps:
1. **C-G2 upload:** one `cudaMemcpy` H→D of the fine operator (fp32 ~1.2 GB, or fp16/tf32 ~0.6 GB).
   Target ≤ 0.1 s (one contiguous buffer; tighter than TT's 3-stream 0.62 s).
2. **C-G3 kernel — start with the library baseline, then hand-tune:**
   - (a) cuSPARSE `bsrmv` (block size 3) as the correctness oracle + first speed bar (the CUDA analogue of
     TT Stage A / "what the high-level API gives for free").
   - (b) hand kernel: one warp/block per node, load the 27 neighbor 3-vectors via coalesced/L2-cached
     `x[3·nbr[o,n]+c]`, do the 3×3 register-block MACs, write `y[3n+r]`. BW-bound: fp32 2.4 GB/apply ÷
     ~1 TB/s ≈ 2.4 ms; fp16 0.6 GB ≈ 0.6 ms.
   - Keep the **component-major (de-interleaved) layout** (Lesson L3) for coalescing — reuse the Phase-1a
     `make_abref.py` layout so TT and CUDA consume the *same* operator files.
3. **C-COR (SpMV level):** validate GPU `y` vs the fp64 `ref` from `make_abref.py`, `rel_err ≤ 1e-6`
   (fp32) — this is the direct CUDA analogue of the TT gather/MAC correctness check.

**Exit gate 9 (== C-G3):** GPU fine SpMV ≤ 3 ms (target ≤ ~1 ms), `rel_err ≤ 1e-6` vs fp64 ref; wrapped
as `CudaSpmv` behind the Phase-7 ABI (`spmv_init/apply/shutdown`), vectors resident in GPU global memory.

--------------------------------------------------------------------------------
## PHASE 10 — CUDA PCG + acceptance  (C-G4, C-G5, C-COR, C-ACCEPT)  (1 week)

**Objective:** full row236 GMG solve on the GPU through the shared fp64 gate, hitting the CUDA gates.

Steps:
1. **C-G4:** run the Phase-7 shared fp64 PCG with `GMG_BACKEND=cuda`. Keep vectors resident on the GPU;
   cuBLAS `dot`/`axpy` on device; only scalar dots + residual norm cross PCIe to the host fp64 gate
   (the same "vectors stay on device" discipline as TT Option B, but native here). Target ≤ 1 s.
2. **C-G5:** `cudaMemcpy` D→H of the solution + un-permute ≤ 0.2 s.
3. **C-COR:** full solve `maxU` == golden to tol; fp64 host residual gate verifies every solve; on failure
   fall back to CPU GMG/SPOOLES. Bit-for-bit *decision* parity with TT (the numbers converge to the same
   golden, not necessarily the same iteration trace).
4. **C-ACCEPT:** cold ≤ 5 s / warm ≤ 1.5 s / stretch ≤ 2 s, 3× median each.

**Exit gate 10 (== C-ACCEPT):** row236 solves on the GPU, cold ≤ 5 s / warm ≤ 1.5 s, `maxU` == golden,
fp64 gate intact, behind the shared interface.

--------------------------------------------------------------------------------
## PHASE 11 — Cross-platform parity + repro harness  (X-3, X-4, X-5)  (3–5 days)

**Objective:** prove the project is genuinely transferable — same CCX build, `GMG_BACKEND` swap, identical
golden on both — and lock it behind one reproducible command.

Steps:
1. **X-3 parity:** from *one* CCX build, run row236 with `GMG_BACKEND=tt` and `GMG_BACKEND=cuda`; assert
   `maxU` matches golden to tol on both. Record both gate tables side by side.
2. **X-4 lessons:** finalize the TT→CUDA lessons table in `GMG_ACCEL_GOAL.md` with the *measured* CUDA
   outcomes (did fp32 suffice? did SIMT erase the gather wall? actual C-G2/G3/G4 numbers vs TT).
3. **X-5 CI/repro:** one script `run_gates.sh <backend>` that performs non-destructive health checks, builds,
   runs row236, and prints the correctness + G1–G5 + E2E gate table. TT must delegate lifecycle to the hardened
   fresh-boot runner—no `pkill`, manual interruption, or `tt-smi -r`. Any CUDA reset must be separately authorized
   and gated on exclusive ownership; it is not automatic CI hygiene.

**Exit gate 11 (== X-3/X-4/X-5):** one command reproduces the full gate table on each available backend;
identical golden proven on both; lessons table reflects measured CUDA reality.

--------------------------------------------------------------------------------
## PHASE 12 — CCX end-to-end warm ≤ 0.5 s (the ★ north star)  (E2E-1, E2E-WARM, E2E-COLD)  (1–2 weeks)

**Objective:** the user-facing headline — a full **warm `ccx` run of row236 in ≤ 0.5 s vs ~22 s CPU
(≥ 44×)**, correct, on the faster backend (whichever hits it first; expected CUDA given native fp32/SIMT).

**Why this is its own phase:** G1–G5 are the *solver* gates; E2E-WARM is the *whole ccx binary* wall-time,
which also includes input parse, assembly, and FRD output. The fork already has the enablers (Arrow
binary mesh input + binFRD output, per the CCX rules) — this phase wires the accelerated solver into that
fast-I/O path and measures the binary end-to-end.

Steps:
1. **E2E-1 (correct, any speed):** `ccx` solves row236 through the accelerated GMG backend (opt-in flag),
   `maxU` == golden, FRD read-back correct. This is the integration proof — the solver `.so` is called
   from CCX's solve path, not a standalone harness.
2. **Warm-path plumbing:** ensure setup/hierarchy/BCSR/operator are **cached & device-resident** across
   solves (the optimization-loop case), and the run uses **binary** mesh input + binFRD output (no ASCII
   parse/write in the hot path). Confirm the fp64 gate still runs.
3. **E2E-WARM measure:** full `ccx` wall-time, warm, 3× median. Attribute any miss to input/assembly/solve/
   output with the timing contract; close the dominant term. To hit 0.5 s: G3 ~1 ms · G4 ~0.3 s
   (device-resident vectors) · binary I/O · setup cached.
4. **E2E-COLD:** first-solve wall-time (upload + setup included) ≤ 5 s, correct.

**Exit gate 12 (== E2E-WARM ★):** `ccx` row236 **warm ≤ 0.5 s, ≥ 44× vs 22 s CPU**, correct to golden,
fp64 gate intact, on at least one backend; E2E-COLD ≤ 5 s. **This is the project's definition of success.**

--------------------------------------------------------------------------------
## PHASE 13 — Scale to a bigger model, both backends  (E2E-BIG)  (post-acceptance)

**Objective:** prove the design isn't row236-specific. TT: shard x + ethernet **halo** (Phase 6 Stage C).
CUDA: **NCCL** multi-GPU halo (only needed once a model exceeds one GPU's VRAM). Add the model-size
preflight (replicated-x vs sharded-x+halo) on both. Re-run E2E-WARM on a 2× and 4× model.

**Exit gate 13 (== E2E-BIG):** a larger model solves within the same gate structure on both backends;
halo/NCCL paths validated.

--------------------------------------------------------------------------------
## APPENDIX — per-gate verification method (how each green is *proven*, not claimed)

| gate | verified by | artifact |
|------|-------------|----------|
| correctness (TT & CUDA) | full solve `maxU` vs golden to tol; SpMV `rel_err ≤ 1e-6` vs `make_abref.py` fp64 `ref` | solve log + rel_err print |
| T-G3 / C-G3 | standalone `apply()` timed on real HW, repeated calls (no rebuild), `rel_err ≤ 1e-6` | SpMV-MAC ms + GB/s line |
| G3-XCHIP | 8-chip full gather over all 3782 tiles vs fp64 ref | rel_err over full n_out |
| G4 | PCG wall-time, 3× median; device-resident vector path confirmed | gate_timing table |
| G2 / G5 | upload/download timed on real HW | DBG G2 / G5 lines |
| cold/warm/stretch | full solve 3× median, host timing contract | gate table |
| E2E-1 | ccx FRD read-back == golden | frd diff |
| E2E-WARM ★ | full ccx binary wall-time warm, 3× median, vs 22 s CPU | timing_repord.md |
| X-3 | one build, GMG_BACKEND swap, both == golden | side-by-side gate tables |
| X-5 | `run_gates.sh <backend>` reproduces the whole table | CI output |

**Discipline (applies to every phase):** a gate is GREEN only with a **measured number on the real
target**; no green from projection or single-run noise (use 3× median for wall-times); the fp64 host gate
is never removed by any optimization; and every phase leaves the box in a clean state (TT stale-lock,
CUDA reset) so the next run is reproducible.

## Definition of done (the whole project)
`ccx` solves row236 **warm in ≤ 0.5 s (≥ 44× vs 22 s CPU)**, correct to golden, on **both** the
Tenstorrent and NVIDIA backends behind one SpMV interface with one shared fp64 host gate, reproducible via
`run_gates.sh`, with the same path proven to scale to a larger model (Phase 13).
