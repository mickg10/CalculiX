# TT-GMG row236 — RESUME STATE (ultra-thorough handoff)

> **CURRENT CORRECTION — read this before the historical handoff below (2026-07-15, Run67 update):** this file captured an
> optimistic earlier checkpoint. `tt_gmg/GATE_CONTRACT.md`, `tt_gmg/gates/gate_contract.json`, and the generated
> `tt_gmg/gates/current_report.md` are now authoritative. The original program has eight performance gates;
> correctness is a separate invariant. The honest score is **2/8 green** (G3 and G5), with reduced-dump
> correctness separately green. G1 is red at about `8.25 s`; G2 is red at Run66's measured `0.547659 s`; and G3
> is green from Run66's correct persistent fixed-resident-x samples `1.945499/1.822259/1.858679 ms`, median
> `1.858679 ms` versus `3 ms`. G4, cold, warm, stretch, and full-CCX acceptance remain open. Any `4/8`, `5/8`,
> “G1/G2 closed,” projected-pass, or “single remaining uncertainty” wording below is superseded history.
>
> **Current implementation state after Run67:** Run66 remains the qualifying G3 frontier. The affine A-low family
> was vectorized/precomputed, but run54 deadlocked and runs55–57 repeated the same two device errors under fixed,
> zero, and dummy-NOC cadence; that family is closed device-incorrect. Run58/58b and run59 were infrastructure-only
> negatives. The split-A retry run59b was correct but slower (`3.505936 ms` median). A complete raw-operator audit
> then proved every corner entry is one of three exact BF16x3 triples sharing one presence pattern. The lossless
> compressed-buffer run60 was correct but slower (`3.532192 ms`); the fixed-27-page, original-buffer reuse run61
> removed its CB/source-switch cost but still measured `3.405163 ms`. Exact corner-coefficient traffic is therefore
> closed as the G3 wall. A joint inversion-symmetric A-low/B-low host search found four-vector-green masks, but the
> precision ceiling is only 36 omitted low products versus run52's 18—far too little at the measured slope to
> justify another disruptive boot. Run66 then changed the representation to canonical-base plus exact
> dense-fallback and closed G3. The exact changing-x page reader then reached hardware in Run67: plan preflight
> passed; upload measured `282.974 ms` for `411,188,224` bytes; build measured `7.103 ms`; warmup correctness was
> RED with `50,858` nonfinite outputs. The failure is diagnosed as a Tenstorrent circular-buffer cycle violation:
> variable `7..62` page advances crossed a 62-page staging CB end, and compact mode-3 `1/2/3` low-split advances
> crossed a three-page CB end. The group-84 `-inf` concentration matches the simulated crossing. The local fix
> reserves each complete stage allocation once as fixed-address L1 scratch, never pushes that scratch CB, and uses
> full three-page low-split publish cycles; all `29/29` local tests pass.
> Remote host/offline compilation and a new fresh-boot device confirmation are pending. Persistent device-vector
> PCG/G4 remains gated on that correctness proof, adjacent-chip halo exchange, resident vector operations/reductions,
> and the complete fp64 true-residual path. See `tt_gmg/RUN67_CANONICAL_DYNAMIC_CB_FAILURE_ANALYSIS.md`.
>
> **Names/infrastructure:** “BNC” means the QuietBox **BMC**; “GVM” means the GL.iNet **KVM** (`glmkvmigor`);
> **GMG** is geometric multigrid. There is no GVM compute subsystem. A live AWS `default` IAM-user profile is
> present and STS identity succeeds, but that principal is denied billing and global resource discovery, so account
> existence is proven while inventory and spend remain unknown. The reachable solver path is private physical/
> virtual compute. See
> `tt_gmg/CLOUD_AND_MANAGEMENT_AUDIT_2026-07-14.md`. Passwordless SSH and `sudo -n` are sufficient; no supplied
> password is needed or recorded. BMC cycles remain subject to the strict idle/jobs/D-state/holder preflight,
> `tt-fold` must be restored after every hardware window, and `tt-smi -r` remains forbidden.
>
> **Current site reachability:** after Run67's runner restored and verified `tt-fold`, both `tt-quietbox` and
> `glmkvmigor` went tailnet-offline within 12 seconds of each other (last seen `15:59:50.1Z` and `15:59:38.1Z`).
> The Mac Tailscale backend and unrelated peers remain healthy. Treat this as a shared remote-site power/WAN/LAN/
> Tailscale-path outage until one management path returns. Do not issue a blind power action. The Run67/BMC logs
> remain remote and must be recovered verbatim rather than reconstructed.
>
> **Target-pack publication (2026-07-19):** large targets and portable repro
> packs now live in `github.com/mickg10/calculi_target_packs` at registry
> commit `1fb01bf0d908ae9b555ce118a4db9fb9b06e3fa7`, release
> `packs-v1-20260719`. This fork binds them by manifest/object SHA in
> `TARGET_PACKS.md` and `tt_gmg/target_pack_registry.json`. The primary
> row236 fine-dump object is
> `8f13780171520a82d349e69b3c581e3278bf3b927cef8d9d8a6cc987e85ce676`
> compressed and
> `e58c212d593dd39884f9806558f1bfc30e6f2456708e61ed2ca1f32220f13638`
> decompressed. Pack publication changes no solver gate.

## THE GOAL (the standing directive this work was driving toward — recorded verbatim)

> read and reread calculix-fork/TT_GMG_STRATEGY.md. Make sure to require ALL gates it says. Do not complete until
> done. DO NOT GIVE UP - IF YOU LOOK AT a 3 month integration - GO AHEAD AND PLOW INTO IT! You like hard problems -
> diving into hard things is your job!

Meaning under the reconciled contract: close all eight **performance** gates (G1–G5, cold, warm, stretch) for
row236 on the 8×Wormhole QuietBox while retaining correctness as a separate mandatory acceptance invariant. Current
status is 2/8 green, not 4/8. G3/G5 are green; G1/G2 are measured red; G4/cold/warm/stretch are open; reduced-dump
correctness is separately green; full-CCX correctness/integration is open. (The old goal was `/goal clear`-ed at that
session end; the directive is retained here so the resume is self-contained, but status comes from the gate report.)

---

**Last updated:** 2026-07-19 (publication audit). **Branch:**
`tt-gmg-run67-pack-registry-20260719` (fork
`github.com/mickg10/CalculiX.git`, branched from `accel-gmg-backend` at
`7c77acff8df18a9edd5fe995db6974cd02d086a8`).
**Goal file:** `calculix-fork/TT_GMG_STRATEGY.md` (the 8 gates). **Forensic trail:** `tt_gmg/GATHER_DESIGN.md`.
**Executable plan:** `tt_gmg/stencil/IMPLEMENTATION_PLAN.md`.

---

## 0. ONE-PARAGRAPH SUMMARY

Task: close the exact eight-gate TT contract for row236 (3.87M-DOF near-singular grid elasticity). The brick-major
resident-x implementation is real and correct, not projected. Run66 is the current G3 frontier at a `1.858679 ms`
median on all eight chips, below the `3 ms` budget. It uses the canonical-base plus exact dense-fallback packed
operator and one fixed pre-shifted resident vector. Runs54–61 close affine folding, split-A redistribution, and exact
corner-coefficient compression as device-negative/slower lever families; Run52 remains historical arithmetic
evidence. The exact host changing-x map and selected page plan replay the fixed-vector bundle bit-for-bit. Run67
measured its upload inside G2's budget but failed the first device apply because of the now-fixed CB cycle defect.
The current durable score remains **2/8** (G3 and G5), with reduced-dump correctness green, until the original
Run67 log is recovered and hashed. The program still needs fixed-reader device correctness, resident device-vector
PCG, full-CCX integration, and separate cold/cache-hit-warm/overlap measurements. No estimate, missing log,
timeout, infrastructure failure, or host-only plan is a pass.

---

## 1. THE 8 GATES (from TT_GMG_STRATEGY.md, lines ~48-60)

North-star: solve row236 in ≤5s cold, golden `maxU`, fp64 outer PCG + true-residual gate on host.

| id | what | budget | STATUS |
|----|------|-------:|--------|
| G1 | setup (host: hierarchy + BCSR + bf16×3 terms) | ≤3.0s | RED — `8.25 s` cold. Cacheability does not close a cold gate. |
| G2 | complete matrix upload host→8 chips | ≤0.3s | RED in the durable report at Run66 `0.547659 s`; Run67 measured `0.282974 s` for `411,188,224` bytes, but the original log is still remote and must be recovered/hashed before banking the independent G2 gate. |
| G3 | correct persistent resident-x fine SpMV | ≤3ms | GREEN — Run66 median `1.858679 ms` from three correct samples. |
| G4 | PCG solve including device vectors + fp64 true-residual decision | ≤1.0s | OPEN — persistent device-vector PCG not implemented/measured. |
| G5 | output / un-permute | ≤0.2s | GREEN — `0.013 s`, component scope; remeasure in final bundle. |
| COLD | full accelerated first run | ≤5.0s | OPEN — no qualifying complete run. |
| WARM | full accelerated cache-hit run | ≤1.5s | OPEN — no proven cache-hit/resident-operator complete run. |
| STRETCH | cold run with measured setup/upload overlap | ≤2.0s | OPEN — no overlap/critical-path report. |

**Golden note:** strategy line 6 says `maxU=107.0569734213` for the FULL row236; the DUMP on the box
(`row236_fine.bin`) is the REDUCED problem with golden `maxU=95.8129714`. The solve_driver labels 95.8129714 as
GOLDEN. Reduced-dump correctness is banked separately (that's what's on the box). It is not a performance point and
does not close the full-CCX golden/output-path invariant.

---

## 2. HARDWARE / ACCESS (all verified this session)

- **Box:** tt-quietbox = 8×Wormhole (4× n300 boards → `/dev/tenstorrent/0..3`, 2 chips each). The currently
  reliable SSH path uses `glmkvmigor` as a key-authenticated `ProxyCommand`; see §11. CPU:
  **AMD EPYC 8124P 16-core / 32 threads**.
- **tt-metal:** `/home/ttuser/src/tt-metal-073`, v0.73.1; build target at
  `build_Release/programming_examples/metal_example_brick_spmv`. Do not substitute the older `~/src/tt-metal` tree.
- **g++:** Ubuntu 13.3.0. LAPACK/BLAS: `liblapack libopenblas libblas` present.
- **PASSWORDLESS SUDO WORKS** on the box: `sudo -n ipmitool ...`, `sudo -n systemctl ...`, `sudo -n fuser ...`
  all work with no password. (This is how device recovery + tt-fold management were done — no password ever
  written to a file.)
- **BMC:** ASRock Rack/SIENAD8-2L2T, firmware 2.05, IPMI 2.0, LAN 10.0.0.48. Recovery for a genuinely wedged
  device is passwordless `sudo -n ipmitool chassis power cycle`, but only after the strict jobs/workload/D-state/
  foreign-holder/users preflight. Do not issue repeated cycles casually; one acknowledged cycle is disruptive.
  DO NOT use `tt-smi -r` (re-wedges healthy cards via AER). After cycling, poll `test -e /dev/tenstorrent/0`;
  server cold-boot POST+boot ~150-170s.
- **tt-fold** = the USER'S OTHER PROJECT ("TT-Fold portal — tt-bio controller (loopback) + FastAPI gateway
  (Tailscale), local MSA", ExecStart `/home/ttuser/ttbio/tt-bio/webportal/run.sh`). It is **enabled** (auto-starts
  on boot) and its python workers (`~/ttbio/tt-bio/.venv/bin/python`) hold `/dev/tenstorrent/0,2,3` (device 1 was
  free once). To run the TT-GMG solve you must `sudo -n systemctl stop tt-fold` (idle-holds the devices, ~3-4% CPU,
  no active fold observed → clean stop). **ALWAYS `sudo -n systemctl start tt-fold` after.** This session stopped/
  started it ~9 times. Minimize this — each stop disrupts the user's bio project.

### Hard security/ops constraints (carry forward verbatim)
- sudo passwords must NEVER be written to any file (passwordless sudo sidesteps this).
- Do NOT use `tt-smi -r`.
- `pkill -9 -f <pattern>` can MATCH THE SSH COMMAND'S OWN cmdline → kills its own shell → truncated output. Kill by
  exact PID, or use a pattern the launch cmd doesn't contain, or `for p in $(pgrep -f solve_driver); do kill $p; done`.
- **NEVER manually interrupt a multi-chip run MID-WORKLOAD.** It leaves Tensix cores in a bad run-state → EVERY later run
  (incl. known-good) hangs at first apply with `TT_FATAL Read unexpected run_mailbox value 0x40`. Only recovery =
  BMC cold-cycle. Let the whrun 850s `timeout` end a hang (exit=124) — that path kept the device healthy.
- A timed-out multi-chip run can leave a D-state proc holding all `CHIP_IN_USE_*_PCIe` locks; only BMC recovers.
  (This session never hit true D-state after the fix; `dstate=0 busy=0` throughout the later runs.)
- Avoid `NCHIP=1` (ethernet timeout / wedge) and large sharded-fp32 host writes.

---

## 3. FILE / BUFFER LOCATIONS ON THE BOX

- **Durable (survive reboot), in HOME:**
  - `~/ttgmg/staged/row236_fine.bin` (2.61GB) — the GMG dump (BCSR + ijk + rhs). Header h[5]={nb,nblk,nx,ny,nz}.
  - `~/ttgmg/staged/row236_real_op.bin` (1.25GB) — the SpMV operator a (fp32, [n_out,K,1024] tile-major).
  - `~/ttgmg/staged/row236_nbr.bin` (139MB) — nbr int32[27,NBn] (+8-byte header [_,NBn]).
  - `~/ttgmg/libgmg.so` — the CPU GMG solver (`ccx_gmg_solve_from_dump`, `ccx_gmg_solve_mem`).
  - `~/ttgmg/gmg_solve.cpp` — CPU GMG source (Jul-1 version; DIFFERS from local `src/gmg_solve.cpp` Jul-5. md5
    box=afda94..., local=177678...). Rebuild: `g++ -O3 -march=native -fopenmp -shared -fPIC ~/ttgmg/gmg_solve.cpp
    -o /tmp/libgmg_test.so -llapack -lblas` (VALIDATED: rebuilds baseline exactly, 56 it, maxU=95.8129716, 10.14s PCG).
- **Ephemeral `/tmp` (WIPED by reboot — re-stage from repo + HOME):**
  - `/tmp/libtt_spmv.so` (built by whrun from `/tmp/tt_spmv.cpp`), `/tmp/kernels/*.cpp` (JIT kernels),
    `/tmp/whrun_solve.sh`, `/tmp/solve_driver.py`, `/tmp/row236_*.bin` (symlink these to `~/ttgmg/staged/`!),
    `/tmp/whhome/.cache` (JIT cache).
  - **NOTE: `/tmp` is on the nvme root (3.6T, 96% full, 161G free) — NOT tmpfs.** But copying 1.25GB was slow/hung;
    use **symlinks**: `ln -sf ~/ttgmg/staged/row236_real_op.bin /tmp/row236_real_op.bin` (and nbr). Instant.
- **Repo copies (durable source of truth):** `tt_gmg/tt_spmv.cpp`, `tt_gmg/kernels/{gather_reader_galaxy.cpp,
  mac_compute.cpp,mac_writer.cpp}`, `tt_gmg/harness/{whrun_solve.sh,solve_driver.py,make_cancel_op.py}`,
  `tt_gmg/stencil/{stencil_poc.py,build_tiled_stencil.py,serialize_tiled.py,IMPLEMENTATION_PLAN.md}`,
  `tt_gmg/emu_bf16x2b_precision.py`, `tt_gmg/GATHER_DESIGN.md`.
- **Efficient reader ON BOX:** `~/src/tt-metal/tt_metal/programming_examples/spmv_mac/kernels/mac_reader.cpp`
  (the 3.46ms pre-stored-a+b DMA reader). The `spmv_mac` HOST harness (metal_example) is GONE — only kernels remain.

---

## 4. THE OPERATOR (row236) — measured structure

- Fine: **nb (NBn) = 1,290,738** block-rows (×3 DOF → n=3,872,214), **~33M** 3×3 blocks (~297M scalar nnz),
  27-point stencil. Hierarchy 5 levels, coarsest 4284. PCG 38 iters (strategy) / 56 iters (measured CPU baseline).
- **It IS a structured grid** (my earlier "not shiftable / impossible" was WRONG — retracted): the nbr/BCSR is
  stored COLUMN-SORTED, so per-slot deltas look scattered, but mapping every stored block's (node,col) through the
  dump's `ijk` gives EXACTLY **27 fixed geometric offsets** (di,dj,dk ∈ {-1,0,1}). Node numbering correlates 0.97
  with lattice lin-index (mostly z-fastest, gaps from inactive voxels).
- **Grid box = nx×ny×nz = 559×273×229 = 34,947,003 voxels.** Active nb = 1,290,738 → **density 3.7%** (a lattice-
  cloud holder, mostly air).
- The historical **“101 distinct 3×3 blocks”** statement was not computed: `stencil_poc.py` printed the literal
  value. A full BF16x3 audit found many raw noise-distinct blocks, while the eight corner offsets alone have the
  narrower exact three-triple structure used by runs60/61. Do not claim a global 101-entry dictionary without a
  new measured canonicalization contract.
- **Stencil-shift SpMV reproduces bspmv MACHINE-EXACT: rel_err 2.97e-16 (untiled) / 3.83e-16 (tiled 4^3 w/ halos).**
  (Validated on host in `stencil_poc.py` / `build_tiled_stencil.py`. The einsum must be `nj,nij->ni` = A·x, not Aᵀ·x.)
- **Brick occupancy (spatial coherence):** 4^3 bricks → 25,284 occupied, covered_vol=1.62M = **1.3× active** (only
  30% zero-pad, NOT 27×). 8^3→1.6×, 16^3→2.5×. So the tiled dense-shift operator = 27×1.62M×9×6B(bf16×3) = **2.36GB**.

---

## 5. THE MEASURED TIMING LADDER (all on real 8×Wormhole, PHASE workload ms/apply)

| kernel | workload | notes |
|--------|---------:|-------|
| gather_reader (scalar b-materialize, per-node) | **40-44ms** | the ORIGINAL path. b materialized by a 1024-elem scalar loop/k. |
| contiguous-read diagnostic (no scatter, still materializes) | 40ms | **proves gather-reads are NOT the wall (~4ms)** |
| 1-MAC-term vs 6 | 44ms | **proves the eltwise MAC compute is NOT the wall** |
| **mac_reader (pre-stored a+b, DMA tiles)** | **4.4ms** | **9× — the b-MATERIALIZE was the wall; DMA-tile structure works.** depth-3 and depth-24 CB both 4.4ms → BW-bound on a+b (3.6GB, ~100 GB/s/chip). |
| **a-only (mac_reader_aonly, stream 3 a-tiles/k, b=a placeholder)** | **2.8ms** | **the resident-x floor** (only a streams). 1.8GB @ ~80 GB/s/chip (has fixed overhead). |
| strategy's own spmv_mac (historical) | 3.46ms | 3.6GB a+b @ **130 GB/s/chip** — HIGHER BW than my kernel → the target efficiency EXISTS. |

Per-apply overhead (measured, separate from workload): **xwrite=5.1ms, readback=8.9ms** (per-shard ReadShard is
inefficient). Full apply ≈ 4.4 + 5.1 + 8.9 + host ≈ 20ms with the current (non-resident) path.

### Cascade arithmetic (to be VERIFIED by building the kernel)
- **G3-timing ≤3ms:** correct brick-stencil a=2.36GB. At my measured ~80-102 GB/s/chip → **~3.67ms (OVER)**. At the
  strategy's demonstrated 130 GB/s/chip → **~2.27ms (UNDER)**. So ≤3ms is ACHIEVABLE with batched-DMA BW tuning
  (the strategy proved that BW is attainable), NOT guaranteed by current kernel efficiency. **This is the crux
  uncertainty.**
- **G4/cold/warm:** with workload ~2.8-4.4ms + **x-resident PCG** (drop xwrite 5.1 + readback 8.9 + host round-trip)
  → ~3-5ms/apply. G4 = ~156-228 applies × ~3-5ms ≈ **0.68-1.14s** (borderline-to-under 1s). cold ≈ setup(3s)+0.7s
  ≈ 3.7s ≤5s. warm ≈ 0.7s ≤1.5s. **Plausible but UNMEASURED — must build x-resident PCG.**

---

## 6. THE tt_spmv.cpp HARNESS — current state (what's in the repo NOW)

`tt_gmg/tt_spmv.cpp` — extern C `tt_spmv_init(real_op_path, nbr_path, x_path)` + `tt_spmv(const double* x, double* y)`.
Structure: `TtSpmvCtx` (dev, buffers ah/am/al/**bh/bm/bl**/xh/xm/xl/nbrbuf/c, MeshWorkload wl, sizes). Init loads a
(split3 → bf16×3 ahd/amd/ald), nbr, creates SHARDED a buffers (`MakeBuf(..., NCHIP)`), uploads a PER-SHARD via
`WriteShard` (EnqueueWriteMeshBuffer only populates edges — the "85-at-123i" bug). x REPLICATED (`MakeReplBuf`),
nbr replicated. Apply: adaptive vscale=1e3/xmax, split x→bf16×3, EnqueueWrite x, `tt_build_wl` (rebuild per apply
unless SPMV_REUSE_WORKLOAD — reusing corrupts gather after ~4 applies), EnqueueMeshWorkload, per-shard ReadShard c→y.

### Env-var paths ADDED this session (all committed):
- **`SPMV_MAC_READER=1`** — init creates+uploads bh/bm/bl (bf16×3, = a's data placeholder), tt_build_wl uses
  `/tmp/kernels/mac_reader.cpp` (DMA a+b tiles) instead of gather_reader, with args `{ah,am,al,bh,bm,bl,npc,K,start}`
  and 6 TensorAccessorArgs. Reader CB depth = `ab_depth` = `SPMV_MAC_READER ? (SPMV_ABDEPTH or 24) : 3` (mac_reader
  has no x-window → L1 free → deep prefetch OK). **This is how the 4.4ms was measured.** Output is WRONG (b is a
  placeholder) — TIMING ONLY.
- **`SPMV_ABDEPTH=<n>`** — override the mac_reader CB depth (default 24).
- The a-ONLY (2.8ms) measurement REPLACED `/tmp/kernels/mac_reader.cpp` and `/tmp/kernels/mac_compute.cpp` with
  `mac_reader_aonly` (reads only 3 a-tiles/k to cb_a) + `mac_compute_aonly` (b=a from cb_a, 6 terms). Those two
  diagnostic kernels are in `/tmp/mac_reader_aonly.cpp` / `/tmp/mac_compute_aonly.cpp` in the last session (NOT
  committed to repo — recreate from GATHER_DESIGN.md / this file if needed). Run with `SPMV_MAC_READER=1` after
  overwriting the two /tmp/kernels files.

### whrun_solve.sh (harness, in repo `tt_gmg/harness/`, deployed to /tmp on box)
Rebuilds libtt_spmv from `/tmp/tt_spmv.cpp` (g++), JIT-copies `/tmp/kernels/*.cpp` to the spmv_mac example dir,
decompresses row236_*.bin.zst (or use symlinks), runs solve_driver.py with 850s `timeout`, env:
`SPMV_TIMING=1 SPMV_REUSE_WORKLOAD=1 GMG_DEFL_CORR=1 GMG_DEFL_EIG=1 GMG_DEFL_K=24 GMG_HYBRID_TOL=1e-2`.
**FIXED this session:** fail-closed rebuild check (`rm -f libtt_spmv.so` before g++, check `$?`==0 AND file exists —
was only checking file-existence, masking compile failures). To toggle the mac_reader path, `sed -i` the env line to
add `SPMV_MAC_READER=1` (remember to sed it back out after).

---

## 7. THE CORRECTNESS PATH (banked — how to reproduce G3-correctness)

`solve_driver.py` (repo `tt_gmg/harness/`): loads `/tmp/libtt_spmv.so` + `tt_spmv_init`, loads `~/ttgmg/libgmg.so`,
sets `g_tt_fine_spmv` (extern C fn ptr in gmg_solve.cpp) to a ctypes callback wrapping `tt.tt_spmv`, sets
`g_tt_fine_n = n`, calls `ccx_gmg_solve_from_dump(FINE=~/ttgmg/staged/row236_fine.bin, 300, 1e-6, 1, None, &maxu)`.
Prints maxU + per-apply/PHASE timing.

**The convergence mechanism (CRITICAL — bf16 needs this):** faithful bf16×1 and bf16×2 smoothers DIVERGE (~400×).
**bf16×3 compensated** (Ozaki/3×TF32: hi/mid/lo split, 6 bf16 sub-products level-sum≤2, fp32 accumulate) →
converges. AND the run needs **eig-deflation + hybrid PCG**: `GMG_DEFL_CORR=1 GMG_DEFL_EIG=1 GMG_DEFL_K=24
GMG_HYBRID_TOL=1e-2`. The near-null eigenvectors (6 rigid-body modes) drive the extreme cancellation floor; the
deflated PCG (Saad DCG) absorbs it. **Env-var prefix matters:** it's `GMG_DEFL_EIG` NOT bare `EIG` (a past bug used
bare vars → default deg=1 POLYNOMIAL deflation → FAILS/diverges). NOTE: eig-deflation HELPS the TT bf16 path but on
exact-fp64 CPU it HURTS (56 it → 300 it, rc=4) — only use it for TT.

**mac_compute.cpp (the bf16×3 compensated MAC, in repo):** `mac_term` calls the LLK directly
(`llk_math_eltwise_binary<ELWMUL,...>(cb_a,cb_b,0,first)`) with `clear_fp32_dst_acc=first` so 6*K cross-terms
accumulate in fp32 dst. 6 terms: ah·bh, ah·bm, am·bh, ah·bl, am·bm, al·bh. **Include is
`compute_kernel_api/eltwise_binary.h`** (NOT `api/compute/...` — that was a repo bug fixed this session).

**Kernel bugs FIXED this session (repo kernels now actually build against the box tt-metal):**
1. `mac_compute.cpp`: `api/compute/eltwise_binary.h` → `compute_kernel_api/eltwise_binary.h`.
2. `mac_compute.cpp`: `mul_tiles_init(cb_a,cb_b,1,0)` (4 args) → `mul_tiles_init(cb_a,cb_b)` (API is
   `(icb0,icb1,call_line=__LINE__)` = 2 real args).
3. `mac_writer.cpp`: `TensorAccessor(c_args,c_addr)` CTAD-failed → `TensorAccessor(c_args,c_addr,4096)` (fp32 tile
   = 32*32*4 bytes; page size required for CTAD).
(These were latent because the box's ORIGINAL /tmp/kernels versions had the right forms; the reboot wiped /tmp and
re-staging the repo's stale copies exposed them.)

---

## 8. bf16×2-b PRECISION (host-validated — path B)

`tt_gmg/emu_bf16x2b_precision.py`: emulates the exact device compensated MAC (bf16 RNE split, exact bf16×bf16
products, fp32 accumulate over device k-outer order) vs fp64 on an orthogonalized extreme-cancellation operator.
Result: bf16×3 err/term ~7.5e-8; **bf16×2-b** (b in 2 levels, 5 cross-terms: ah·bh, ah·bm, am·bh, al·bh, am·bm —
drop ah·bl) err/term ~1e-6, **HOLDS fp32-class at apply2 depth (RATIO 1e-4..7e-5, 8-10× margin)**, marginal at 3e-5.
So bf16×2-b is precision-viable (a:3 levels, b:2 levels). Note bf16×2-BOTH (a:2,b:2) DIVERGES (strategy). Using
bf16×2-b for the b-stream cuts 1/3 of the b traffic; but a-only (resident-x) is the bigger win.

---

## 9. WHAT I GOT WRONG (retracted — don't repeat these dead ends)

1. **"Timing gates physically impossible"** — WRONG. Based on the general-sparse GATHER representation. The stencil
   representation eliminates the gather. RETRACTED.
2. **"The scattered gather is the 40-45ms wall"** — WRONG (measured). Contiguous reads = 40ms too. The scalar
   **b-materialize** (1024-elem write loop/k) is the wall. mac_reader (DMA b-tiles) = 4.4ms.
3. **"SFPU can vectorize the gather"** — dead: Wormhole sfpi has NO indexed/gather load (only immediate/const loads
   + fixed dst_reg lanes). Unpacker does strided/tilize, not arbitrary gather. ttnn has no indexed-gather op. So the
   SCALAR gather can't be vectorized — but the STENCIL avoids needing a gather (contiguous shifts).
4. **"offset-grouping (cb depth 9)"** — L1-INFEASIBLE for the gather reader (x-window ~444KB + nbr ~360KB/core is
   tight; +24KB overflows → hang). BUT mac_reader has no x-window → deep CB (24) fits fine.
5. **"the a-only floor 2.8ms guarantees G3≤3ms"** — OVERSTATED. That was the COMPACT a (1.8GB). The correct
   brick-stencil a is 2.36GB → ~3.67ms at my measured BW, ~2.27ms at the strategy's proven BW. So ≤3ms is
   ACHIEVABLE-with-BW-tuning, not guaranteed.
6. CPU-GMG fallback: measured ~20s (56 it, memory-bound on 2.4GB fp64 val); fp32-val optimization BREAKS
   (global→coarse dpotrf info=1990 not SPD; fine-only→PCG rel=1.0 no convergence); deflation on CPU makes it WORSE.
   The TT offload for THIS sparse scattered operator is a net loss vs CPU UNLESS the stencil+resident-x lands.

---

## 10. THE REMAINING WORK — current measured route

The original brick-major Step A is superseded by Run66's canonical packed base/fallback kernel. Run66 is correct,
persistent, and G3-green at a `1.858679 ms` median. Its input is still one fixed pre-shifted vector, so it does not
close changing-x residency or G4.

### STEP A — G3 complete; changing-x reader hardware fix pending

Run66 is the measured frontier; Run52 and Runs59b–61 remain closed historical evidence. The default-off dynamic
page reader specified in `tt_gmg/CANONICAL_PCG_RESIDENCY_DESIGN.md` has now reached hardware. Its host contract is
complete: all `314,523,648` fixed-vector BF16 words replay exactly; raw, unique-node, and page-stream maps reconstruct
with zero mismatches; canonical order has better chip locality than a second spatial vector. Run67 exposed and
localized the device CB-cycle bug described above. The fix is locally tested. Before another workload, rebuild the
host harness, repeat host preflight, and compile both dynamic BRISC/TRISC role sets offline on QuietBox.

### STEP B — persistent device-vector PCG

G3 is green. Keep canonical PCG vectors on device, implement adjacent-chip halo exchange plus device vector
operations/reductions, and include scalar transfer and the fp64 true-residual decision in the G4 timing boundary.
No traffic projection closes G2 or G4.

### STEP C — full acceptance bundle

Run the unchanged opt-in CCX path, verify reduced and full goldens at their proper scopes, exercise CPU fallback,
and collect separate three-sample G4, cold, proven-cache-hit warm, and explicit setup/upload-overlap reports. Re-run
G5 in the final bundle. Generate the score only with `tt_gmg/gates/evaluate_gates.py`.

---

## 11. SAFE COMMANDS CHEAT-SHEET (current)

```bash
TT="ttuser@100.117.137.85"
PROXY="ssh -i ~/.ssh/glmkvm -o BatchMode=yes -W %h:%p root@glmkvmigor"

# Read-only health. Seeing this command itself in pgrep output is harmless.
ssh -i ~/.ssh/id_rsa -o BatchMode=yes -o "ProxyCommand=$PROXY" "$TT" \
  'cat /proc/sys/kernel/random/boot_id; systemctl is-active tt-fold; \
   curl -fsS http://100.117.137.85:8099/api/jobs; \
   ps -eo stat= | awk '\''/^D/{n++} END{print "dstate=" n+0}'\''; \
   pgrep -af "metal_example_brick_spmv|run_brick_hw" || true'

# Hardware workloads use only the hardened runner. It checks empty jobs, a fresh boot ID,
# service settle time, conflicts, D-state, holders, and devices; its EXIT trap restores tt-fold.
ssh -i ~/.ssh/id_rsa -o BatchMode=yes -o "ProxyCommand=$PROXY" "$TT" \
  '/home/ttuser/ttgmg/brick_v1/run_brick_hw.sh \
   /home/ttuser/ttgmg/evidence/device_v1 <new_unique_run_tag>'
```

Never manually interrupt the TT workload, never run a second TT workload on the same boot ID, and never use
`tt-smi -r`. A BMC cold cycle is not a routine pre-run step: use passwordless `sudo -n ipmitool` only after a
separate strict portal-jobs/workload/D-state/foreign-holder/users audit. The BMC can reboot the user's `tt-fold`
host, so confirm the portal returns active with empty jobs afterward. The runner, not an ad-hoc `stop/start`, owns
normal service restoration.

---

## 12. COMMIT LOG (this session, on accel-gmg-backend — most recent first, key ones)

- `*** HW MEASURED: efficient mac_reader MAC = 4.4ms vs 40ms gather (9x) ***` + a-only 2.8ms resident-x floor.
- `SPMV_MAC_READER path` (pre-stored-b DMA reader) + `deep CB prefetch depth 24`.
- `*** GATES REACHABLE ***` (4^3 tiled stencil-shift 1.3× active → 2.36GB → ~1.4ms) + retracts impossibility.
- `serialize tiled-stencil operator to TT-uploadable binary` + `build+validate tiled-stencil (rel_err 3.83e-16)`.
- `*** BREAKTHROUGH: timing gates ARE reachable ***` (historical commit title; its 27-offset/exact-shift result is
  valid, but the literal “101 blocks” subclaim is retracted by the full audit).
- `CORRECT earlier error (nbr is column-sorted, not geometric)` + `3.7%-dense grid root cause`.
- `IMPLEMENTATION_PLAN.md` (Step 1/2/3, exact edits).
- `device recovered via BMC cold-cycle; both readers re-validated on silicon; 3 stale repo-kernel bugs fixed`.
- `preserve validation harness from /tmp into repo` + `fail-closed rebuild check`.
- `fix mac_compute include; mul_tiles_init 4->2 args; mac_writer TensorAccessor page-size`.
- Earlier: per-node gather (64→44ms), offset-grouping L1-infeasible, bf16x2-b host-validated, CPU-GMG measured.

All pushed to `origin/accel-gmg-backend`.

---

## 13. CURRENT STATE (verified 2026-07-15)

- The best durable qualifying evidence is run66, correct at a `1.858679 ms` median. The generated gate evaluator
  reports exactly `2/8` green and acceptance false.
- Run52's selective b-low path retains `63/81` low-term products, passes all four host vectors, and is default-off.
  Its device L2 relative error is `9.338409427e-7`.
- Runs54–57 close affine A-low as device-incorrect. Run59b closes split-A as correct but slower. Runs60/61 close
  exact corner compression/reuse as correct but slower. The host-only joint-mask search raises the omission ceiling
  from 18 to 36 low products but is far too small at the measured slope to merit a hardware run.
- Run66 consumed QuietBox boot ID `b9b8dca3-ec0a-49a8-9635-2789d6d17841`; it is **not** reusable. The final
  closeout restored `tt-fold` with three online workers and three device holders; global controller/portal state is
  idle.
- Run67 consumed fresh boot `c94ec4dd-...`; it is **not** reusable. Its runner restored `tt-fold` and verified the
  service/API/workers/holders/jobs state before the later simultaneous QuietBox/KVM outage. Recover the original
  logs after the site reconnects; do not fabricate them from this handoff.
- Runs45/46/48 are durable negative logs. Run49 repeated their exact fault after explicit FP32 DEST clearing, but
  its raw tee was not retained or recoverable; it is recorded as an observation, never fabricated as a log.
- Physical holder geometry and the Row681/Row682 authority were not modified by this APHYSICAL solver investigation.

## 14. THE SINGLE MOST IMPORTANT NEXT ACTION

Keep Run66 as the G3 frontier. Wait for the shared QuietBox/KVM site path to return; recover and hash the original
Run67/BMC logs; verify `tt-fold`, global portal/controller state, session jobs, D-state, workload, and holders
read-only. Deploy the CB-cycle fix, rebuild the host harness, run host preflight, and offline-compile both dynamic
role sets. Only then use one strict BMC cycle and one guarded fresh boot for the fixed changing-x measurement. If it
is correct and the measured budget supports G4, proceed to adjacent-chip halo exchange, persistent device vector
operations, complete PCG, and the fp64 true-residual gate. Do not claim G2 before its original durable log is
indexed, and do not claim G4/cold/warm/stretch/full CCX from Run67.

## 15. RUN66 / RUN67 PCG RESIDENCY CHECKPOINT (2026-07-15)

- Run66 G3: `1.858679 ms` median, correct; G2: `0.547659 s`, red.
- Exact dynamic map report: `tt_gmg/evidence/device_v1/canonical_pcg_gather_analysis_v1.json`.
- Selected design: `tt_gmg/CANONICAL_PCG_RESIDENCY_DESIGN.md`.
- Same-chip references: `97.9591%`; adjacent-chip halo: `88,776` nodes / `1.598 MB` BF16×3.
- Device page plan: `70.330180 MB` (`70.329036 MB` unpadded host factorization); vector page stream:
  `344.181 MB/apply`; all reconstructions and all-word replay green.
- Run67: upload `0.282974 s` / `411,188,224` bytes, build `0.007103 s`, warmup correctness red with `50,858`
  nonfinite outputs; CB-cycle source fix locally green `29/29`, device rerun pending.
- Local Row236 dump hashes match the QuietBox exactly; unmodified macOS CPU solve is `rc=0`, reduced
  `maxU=95.8129717`, true relative residual `1.02e-6`.
- AWS identity is live but inventory/billing authorization is denied; resource count and spend remain unknown.
  BMC/KVM are private management paths, not compute clouds. The physical lattice cloud remains Row681 v18b;
  Row236 is an APHYSICAL sparse voxel/operator workload.
