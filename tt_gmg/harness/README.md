# TT-GMG validation harness (tt-quietbox, 8×Wormhole)

Preserved from the box's ephemeral `/tmp` on 2026-07-10 (a BMC cold-cycle wipes `/tmp`, so this is the durable copy).

- `whrun_solve.sh FABRIC_1D 8` — rebuild `libtt_spmv` from `/tmp/tt_spmv.cpp`, JIT-copy `/tmp/kernels/*.cpp`, run `solve_driver.py` with the eig-deflation env + `SPMV_TIMING`, 850 s timeout. Fail-closed rebuild check.
- `solve_driver.py` — loads `/tmp/libtt_spmv.so` + `~/ttgmg/libgmg.so`, wires `g_tt_fine_spmv`, runs `ccx_gmg_solve_from_dump`, prints maxU + per-apply/phase timing.
- `make_cancel_op.py` — synthetic extreme-cancellation operator (precision-gate input).

## Golden result (banked on silicon)

`maxU = 95.8129714` (row236 reduced), `true_rel ≈ 1.1e-6`, ~89 PCG iters. Env: `GMG_DEFL_CORR=1 GMG_DEFL_EIG=1 GMG_DEFL_K=24 GMG_HYBRID_TOL=1e-2`.

## Post-reset validation sequence (device must be BMC cold-cycled first)

The device is currently in a dirty core-state (every run hangs at the first apply, `run_mailbox 0x40`) from
mid-workload-kill / L1-overflow hangs. **Recovery = BMC cold-cycle 10.0.0.48 only** (`tt-smi -r` is forbidden).
After `test -e /dev/tenstorrent/0` returns true and `boards_busy=0`:

1. **Re-confirm baseline** — proven per-node reader (repo `tt_gmg/kernels/gather_reader_galaxy.cpp` @ HEAD) →
   `/tmp/kernels/gather_reader.cpp`; depth-3 `tt_spmv.cpp` → `/tmp/tt_spmv.cpp`. Run `whrun_solve.sh FABRIC_1D 8`.
   Expect maxU=95.8129714, workload ≈ 44 ms. If this hangs, the device is still dirty — re-cycle.
2. **Validate the pipelined gather** (`git show 92d9a76:tt_gmg/kernels/gather_reader_galaxy.cpp`) — the
   dependent-load-stall fix (prefetch node+2 nbr / node+1 x while writing node b). Correct-by-inspection & OOB-safe.
   Confirm maxU unchanged; read the new workload ms. Target: well under 44 ms → measure G4/cold/warm.
3. **If more headroom needed** — stack bf16×2-b (path B, host-validated `emu_bf16x2b_precision.py`): b in 2 levels
   (5 cross-terms), which cuts gather writes 1/3 AND frees ~148 KB L1 (re-enabling depth-9 offset-grouping).
4. **NEVER** `kill -9` a multi-chip run mid-workload — let the 850 s timeout end a hang (exit=124), else the
   cores are left dirty and need another BMC cycle.
