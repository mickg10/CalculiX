# TT-GMG gate report

Performance score: **2/8 green**.
Acceptance complete: **false**.

| Gate | Status | Median | Budget | Evidence/blocker |
|---|---|---:|---:|---|
| G1 | red | 8.25 | ≤ 3 seconds | median 8.25 exceeds 3 seconds |
| G2 | red | 0.547659 | ≤ 0.3 seconds | median 0.547659 exceeds 0.3 seconds |
| G3 | green | 0.001858679 | ≤ 0.003 seconds | ../evidence/device_v1/brick_spmv_run66_canonical_packed_base_fallback_reverse_freshboot_infra_retry.log |
| G4 | open | — | ≤ 1 seconds | needs 3 sample(s), has 0; missing evidence tags: real_row236, real_8x_wormhole, complete_pcg, device_resident_vectors, fp64_true_residual_gate, correct_output, measured, durable |
| G5 | green | 0.013 | ≤ 0.2 seconds | ../../resume.md#1-the-8-gates-from-tt_gmg_strategymd-lines-48-60 |
| COLD | open | — | ≤ 5 seconds | needs 3 sample(s), has 0; missing evidence tags: real_row236, real_8x_wormhole, complete_solver_run, cold_setup, full_upload, fp64_true_residual_gate, correct_output, measured, durable |
| WARM | open | — | ≤ 1.5 seconds | needs 3 sample(s), has 0; missing evidence tags: real_row236, real_8x_wormhole, complete_solver_run, cache_hit_proven, operator_resident, fp64_true_residual_gate, correct_output, measured, durable |
| STRETCH | open | — | ≤ 2 seconds | needs 3 sample(s), has 0; missing evidence tags: real_row236, real_8x_wormhole, complete_solver_run, cold_setup, full_upload, overlap_measured, critical_path_reported, fp64_true_residual_gate, correct_output, measured, durable |

## Correctness/integration

```json
{
  "full_ccx": {
    "evidence": null,
    "note": "The full 107.0569734213 golden has not run through the TT backend and CCX output path.",
    "status": "open"
  },
  "reduced_dump": {
    "evidence": "../TT_GMG_STRATEGY.md#correctness-gate-banked-on-real-tt--2026-07-03",
    "max_u": 95.8129714,
    "rc": 0,
    "status": "green",
    "tol": 1e-06,
    "true_relative_residual": 1.13e-06
  },
  "safety_path": {
    "note": "The fp64 gate exists in gmg_solve.cpp, but final TT fallback behavior must be exercised in CCX.",
    "status": "code_present_unaccepted"
  }
}
```
