# TT-GMG row236 gate contract

Status: authoritative for the original 8×Wormhole TT-GMG closure program as of 2026-07-15.

This file resolves the conflicting gate counts and optimistic status labels in `resume.md`,
`TT_GMG_STRATEGY.md`, `tt_gmg/GMG_ACCEL_GOAL.md`, and `tt_gmg/TT_GMG_PLAN.md`. The original strategy has
exactly **eight performance gates**:

| Gate | Exact measured workload | Budget |
|---|---|---:|
| G1 | Cold host setup: hierarchy, BCSR, and all bf16×3 operator terms | ≤ 3.0 s |
| G2 | One-time upload of the complete bf16×3 operator to all 8 chips | ≤ 0.3 s |
| G3 | One correct, persistent, resident-x fine SpMV on the real row236 operator | ≤ 3 ms |
| G4 | Complete accelerated PCG solve, including device vector work and the fp64 true-residual gate | ≤ 1.0 s |
| G5 | Complete solution download and un-permute | ≤ 0.2 s |
| COLD | First complete accelerated solver run, G1/G2 included | ≤ 5.0 s |
| WARM | Repeated complete solver run with an actually reused cache and resident operator | ≤ 1.5 s |
| STRETCH | Cold run with measured setup/upload overlap | ≤ 2.0 s |

Correctness is not a ninth performance gate. It is a non-negotiable acceptance invariant:

- Reduced-dump component correctness: `maxU = 95.8129714`, `rc = 0`, and
  `true_rel < min(2*tol, 1e-2)`.
- Full CCX correctness: `maxU = 107.0569734213`, the same residual rule, and correct output read-back.
- The fp64 true-residual check and safe CPU fallback remain active.
- The accelerated result must enter the unchanged, opt-in CCX solve path. A standalone dump solve cannot close
  full acceptance.

## Evidence rules

A performance gate is green only when all of the following hold:

1. The measurement is from the real 8×Wormhole target and the real row236 workload named above.
2. The measured interval matches the gate boundary. Setup excludes dump-file I/O only when the report says so,
   but it must include hierarchy, BCSR, and all three bf16 coefficient terms. G4 excludes G1/G2 but includes all
   per-solve PCG work and the fp64 residual decision.
3. The kernel produced a correct output during the timed run. Placeholder-b, a-only, zero-c, subset, projected,
   or synthetic runs are diagnostics and cannot close G3.
4. Persistent claims use a reused program and resident state. A per-apply program rebuild or host x/y round trip
   cannot be silently excluded.
5. G3, G4, COLD, WARM, and STRETCH use at least three completed samples and report the median. A timeout, wedge,
   or interrupted workload is a failed sample, not discarded noise.
6. Every green cites a durable log or machine-readable report. Projections and arithmetic estimates are never
   promoted to measurements.
7. WARM requires a cache-hit proof. “Cacheable in principle” is not a warm measurement.
8. STRETCH cannot reuse the COLD number; the report must show the overlap interval and critical path.

The machine-readable form is `tt_gmg/gates/gate_contract.json`; evaluate evidence with
`python3 tt_gmg/gates/evaluate_gates.py --evidence <file> --json-out <report.json> --md-out <report.md>`.

## Current reconciled status

| Gate/invariant | Status | Reason |
|---|---|---|
| G1 | Red | Measured setup is about 7.7–8.25 s, above 3 s; the earlier label treated cacheability as a pass. |
| G2 | Red | Run66 measured `0.547659 s` for the conservative complete timed bundle (`943,718,400` bytes: exact operator, fixed resident packed vector, and zero output), above `0.3 s`. The dynamic-vector bundle reduction is host-proved but not device-measured. |
| G3 | Green | Run66 is correct, persistent, resident-x, and complete: samples `1.945499/1.822259/1.858679 ms`, median `1.858679 ms`, below `3 ms`. Its fixed-vector representation does not by itself close G4. |
| G4 | Open | Resident-vector PCG is not implemented or measured. |
| G5 | Green, component scope | A complete solution read measured 0.013–0.018 s, below 0.2 s. It must be remeasured in the final bundle. |
| COLD | Open | No qualifying complete fast run. |
| WARM | Open | No persistent setup/operator cache-hit run. |
| STRETCH | Open | No measured overlap run. |
| Reduced correctness | Green | Real TT reduced dump reached `rc=0`, `maxU=95.8129714`, true relative residual about `1.13e-6`. |
| Full CCX correctness/integration | Open | Full golden `107.0569734213` has not run through the TT backend. |

Therefore the honest performance score is **2/8 green**, with reduced-dump correctness separately green. The
score changes only through a generated evidence report that satisfies this contract.

## 2026-07-14 resident-x addendum

At this dated checkpoint, `tt_gmg/gates/current_evidence.json` cited the durable run41 log for G3. It was then the
best qualifying result:
full row236 brick operator, all eight Wormhole chips, persistent program, resident x, no host x/y round trip, correct
output, and three timed samples. Its `3.402191 ms` median is honestly red by `0.402191 ms`.

The mixed-fidelity investigation did not close that gap. Per-term HiFi3 (run44) and HiFi2 (run47) remained correct
but slower, at `3.534256 ms` and `3.550619 ms` medians. Grouping three terms across a fidelity transition passed the
host arithmetic oracle but produced the same single-lane device corruption in four fresh-boot observations:
chip 2, local group 162, component 0, lane 204 was `-19.27197456` instead of `12.72800064`, a
`-31.99997520` error. MOP-only switching, full LLK reinitialization, a full Tensix pipeline drain, and an explicit
FP32 destination clear did not change it. That schedule is closed negative evidence, not a candidate for another
hardware retry. The source/deployed host binary at that checkpoint was restored to the run41-equivalent all-HiFi4
schedule; the later current frontier is recorded in the 2026-07-15 addendum below.

## 2026-07-15 selective-low and affine-fold addendum

The default-off selective b-low path retains the terms with Manhattan distance at most two plus the opposite-corner
pair (`63/81` low-term products). It passed the random, smooth, boundary, and constant host oracles, compiled the host
and all BRISC/TRISC roles, and produced that checkpoint's best qualifying device result in run52. Its samples were
`3.379803/3.377273/3.256412 ms`; median `3.377273 ms`; L2 relative error `9.338409427e-7`; maximum relative error
`1.590476355e-6`; and maximum absolute error `1.525878906e-4`. This improves run41 by `0.024918 ms` but leaves G3
red by `0.377273 ms`.

The affine A-low candidate folds omitted corner coefficients into center and signed-face coefficients while
preserving affine fields. Its four-vector host oracle and offline roles pass, but the device family is now closed:
run53 timed out in scalar preprocessing; vectorized/precomputed run54 deadlocked; and runs55–57 repeated the same two
output errors under fixed, zero, and dummy-NoC cadence. Run59b's split-A redistribution was correct but slower at a
`3.505936 ms` median.

A complete raw-operator proof found that the eight corner offsets—not the full operator—share three exact BF16x3
coefficient triples and one presence mask. Lossless compressed-buffer run60 was correct at `3.532192 ms`; fixed-page
representative-reuse run61 removed its source/CB penalty but remained slower at `3.405163 ms`. Exact corner transport
is therefore closed as the G3 wall. A host-only joint inversion-symmetric A-low/B-low search found four-vector-green
masks, but its tested precision ceiling is only 36 omitted low products versus run52's 18. The extra 18 cannot close
the `0.377273 ms` gap at the measured run52 slope, so no hardware claim or boot was made for that search.

The final guarded run61 consumed boot ID `f60ce545-7ba3-4307-9ca6-bdcbdfeb722e`; it is not reusable for another
workload. The runner restored `tt-fold`. The next G3 candidate must structurally reduce remaining arithmetic or
B-materialization, pass the complete host/offline proof, and show a budget large enough to close the measured gap
before a new guarded boot is spent.

## 2026-07-15 canonical packed Run66 and PCG-residency addendum

Run66 supersedes Run52 as the qualifying G3 frontier. It partitions each chip into 42 canonical-base and 14 exact
dense-fallback compute roles. The complete measured samples are `1.945499`, `1.822259`, and `1.858679 ms`; median
`1.858679 ms`. L2 relative error is `9.314343407e-7`, maximum relative error `1.590476355e-6`, maximum absolute
error `1.525878906e-4`, with no nonfinite output. Durable log:
`evidence/device_v1/brick_spmv_run66_canonical_packed_base_fallback_reverse_freshboot_infra_retry.log`, SHA-256
`ca218af7a9fb2ee1840e18604a31d6c7c3d40a556153920940abd6df1fe3b48d`.

The same run measured upload `547.659 ms` for `943,718,400` bytes and program build `5.989 ms`. G2 therefore stays
red under the conservative measured boundary. A smaller dynamic bundle may close it, but projections are not gate
evidence.

Run66 consumes one fixed pre-shifted vector and is not a complete PCG. The host-only dynamic-vector contract in
`CANONICAL_PCG_RESIDENCY_DESIGN.md` and
`evidence/device_v1/canonical_pcg_gather_analysis_v1.json` derives the exact changing-x map, verifies all
`314,523,648` BF16 words with zero mismatch, and reduces the selected page-gather runtime plan to `70,329,036`
bytes with a `1,597,968`-byte BF16×3 adjacent-chip halo. Those are implementation facts, not G4 or device-timing
evidence. G4 remains open until device vector operations, halo exchange, the complete preconditioned solve, fp64
true-residual decision, reduced golden, and three measured samples all pass.
