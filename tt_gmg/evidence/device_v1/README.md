# Row236 resident-x device evidence v1

Scope: APHYSICAL TT-GMG tooling for the real row236 brick operator on the private eight-chip Wormhole QuietBox.
This folder is not physical holder geometry, product evidence, or public-cloud evidence.

## Qualifying G3 frontier

Run66 supersedes Run52 as the qualifying G3 frontier. It is the complete canonical-base plus exact
dense-fallback packed operator with a fixed resident pre-shifted vector:

- samples `1.945499`, `1.822259`, and `1.858679 ms`, median `1.858679 ms`;
- G3 green by `1.141321 ms`;
- L2 relative error `9.314343407e-7`, maximum relative error `1.590476355e-6`, maximum absolute error
  `1.525878906e-4`, and no nonfinite output;
- upload `547.659 ms` for `943,718,400` bytes, so G2 remains red;
- log SHA-256 `ca218af7a9fb2ee1840e18604a31d6c7c3d40a556153920940abd6df1fe3b48d`.

`canonical_pcg_gather_analysis_v1.json` is the host-only successor contract for changing PCG vectors. Its exact
map reproduces all `314,523,648` previously proved BF16 words with zero mismatch. The selected page-gather plan is
`70,330,180` device bytes (`70,329,036` before fixed-width page-table padding), uses a `1,597,968`-byte BF16×3
inter-chip halo, and reconstructs the raw map with zero mismatch.
It is not device timing or G4 evidence; see `../../CANONICAL_PCG_RESIDENCY_DESIGN.md`.

## Run67 dynamic page-reader incident

Run67 used the exact dynamic plan on a fresh guarded boot. Its preflight passed and it measured a complete
`411,188,224`-byte upload in `282.974 ms` plus a `7.103 ms` program build, but the warmup was correctness-red with
`50,858` nonfinite outputs. It is not G3 evidence. The original Run67 and BMC logs remain on QuietBox and are not
yet indexed here because the QuietBox and RM10 KVM became unreachable together shortly after the runner restored
`tt-fold`. Do not reconstruct or fabricate those logs.

Source/cadence analysis identified a circular-buffer cycle violation: variable `7..62` page-count advances on a
62-page staging CB and compact `1/2/3` mode-3 advances on a three-page low-split CB can cross the physical FIFO end.
The dominant group-84 `-inf` region matches the simulated stage crossing. The fixed source reserves each complete
stage allocation once as fixed-address L1 scratch, never pushes it, and uses full three-page low-split publish
cycles. Local tests pass `29/29`; remote recompile and fresh-boot device confirmation are pending. Detailed record:
`../../RUN67_CANONICAL_DYNAMIC_CB_FAILURE_ANALYSIS.md`.

The Run67 upload is numerically inside G2's budget, and G2 does not require a correct-output tag. G2 is nevertheless
left unbanked until the original durable Run67 log is recovered and hashed.

## Historical Run52 frontier

Run52 was the qualifying frontier before Run66 and remains useful historical evidence:

- correct full bf16×3 reverse-order resident-x apply;
- persistent device program, all eight chips, no host x/y round trip;
- default-off selective b-low policy retaining Manhattan-distance-≤2 terms plus the opposite-corner pair (`63/81`);
- samples `3.379803`, `3.377273`, and `3.256412 ms`;
- median `3.377273 ms`, therefore G3 red by `0.377273 ms` against `3.000000 ms`;
- L2 relative error `9.338409427e-7`, max relative error `1.590476355e-6`, maximum absolute error
  `1.525878906e-4`, and no nonfinite output;
- log SHA-256 `922c274ac1c6eb7ff216456a58a944170335c3c7293f0a8d45e5b88b6f6ee68a`.

Run41 remains the correct all-term baseline (`3.402191 ms` median, log SHA-256
`405ec1f034b64f08ca5db91439d92fda77450312aaf4a4253934628370a3b5b4`). Run52 improves its median by
`0.024918 ms` (`0.73%`) without closing G3.

## Later conservative candidates

| Run | Candidate | Result | Interpretation |
|---|---|---|---|
| 50 | Fixed HiFi2 with exact leading product | Correct, median `18.312919 ms` | Numerically useful but much slower; log SHA-256 `eaa8cafc2bb36e63e52b14b5175d042e09c755fdc7ccfaf21bd00fd08932e5d9`. |
| 51 | Selective b-low | Mesh-open failure after the then-current `300 s` settle interval | Infrastructure-negative before kernel execution; log SHA-256 `eb4b8d381d3978b44bb0df6ad8f06b837d3db6c4ee57f53e6b68f0b8902c1c01`. |
| 52 | Selective b-low after fresh-boot recovery and `420 s` settle | Correct, median `3.377273 ms` | Historical pre-Run66 frontier; G3-red. |
| 53 | Affine A-low corner fold plus run52 b-low policy | Guarded timeout after `850 s`, before any device sample | The scalar host transform is implementation-negative; not a G3 or device-correctness result. Log SHA-256 `e5c0505ff07bd4833b841292de7780db2aabb0f53c66ba2f077f1c7a970648ab`. |
| 54 | Vectorized/precomputed affine A-low fold | Device deadlock; no qualifying sample | Device-negative; SHA-256 `ff24cc2070ef947618e012faf651d0cf0a5ec81626897d93396872e8ee1afc34`. |
| 55 | Affine A-low with fixed CB cadence | Same two device output errors | Correctness-red; SHA-256 `cbb8a4a02aff0a98a5a37df648c3da284f843d290bcd17362c5eaaf4b8a2a1b4`. |
| 56 | Affine A-low with zero cadence | Same two device output errors | Correctness-red; SHA-256 `25ab888de996ea10758217c8cd6027d63aaba6272060cf7f757c35d8a44c19f5`. |
| 57 | Affine A-low with dummy-NoC cadence | Same two device output errors | Correctness-red; affine family closed; SHA-256 `e76cd1f092f0734d697959a501a1da49fac59f8acd73e17e65f68268371cf197`. |
| 58 | Split-A redistribution | Infrastructure-negative | No G3 evidence; SHA-256 `4d5c63bc59395c78bed5179242e5f09dfdcc4194ae1d008487218c6284c8dc77`. |
| 58b | Split-A redistribution retry | Infrastructure-negative | No G3 evidence; SHA-256 `cdb43f04d6d658ff6bfd3d95f1c014f871174d99892759fe5834953bbaca4717`. |
| 59 | Split-A after extended service stop | Preflight aborted on transient D-state | No mesh opened and no workload ran; SHA-256 `800314c2b77299bdd49b9f071551947aa4d364bd687b427ab150bf6d8a5c1d2e`. |
| 59b | Same-boot pre-workload split-A retry | Correct, median `3.505936 ms` | Slower than run52; SHA-256 `04e9ef9d26ceae76634bb0e294328c1329c66d5b482ce941749d7227a2c9ca28`. |
| 60 | Lossless exact-corner compressed buffer | Correct, median `3.532192 ms` | Slower; SHA-256 `98fd6ad5547a9695502a90334886aae667394fcb162eedaec0c9d92c354ef47f`. |
| 61 | Exact-corner representative reuse from original fixed buffer | Correct, median `3.405163 ms` | Removes run60's source/CB penalty but remains slower than run52; SHA-256 `1954e27aa51f4ca571f22d8ddf54f3e92391a97627c9fc8c19176511e63c4944`. |

The original run53 prerequisite evidence remains valid: its random/smooth/boundary/constant host oracle passes
(JSON/log SHA-256 `73b102f42fdcda4f127f8d01fd9df62882b72840b4c3c51cae814f0653a8ebe9`) and its BRISC plus TRISC0/1/2 roles
compile offline (log SHA-256 `ec33fe20515f32dd3e053a4ba38a6cee540bbfbea2d4a5fc52cca328fdde6e47`). Runs54–57 add the decisive device evidence:
vectorizing/precomputing the transform did not make the family correct. Do not retry it.

## Exact corner-structure proof and device result

`build_corner_constant_compression.py` audited the complete raw operator rather than sampling it. Every present
coefficient at the eight corner offsets uses one of exactly three BF16x3 split triples, with a shared presence mask.
The generated `corner_a_exact.bf16` artifact is `233,570,304` bytes and reconstructs all corner coefficients with
zero bit mismatches. This is a corner-only fact: the old global “101 distinct blocks” statement was a literal print,
not a measured dictionary, and must not be reused.

The compression manifest SHA-256 is `b4c3e576bcbc0c7a691366c980d0be716ec155ec5674cd3a453f173587b5557c`;
the run60 and run61 all-role compile logs are respectively
`ce4bc51990a9ebb182d93ba223ea793e998fd425119e08881bfb3757eb27c8ca` and
`7b21f9c7e8b6cd5d8d7b5115af08efe602d4876fe3e35c52cd2a64ad84de6117`.
Both device candidates were correct but slower, so exact corner A-coefficient transport is closed as the G3 wall.

## Joint low-product search

`search_joint_low_masks.py` exhaustively formed the allowed single-action schedules over 14 inversion groups, then
ran the exact ordered oracle on the leading candidates across random, smooth, boundary, and constant vectors. The
tested model's precision ceiling is `36/162` omitted low products; run52 already omits `18/162`. The best eligible
schedule passed all four vectors (smooth L2 `1.772199631877898e-5`) but provides only 18 additional omissions. At
run52's measured `0.024918 ms` gain for the first 18, this cannot credibly remove the remaining `0.377273 ms`.
No BMC cycle or hardware workload was justified. Full search log SHA-256:
`371374bb755c9918483c378ace30bdde381fb24a82eea97ca16cf517f3eecac0`.

## Mixed-fidelity diagnostics

| Run | Schedule | Correct | Median/result | Interpretation |
|---|---|---:|---:|---|
| 44 | Per-term HiFi4 leading + HiFi3 degree-2 | yes | `3.534256 ms` | Red and slower than run41. |
| 45 | Three-term HiFi2 grouping, direct MOP update | no | one `-31.99997520` lane error | Negative. |
| 46 | Three-term HiFi2 grouping, full LLK reinit | no | identical lane/error | Full reinit falsified as the cause. |
| 47 | Per-term HiFi4 leading + HiFi2 degree-2 | yes | `3.550619 ms` | Numerically valid, red, slower. |
| 48 | Three-term HiFi2 grouping + full Tensix drain | no | identical lane/error | Pipeline-drain hypothesis falsified. |
| 49 | Run48 schedule + explicit full FP32 DEST clear | no | identical lane/error | Initial-DEST-residue hypothesis falsified. |

The raw run49 tee was not retained by the remote runner and could not be recovered from the host, `/tmp`, or the
local workspace. It is therefore recorded only as an observed diagnostic, not cited as durable gate evidence. The
machine was subsequently verified healthy: boot ID guard still consumed, `tt-fold` active, portal jobs empty, zero
D-state tasks, and no GMG workload. No log has been reconstructed or fabricated.

Runs45/46/48 have durable logs with SHA-256 values
`c31fd2cc0d8dcb51d6fc81d9f2de6180acdab86201db9e602cf4c2b6b8538105`,
`13276d29ae3f9a406483ef6fcc99013e2bbb5cc155e7fb3bd4392f32bb0e7a0a`, and
`a69303a3b20137f92e8b2b6d556c180d1620fad1653e961e3e9ed0e9abc541a9`.

## Lane reconstruction and host oracles

`brick_spmv_run48_bad_lane_chip2_group162_c0_lane204.json` maps the corrupt output to core 44, group slot 0,
brick 3, local `(0,3,0)`. The strict FP32 reconstruction is `12.72800064086914`, exactly the stored reference;
hardware returned `-19.27197456`. No individual product is near `32`, so the symptom is not one missing ±32
coefficient product. Diagnostic SHA-256:
`de2d8232d39e682447fb0d14c98b0ec40aab9862ce211db945074d54e6eb3dc2`.

The per-term, batch-one, and batch-three mixed-HiFi2 host schedules all pass random, smooth, boundary, and constant
vectors. Their JSON SHA-256 values are respectively
`e37d7abd752b80ef293e0f7841323ff0a1de4c224fbb5fc9683938464d0a37b8`,
`26f766c87ebe0f3d720c10a67f7349cb4944409db61b0c4a4549dd13b7784f10`, and
`d8819887af86eb332362389cfd5466ac60532bee34784dd48961c65d4665264b`.
That proves the arithmetic/order model, not device safety. The repeated hardware failure closes cross-term fidelity
batching as a usable optimization family for this kernel.
