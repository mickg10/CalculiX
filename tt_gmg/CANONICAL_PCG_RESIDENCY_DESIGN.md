# Canonical packed PCG residency design

Status: Run67 measured the dynamic upload but failed device correctness; its circular-buffer root cause is fixed
locally, with remote compile and fresh-boot hardware confirmation pending (2026-07-15).

This is APHYSICAL solver tooling for the real Row236 dump. It does not alter or qualify the physical Row681 v18b
holder frontier. No result in this document is a product, geometry, or public-cloud claim.

## What Run66 proved

Run66 is the qualifying standalone canonical packed SpMV frontier on the private eight-chip Wormhole QuietBox:

- samples `1.945499`, `1.822259`, and `1.858679 ms`, median `1.858679 ms`;
- L2 relative error `9.314343407e-7`, maximum relative error `1.590476355e-6`, maximum absolute error
  `1.525878906e-4`, and no nonfinite output;
- complete canonical-base plus exact dense-fallback arithmetic over all `1,290,738` active nodes;
- persistent program and fixed resident pre-shifted vector, with no timed host x/y round trip;
- upload `547.659 ms` for `943,718,400` bytes and program build `5.989 ms`;
- durable log `evidence/device_v1/brick_spmv_run66_canonical_packed_base_fallback_reverse_freshboot_infra_retry.log`,
  SHA-256 `ca218af7a9fb2ee1840e18604a31d6c7c3d40a556153920940abd6df1fe3b48d`.

This closes G3. It does not close G2, G4, cold, warm, stretch, or full CCX integration. The Run66 B files are
pre-shifted for one fixed vector and therefore cannot be reused when PCG changes its search vector.

## Exact dynamic-vector map

`stencil/canonical_pcg_layout.py` derives the missing mapping

```text
[output chip, output group, geometric offset, output lane]
    -> packed logical input-vector position
```

from the brick-neighbor records and canonical permutation. The durable report is
`evidence/device_v1/canonical_pcg_gather_analysis_v1.json`. The remote generated artifacts live under
`/home/ttuser/ttgmg/canonical_pcg_v1/`.

The map is not inferred from one vector. It was checked bit-for-bit against every term plane of the previously
proved random-vector bundle: `314,523,648` BF16 words verified, `0/0/0` split mismatches. The compact unique-node
and page-stream factorizations also reconstruct the raw map with zero mismatches.

Measured Row236 facts:

| Quantity | Result |
|---|---:|
| Active / padding packed lanes | `1,290,738 / 3,598` |
| Valid 27-neighbor references | `33,553,154` |
| Same-chip reference fraction | `0.9795910691` |
| Remote reference topology | adjacent chips only (`-1/+1`) |
| Distinct remote halo nodes, summed by destination chip | `88,776` |
| Remote halo traffic, BF16×3 | `1,597,968 bytes/apply` |
| Unique input nodes per 1,024-node output group, mean / max | `2,442.32 / 3,224` |
| Reference reuse per output group, mean | `11.0433x` |
| Unique input pages per output group, mean / max | `15.40 / 62` |
| Contiguous input-page runs per group, mean / p99 | `4.79 / 7` |

A second spatial/source-order vector is not selected. The measured source-order alternative reduces contiguous
page runs slightly but worsens same-output-chip locality (`95.92%` versus `97.96%`) and expands the distinct remote
halo (`106,680` versus `88,776` nodes). It would also require an extra canonical-to-spatial conversion after each
operator application.

## Selected runtime representation

Keep the logical PCG vectors in canonical packed order on the device. For each destination chip:

1. exchange only the distinct adjacent-chip halo nodes;
2. append that compact halo after the chip's `158 * 1024` local vector lanes;
3. use `pcg_group_vector_page_local_index.u32` to load the full vector pages needed by one output group;
4. use `pcg_neighbor_group_page_lane.u16` to form the 27 shifted term tiles from the staged pages;
5. feed those tiles into the already-proved canonical-base and dense-fallback compute roles;
6. write fp32 output in canonical order for the next device vector operation.

The page plan is selected over scalar unique-node gathering because it uses page DMA instead of millions of tiny
NoC reads. Its exact host artifacts are:

- `pcg_device_group_vector_page_table.u8` (`80,896` bytes, `18,673` page references/apply after padding);
- `pcg_neighbor_group_page_lane.u16` (`69,894,144` bytes);
- `pcg_halo_offsets.u32` plus `pcg_halo_packed_position.u32` (`355,140` bytes).

The selected device runtime metadata is `70,330,180` bytes, half the `139,788,288`-byte raw int32 map. The
unpadded host factorization is `70,329,036` bytes; the extra `1,144` bytes are the fixed-width device page-table
padding required by the kernel contract. It streams
`344,180,736` bytes of BF16×3 vector pages per apply, versus `629,047,296` bytes for Run66's three fixed
pre-shifted-B files. Metadata plus vector-page traffic is about `414.5 MB/apply`, a `34.1%` reduction from the
fixed-B traffic before counting the small `1.60 MB` halo exchange. This is a device budget, not a timing claim.

The alternative scalar unique-node plan is retained as a correctness/control representation. It is `82,573,408`
bytes and stages only the `3,079,766` group-local unique-node occurrences, but its fine-grained NoC transaction
pattern is the known Wormhole failure mode.

## Upload consequence (Run67 measured)

Run66 uploaded `943,718,400` bytes because it included the `629,047,296`-byte fixed vector expansion. Replacing
that expansion with the padded device plan and changing-vector bundle produced a measured Run67 upload of
`411,188,224` bytes in `282.974 ms`, inside G2's `300 ms` numerical budget. Program build was `7.103 ms`.

Run67's warmup output was correctness-red (`50,858` nonfinite values), so it is not G3 evidence. The formal G2
contract does not require correct output, but G2 remains unbanked until the original complete Run67 log is copied
from QuietBox, hashed, and indexed as durable evidence. See `RUN67_CANONICAL_DYNAMIC_CB_FAILURE_ANALYSIS.md`.

The correctness failure is not an address-map error. The initial kernel advanced a `62`-page staging circular
buffer by variable group page counts and advanced a three-page low-split CB by compact mode-3 counts of one, two,
or three. Both sequences can cross the physical FIFO end without completing an exact CB-capacity cycle. The
observed `-inf` concentration near chip 0/group 84 matches the simulated stage-pointer crossing. The source fix
reserves each complete stage allocation once as fixed-address L1 scratch, never pushes that scratch CB, and
publishes a complete three-page low-split cycle whenever a mode-3 batch has any low terms. Local tests pass
`29/29`; device confirmation is still open.

## The G4 integration gap

`src/gmg_solve.cpp` currently owns fp64 outer PCG and the full V-cycle on the host. Its TT hook replaces only a
fine-level smoother SpMV and passes host `double* x/y` through a callback. That path cannot satisfy the
`device_resident_vectors` G4 tag, even if the standalone SpMV is fast.

A qualifying G4 implementation needs:

- canonical packed fp32 outer vectors resident on-device;
- device AXPY, scale, copy, and partial-dot/reduction kernels;
- the dynamic page-gather SpMV above;
- a defined fine/coarse V-cycle boundary (coarse levels may remain host-side only if all transfers are included in
  G4 and the gate report says so);
- scalar reductions and the final fp64 true-residual decision on the host;
- safe CPU fallback on any device, convergence, or true-residual failure;
- three complete correct measured samples and the reduced golden `maxU=95.8129714` before full CCX integration.

The locally transferred Row236 dump is byte-identical to the QuietBox source: raw SHA-256
`e58c212d593dd39884f9806558f1bfc30e6f2456708e61ed2ca1f32220f13638`; zstd SHA-256
`8f13780171520a82d349e69b3c581e3278bf3b927cef8d9d8a6cc987e85ce676`. A current macOS CPU checkpoint using
the unmodified exact solver reached `rc=0`, `maxU=95.8129717`, `56` PCG iterations, setup `4.44 s`, solve
`11.61 s`, and true relative residual `1.02e-6`. This is algorithm/integration evidence, not a TT gate sample.

## Ordered implementation gates

1. **Complete:** exact map, page-plan factorization, all-word fixed-vector replay, compact-plan reconstruction,
   source-order comparison, and local stencil/source-contract tests.
2. **Complete before Run67:** default-off dynamic page reader, host buffer contract, exact Run66-vector replay,
   and offline compilation of both role sets. **Repeat pending after the CB fix:** rebuild host and recompile both
   role sets on QuietBox before another workload.
3. **Run67 complete and red:** the guarded standalone run measured the upload but exposed the CB cycle defect.
   One new guarded fresh boot is required to validate the fixed standalone apply. Do not proceed if correctness is
   still red or the measured apply budget cannot support G4.
4. **Resident vector mechanics:** device vector update, halo exchange, partial dots, and at least two different
   device-generated x vectors without host round trips.
5. **G4 reduced solve:** complete PCG/V-cycle path, fp64 true-residual gate, reduced golden, three samples.
6. **Acceptance bundle:** G2/cold/warm/stretch, cache-hit proof, then opt-in unchanged CCX path and full golden
   `107.0569734213`.

Only one TT workload may run per boot. Never use `tt-smi -r`; use the hardened global-idle runner and restore the
three `tt-fold` workers/device holders after any authorized BMC cycle.
