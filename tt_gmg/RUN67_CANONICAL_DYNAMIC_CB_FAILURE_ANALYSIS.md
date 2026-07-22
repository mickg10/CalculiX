# Run67 canonical dynamic circular-buffer failure analysis

Status: root cause identified and source fix locally verified; remote offline compile and fresh-boot hardware
confirmation are pending (2026-07-15).

Scope: APHYSICAL TT-GMG solver tooling for the real Row236 operator. This is not physical holder geometry,
structural/acoustic evidence, or a public-cloud result.

## Measured Run67 result

Run67 used fresh QuietBox boot `c94ec4dd-...` and the hardened global-idle runner. Its host/device-plan preflight
passed with `33,553,154` valid lane references, `1,393,918` absent references, `18,673` group pages, `1,261`
nonempty groups, `3` padding groups, and a maximum of `62` pages per group.

- Complete dynamic-bundle upload: `282.974 ms` for `411,188,224` bytes.
- Program build: `7.103 ms`.
- Warmup correctness: RED, with L2 relative error `1.492614469e+36`, maximum relative error
  `3.546880582e+36`, maximum absolute error `3.402823466e+38`, and `50,858` nonfinite outputs.
- The largest failures were `-inf` values concentrated around chip 0, group 84, component 0.
- Workload exit: `2`.

The runner then restored `tt-fold` and verified the service active, its API ready, three online workers, three
expected device holders, and an empty session jobs list. The original Run67 and BMC logs remain on QuietBox and
must be copied and hashed before this incident is used as durable gate evidence.

The `282.974 ms` measurement is numerically inside G2's `300 ms` budget. The gate contract does not require a
correct-output tag for G2, but G2 remains unbanked until the original complete log is recovered and indexed.
Run67 is not G3 evidence.

## Root cause

Tenstorrent circular buffers require producer/consumer pointer movement to complete exact cycles of the declared
CB capacity. A reserve operation can wait for free pages, but it does not make a variable-size transfer that
crosses the physical end of the CB legal.

The first dynamic readers allocated a `62`-page CB for staged vector pages and then advanced its FIFO pointer by
each group's variable `page_count` (`7..62`) once per component. The actual chip-0 sequence contains adjacent
groups with page counts `16`, `18`, and `18`; simulation of the accumulated pointer exposes multiple end
crossings. The crossings include the group-83/component-0 and group-84/component-1 region, matching the dominant
Run67 corruption location.

Runtime order mode 3 exposed a second instance of the same bug. Its component-major low-split batch counts are

```text
1, 3, 2, 3, 3, 3, 2, 3, 1
```

for each component. The low CB has capacity three pages, so compactly advancing by `1`, then `3`, crosses its end.
Run66 used order mode 2, where the corresponding low-count cadence was zero or three, and therefore did not expose
this path.

Host replay proves address mapping and BF16 words, while offline compilation proves kernel syntax and role
descriptors. Neither executes the on-device CB pointer protocol, so both correctly passed before Run67 without
detecting this runtime defect.

Primary API basis: Tenstorrent documents that a complete CB push cycle must sum exactly to the configured CB size
([`cb_push_back`](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/circular_buffers/cb_push_back.html))
and that `get_write_ptr` is valid only after a reservation and before the matching push
([`get_write_ptr`](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/circular_buffers/get_write_ptr.html)).
The fix below satisfies both rules without advancing the variable-count staging region.

## Fix

- The four dynamic readers now treat the `62`-page stage allocations as dedicated fixed-address L1 scratch.
  Each allocation is reserved once at its complete `62`-page capacity so `get_write_ptr` is valid under the
  documented TT-Metal API, then it is never pushed, waited, or popped; its pointer therefore never advances.
- A mode-3 low-split batch publishes a complete three-page CB cycle whenever it contains any retained low terms.
  Only the first `low_count` pages are materialized or used by the MAC schedule; writer and compute both advance
  the FIFO by three.
- Shared compute behavior changes only for runtime order mode 3. Run66's mode-2 fixed-vector path retains its
  prior cadence and arithmetic.
- The harness reports `CANONICAL_DYNAMIC_RING_CONTRACT persistent_stage_scratch=1
  stage_full_reservation_pages=62 stage_fifo_pushes=0 low_split_publish_pages=3 mode3_fifo_wrap_safe=1` during
  host preflight.

## Pinned Run68 deployment set

Copy these exact local sources into the matching QuietBox TT-Metal example paths and verify their SHA-256 values
before rebuilding:

| Relative source | SHA-256 |
|---|---|
| `canonical_dynamic_spmv.cpp` | `af4399eef7b56ceda9ee4c79f5da1945b4da589cc0eee9d07be12b35fc984155` |
| `kernels/canonical_dynamic_page_common.hpp` | `bce2b0a9364a55f0657595e1f60b8166b3b817702b54e9c81e1c127bdd162e5b` |
| `kernels/brick_reader_canonical_dynamic_base_bhi.cpp` | `49617138f18167e5d7d5a0b82f8e448cb57f893ce539f51b5d59d5bfe907ae51` |
| `kernels/brick_reader_canonical_dynamic_base_bmid_blow_writer.cpp` | `03a6b2f8dcba4b5b45734767918bd7ec5e3b841061500f494c841975f99ee80c` |
| `kernels/brick_reader_canonical_dynamic_fallback_bhi.cpp` | `65e5c1368e2639ba3a352ef417cb079042239c5ff18a346f574c76eaae1554da` |
| `kernels/brick_reader_canonical_dynamic_fallback_bmid_blow_writer.cpp` | `7bb09a178942d114b00669ff0ae2f6aed990a374d519a2408f02ac743b06df40` |
| `kernels/brick_compute_canonical_packed_base.cpp` | `12fdd59cb20a42518ba29bea309ed0e8aff76b5448b143241813d6fb3b747929` |
| `kernels/brick_compute_bf16x3_preexpanded_b.cpp` | `925210b5516da9eef8b295643febdcf01a43f8d069a9cfde516430b397afa292` |

Do not deploy by directory replacement: inspect the remote files first and copy only this scoped set so unrelated
QuietBox TT-Metal work is preserved.

## Verification and next gate

Local stencil discovery passes `29/29` tests after the fix. The source-contract test requires exactly one full-size
reservation before each stage `get_write_ptr`, forbids push/wait/pop on stage CBs, and simulates the mode-3
low-split cycle.

When the QuietBox/KVM site reconnects:

1. recover and hash the original Run67 and BMC logs;
2. verify `tt-fold`, global portal/controller databases, session jobs, D-state, and device-holder state read-only;
3. deploy the fixed sources, rebuild the host harness, run host preflight, and offline-compile both dynamic role
   sets;
4. do not reuse Run67's consumed boot;
5. only after the strict global idle preflight, perform one BMC cycle and run one fresh-boot hardware validation.

No BMC or workload action is authorized merely by this diagnosis; all normal safety gates remain mandatory.
