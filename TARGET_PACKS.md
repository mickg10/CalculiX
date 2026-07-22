# CalculiX target-pack registry

Large targets and portable reproduction packs are stored outside this source
fork in the content-addressed repository:

- repository: `https://github.com/mickg10/calculi_target_packs`;
- registry commit:
  `97a2e85b19168e9a081c72252e6eb1e164c07e39`;
- immutable release tag: `packs-v12-20260722`;
- index SHA-256:
  `01a03e0128db49718e2f412964f1572d158db278be5f8b08fecaf3f0d92bb5b4`;
- machine-readable binding: `tt_gmg/target_pack_registry.json`.

Do not identify a target by a local pathname, a branch, “latest,” or a release
asset label alone. The minimum identity is the pack ID plus manifest SHA-256;
each manifest then binds every release object by full SHA-256 and byte count.

## Published pack manifests

| Pack ID | Status | Manifest SHA-256 |
|---|---|---|
| `row236_ttgmg_fine_v1` | target | `9cdf3a75e721acd5649f02f71d68c6f84201247d70b4e99dbefe7d0b27e08453` |
| `row236_path02_fulltrunk_hubfan_20260615` | reproducer | `9e065a0bcc33d02887ac3d79618e335fa2c06f50b5e56a3f981f1ed7856e30d1` |
| `rigid_body_modal_c3d8_20260716` | reproducer | `7918703441c7c4f527e4b55a254165b8c96bfa83664cc4bc9f7be94718782841` |
| `modal_v18b_solver_targets_20260715` | frozen target ladder | `cc3e4678f72ab25fcba17e71da90a209ab8d609e3852ed2c74b055f3acab3251` |
| `payloaded_modal_v18b_exact_decks_20260716` | frozen exact anchors | `1bd1d1c087e15cac5ebf36796d0d3aa012023f2c8ac54c6081f97130a9811192` |
| `matrix_free_payloaded_transfer_wip_20260719` | superseded work-in-progress transport snapshot | `7813f50775c665d698fd18ac082950b867945803bc9e6b3201d7c756ab74ce79` |
| `matrix_free_payloaded_transfer_wip_20260720` | superseded work-in-progress continuation pack; scalar Stage 1 red | `d09b727e446c040b6f507bebeca35d729c56c0e7d0302787ed30f909328fb8e7` |
| `matrix_free_payloaded_transfer_block_schwarz_wip_20260720` | superseded generation-2 WIP; p010 topology storage green, Stage 1 convergence red | `d4f373fc1745bcdbdce3dc451e756447f583ee30ff63af4e0574603ccf27e6b0` |
| `matrix_free_payloaded_transfer_modal_schwarz_red_wip_20260721` | superseded generation-3 WIP; exact modal-Schwarz algebra green, Stage 1 convergence red and consumed | `d3122b2855a08908e46a57d3feea67964d372ba280a2390d5a3dcd8df106223b` |
| `matrix_free_payloaded_transfer_smoothed_aggregation_red_wip_20260721` | superseded generation-4 WIP; p035/p030 fixed calibration complete, no candidate selected, no target response | `fd5502b9a6f4f2f46069a6f5d107c7b021c908cb456c3f3b0ad88cd096a73ffa` |
| `matrix_free_payloaded_transfer_coarse_polynomial_red_wip_20260721` | superseded generation-6 WIP; P1 consumed terminal RED before PCG, no candidate selected, no p010 or target response | `e42e667f2a14da4f4e368fc499a44db0ad68f2e8ede6f25603ca9e3ddc2ddd9f` |
| `matrix_free_payloaded_transfer_generation7_execution_repair_red_wip_20260721` | superseded generation-7 WIP; P1A stopped terminal RED at setup case four before RHS/operator/PCG, no P1B, p010, or target response | `4571cd7ea53c09237c16b8c1602d15b49ba6755562327e07809b38ef959a862c` |
| `matrix_free_payloaded_transfer_generation8_p0_green_wip_20260722` | superseded generation-8 WIP; marker-free p035/p030 P0 GREEN and immutable, no topology/operator/Krylov/p010/response | `ebcfa948df07075ef9a4a944cf1dd3227c3dcd4caffd59281b5f80732c77eba9` |
| `matrix_free_payloaded_transfer_generation8_p1a_green_wip_20260722` | superseded generation-8 WIP; all ten p035 P1A setup/RHS/resource cases GREEN and immutable, no action/Krylov/p010/response | `99310dd22ddb4a595055ab0ad2386fd62ecb83b302fda9f4c60167221f50c1d5` |
| `matrix_free_payloaded_transfer_generation8_p1b_preparation_green_wip_20260722` | superseded generation-8 WIP; response-blind P1B plan/freeze/exact logical ledger GREEN without launch | `28c185c0cacf16d7a503a462e91343fe2a857086f7bf688af7457d696ec11936` |
| `matrix_free_payloaded_transfer_generation8_p1b_terminal_red_wip_20260722` | superseded generation-8 WIP; all ten p035 fixed-screen cases integrity-GREEN, terminal no-candidate RED, P2 and later work locked | `d768717ab5a36abe53a5bb29aff0490d33ff9bd16abb09e85f128dcd7415ff87` |
| `matrix_free_payloaded_transfer_generation9_global_polynomial_p035_green_wip_20260722` | current generation-9 WIP; D3 and D2 p035 screens GREEN, only a new p030 plan is authorized | `6e76954791cecc100b47f5ea243dec1202cafea5062206ad5e06adf8190be40f` |

The primary TT-GMG target is the compressed row236 fine dump:

- object SHA-256:
  `8f13780171520a82d349e69b3c581e3278bf3b927cef8d9d8a6cc987e85ce676`;
- object bytes: `748603422`;
- decompressed SHA-256:
  `e58c212d593dd39884f9806558f1bfc30e6f2456708e61ed2ca1f32220f13638`;
- decompressed bytes: `2611250272`.

## Fetch and verify

```sh
git clone https://github.com/mickg10/calculi_target_packs.git
cd calculi_target_packs
git checkout 97a2e85b19168e9a081c72252e6eb1e164c07e39
python3 scripts/rebuild_index.py --check
python3 scripts/verify_registry.py
python3 scripts/fetch_pack.py row236_ttgmg_fine_v1 --dest downloads
```

For the current transfer continuation, fetch the SHA-bound Generation-9
p035 terminal-GREEN archive and the unchanged p010 model and payload together:

```sh
python3 scripts/fetch_pack.py \
  matrix_free_payloaded_transfer_generation9_global_polynomial_p035_green_wip_20260722 \
  --dest downloads --extract
```

Its archive is `13,334,004` bytes at SHA-256
`a4f11bdd30c2f1baadc2e258c840691381d7622bf47a8fc9c86d4237e8a0b1f8`.
The normalized tar is `101,345,280` bytes at SHA-256
`f251920627e86266d306ee25543648495717deec9d406652b4e0f5bf48e21a08`
with 877 members. Two independent archive builds are byte-identical. The
archive's exact 843-entry Generation-9 manifest is SHA-256
`a444b327cfc51c3e63d794379cc9bc0a636382d0807c74948bdc1ab87b9d9a08`.
It retains frozen predecessor pointer v4 and excludes its v5 self-pointer.

The p010 model/payload remain exact v3 SHA references
`1515b4d2b1385a5303ffbe65b22094a298f7e156c709ac65b3328894cf63679a`
and `397fb88719d1e041e628bb001f1dd98a251fe7bbb403441aa8b31126609ded8f`;
Generation 9 did not open or evaluate them.

Generation-9 plan/audit and source-freeze/audit SHA-256 are
`ecd74d279cee41e96da65689b96b8f065b99001011b3b30ff04e7d99c823a6fc`,
`5c11a32ba403cc12a657b29d1a18968250af1fe19873590d4fd002debfe51adf`,
`be1de4daae4c0ca94f6822a47c7be633e2664ba4675c381cdec8c14f01cc4d96`,
and `60d2752eca604388ecf2113cfee365034fcb9d39d0bccb6a68230ee379a54dd3`.
Four frozen-order p035 cases completed with exact 15-record streams, 36
fixed solves, 576 PCG/preconditioner applications, and zero swaps.

Full-grid audit SHA-256 is
`d66075a310e7216e20b06a203fc4719930325588aabe8281166bec62a58fb65e`;
the decision is `GREEN_G9_P035_CANDIDATE_FOR_NEW_P030_PLAN`. Degree 3 ranks
first at maximum residual `0.195882489`, `0.260456584` of its frozen
Generation-8 baseline, median wall fraction `2.828290496`, `683,167,673`
persistent bytes, and `1,287,663,616` peak bytes. Degree 2 also passes at
`0.365562827`, `0.534374426`, `1.939851712`, `683,167,673`, and
`1,287,630,848` bytes. Both are p035 calibration results only.

A fresh exact-tag clone passed the registry checks and `3/3` tests. Public
Release downloads of all three objects matched compressed and unpacked
identities and extracted safely. The public package and its independently
reconstructed manifest-only tree each pass `350/350` tests plus T1 and every
auxiliary package check. Verification transcript SHA-256 is
`63fd5d441ee58f5d395e250e8cf186d3661f281c007c850e578f20684ad615a4`.

The only newly granted numerical authority is to write and independently
audit a new response-blind p030 plan. P030 execution, p010, physical forcing
or response, GPU/Metal acceptance, T5, T6-T14, N1/N2/N3, and product
authority remain locked.

For the preceding Generation-8 v11 checkpoint, fetch the terminal-RED
archive and the unchanged p010 model and payload together:

```sh
python3 scripts/fetch_pack.py \
  matrix_free_payloaded_transfer_generation8_p1b_terminal_red_wip_20260722 \
  --dest downloads --extract
```

Its archive SHA-256 is
`699f56a19024df2bec0bc8a43e6267dba2b23a2c7446db7e3d0a85ca94d61e41`;
the compressed p010 model/payload SHAs are
`1515b4d2b1385a5303ffbe65b22094a298f7e156c709ac65b3328894cf63679a`
and `397fb88719d1e041e628bb001f1dd98a251fe7bbb403441aa8b31126609ded8f`.
The p010 objects remain immutable v3 release objects; v11 references rather
than duplicates them. Generation 8 P1B did not open, scan, or evaluate p010.

Selected-host preflight/audit SHA-256 are
`b3800ce62f3bb21d1d65dca1204ca5d9beef14b7089b888410fa5bf56bd1bc52`
and `95040f2affda5a1c46a86f2765f8540c85829d8fb296c2803f7a8be41996ea47`;
the audit passed `31/31` on quiet Linux `research3`. All ten frozen p035
fixed-screen cases were then consumed exactly once in the preregistered
order. All native exits are zero, all ten integrity records are GREEN, and
the run contains exactly 90 fixed solves, 1,440 PCG iterations, 1,440
preconditioner applications, and zero process swaps. Terminal prefix audit
SHA-256 is
`42faca1e660b27d7ce9a892e6c3f825e5e477cfa10120ff117634efe1d80f39d`.

The independent full-grid audit SHA-256 is
`8cb6f52cf584286e1078bafea2ad8fb66a39f452bb283156cc6576f1f3eef46f`;
its exact decision is `RED_P1B_NO_CANDIDATE_P2_LOCKED` and promotion count
is zero. Candidate maximum true residuals are `0.928473`, `0.967945`,
`0.922097`, and `0.897765` against the fixed `0.7` limit. Worst matched-
baseline fractions are `1.47152`, `1.56919`, `1.55049`, and `1.52893`
against the fixed `0.9` limit. All four pass wall-time, persistent-memory,
peak-RSS, and zero-swap gates; the three `h1.20` variants use about
`1.187 GB` persistent / `1.654 GB` peak, and `h0.70` uses `2.460 GB` /
`2.931 GB`. This is a convergence-quality failure, not a RAM or GPU-
throughput failure. More GPU cannot make the same fixed preconditioner pass.

The exact 64-file raw set is preserved by file-manifest/evidence/audit/nested-
archive SHA-256
`1cf5f8572e5bdb93d54578f7342417ad16782dfb71367d641017d1216868230e`,
`0dbd9ee0db112cb5a26686d6dcda50d10acb0a7998e55b56d20f054c24e446b0`,
`68c9b85caff979b68841bc9347616aa7bd88f1098e7b2c66e7a873efb023c7d6`,
and `762682b1cecc759612e9c254ab4851029649d4b19f41824d3319867310cdb1d0`.
No P1B marker or raw record may be removed, changed, reused, or rerun.

A fresh public clone at the exact v11 tag downloaded the release archive and
both reused p010 SHA objects, verified all compressed and unpacked identities,
and safely extracted the package. The extracted tree and its recursive
manifest-only reconstruction each passed `337/337` tests plus native, T1,
package-schema, manifest, input-identity, and no-heavy-work checks. On Linux,
point the verifier to the packaged frozen executables before running it:

```sh
export GENERATION8_BUILD_DIR="$PWD/reference/linux_x86_64_cpu_fp64_gen8"
./scripts/verify_release.sh
```

The archive is `12,923,454` bytes and unpacks to normalized tar SHA-256
`a536e129242d4ffba74c85f2f07d2ef364025a5f757cf898d9f245cf595ce9a3`,
`98,426,880` bytes, with `812` members. Its 781-entry internal manifest is
SHA-256
`90424ff731f29caf9df4e2943c0967293011307176af31ad82addbc68bc03545`.
The archive intentionally retains exact v10 predecessor pointer
`reference/target_pack_registry_v3.json`, SHA-256
`15916b0927f8accbec3fe843bc4e8208dd48e9ce14858b302fe8b158c2d3a98c`,
and contains no v11 self-pointer. After public verification, the live package
added distinct successor pointer `reference/target_pack_registry_v4.json`,
SHA-256
`27903f73e989871106f6719b1f8e7166495c466993d29cbffc1d35d52198e649`,
and its resulting 782-entry live manifest is SHA-256
`78f6e13afae3abf9ef3ed6ef6a633366a35d43730f58a35ce9de484d557ff76b`.
Those live successor bytes are intentionally not self-included in v11.

This pack is transport/evidence, not T5 or product authority. P2, p030,
p010, physical forcing/response, GPU/Metal acceptance, T5, T6-T14, and
N1/N2/N3 remain locked. Any continuation requires a new response-blind
mathematical preregistration and a materially different preconditioner or
bounded block-Krylov design.

The fetcher selects the exact release tag/asset from the pinned manifest and
checks compressed identity plus the declared unpacked identity before safe
extraction. It does not execute a solver or touch Tenstorrent hardware.

Pack storage is transport/provenance only. It does not change the current
TT-GMG score (2/8 performance gates green), close full-CCX correctness, or
grant any vibration-mount product authority.
