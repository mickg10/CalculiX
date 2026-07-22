# CalculiX target-pack registry

Large targets and portable reproduction packs are stored outside this source
fork in the content-addressed repository:

- repository: `https://github.com/mickg10/calculi_target_packs`;
- registry commit:
  `98545757bbf0f0d0b2dcaa9d0c2cfb91e3b63f55`;
- immutable release tag: `packs-v7-20260721`;
- index SHA-256:
  `d1da250383e973c5cae6d1c7660ba2a884de7588e530a27b1201690921f8a747`;
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
| `matrix_free_payloaded_transfer_generation7_execution_repair_red_wip_20260721` | current generation-7 WIP; P1A stopped terminal RED at setup case four before RHS/operator/PCG, no P1B, p010, or target response | `4571cd7ea53c09237c16b8c1602d15b49ba6755562327e07809b38ef959a862c` |

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
git checkout 98545757bbf0f0d0b2dcaa9d0c2cfb91e3b63f55
python3 scripts/rebuild_index.py --check
python3 scripts/verify_registry.py
python3 scripts/fetch_pack.py row236_ttgmg_fine_v1 --dest downloads
```

For the current transfer continuation, fetch the SHA-bound generation-7
source/evidence archive and the unchanged p010 model and payload together:

```sh
python3 scripts/fetch_pack.py \
  matrix_free_payloaded_transfer_generation7_execution_repair_red_wip_20260721 \
  --dest downloads --extract
```

Its archive SHA-256 is
`34e37135598621bd6d721f2ffc401b57c9e0b2c9553501d90a7b3707d2b3aca6`;
the compressed p010 model/payload SHAs are
`1515b4d2b1385a5303ffbe65b22094a298f7e156c709ac65b3328894cf63679a`
and `397fb88719d1e041e628bb001f1dd98a251fe7bbb403441aa8b31126609ded8f`.
The p010 objects remain immutable v3 release objects; v7 references rather
than duplicates them. Generation 7 did not scan or evaluate them. Its frozen
p035 P1A setup/RHS census stopped at case four of ten. The two Generation-2
baselines and structure-only `h0.70/d3/k20` are GREEN; the distributed
`0.372 kg` version is RED because the level-0 mass Galerkin relative value
`2.32119363768037e-12` exceeds the frozen `2e-12` limit. The RED process
returned before RHS allocation, configuration, operator/preconditioner
application, or PCG. Six later P1A cases and all P1B/p030/p010/target work are
absent. Maximum setup RSS was `2,490,826,752` bytes, maximum persistent state
was `1,995,736,395` bytes, and swaps were zero; this is a setup-audit failure,
not a memory or convergence result.

A fresh public clone/download verified every compressed and decompressed
object and passed `238/238` Linux tests plus `238/238` again recursively. The
archive preserves both platform freezes. On Linux, point the verifier to the
packaged frozen executables before running it:

```sh
export GENERATION7_BUILD_DIR="$PWD/reference/linux_x86_64_cpu_fp64_gen7"
./scripts/verify_release.sh
```

The successful public Linux log SHA-256 is
`bf9812f9fdabe96f1cf921d98613fb5d4f7b2ed86de40dcd371b49e29206025b`.
This pack is transport/evidence, not T5 or product authority. Generation-7
P1A rerun/completion, P1B, p030, p010, and target response are forbidden.

The fetcher selects the exact release tag/asset from the pinned manifest and
checks compressed identity plus the declared unpacked identity before safe
extraction. It does not execute a solver or touch Tenstorrent hardware.

Pack storage is transport/provenance only. It does not change the current
TT-GMG score (2/8 performance gates green), close full-CCX correctness, or
grant any vibration-mount product authority.
