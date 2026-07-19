# CalculiX target-pack registry

Large targets and portable reproduction packs are stored outside this source
fork in the content-addressed repository:

- repository: `https://github.com/mickg10/calculi_target_packs`;
- registry commit:
  `1fb01bf0d908ae9b555ce118a4db9fb9b06e3fa7`;
- immutable release tag: `packs-v1-20260719`;
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
| `payloaded_modal_v18b_exact_decks_20260716` | frozen exact anchors | `b958ec3447c9f1d45c38f3ee5b20a42932c597e7bcf051c514a9fe734fb4ef84` |
| `matrix_free_payloaded_transfer_wip_20260719` | work in progress | `9ebd9cfe01fa9ba57f1ffcf97e91f46f57f229802c5110f4a59935cea45bdad0` |

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
git checkout 1fb01bf0d908ae9b555ce118a4db9fb9b06e3fa7
python3 scripts/rebuild_index.py --check
python3 scripts/verify_registry.py
python3 scripts/fetch_pack.py row236_ttgmg_fine_v1 --dest downloads
```

The fetcher selects the exact release tag/asset from the pinned manifest and
checks byte count plus SHA-256 after download. It does not execute a solver or
touch Tenstorrent hardware.

Pack storage is transport/provenance only. It does not change the current
TT-GMG score (2/8 performance gates green), close full-CCX correctness, or
grant any vibration-mount product authority.
