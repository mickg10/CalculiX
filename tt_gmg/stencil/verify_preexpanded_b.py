#!/usr/bin/env python3
"""Exhaustively verify shifted resident-vector tiles against a halo artifact.

This checker is intentionally independent of ``build_preexpanded_b.py``.  It
replays the scalar address formula from the qualifying device reader and
compares every 64-bit word in all three preexpanded files.  It never opens a
Tenstorrent device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


CHIPS = 8
CORES_PER_CHIP = 56
GROUPS_PER_CHIP = 198
MAX_GROUPS_PER_CORE = 4
HALO_PLANES = 9
HALO_PLANE_VALUES = 8192
TERMS = 81
TILE_VALUES = 1024
BRICKS_PER_GROUP = 16
WORDS_PER_BRICK = 432 // 4
WORDS_PER_TILE = TILE_VALUES // 4
WORD_OFFSETS = np.asarray(
    (0, 3, 6, 9, 18, 21, 24, 27, 36, 39, 42, 45, 54, 57, 60, 63),
    dtype=np.int64,
)


def _sha256(path: Path, chunk_bytes: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _group_owners(layout: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    partition = layout["partition"]
    expected = {
        "chips": CHIPS,
        "cores_per_chip": CORES_PER_CHIP,
        "groups_per_chip": GROUPS_PER_CHIP,
        "max_groups_per_core": MAX_GROUPS_PER_CORE,
    }
    for key, value in expected.items():
        if int(partition[key]) != value:
            raise ValueError(f"layout {key}={partition[key]} != {value}")
    counts = np.asarray(partition["core_group_counts"], dtype=np.int64)
    starts = np.asarray(partition["core_group_starts"], dtype=np.int64)
    if counts.shape != (CORES_PER_CHIP,) or starts.shape != (CORES_PER_CHIP,):
        raise ValueError("invalid per-core partition arrays")
    owners = np.full(GROUPS_PER_CHIP, -1, dtype=np.int64)
    slots = np.full(GROUPS_PER_CHIP, -1, dtype=np.int64)
    for core in range(CORES_PER_CHIP):
        start = int(starts[core])
        count = int(counts[core])
        owners[start : start + count] = core
        slots[start : start + count] = np.arange(count, dtype=np.int64)
    if np.any(owners < 0) or np.any(slots < 0) or int(counts.sum()) != GROUPS_PER_CHIP:
        raise ValueError("layout partition does not cover all local groups")
    return owners, slots


def _scalar_kernel_word_indices() -> np.ndarray:
    """Return [term, tile-word] indices from the device scalar formula."""

    indices = np.empty((TERMS, WORDS_PER_TILE), dtype=np.int64)
    for term in range(TERMS):
        offset_id = term // 3
        di = offset_id // 9
        dj = (offset_id // 3) % 3
        dk = offset_id % 3
        window = (di * 6 + dj) * 3 + dk
        cursor = 0
        for brick in range(BRICKS_PER_GROUP):
            source = brick * WORDS_PER_BRICK + window
            indices[term, cursor : cursor + len(WORD_OFFSETS)] = (
                source + WORD_OFFSETS
            )
            cursor += len(WORD_OFFSETS)
    return indices


def verify(
    halo_path: Path,
    layout_path: Path,
    manifest_path: Path,
    *,
    hash_files: bool = True,
) -> dict[str, Any]:
    halo_path = halo_path.expanduser().resolve()
    layout_path = layout_path.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "tt_gmg_preexpanded_resident_vector_v1":
        raise ValueError("unexpected preexpanded-vector manifest schema")
    if manifest.get("shape_per_split") != [CHIPS, GROUPS_PER_CHIP, TERMS, TILE_VALUES]:
        raise ValueError("manifest shape does not match the kernel contract")

    expected_halo_bytes = (
        CHIPS
        * CORES_PER_CHIP
        * MAX_GROUPS_PER_CORE
        * HALO_PLANES
        * HALO_PLANE_VALUES
        * 2
    )
    if halo_path.stat().st_size != expected_halo_bytes:
        raise ValueError("halo byte count does not match the kernel contract")
    if hash_files and manifest.get("source_halo_sha256") != _sha256(halo_path):
        raise ValueError("halo SHA-256 does not match the manifest")

    owners, slots = _group_owners(layout)
    halo_u16 = np.memmap(
        halo_path,
        dtype="<u2",
        mode="r",
        shape=(
            CHIPS,
            CORES_PER_CHIP,
            MAX_GROUPS_PER_CORE,
            HALO_PLANES,
            HALO_PLANE_VALUES,
        ),
    )
    halo_words = halo_u16.view("<u8").reshape(
        CHIPS,
        CORES_PER_CHIP,
        MAX_GROUPS_PER_CORE,
        HALO_PLANES,
        HALO_PLANE_VALUES // 4,
    )
    word_indices = _scalar_kernel_word_indices()
    term_components = np.arange(TERMS, dtype=np.int64) % 3
    term_rows = np.arange(TERMS, dtype=np.int64)[:, None]
    expected_split_bytes = CHIPS * GROUPS_PER_CHIP * TERMS * TILE_VALUES * 2

    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, list) or len(manifest_files) != 3:
        raise ValueError("manifest must contain exactly three split records")

    compared_words = 0
    split_reports: list[dict[str, Any]] = []
    for split in range(3):
        record = next(
            (item for item in manifest_files if int(item.get("split", -1)) == split),
            None,
        )
        if record is None:
            raise ValueError(f"manifest is missing split {split}")
        path = Path(record["path"]).expanduser().resolve()
        if path.stat().st_size != expected_split_bytes:
            raise ValueError(f"split {split} has the wrong byte count")
        actual_sha256 = _sha256(path) if hash_files else None
        if hash_files and record.get("sha256") != actual_sha256:
            raise ValueError(f"split {split} SHA-256 does not match the manifest")
        output_u16 = np.memmap(
            path,
            dtype="<u2",
            mode="r",
            shape=(CHIPS, GROUPS_PER_CHIP, TERMS, TILE_VALUES),
        )
        output_words = output_u16.view("<u8").reshape(
            CHIPS, GROUPS_PER_CHIP, TERMS, WORDS_PER_TILE
        )
        for chip in range(CHIPS):
            for local_group in range(GROUPS_PER_CHIP):
                source_planes = halo_words[
                    chip,
                    int(owners[local_group]),
                    int(slots[local_group]),
                    term_components * 3 + split,
                ]
                expected = source_planes[term_rows, word_indices]
                actual = output_words[chip, local_group]
                if not np.array_equal(actual, expected):
                    mismatch = np.argwhere(actual != expected)[0]
                    term = int(mismatch[0])
                    tile_word = int(mismatch[1])
                    raise ValueError(
                        "preexpanded word mismatch "
                        f"split={split} chip={chip} group={local_group} "
                        f"term={term} tile_word={tile_word} "
                        f"actual=0x{int(actual[term, tile_word]):016x} "
                        f"expected=0x{int(expected[term, tile_word]):016x}"
                    )
                compared_words += actual.size
        split_reports.append(
            {
                "split": split,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": actual_sha256,
                "bitwise_match": True,
            }
        )

    return {
        "schema": "tt_gmg_preexpanded_resident_vector_verification_v1",
        "pass": True,
        "device_opened": False,
        "source_halo": str(halo_path),
        "source_layout": str(layout_path),
        "manifest": str(manifest_path),
        "scalar_kernel_address_formula_replayed": True,
        "compared_u64_words": compared_words,
        "compared_bytes": compared_words * 8,
        "splits": split_reports,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--halo", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--write-report", type=Path)
    parser.add_argument("--skip-hashes", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = verify(
        args.halo,
        args.layout,
        args.manifest,
        hash_files=not args.skip_hashes,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.write_report is not None:
        report_path = args.write_report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
