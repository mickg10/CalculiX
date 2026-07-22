#!/usr/bin/env python3
"""Build the exact shifted resident-vector tiles consumed by the brick SpMV.

The qualifying brick kernel keeps a 6x6x6 halo for every 4x4x4 brick and
materializes each shifted 4x4x4 input tile with scalar RISC/PACK copies during
the timed apply.  This tool performs the same bitwise permutation ahead of
time and stores the three BF16 resident-vector splits as direct TT tile pages.

The resulting representation is intentionally redundant: PCG vectors would
be maintained in shifted form so their elementwise updates remain regular and
the fine SpMV no longer contains the scalar halo-to-tile materializer.  This
builder is host-side evidence and never opens a Tenstorrent device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
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
SPLITS = 3
BRICKS_PER_GROUP = 16
HALO_WORDS_PER_BRICK = 432 // 4

# Exact uint64 source-word cadence in materialize_b_tile().  Each word carries
# the four BF16 k values of one aligned halo window.
MATERIALIZE_WORD_OFFSETS = np.asarray(
    [0, 3, 6, 9, 18, 21, 24, 27, 36, 39, 42, 45, 54, 57, 60, 63],
    dtype=np.int64,
)


def _sha256(path: Path, chunk_bytes: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _load_partition(layout_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    partition = layout["partition"]
    if (
        int(partition["chips"]) != CHIPS
        or int(partition["cores_per_chip"]) != CORES_PER_CHIP
        or int(partition["groups_per_chip"]) != GROUPS_PER_CHIP
        or int(partition["max_groups_per_core"]) != MAX_GROUPS_PER_CORE
    ):
        raise ValueError("device layout does not match the row236 brick-v1 constants")
    counts = np.asarray(partition["core_group_counts"], dtype=np.int64)
    starts = np.asarray(partition["core_group_starts"], dtype=np.int64)
    if counts.shape != (CORES_PER_CHIP,) or starts.shape != (CORES_PER_CHIP,):
        raise ValueError("invalid core group partition")
    owner = np.empty(GROUPS_PER_CHIP, dtype=np.int64)
    slot = np.empty(GROUPS_PER_CHIP, dtype=np.int64)
    for core, (start, count) in enumerate(zip(starts, counts, strict=True)):
        owner[start : start + count] = core
        slot[start : start + count] = np.arange(count, dtype=np.int64)
    if int(counts.sum()) != GROUPS_PER_CHIP:
        raise ValueError("core group partition does not cover every local group")
    return owner, slot, layout


def _term_word_indices() -> np.ndarray:
    result = np.empty((TERMS, BRICKS_PER_GROUP, 16), dtype=np.int64)
    brick_base = (
        np.arange(BRICKS_PER_GROUP, dtype=np.int64)[:, None]
        * HALO_WORDS_PER_BRICK
    )
    for term in range(TERMS):
        offset = term // 3
        di = offset // 9
        dj = (offset // 3) % 3
        dk = offset % 3
        window = (di * 6 + dj) * 3 + dk
        result[term] = brick_base + window + MATERIALIZE_WORD_OFFSETS[None, :]
    return result


def build(
    halo_path: Path,
    layout_path: Path,
    output_dir: Path,
    *,
    vector_tag: str,
    force: bool = False,
    hash_files: bool = True,
) -> dict[str, Any]:
    started = time.time()
    halo_path = halo_path.expanduser().resolve()
    layout_path = layout_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_halo_bytes = (
        CHIPS
        * CORES_PER_CHIP
        * MAX_GROUPS_PER_CORE
        * HALO_PLANES
        * HALO_PLANE_VALUES
        * np.dtype("<u2").itemsize
    )
    if halo_path.stat().st_size != expected_halo_bytes:
        raise ValueError(
            f"unexpected halo size {halo_path.stat().st_size}; "
            f"expected {expected_halo_bytes}"
        )
    owner, slot, layout = _load_partition(layout_path)
    halo = np.memmap(
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
    halo_words = halo.view("<u8").reshape(
        CHIPS,
        CORES_PER_CHIP,
        MAX_GROUPS_PER_CORE,
        HALO_PLANES,
        HALO_PLANE_VALUES // 4,
    )
    word_indices = _term_word_indices()
    term_components = np.arange(TERMS, dtype=np.int64) % 3
    output_shape = (CHIPS, GROUPS_PER_CHIP, TERMS, TILE_VALUES)
    records: list[dict[str, Any]] = []

    for split in range(SPLITS):
        final = output_dir / f"shifted_b{split}_{vector_tag}.bf16"
        partial = final.with_name(final.name + ".partial")
        if final.exists() and not force:
            raise FileExistsError(f"{final} exists; pass --force to replace it")
        if partial.exists():
            partial.unlink()
        destination = np.memmap(partial, dtype="<u2", mode="w+", shape=output_shape)
        destination[:] = 0
        for chip in range(CHIPS):
            for local_group in range(GROUPS_PER_CHIP):
                planes = halo_words[
                    chip,
                    int(owner[local_group]),
                    int(slot[local_group]),
                    term_components * SPLITS + split,
                ]
                gathered_words = np.take_along_axis(planes, word_indices.reshape(TERMS, -1), axis=1)
                destination[chip, local_group] = (
                    np.ascontiguousarray(gathered_words)
                    .view("<u2")
                    .reshape(TERMS, TILE_VALUES)
                )
        destination.flush()
        del destination
        os.replace(partial, final)
        records.append(
            {
                "split": split,
                "path": str(final),
                "bytes": final.stat().st_size,
                "sha256": _sha256(final) if hash_files else None,
            }
        )

    manifest = {
        "schema": "tt_gmg_preexpanded_resident_vector_v1",
        "created_unix_s": time.time(),
        "source_halo": str(halo_path),
        "source_halo_sha256": _sha256(halo_path) if hash_files else None,
        "source_layout": str(layout_path),
        "source_layout_schema": layout.get("schema"),
        "vector_tag": vector_tag,
        "representation": (
            "three BF16 splits in exact [chip][local_group][offset*3+input]"
            "[TT-tile-lane] order"
        ),
        "shape_per_split": list(output_shape),
        "files": records,
        "bitwise_source_transform": True,
        "timed_kernel_scalar_materialization_required": False,
        "build_seconds": time.time() - started,
    }
    manifest_path = output_dir / f"preexpanded_b_{vector_tag}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--halo", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vector-tag", default="random_seed351")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-hashes", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build(
        args.halo,
        args.layout,
        args.output_dir,
        vector_tag=args.vector_tag,
        force=args.force,
        hash_files=not args.skip_hashes,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
