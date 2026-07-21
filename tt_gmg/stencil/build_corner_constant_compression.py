#!/usr/bin/env python3
"""Build and prove the exact Row236 corner-coefficient compression.

The brick device layout stores each A split as
``[chip, group, term, output_component, lane]`` with
``term = offset * 3 + input_component``.  For the eight 3-D corner offsets,
the nonzero lanes of every 3x3 block share one presence mask and every matrix
entry is one of only three constant BF16x3 triples.  This tool proves those
properties over the complete device artifact before emitting the lossless
``[chip, group, corner, split, coefficient_type, lane]`` stream consumed by
the corner-constant kernels.

This is representation compression, not an approximation: the final check
reconstructs every corner tile and requires bitwise identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


CHIPS = 8
GROUPS_PER_CHIP = 198
TERMS = 81
COMPONENTS = 3
LANES = 1024
CORNER_OFFSETS = (0, 2, 6, 8, 18, 20, 24, 26)


def representative_entry(offset: int, coefficient_type: int) -> tuple[int, int]:
    input_component = 0
    output_component = 0
    if coefficient_type != 1:
        signs = (offset // 9 == 2, (offset // 3) % 3 == 2, offset % 3 == 2)
        seek_equal = coefficient_type == 2
        if (signs[0] == signs[1]) == seek_equal:
            output_component = 1
        elif (signs[0] == signs[2]) == seek_equal:
            output_component = 2
        elif seek_equal:
            input_component = 1
            output_component = 2
    return input_component, output_component


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def open_split(path: Path) -> np.memmap:
    expected_elements = CHIPS * GROUPS_PER_CHIP * TERMS * COMPONENTS * LANES
    expected_bytes = expected_elements * np.dtype("<u2").itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"{path}: expected {expected_bytes} bytes, got {path.stat().st_size}"
        )
    return np.memmap(
        path,
        dtype="<u2",
        mode="r",
        shape=(CHIPS, GROUPS_PER_CHIP, TERMS, COMPONENTS, LANES),
    )


def discover(
    splits: list[np.memmap],
) -> tuple[list[tuple[int, int, int]], np.ndarray, list[np.ndarray]]:
    entry_triples: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    presence_by_corner: list[np.ndarray] = []

    for corner_index, offset in enumerate(CORNER_OFFSETS):
        reference_presence: np.ndarray | None = None
        for input_component in range(COMPONENTS):
            term = offset * COMPONENTS + input_component
            for output_component in range(COMPONENTS):
                arrays = [
                    np.asarray(split[:, :, term, output_component, :])
                    for split in splits
                ]
                presence = np.logical_or.reduce([array != 0 for array in arrays])
                if reference_presence is None:
                    reference_presence = presence.copy()
                elif not np.array_equal(reference_presence, presence):
                    mismatch = int(np.count_nonzero(reference_presence != presence))
                    raise ValueError(
                        f"corner {offset} entry ({input_component},{output_component}) "
                        f"has {mismatch} presence-mask mismatches"
                    )

                triple: list[int] = []
                for split_index, array in enumerate(arrays):
                    if np.any((array != 0) != presence):
                        raise ValueError(
                            f"corner {offset} entry ({input_component},{output_component}) "
                            f"split {split_index} does not share the block-presence mask"
                        )
                    values = np.unique(array[presence])
                    if values.size != 1:
                        raise ValueError(
                            f"corner {offset} entry ({input_component},{output_component}) "
                            f"split {split_index} has {values.size} nonzero BF16 values"
                        )
                    triple.append(int(values[0]))
                entry_triples[(corner_index, input_component, output_component)] = (
                    triple[0],
                    triple[1],
                    triple[2],
                )
        if reference_presence is None:
            raise AssertionError("unreachable empty corner")
        presence_by_corner.append(reference_presence)

    triples = sorted(set(entry_triples.values()))
    if len(triples) != 3:
        raise ValueError(f"expected exactly three corner BF16x3 triples, got {triples}")
    triple_to_type = {triple: index for index, triple in enumerate(triples)}
    type_map = np.empty((len(CORNER_OFFSETS), COMPONENTS, COMPONENTS), dtype=np.uint8)
    for key, triple in entry_triples.items():
        type_map[key] = triple_to_type[triple]
    for corner_index, offset in enumerate(CORNER_OFFSETS):
        for coefficient_type in range(len(triples)):
            if not np.any(type_map[corner_index] == coefficient_type):
                continue
            input_component, output_component = representative_entry(
                offset, coefficient_type
            )
            if (
                int(type_map[corner_index, input_component, output_component])
                != coefficient_type
            ):
                raise ValueError(
                    f"corner {offset} representative for type {coefficient_type} "
                    f"selects type {type_map[corner_index, input_component, output_component]}"
                )
    return triples, type_map, presence_by_corner


def emit_and_verify(
    output_path: Path,
    splits: list[np.memmap],
    triples: list[tuple[int, int, int]],
    type_map: np.ndarray,
    presence_by_corner: list[np.ndarray],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    compressed = np.memmap(
        temporary,
        dtype="<u2",
        mode="w+",
        shape=(
            CHIPS,
            GROUPS_PER_CHIP,
            len(CORNER_OFFSETS),
            COMPONENTS,
            len(triples),
            LANES,
        ),
    )

    for corner_index, presence in enumerate(presence_by_corner):
        for split_index in range(COMPONENTS):
            for type_index, triple in enumerate(triples):
                compressed[:, :, corner_index, split_index, type_index, :] = np.where(
                    presence,
                    np.uint16(triple[split_index]),
                    np.uint16(0),
                )
    compressed.flush()

    mismatches = 0
    for corner_index, offset in enumerate(CORNER_OFFSETS):
        for input_component in range(COMPONENTS):
            term = offset * COMPONENTS + input_component
            for output_component in range(COMPONENTS):
                type_index = int(type_map[corner_index, input_component, output_component])
                for split_index, source in enumerate(splits):
                    expected = np.asarray(source[:, :, term, output_component, :])
                    actual = np.asarray(
                        compressed[:, :, corner_index, split_index, type_index, :]
                    )
                    mismatches += int(np.count_nonzero(expected != actual))
    if mismatches:
        del compressed
        temporary.unlink(missing_ok=True)
        raise ValueError(f"bitwise corner reconstruction has {mismatches} mismatches")

    del compressed
    os.replace(temporary, output_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=Path,
        help="directory containing operator_a0.bf16, operator_a1.bf16, operator_a2.bf16",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="compressed BF16 output (omit for an audit-only run)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="JSON manifest path (defaults next to --output)",
    )
    args = parser.parse_args()

    paths = [args.root / f"operator_a{split}.bf16" for split in range(3)]
    splits = [open_split(path) for path in paths]
    triples, type_map, presence_by_corner = discover(splits)

    if args.output is not None:
        emit_and_verify(
            args.output,
            splits,
            triples,
            type_map,
            presence_by_corner,
        )

    manifest_path = args.manifest
    if manifest_path is None and args.output is not None:
        manifest_path = args.output.with_suffix(args.output.suffix + ".json")

    manifest: dict[str, Any] = {
        "schema": "tt_gmg_corner_constant_compression_v1",
        "lossless": True,
        "source_layout": "chip,group,term,output_component,lane",
        "compressed_layout": "chip,group,corner,split,coefficient_type,lane",
        "chips": CHIPS,
        "groups_per_chip": GROUPS_PER_CHIP,
        "corner_offsets": list(CORNER_OFFSETS),
        "coefficient_triples_bf16_bits": [list(triple) for triple in triples],
        "type_map_corner_input_output": type_map.tolist(),
        "representative_entry_corner_type": [
            [list(representative_entry(offset, coefficient_type)) for coefficient_type in range(3)]
            for offset in CORNER_OFFSETS
        ],
        "representative_mapping_verified": True,
        "source_sha256": {path.name: sha256_file(path) for path in paths},
        "full_corner_reconstruction_bitwise_mismatches": 0,
    }
    if args.output is not None:
        manifest.update(
            {
                "artifact": str(args.output),
                "artifact_bytes": args.output.stat().st_size,
                "artifact_sha256": sha256_file(args.output),
                "compressed_tiles_per_chip": (
                    GROUPS_PER_CHIP
                    * len(CORNER_OFFSETS)
                    * COMPONENTS
                    * len(triples)
                ),
                "original_corner_tiles_per_chip": (
                    GROUPS_PER_CHIP
                    * len(CORNER_OFFSETS)
                    * COMPONENTS
                    * COMPONENTS
                    * COMPONENTS
                ),
            }
        )
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
