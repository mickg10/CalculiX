#!/usr/bin/env python3
"""Build and exhaustively verify the Row236 canonical-stencil packed layout.

The layout is deliberately a permutation, not a new discretization:

* q1e-7-canonical base-stencil nodes are packed into homogeneous tiles;
* every other active node is packed into a dense exact-BF16x3 fallback;
* shifted resident-vector pages and references follow the same permutation;
* inverse maps preserve the G5 component/output-scope contract.

No command in this module opens a Tenstorrent device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from brick_format import BrickOperator, _bf16_values, split_bf16x3


TERMS = 81
OUTPUT_COMPONENTS = 3
LANES = 1024
A_SPLITS = 3
Q1E7 = 1.0e-7
GROUP_PADDING = 0
GROUP_BASE = 1
GROUP_FALLBACK = 2


def _sha256(path: Path, chunk_bytes: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prepare_output(path: Path, force: bool) -> Path:
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; pass --force to replace it")
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        partial.unlink()
    return partial


def _write_array(path: Path, array: np.ndarray, force: bool) -> dict[str, Any]:
    partial = _prepare_output(path, force)
    np.asarray(array).tofile(partial)
    os.replace(partial, path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "dtype": np.asarray(array).dtype.str,
        "shape": list(np.asarray(array).shape),
    }


def _distribute_tiles(tile_count: int, chips: int) -> tuple[int, ...]:
    base, remainder = divmod(tile_count, chips)
    return tuple(base + (chip < remainder) for chip in range(chips))


def build_packed_mapping(
    active_mask: np.ndarray,
    canonical_base_mask: np.ndarray,
    *,
    chips: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return packed->source, source->packed, group classes, and its plan."""

    active = np.asarray(active_mask, dtype=bool).reshape(-1)
    base_mask = np.asarray(canonical_base_mask, dtype=np.uint8).reshape(-1)
    if active.size != base_mask.size:
        raise ValueError("active and canonical masks differ in length")
    if np.any((base_mask != 0) & (base_mask != 1)):
        raise ValueError("canonical base mask must contain only 0/1")
    if np.any((base_mask == 1) & ~active):
        raise ValueError("canonical base mask selects inactive lanes")

    base_positions = np.flatnonzero(base_mask == 1).astype(np.int32)
    fallback_positions = np.flatnonzero(active & (base_mask == 0)).astype(np.int32)
    base_tiles = math.ceil(base_positions.size / LANES)
    fallback_tiles = math.ceil(fallback_positions.size / LANES)
    base_groups = _distribute_tiles(base_tiles, chips)
    fallback_groups = _distribute_tiles(fallback_tiles, chips)
    total_groups = tuple(
        base_groups[chip] + fallback_groups[chip] for chip in range(chips)
    )
    groups_per_chip = max(total_groups)

    packed_to_source = np.full(
        (chips, groups_per_chip, LANES),
        -1,
        dtype="<i4",
    )
    group_class = np.full(
        (chips, groups_per_chip),
        GROUP_PADDING,
        dtype=np.uint8,
    )

    base_cursor = 0
    fallback_cursor = 0
    for chip in range(chips):
        base_capacity = base_groups[chip] * LANES
        base_count = min(base_capacity, base_positions.size - base_cursor)
        if base_count:
            packed_to_source[chip, : base_groups[chip]].reshape(-1)[:base_count] = (
                base_positions[base_cursor : base_cursor + base_count]
            )
        group_class[chip, : base_groups[chip]] = GROUP_BASE
        base_cursor += base_count

        fallback_capacity = fallback_groups[chip] * LANES
        fallback_count = min(
            fallback_capacity,
            fallback_positions.size - fallback_cursor,
        )
        fallback_start = base_groups[chip]
        fallback_stop = fallback_start + fallback_groups[chip]
        if fallback_count:
            packed_to_source[chip, fallback_start:fallback_stop].reshape(-1)[
                :fallback_count
            ] = fallback_positions[
                fallback_cursor : fallback_cursor + fallback_count
            ]
        group_class[chip, fallback_start:fallback_stop] = GROUP_FALLBACK
        fallback_cursor += fallback_count

    if base_cursor != base_positions.size:
        raise AssertionError("base-node packing did not consume every node")
    if fallback_cursor != fallback_positions.size:
        raise AssertionError("fallback packing did not consume every node")

    source_to_packed = np.full(active.size, -1, dtype="<i4")
    packed_flat = packed_to_source.reshape(-1)
    valid = packed_flat >= 0
    source_to_packed[packed_flat[valid]] = np.flatnonzero(valid).astype(np.int32)
    if np.any(source_to_packed[active] < 0):
        raise AssertionError("active source lane is absent from inverse map")
    if np.any(source_to_packed[~active] >= 0):
        raise AssertionError("inactive source lane appears in inverse map")

    plan = {
        "schema": "tt_gmg_canonical_packed_plan_v1",
        "chips": chips,
        "lanes_per_group": LANES,
        "source_lanes": int(active.size),
        "active_lanes": int(active.sum()),
        "base_lanes": int(base_positions.size),
        "fallback_lanes": int(fallback_positions.size),
        "base_tiles": base_tiles,
        "fallback_tiles": fallback_tiles,
        "packed_tiles": base_tiles + fallback_tiles,
        "groups_per_chip": groups_per_chip,
        "base_groups_per_chip": list(base_groups),
        "fallback_groups_per_chip": list(fallback_groups),
        "total_groups_per_chip": list(total_groups),
        "packed_padding_lanes": int(packed_flat.size - active.sum()),
    }
    return packed_to_source, source_to_packed, group_class, plan


def _active_mask_from_operator(
    brick_path: Path,
    *,
    chips: int,
    source_groups_per_chip: int,
) -> tuple[np.ndarray, BrickOperator]:
    operator = BrickOperator(brick_path)
    active = np.zeros(chips * source_groups_per_chip * LANES, dtype=bool)
    source_slots = np.asarray(
        operator.section("slot_to_node")[: operator.ngroup * 16],
        dtype=np.int64,
    ).reshape(-1)
    if source_slots.size != operator.ngroup * LANES:
        raise ValueError("brick slot-to-node section has unexpected size")
    active[: source_slots.size] = source_slots >= 0
    if int(active.sum()) != operator.nb:
        raise ValueError("active mask does not reproduce operator node count")
    return active, operator


def _signature_term_output(signature_q: np.ndarray) -> np.ndarray:
    signature = np.asarray(signature_q, dtype="<i4").reshape(27, 3, 3)
    return np.ascontiguousarray(signature.transpose(0, 2, 1).reshape(TERMS, 3))


def _build_palette(
    signature_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    term_output = _signature_term_output(signature_q)
    palette_q = np.unique(term_output).astype("<i4")
    index = {int(value): slot for slot, value in enumerate(palette_q)}
    schedule = np.asarray(
        [[index[int(value)] for value in row] for row in term_output],
        dtype=np.uint8,
    )
    values = (palette_q.astype(np.float64) * Q1E7).astype(np.float32)
    splits = split_bf16x3(values)
    tiles = tuple(
        np.repeat(np.asarray(bits, dtype="<u2")[:, None], LANES, axis=1)
        for bits in splits
    )
    return palette_q, schedule, tiles


def _gather_coefficients(
    source: np.memmap,
    packed_to_source: np.ndarray,
    base_groups: Iterable[int],
    fallback_groups: Iterable[int],
    destination: np.memmap,
) -> None:
    terms = np.arange(TERMS, dtype=np.int64)
    outputs = np.arange(OUTPUT_COMPONENTS, dtype=np.int64)
    for chip, (base_count, fallback_count) in enumerate(
        zip(base_groups, fallback_groups)
    ):
        for fallback_group in range(fallback_count):
            packed_group = base_count + fallback_group
            positions = packed_to_source[chip, packed_group]
            valid = positions >= 0
            if not np.any(valid):
                continue
            source_group = positions[valid] // LANES
            source_lane = positions[valid] % LANES
            gathered = source[
                source_group[:, None, None],
                terms[None, :, None],
                outputs[None, None, :],
                source_lane[:, None, None],
            ]
            block = np.zeros(
                (TERMS, OUTPUT_COMPONENTS, LANES),
                dtype="<u2",
            )
            block[:, :, valid] = gathered.transpose(1, 2, 0)
            destination[chip, fallback_group] = block


def build_layout(
    brick_path: Path,
    device_dir: Path,
    canonical_dir: Path,
    output_dir: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    started = time.time()
    device_manifest_path = device_dir / "device_layout.json"
    device_manifest = _load_json(device_manifest_path)
    partition = device_manifest["partition"]
    chips = int(partition["chips"])
    source_groups_per_chip = int(partition["groups_per_chip"])
    active, operator = _active_mask_from_operator(
        brick_path,
        chips=chips,
        source_groups_per_chip=source_groups_per_chip,
    )
    canonical_metadata_path = canonical_dir / "canonical_base_metadata.json"
    canonical_metadata = _load_json(canonical_metadata_path)
    canonical_mask_path = canonical_dir / "canonical_base_lane_mask.u8"
    canonical_mask = np.fromfile(canonical_mask_path, dtype=np.uint8)
    signature_path = canonical_dir / "canonical_base_signature_q1e7.i32"
    signature_q = np.fromfile(signature_path, dtype="<i4")
    if signature_q.size != 27 * 3 * 3:
        raise ValueError("canonical signature must contain 243 int32 scalars")
    packed_to_source, source_to_packed, group_class, plan = build_packed_mapping(
        active,
        canonical_mask,
        chips=chips,
    )
    if plan["active_lanes"] != int(canonical_metadata["active_lanes"]):
        raise ValueError("canonical metadata active-lane count mismatch")
    if plan["base_lanes"] != int(canonical_metadata["base_lanes"]):
        raise ValueError("canonical metadata base-lane count mismatch")

    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    files["packed_to_source"] = _write_array(
        output_dir / "packed_to_source_position.i32",
        packed_to_source,
        force,
    )
    files["source_to_packed"] = _write_array(
        output_dir / "source_to_packed_position.i32",
        source_to_packed,
        force,
    )
    files["group_class"] = _write_array(
        output_dir / "packed_group_class.u8",
        group_class,
        force,
    )

    palette_q, palette_schedule, palette_tiles = _build_palette(signature_q)
    files["palette_q1e7"] = _write_array(
        output_dir / "canonical_base_palette_q1e7.i32",
        palette_q,
        force,
    )
    files["palette_schedule"] = _write_array(
        output_dir / "canonical_base_palette_index.u8",
        palette_schedule,
        force,
    )
    for split, tiles in enumerate(palette_tiles):
        files[f"palette_a{split}"] = _write_array(
            output_dir / f"canonical_base_palette_a{split}.bf16",
            tiles,
            force,
        )

    fallback_groups = tuple(plan["fallback_groups_per_chip"])
    max_fallback_groups = max(fallback_groups)
    fallback_shape = (
        chips,
        max_fallback_groups,
        TERMS,
        OUTPUT_COMPONENTS,
        LANES,
    )
    source_shape = (
        chips * source_groups_per_chip,
        TERMS,
        OUTPUT_COMPONENTS,
        LANES,
    )
    for split in range(A_SPLITS):
        source_path = device_dir / f"operator_a{split}.bf16"
        source = np.memmap(source_path, dtype="<u2", mode="r", shape=source_shape)
        final = output_dir / f"fallback_operator_a{split}.bf16"
        partial = _prepare_output(final, force)
        destination = np.memmap(
            partial,
            dtype="<u2",
            mode="w+",
            shape=fallback_shape,
        )
        destination[:] = 0
        _gather_coefficients(
            source,
            packed_to_source,
            plan["base_groups_per_chip"],
            fallback_groups,
            destination,
        )
        destination.flush()
        del destination, source
        os.replace(partial, final)
        files[f"fallback_a{split}"] = {
            "path": str(final.resolve()),
            "bytes": final.stat().st_size,
            "sha256": _sha256(final),
            "dtype": "<u2",
            "shape": list(fallback_shape),
        }

    manifest = {
        "schema": "tt_gmg_canonical_packed_layout_v1",
        "created_unix_s": time.time(),
        "host_only": True,
        "brick_operator": {
            "path": str(operator.path),
            "sha256": _sha256(operator.path),
            "active_nodes": operator.nb,
            "source_groups": operator.ngroup,
        },
        "source_device_layout": {
            "path": str(device_manifest_path.resolve()),
            "sha256": _sha256(device_manifest_path),
            "groups_per_chip": source_groups_per_chip,
        },
        "canonical_export": {
            "metadata_path": str(canonical_metadata_path.resolve()),
            "metadata_sha256": _sha256(canonical_metadata_path),
            "mask_path": str(canonical_mask_path.resolve()),
            "mask_sha256": _sha256(canonical_mask_path),
            "signature_path": str(signature_path.resolve()),
            "signature_sha256": _sha256(signature_path),
            "quantum": Q1E7,
        },
        "plan": plan,
        "palette": {
            "size_including_zero": int(palette_q.size),
            "zero_index": int(np.flatnonzero(palette_q == 0)[0]),
            "nonzero_scalar_planes": int(np.count_nonzero(signature_q)),
            "term_output_schedule_shape": [TERMS, OUTPUT_COMPONENTS],
        },
        "fallback_operator_layout": {
            "shape": list(fallback_shape),
            "order": "[chip][fallback_group][term][out_component][lane]",
        },
        "files": files,
        "build_seconds": time.time() - started,
    }
    manifest_path = output_dir / "canonical_packed_layout.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _gather_shifted_vector(
    source: np.memmap,
    packed_to_source: np.ndarray,
    destination: np.memmap,
) -> None:
    terms = np.arange(TERMS, dtype=np.int64)
    for chip in range(packed_to_source.shape[0]):
        for group in range(packed_to_source.shape[1]):
            positions = packed_to_source[chip, group]
            valid = positions >= 0
            if not np.any(valid):
                continue
            source_group = positions[valid] // LANES
            source_lane = positions[valid] % LANES
            gathered = source[
                source_group[:, None],
                terms[None, :],
                source_lane[:, None],
            ]
            block = np.zeros((TERMS, LANES), dtype="<u2")
            block[:, valid] = gathered.T
            destination[chip, group] = block


def _gather_reference(
    source: np.memmap,
    packed_to_source: np.ndarray,
    destination: np.memmap,
) -> None:
    components = np.arange(OUTPUT_COMPONENTS, dtype=np.int64)
    for chip in range(packed_to_source.shape[0]):
        for group in range(packed_to_source.shape[1]):
            positions = packed_to_source[chip, group]
            valid = positions >= 0
            if not np.any(valid):
                continue
            source_group = positions[valid] // LANES
            source_lane = positions[valid] % LANES
            gathered = source[
                source_group[:, None],
                components[None, :],
                source_lane[:, None],
            ]
            block = np.zeros((OUTPUT_COMPONENTS, LANES), dtype="<f4")
            block[:, valid] = gathered.T
            destination[chip, group] = block


def pack_vector(
    device_dir: Path,
    layout_manifest_path: Path,
    output_dir: Path,
    *,
    vector_tag: str,
    force: bool = False,
) -> dict[str, Any]:
    started = time.time()
    layout = _load_json(layout_manifest_path)
    plan = layout["plan"]
    chips = int(plan["chips"])
    source_groups_per_chip = int(
        layout["source_device_layout"]["groups_per_chip"]
    )
    packed_groups_per_chip = int(plan["groups_per_chip"])
    packed_map_file = layout["files"]["packed_to_source"]
    packed_to_source = np.memmap(
        packed_map_file["path"],
        dtype="<i4",
        mode="r",
        shape=tuple(packed_map_file["shape"]),
    )
    source_b_shape = (
        chips * source_groups_per_chip,
        TERMS,
        LANES,
    )
    packed_b_shape = (
        chips,
        packed_groups_per_chip,
        TERMS,
        LANES,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    for split in range(A_SPLITS):
        source_path = device_dir / f"shifted_b{split}_{vector_tag}.bf16"
        source = np.memmap(source_path, dtype="<u2", mode="r", shape=source_b_shape)
        final = output_dir / f"packed_shifted_b{split}_{vector_tag}.bf16"
        partial = _prepare_output(final, force)
        destination = np.memmap(
            partial,
            dtype="<u2",
            mode="w+",
            shape=packed_b_shape,
        )
        destination[:] = 0
        _gather_shifted_vector(source, packed_to_source, destination)
        destination.flush()
        del destination, source
        os.replace(partial, final)
        files[f"packed_b{split}"] = {
            "path": str(final.resolve()),
            "bytes": final.stat().st_size,
            "sha256": _sha256(final),
            "dtype": "<u2",
            "shape": list(packed_b_shape),
        }

    source_reference_path = device_dir / f"reference_{vector_tag}.fp32"
    source_reference = np.memmap(
        source_reference_path,
        dtype="<f4",
        mode="r",
        shape=(
            chips * source_groups_per_chip,
            OUTPUT_COMPONENTS,
            LANES,
        ),
    )
    packed_reference_shape = (
        chips,
        packed_groups_per_chip,
        OUTPUT_COMPONENTS,
        LANES,
    )
    final_reference = output_dir / f"packed_reference_{vector_tag}.fp32"
    partial_reference = _prepare_output(final_reference, force)
    destination_reference = np.memmap(
        partial_reference,
        dtype="<f4",
        mode="w+",
        shape=packed_reference_shape,
    )
    destination_reference[:] = 0
    _gather_reference(
        source_reference,
        packed_to_source,
        destination_reference,
    )
    destination_reference.flush()
    del destination_reference, source_reference, packed_to_source
    os.replace(partial_reference, final_reference)
    files["packed_reference"] = {
        "path": str(final_reference.resolve()),
        "bytes": final_reference.stat().st_size,
        "sha256": _sha256(final_reference),
        "dtype": "<f4",
        "shape": list(packed_reference_shape),
    }

    report = {
        "schema": "tt_gmg_canonical_packed_vector_v1",
        "created_unix_s": time.time(),
        "host_only": True,
        "vector_tag": vector_tag,
        "layout_manifest": {
            "path": str(layout_manifest_path.resolve()),
            "sha256": _sha256(layout_manifest_path),
        },
        "source": {
            "shifted_b": [
                {
                    "path": str((device_dir / f"shifted_b{split}_{vector_tag}.bf16").resolve()),
                    "sha256": _sha256(
                        device_dir / f"shifted_b{split}_{vector_tag}.bf16"
                    ),
                }
                for split in range(A_SPLITS)
            ],
            "reference": {
                "path": str(source_reference_path.resolve()),
                "sha256": _sha256(source_reference_path),
            },
        },
        "files": files,
        "pack_seconds": time.time() - started,
    }
    report_path = output_dir / f"canonical_packed_vector_{vector_tag}.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _round_half_away_from_zero(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(
        values >= 0.0,
        np.floor(values + 0.5),
        np.ceil(values - 0.5),
    ).astype("<i4")


def verify_layout(
    device_dir: Path,
    layout_manifest_path: Path,
    canonical_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Exhaustively replay classification, permutation, palette, and fallback A."""

    started = time.time()
    layout = _load_json(layout_manifest_path)
    plan = layout["plan"]
    chips = int(plan["chips"])
    source_groups_per_chip = int(
        layout["source_device_layout"]["groups_per_chip"]
    )
    source_groups = chips * source_groups_per_chip
    packed_file = layout["files"]["packed_to_source"]
    inverse_file = layout["files"]["source_to_packed"]
    class_file = layout["files"]["group_class"]
    packed = np.memmap(
        packed_file["path"],
        dtype="<i4",
        mode="r",
        shape=tuple(packed_file["shape"]),
    )
    inverse = np.memmap(
        inverse_file["path"],
        dtype="<i4",
        mode="r",
        shape=tuple(inverse_file["shape"]),
    )
    group_class = np.memmap(
        class_file["path"],
        dtype=np.uint8,
        mode="r",
        shape=tuple(class_file["shape"]),
    )
    canonical_mask = np.fromfile(
        canonical_dir / "canonical_base_lane_mask.u8",
        dtype=np.uint8,
    )
    signature_q = np.fromfile(
        canonical_dir / "canonical_base_signature_q1e7.i32",
        dtype="<i4",
    )
    expected_term_output = _signature_term_output(signature_q)

    packed_flat = np.asarray(packed).reshape(-1)
    valid = packed_flat >= 0
    unique_sources = np.unique(packed_flat[valid])
    permutation_ok = (
        unique_sources.size == int(plan["active_lanes"])
        and np.array_equal(
            np.asarray(inverse)[packed_flat[valid]],
            np.flatnonzero(valid).astype(np.int32),
        )
    )
    base_group_ok = True
    fallback_group_ok = True
    for chip in range(chips):
        for group in range(packed.shape[1]):
            positions = np.asarray(packed[chip, group])
            active_positions = positions[positions >= 0]
            role = int(group_class[chip, group])
            if role == GROUP_BASE:
                base_group_ok &= bool(
                    active_positions.size == 0
                    or np.all(canonical_mask[active_positions] == 1)
                )
            elif role == GROUP_FALLBACK:
                fallback_group_ok &= bool(
                    active_positions.size == 0
                    or np.all(canonical_mask[active_positions] == 0)
                )
            elif role != GROUP_PADDING:
                raise ValueError("unknown packed group role")

    source_shape = (
        source_groups,
        TERMS,
        OUTPUT_COMPONENTS,
        LANES,
    )
    source_a = [
        np.memmap(
            device_dir / f"operator_a{split}.bf16",
            dtype="<u2",
            mode="r",
            shape=source_shape,
        )
        for split in range(A_SPLITS)
    ]
    recomputed_base_mask = np.ones(source_groups * LANES, dtype=bool)
    group_chunk = 8
    for group0 in range(0, source_groups, group_chunk):
        group1 = min(group0 + group_chunk, source_groups)
        matches = np.ones((group1 - group0, LANES), dtype=bool)
        for term0 in range(0, TERMS, 9):
            term1 = min(term0 + 9, TERMS)
            values = sum(
                _bf16_values(split[group0:group1, term0:term1]).astype(
                    np.float64
                )
                for split in source_a
            )
            quantized = _round_half_away_from_zero(values / Q1E7)
            expected = expected_term_output[term0:term1][None, :, :, None]
            matches &= np.all(quantized == expected, axis=(1, 2))
        recomputed_base_mask[group0 * LANES : group1 * LANES] = matches.reshape(-1)
    classification_mismatch_count = int(
        np.count_nonzero(recomputed_base_mask != (canonical_mask == 1))
    )

    fallback_shape = tuple(layout["fallback_operator_layout"]["shape"])
    fallback_bit_mismatches = [0, 0, 0]
    terms = np.arange(TERMS, dtype=np.int64)
    outputs = np.arange(OUTPUT_COMPONENTS, dtype=np.int64)
    for split in range(A_SPLITS):
        fallback = np.memmap(
            layout["files"][f"fallback_a{split}"]["path"],
            dtype="<u2",
            mode="r",
            shape=fallback_shape,
        )
        for chip in range(chips):
            base_count = int(plan["base_groups_per_chip"][chip])
            fallback_count = int(plan["fallback_groups_per_chip"][chip])
            for fallback_group in range(fallback_shape[1]):
                if fallback_group >= fallback_count:
                    fallback_bit_mismatches[split] += int(
                        np.count_nonzero(fallback[chip, fallback_group])
                    )
                    continue
                positions = np.asarray(
                    packed[chip, base_count + fallback_group]
                )
                valid_lanes = positions >= 0
                expected = np.zeros(
                    (TERMS, OUTPUT_COMPONENTS, LANES),
                    dtype="<u2",
                )
                if np.any(valid_lanes):
                    source_group = positions[valid_lanes] // LANES
                    source_lane = positions[valid_lanes] % LANES
                    gathered = source_a[split][
                        source_group[:, None, None],
                        terms[None, :, None],
                        outputs[None, None, :],
                        source_lane[:, None, None],
                    ]
                    expected[:, :, valid_lanes] = gathered.transpose(1, 2, 0)
                fallback_bit_mismatches[split] += int(
                    np.count_nonzero(fallback[chip, fallback_group] != expected)
                )

    palette_q, schedule, tiles = _build_palette(signature_q)
    palette_mismatches = 0
    palette_mismatches += int(
        np.count_nonzero(
            np.fromfile(
                layout["files"]["palette_q1e7"]["path"],
                dtype="<i4",
            )
            != palette_q
        )
    )
    palette_mismatches += int(
        np.count_nonzero(
            np.fromfile(
                layout["files"]["palette_schedule"]["path"],
                dtype=np.uint8,
            ).reshape(TERMS, OUTPUT_COMPONENTS)
            != schedule
        )
    )
    for split, expected_tiles in enumerate(tiles):
        actual_tiles = np.fromfile(
            layout["files"][f"palette_a{split}"]["path"],
            dtype="<u2",
        ).reshape(expected_tiles.shape)
        palette_mismatches += int(
            np.count_nonzero(actual_tiles != expected_tiles)
        )

    checks = {
        "permutation_bijective_on_active_lanes": bool(permutation_ok),
        "base_groups_contain_only_base_lanes": bool(base_group_ok),
        "fallback_groups_contain_only_nonbase_lanes": bool(fallback_group_ok),
        "independent_q1e7_classification_exact": classification_mismatch_count == 0,
        "fallback_operator_bitwise_exact": sum(fallback_bit_mismatches) == 0,
        "palette_and_schedule_bitwise_exact": palette_mismatches == 0,
    }
    report = {
        "schema": "tt_gmg_canonical_packed_layout_verification_v1",
        "created_unix_s": time.time(),
        "host_only": True,
        "layout_manifest": {
            "path": str(layout_manifest_path.resolve()),
            "sha256": _sha256(layout_manifest_path),
        },
        "checks": checks,
        "passing": all(checks.values()),
        "classification_mismatch_count": classification_mismatch_count,
        "fallback_bit_mismatches_by_split": fallback_bit_mismatches,
        "palette_mismatches": palette_mismatches,
        "verified_active_lanes": int(unique_sources.size),
        "verify_seconds": time.time() - started,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def verify_vector(
    device_dir: Path,
    layout_manifest_path: Path,
    vector_manifest_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Exhaustively compare every packed B/reference word with its source."""

    started = time.time()
    layout = _load_json(layout_manifest_path)
    vector = _load_json(vector_manifest_path)
    plan = layout["plan"]
    chips = int(plan["chips"])
    source_groups_per_chip = int(
        layout["source_device_layout"]["groups_per_chip"]
    )
    packed_file = layout["files"]["packed_to_source"]
    packed = np.memmap(
        packed_file["path"],
        dtype="<i4",
        mode="r",
        shape=tuple(packed_file["shape"]),
    )
    source_b_shape = (
        chips * source_groups_per_chip,
        TERMS,
        LANES,
    )
    packed_b_shape = tuple(vector["files"]["packed_b0"]["shape"])
    terms = np.arange(TERMS, dtype=np.int64)
    b_mismatches = [0, 0, 0]
    for split in range(A_SPLITS):
        source = np.memmap(
            vector["source"]["shifted_b"][split]["path"],
            dtype="<u2",
            mode="r",
            shape=source_b_shape,
        )
        actual = np.memmap(
            vector["files"][f"packed_b{split}"]["path"],
            dtype="<u2",
            mode="r",
            shape=packed_b_shape,
        )
        for chip in range(chips):
            for group in range(packed.shape[1]):
                positions = np.asarray(packed[chip, group])
                valid = positions >= 0
                expected = np.zeros((TERMS, LANES), dtype="<u2")
                if np.any(valid):
                    source_group = positions[valid] // LANES
                    source_lane = positions[valid] % LANES
                    gathered = source[
                        source_group[:, None],
                        terms[None, :],
                        source_lane[:, None],
                    ]
                    expected[:, valid] = gathered.T
                b_mismatches[split] += int(
                    np.count_nonzero(actual[chip, group] != expected)
                )

    source_reference = np.memmap(
        vector["source"]["reference"]["path"],
        dtype="<f4",
        mode="r",
        shape=(
            chips * source_groups_per_chip,
            OUTPUT_COMPONENTS,
            LANES,
        ),
    )
    packed_reference = np.memmap(
        vector["files"]["packed_reference"]["path"],
        dtype="<f4",
        mode="r",
        shape=tuple(vector["files"]["packed_reference"]["shape"]),
    )
    components = np.arange(OUTPUT_COMPONENTS, dtype=np.int64)
    reference_mismatches = 0
    for chip in range(chips):
        for group in range(packed.shape[1]):
            positions = np.asarray(packed[chip, group])
            valid = positions >= 0
            expected = np.zeros((OUTPUT_COMPONENTS, LANES), dtype="<f4")
            if np.any(valid):
                source_group = positions[valid] // LANES
                source_lane = positions[valid] % LANES
                gathered = source_reference[
                    source_group[:, None],
                    components[None, :],
                    source_lane[:, None],
                ]
                expected[:, valid] = gathered.T
            reference_mismatches += int(
                np.count_nonzero(
                    packed_reference[chip, group].view("<u4")
                    != expected.view("<u4")
                )
            )

    checks = {
        "packed_shifted_b_bitwise_exact": sum(b_mismatches) == 0,
        "packed_reference_bitwise_exact": reference_mismatches == 0,
    }
    report = {
        "schema": "tt_gmg_canonical_packed_vector_verification_v1",
        "created_unix_s": time.time(),
        "host_only": True,
        "vector_tag": vector["vector_tag"],
        "checks": checks,
        "passing": all(checks.values()),
        "b_bit_mismatches_by_split": b_mismatches,
        "reference_bit_mismatches": reference_mismatches,
        "verified_b_words": int(np.prod(packed_b_shape)) * A_SPLITS,
        "verified_reference_words": int(np.prod(packed_reference.shape)),
        "verify_seconds": time.time() - started,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    layout = subparsers.add_parser("build-layout")
    layout.add_argument("--brick", type=Path, required=True)
    layout.add_argument("--device-dir", type=Path, required=True)
    layout.add_argument("--canonical-dir", type=Path, required=True)
    layout.add_argument("--output-dir", type=Path, required=True)
    layout.add_argument("--force", action="store_true")

    vector = subparsers.add_parser("pack-vector")
    vector.add_argument("--device-dir", type=Path, required=True)
    vector.add_argument("--layout-manifest", type=Path, required=True)
    vector.add_argument("--output-dir", type=Path, required=True)
    vector.add_argument("--vector-tag", required=True)
    vector.add_argument("--force", action="store_true")

    verify_l = subparsers.add_parser("verify-layout")
    verify_l.add_argument("--device-dir", type=Path, required=True)
    verify_l.add_argument("--layout-manifest", type=Path, required=True)
    verify_l.add_argument("--canonical-dir", type=Path, required=True)
    verify_l.add_argument("--report", type=Path, required=True)

    verify_v = subparsers.add_parser("verify-vector")
    verify_v.add_argument("--device-dir", type=Path, required=True)
    verify_v.add_argument("--layout-manifest", type=Path, required=True)
    verify_v.add_argument("--vector-manifest", type=Path, required=True)
    verify_v.add_argument("--report", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "build-layout":
        result = build_layout(
            args.brick,
            args.device_dir,
            args.canonical_dir,
            args.output_dir,
            force=args.force,
        )
    elif args.command == "pack-vector":
        result = pack_vector(
            args.device_dir,
            args.layout_manifest,
            args.output_dir,
            vector_tag=args.vector_tag,
            force=args.force,
        )
    elif args.command == "verify-layout":
        result = verify_layout(
            args.device_dir,
            args.layout_manifest,
            args.canonical_dir,
            args.report,
        )
    else:
        result = verify_vector(
            args.device_dir,
            args.layout_manifest,
            args.vector_manifest,
            args.report,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
