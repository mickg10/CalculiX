#!/usr/bin/env python3
"""Build the dynamic-vector gather map for the canonical packed Row236 layout.

Run66 proved a fixed, pre-shifted vector apply.  A real PCG cannot reuse those
fixed ``packed_shifted_b*`` files: every operator application has a new search
vector.  This module derives the missing lossless map

    packed output lane + 27-point offset -> packed logical input lane

from the brick metadata and the canonical output permutation.  It also reports
how much of the gather remains on the owning chip and how many logical vector
pages each output tile touches.  Those are the facts needed to choose between a
structured tile gather, a halo exchange, and a replicated-vector design.

The tool is host-only and never opens a Tenstorrent device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from brick_format import BrickOperator, _slot_shift_tables
from canonical_packed_layout import (
    GROUP_BASE,
    GROUP_FALLBACK,
    LANES,
)


OFFSETS = 27
COMPONENTS = 3
TERMS = OFFSETS * COMPONENTS
TERMS_PER_PUBLISH = 3


def component_major_ordered_term(stream_term: int) -> int:
    """Map device stream order to canonical ``offset * 3 + component``.

    The dynamic reader stages one vector component at a time.  Its compute
    partner therefore consumes all 27 offsets for component 0, then 1, then
    2.  Offsets remain reversed inside each component pass so the changing-x
    path preserves the cancellation direction used by the Run66 reader.
    """

    if not 0 <= stream_term < TERMS:
        raise ValueError(f"stream term must be in [0, {TERMS})")
    component = stream_term // OFFSETS
    reverse_offset = OFFSETS - 1 - stream_term % OFFSETS
    return reverse_offset * COMPONENTS + component


def component_major_stream_order() -> np.ndarray:
    """Return the complete device stream-to-canonical term permutation."""

    return np.asarray(
        [component_major_ordered_term(term) for term in range(TERMS)],
        dtype=np.uint8,
    )


def _sha256(path: Path, chunk_bytes: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _percentiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {key: 0.0 for key in ("min", "p50", "p90", "p95", "p99", "max", "mean")}
    return {
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def build_neighbor_map(
    packed_to_source: np.ndarray,
    source_to_packed: np.ndarray,
    brick_nbr: np.ndarray,
    slot_to_node: np.ndarray,
) -> np.ndarray:
    """Return int32 ``[chip, group, offset, lane]`` packed input positions.

    Positions index the flattened canonical logical-vector layout.  ``-1`` is
    an absent/inactive neighbor and therefore contributes zero to the SpMV.
    """

    packed = np.asarray(packed_to_source, dtype=np.int64)
    inverse = np.asarray(source_to_packed, dtype=np.int64).reshape(-1)
    neighbors = np.asarray(brick_nbr, dtype=np.int64)
    slots = np.asarray(slot_to_node, dtype=np.int64)
    if packed.ndim != 3 or packed.shape[2] != LANES:
        raise ValueError("packed_to_source must have shape [chip, group, 1024]")
    if neighbors.ndim != 2 or neighbors.shape[1] != OFFSETS:
        raise ValueError("brick_nbr must have shape [brick, 27]")
    if slots.ndim != 2 or slots.shape[1] != 64:
        raise ValueError("slot_to_node must have shape [brick, 64]")
    if inverse.size < slots.size:
        raise ValueError("source_to_packed does not cover the brick slot domain")

    delta_id, source_slot = _slot_shift_tables(4)
    result = np.full(
        (packed.shape[0], packed.shape[1], OFFSETS, LANES),
        -1,
        dtype="<i4",
    )
    for chip in range(packed.shape[0]):
        for group in range(packed.shape[1]):
            source_position = packed[chip, group]
            active = source_position >= 0
            if not np.any(active):
                continue
            output_brick = np.zeros(LANES, dtype=np.int64)
            output_slot = np.zeros(LANES, dtype=np.int64)
            output_brick[active] = source_position[active] // 64
            output_slot[active] = source_position[active] % 64
            if np.any(output_brick[active] >= neighbors.shape[0]):
                raise ValueError("packed output position names a padded/nonexistent brick")

            for offset in range(OFFSETS):
                destination = result[chip, group, offset]
                active_lanes = np.flatnonzero(active)
                did = delta_id[output_slot[active_lanes], offset]
                input_brick = neighbors[output_brick[active_lanes], did]
                valid = input_brick >= 0
                if not np.any(valid):
                    continue
                valid_lanes = active_lanes[valid]
                input_position = (
                    input_brick[valid] * 64
                    + source_slot[output_slot[valid_lanes], offset].astype(np.int64)
                )
                # A geometrically present brick may still contain an inactive
                # material slot.  Both the slot table and inverse permutation
                # must agree before the reference is admitted.
                input_slot_active = slots.reshape(-1)[input_position] >= 0
                input_position = input_position[input_slot_active]
                valid_lanes = valid_lanes[input_slot_active]
                packed_input = inverse[input_position]
                if np.any(packed_input < 0):
                    raise AssertionError("active input slot is absent from canonical inverse map")
                destination[valid_lanes] = packed_input.astype("<i4")
    return result


def analyze_neighbor_map(
    neighbor_map: np.ndarray,
    group_class: np.ndarray,
) -> dict[str, Any]:
    mapping = np.asarray(neighbor_map)
    roles = np.asarray(group_class, dtype=np.uint8)
    if mapping.ndim != 4 or mapping.shape[2:] != (OFFSETS, LANES):
        raise ValueError("neighbor_map must have shape [chip, group, 27, 1024]")
    if roles.shape != mapping.shape[:2]:
        raise ValueError("group_class shape does not match neighbor_map")
    chips, groups = mapping.shape[:2]
    positions_per_chip = groups * LANES

    valid = mapping >= 0
    input_chip = np.full(mapping.shape, -1, dtype=np.int16)
    input_chip[valid] = (mapping[valid] // positions_per_chip).astype(np.int16)
    output_chip = np.arange(chips, dtype=np.int16)[:, None, None, None]
    same_chip = valid & (input_chip == output_chip)

    page_counts: list[int] = []
    same_chip_page_counts: list[int] = []
    cross_chip_page_counts: list[int] = []
    group_reference_counts: list[int] = []
    group_unique_position_counts: list[int] = []
    group_unique_page_counts: list[int] = []
    group_page_run_counts: list[int] = []
    group_reference_reuse: list[float] = []
    affine_lane_tiles = 0
    valid_tiles = 0
    per_role: dict[str, dict[str, int]] = {
        "base": {"tiles": 0, "valid_references": 0, "same_chip_references": 0},
        "fallback": {"tiles": 0, "valid_references": 0, "same_chip_references": 0},
    }
    for chip in range(chips):
        for group in range(groups):
            role = int(roles[chip, group])
            role_name = "base" if role == GROUP_BASE else "fallback" if role == GROUP_FALLBACK else None
            if role_name is None:
                continue
            group_values = mapping[chip, group]
            group_valid_values = group_values[group_values >= 0]
            if group_valid_values.size:
                unique_positions = np.unique(group_valid_values)
                unique_pages = np.unique(unique_positions // LANES)
                group_reference_counts.append(int(group_valid_values.size))
                group_unique_position_counts.append(int(unique_positions.size))
                group_unique_page_counts.append(int(unique_pages.size))
                group_page_run_counts.append(
                    1 + int(np.count_nonzero(np.diff(unique_pages) != 1))
                )
                group_reference_reuse.append(
                    float(group_valid_values.size / unique_positions.size)
                )
            for offset in range(OFFSETS):
                tile = mapping[chip, group, offset]
                tile_valid = tile >= 0
                if not np.any(tile_valid):
                    continue
                valid_tiles += 1
                per_role[role_name]["tiles"] += 1
                per_role[role_name]["valid_references"] += int(tile_valid.sum())
                tile_input_chip = tile[tile_valid] // positions_per_chip
                same = tile_input_chip == chip
                per_role[role_name]["same_chip_references"] += int(same.sum())
                pages = np.unique(tile[tile_valid] // LANES)
                page_counts.append(int(pages.size))
                page_chips = pages // groups
                same_chip_page_counts.append(int(np.count_nonzero(page_chips == chip)))
                cross_chip_page_counts.append(int(np.count_nonzero(page_chips != chip)))

                # A single-page affine lane map can be served by one DMA page
                # plus a fixed cyclic/identity lane transform.  Record the
                # strict identity/constant-delta case separately as the cheap
                # floor; more general local permutations remain possible.
                lanes = np.flatnonzero(tile_valid)
                if pages.size == 1:
                    source_lanes = tile[tile_valid] % LANES
                    delta = (source_lanes - lanes) % LANES
                    if np.all(delta == delta[0]):
                        affine_lane_tiles += 1

    for role_report in per_role.values():
        refs = role_report["valid_references"]
        role_report["same_chip_reference_fraction"] = (
            role_report["same_chip_references"] / refs if refs else 0.0
        )

    cross_hist = np.bincount(
        (input_chip[valid].astype(np.int64) - np.broadcast_to(output_chip, mapping.shape)[valid] + chips),
        minlength=2 * chips + 1,
    )
    per_chip_remote_unique_positions = []
    per_chip_local_unique_positions = []
    for chip in range(chips):
        chip_values = mapping[chip]
        chip_valid = chip_values >= 0
        chip_input = input_chip[chip]
        remote = chip_values[chip_valid & (chip_input != chip)]
        local = chip_values[chip_valid & (chip_input == chip)]
        per_chip_remote_unique_positions.append(int(np.unique(remote).size))
        per_chip_local_unique_positions.append(int(np.unique(local).size))

    active_lanes = int(np.count_nonzero(mapping[:, :, OFFSETS // 2, :] >= 0))
    remote_halo_positions = int(sum(per_chip_remote_unique_positions))
    return {
        "schema": "tt_gmg_canonical_pcg_gather_analysis_v1",
        "chips": chips,
        "groups_per_chip": groups,
        "logical_vector_lanes": int(chips * groups * LANES),
        "logical_vector_active_lanes": active_lanes,
        "logical_vector_padding_lanes": int(chips * groups * LANES - active_lanes),
        "logical_vector_fp32_bytes": int(chips * groups * LANES * 3 * 4),
        "logical_vector_bf16x3_bytes": int(chips * groups * LANES * 3 * 3 * 2),
        "neighbor_map_bytes": int(np.asarray(neighbor_map).nbytes),
        "valid_neighbor_node_references": int(valid.sum()),
        "absent_neighbor_node_references": int(valid.size - valid.sum()),
        "same_chip_neighbor_references": int(same_chip.sum()),
        "cross_chip_neighbor_references": int(valid.sum() - same_chip.sum()),
        "same_chip_neighbor_fraction": float(same_chip.sum() / max(valid.sum(), 1)),
        "cross_chip_delta_histogram": {
            str(index - chips): int(count)
            for index, count in enumerate(cross_hist)
            if count
        },
        "valid_gather_tiles": valid_tiles,
        "single_page_affine_lane_tiles": affine_lane_tiles,
        "single_page_affine_lane_tile_fraction": float(affine_lane_tiles / max(valid_tiles, 1)),
        "input_pages_per_gather_tile": _percentiles(np.asarray(page_counts)),
        "same_chip_input_pages_per_gather_tile": _percentiles(np.asarray(same_chip_page_counts)),
        "cross_chip_input_pages_per_gather_tile": _percentiles(np.asarray(cross_chip_page_counts)),
        "valid_references_per_output_group": _percentiles(np.asarray(group_reference_counts)),
        "unique_input_positions_per_output_group": _percentiles(
            np.asarray(group_unique_position_counts)
        ),
        "unique_input_pages_per_output_group": _percentiles(
            np.asarray(group_unique_page_counts)
        ),
        "contiguous_input_page_runs_per_output_group": _percentiles(
            np.asarray(group_page_run_counts)
        ),
        "reference_reuse_per_output_group": _percentiles(
            np.asarray(group_reference_reuse)
        ),
        "per_chip_remote_unique_input_positions": per_chip_remote_unique_positions,
        "per_chip_local_unique_input_positions": per_chip_local_unique_positions,
        "remote_halo_unique_input_positions_total": remote_halo_positions,
        "remote_halo_fp32_bytes_per_apply": remote_halo_positions * 3 * 4,
        "remote_halo_bf16x3_bytes_per_apply": remote_halo_positions * 3 * 3 * 2,
        "per_role": per_role,
    }


def verify_neighbor_map_against_preshifted(
    neighbor_map: np.ndarray,
    vector_manifest_path: Path,
) -> dict[str, Any]:
    """Bitwise-check the map against a previously packed fixed vector.

    The center-offset terms in each ``packed_shifted_b*`` file are the packed
    logical vector itself.  Gathering those center values through the derived
    map must reproduce every one of the 81 pre-shifted term planes, including
    exact zeroes for absent neighbors and padding lanes.
    """

    started = time.time()
    mapping = np.asarray(neighbor_map)
    if mapping.ndim != 4 or mapping.shape[2:] != (OFFSETS, LANES):
        raise ValueError("neighbor_map must have shape [chip, group, 27, 1024]")
    vector = json.loads(vector_manifest_path.read_text(encoding="utf-8"))
    shape = (mapping.shape[0], mapping.shape[1], OFFSETS * 3, LANES)
    center_offset = OFFSETS // 2
    mismatch_counts: list[int] = []
    first_mismatches: list[dict[str, int] | None] = []
    verified_words = 0
    for split in range(3):
        record = vector["files"][f"packed_b{split}"]
        if tuple(record["shape"]) != shape:
            raise ValueError(f"packed_b{split} shape does not match neighbor map")
        bits = np.memmap(record["path"], dtype="<u2", mode="r", shape=shape)
        logical = [
            np.asarray(bits[:, :, center_offset * 3 + component, :]).reshape(-1)
            for component in range(3)
        ]
        split_mismatches = 0
        first: dict[str, int] | None = None
        for chip in range(mapping.shape[0]):
            for group in range(mapping.shape[1]):
                indices = mapping[chip, group]
                valid = indices >= 0
                expected = np.zeros((OFFSETS, 3, LANES), dtype="<u2")
                for component in range(3):
                    expected[:, component, :][valid] = logical[component][indices[valid]]
                expected_terms = expected.reshape(OFFSETS * 3, LANES)
                actual = np.asarray(bits[chip, group])
                mismatch = actual != expected_terms
                count = int(np.count_nonzero(mismatch))
                split_mismatches += count
                verified_words += int(actual.size)
                if count and first is None:
                    term, lane = np.argwhere(mismatch)[0]
                    first = {
                        "chip": chip,
                        "group": group,
                        "term": int(term),
                        "lane": int(lane),
                        "actual_u16": int(actual[term, lane]),
                        "expected_u16": int(expected_terms[term, lane]),
                    }
        mismatch_counts.append(split_mismatches)
        first_mismatches.append(first)
        del bits, logical

    return {
        "schema": "tt_gmg_canonical_pcg_gather_verification_v1",
        "vector_manifest": {
            "path": str(vector_manifest_path.resolve()),
            "sha256": _sha256(vector_manifest_path),
            "vector_tag": vector.get("vector_tag"),
        },
        "verified_bf16_words": verified_words,
        "bit_mismatches_by_split": mismatch_counts,
        "first_mismatch_by_split": first_mismatches,
        "passing": sum(mismatch_counts) == 0,
        "verify_seconds": time.time() - started,
    }


def _write_array(path: Path, array: np.ndarray, *, force: bool) -> dict[str, Any]:
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; pass --force to replace it")
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        partial.unlink()
    np.asarray(array).tofile(partial)
    os.replace(partial, path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "dtype": np.asarray(array).dtype.str,
        "shape": list(np.asarray(array).shape),
    }


def build_compact_neighbor_plan(
    neighbor_map: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Factor the raw int32 map into per-group values and uint16 references.

    Each output group needs only about a few thousand distinct input nodes.
    The runtime plan stores those local/halo addresses once and replaces every
    32-bit term-lane position with a 16-bit index.  Remote values are staged in
    a per-destination-chip halo immediately after that chip's local vector.
    """

    mapping = np.asarray(neighbor_map)
    if mapping.ndim != 4 or mapping.shape[2:] != (OFFSETS, LANES):
        raise ValueError("neighbor_map must have shape [chip, group, 27, 1024]")
    chips, groups = mapping.shape[:2]
    positions_per_chip = groups * LANES
    sentinel = np.iinfo(np.uint16).max

    halo_offsets = np.zeros(chips + 1, dtype="<u4")
    halo_parts: list[np.ndarray] = []
    for chip in range(chips):
        values = mapping[chip]
        valid = values >= 0
        remote = values[valid & (values // positions_per_chip != chip)]
        unique_remote = np.unique(remote).astype("<u4", copy=False)
        halo_parts.append(unique_remote)
        halo_offsets[chip + 1] = halo_offsets[chip] + unique_remote.size
    halo_positions = (
        np.concatenate(halo_parts).astype("<u4", copy=False)
        if halo_parts
        else np.empty(0, dtype="<u4")
    )

    group_offsets = np.zeros(chips * groups + 1, dtype="<u4")
    address_parts: list[np.ndarray] = []
    local_index = np.full(mapping.shape, sentinel, dtype="<u2")
    group_page_offsets = np.zeros(chips * groups + 1, dtype="<u4")
    page_parts: list[np.ndarray] = []
    page_lane_index = np.full(mapping.shape, sentinel, dtype="<u2")
    device_page_table = np.full((chips, groups, 64), 255, dtype=np.uint8)
    # Empty/padding groups still need a valid table record.  Page entries use
    # 255 as their sentinel, while the dedicated count byte must be zero.
    device_page_table[:, :, 63] = 0
    max_unique = 0
    max_pages = 0
    for chip in range(chips):
        chip_halo = halo_parts[chip]
        for group in range(groups):
            flat_group = chip * groups + group
            values = mapping[chip, group]
            valid = values >= 0
            if np.any(valid):
                unique, inverse = np.unique(values[valid], return_inverse=True)
                if unique.size >= sentinel:
                    raise ValueError("per-group unique input count exceeds uint16 plan")
                max_unique = max(max_unique, int(unique.size))
                input_chip = unique // positions_per_chip
                addresses = np.empty(unique.size, dtype="<u4")
                local = input_chip == chip
                addresses[local] = (unique[local] % positions_per_chip).astype("<u4")
                if np.any(~local):
                    halo_index = np.searchsorted(chip_halo, unique[~local])
                    if np.any(chip_halo[halo_index] != unique[~local]):
                        raise AssertionError("group remote input is absent from chip halo")
                    addresses[~local] = (
                        positions_per_chip + halo_index
                    ).astype("<u4")
                address_parts.append(addresses)
                local_index[chip, group][valid] = inverse.astype("<u2")
                group_offsets[flat_group + 1] = (
                    group_offsets[flat_group] + unique.size
                )
                pages = np.unique(addresses // LANES).astype("<u4", copy=False)
                if pages.size * LANES >= sentinel:
                    raise ValueError("per-group page staging exceeds uint16 plan")
                max_pages = max(max_pages, int(pages.size))
                if pages.size > 62:
                    raise ValueError(
                        "device page table reserves only 62 page entries"
                    )
                if pages.size and int(np.max(pages)) >= 255:
                    raise ValueError("device page index exceeds uint8 contract")
                device_page_table[chip, group, : pages.size] = pages.astype(
                    np.uint8
                )
                device_page_table[chip, group, 63] = np.uint8(pages.size)
                page_parts.append(pages)
                unique_page_slot = np.searchsorted(pages, addresses // LANES)
                unique_staging_index = (
                    unique_page_slot * LANES + addresses % LANES
                ).astype("<u2")
                page_lane_index[chip, group][valid] = unique_staging_index[
                    inverse
                ]
                group_page_offsets[flat_group + 1] = (
                    group_page_offsets[flat_group] + pages.size
                )
            else:
                group_offsets[flat_group + 1] = group_offsets[flat_group]
                group_page_offsets[flat_group + 1] = group_page_offsets[
                    flat_group
                ]
    group_addresses = (
        np.concatenate(address_parts).astype("<u4", copy=False)
        if address_parts
        else np.empty(0, dtype="<u4")
    )
    group_pages = (
        np.concatenate(page_parts).astype("<u4", copy=False)
        if page_parts
        else np.empty(0, dtype="<u4")
    )

    mismatches = 0
    invalid_index_mismatches = 0
    page_mismatches = 0
    invalid_page_index_mismatches = 0
    device_page_table_mismatches = 0
    for chip in range(chips):
        chip_halo = halo_parts[chip]
        for group in range(groups):
            flat_group = chip * groups + group
            start = int(group_offsets[flat_group])
            stop = int(group_offsets[flat_group + 1])
            addresses = group_addresses[start:stop]
            global_positions = np.empty(addresses.size, dtype="<i4")
            local = addresses < positions_per_chip
            global_positions[local] = (
                chip * positions_per_chip + addresses[local]
            ).astype("<i4")
            if np.any(~local):
                global_positions[~local] = chip_halo[
                    addresses[~local] - positions_per_chip
                ].astype("<i4")
            expected = mapping[chip, group]
            indices = local_index[chip, group]
            valid = expected >= 0
            invalid_index_mismatches += int(np.count_nonzero(indices[~valid] != sentinel))
            if np.any(valid):
                mismatches += int(
                    np.count_nonzero(global_positions[indices[valid]] != expected[valid])
                )

            page_start = int(group_page_offsets[flat_group])
            page_stop = int(group_page_offsets[flat_group + 1])
            pages = group_pages[page_start:page_stop]
            table_count = int(device_page_table[chip, group, 63])
            table_pages = device_page_table[chip, group, :table_count].astype(
                "<u4"
            )
            device_page_table_mismatches += int(table_count != pages.size)
            if table_pages.shape == pages.shape:
                device_page_table_mismatches += int(
                    np.count_nonzero(table_pages != pages)
                )
            staging = page_lane_index[chip, group]
            invalid_page_index_mismatches += int(
                np.count_nonzero(staging[~valid] != sentinel)
            )
            if np.any(valid):
                local_address = (
                    pages[staging[valid] // LANES] * LANES
                    + staging[valid] % LANES
                )
                reconstructed = np.empty(local_address.size, dtype="<i4")
                is_local = local_address < positions_per_chip
                reconstructed[is_local] = (
                    chip * positions_per_chip + local_address[is_local]
                ).astype("<i4")
                if np.any(~is_local):
                    reconstructed[~is_local] = chip_halo[
                        local_address[~is_local] - positions_per_chip
                    ].astype("<i4")
                page_mismatches += int(
                    np.count_nonzero(reconstructed != expected[valid])
                )

    arrays = {
        "group_unique_offsets": group_offsets,
        "group_unique_local_address": group_addresses,
        "neighbor_group_local_index": local_index,
        "group_page_offsets": group_page_offsets,
        "group_vector_page_local_index": group_pages,
        "neighbor_group_page_lane": page_lane_index,
        "device_group_vector_page_table": device_page_table,
        "halo_offsets": halo_offsets,
        "halo_packed_position": halo_positions,
    }
    common_bytes = int(halo_offsets.nbytes + halo_positions.nbytes)
    unique_plan_bytes = int(
        group_offsets.nbytes
        + group_addresses.nbytes
        + local_index.nbytes
        + common_bytes
    )
    page_plan_bytes = int(
        group_page_offsets.nbytes
        + group_pages.nbytes
        + page_lane_index.nbytes
        + common_bytes
    )
    device_page_plan_bytes = int(
        device_page_table.nbytes + page_lane_index.nbytes + common_bytes
    )
    stream_order = component_major_stream_order()
    stream_batches = stream_order.reshape(-1, TERMS_PER_PUBLISH)
    batch_components = stream_batches % COMPONENTS
    batch_offsets = (stream_batches // COMPONENTS).astype(np.int16)
    report = {
        "schema": "tt_gmg_canonical_pcg_compact_gather_plan_v1",
        "chips": chips,
        "groups_per_chip": groups,
        "positions_per_chip": positions_per_chip,
        "neighbor_index_sentinel": int(sentinel),
        "group_unique_input_positions_total": int(group_addresses.size),
        "max_group_unique_input_positions": max_unique,
        "halo_unique_input_positions_total": int(halo_positions.size),
        "compact_plan_bytes": unique_plan_bytes,
        "page_gather_plan_bytes": page_plan_bytes,
        "device_page_gather_plan_bytes": device_page_plan_bytes,
        "raw_neighbor_map_bytes": int(mapping.nbytes),
        "compact_to_raw_byte_ratio": float(unique_plan_bytes / mapping.nbytes),
        "page_gather_to_raw_byte_ratio": float(page_plan_bytes / mapping.nbytes),
        "reconstruction_mismatches": mismatches,
        "invalid_index_mismatches": invalid_index_mismatches,
        "page_reconstruction_mismatches": page_mismatches,
        "invalid_page_index_mismatches": invalid_page_index_mismatches,
        "device_page_table_mismatches": device_page_table_mismatches,
        "device_page_table_width_bytes": 64,
        "device_page_table_entry_sentinel": 255,
        "device_page_table_count_byte": 63,
        "device_stream_order": {
            "runtime_order_mode": 3,
            "layout": "component-major, reverse offset within component",
            "terms": TERMS,
            "terms_per_publish": TERMS_PER_PUBLISH,
            "stream_to_canonical_term": stream_order.tolist(),
            "is_permutation": bool(
                np.array_equal(np.sort(stream_order), np.arange(TERMS))
            ),
            "each_publish_batch_single_component": bool(
                np.all(batch_components == batch_components[:, :1])
            ),
            "each_publish_batch_descending_offsets": bool(
                np.all(np.diff(batch_offsets, axis=1) == -1)
            ),
        },
        "group_vector_pages_total_per_apply": int(group_pages.size),
        "max_group_vector_pages": max_pages,
        "group_vector_page_bf16x3_bytes_per_apply": int(
            group_pages.size * LANES * 3 * 3 * 2
        ),
        "passing": bool(
            mismatches == 0
            and invalid_index_mismatches == 0
            and page_mismatches == 0
            and invalid_page_index_mismatches == 0
            and device_page_table_mismatches == 0
            and np.array_equal(np.sort(stream_order), np.arange(TERMS))
            and np.all(batch_components == batch_components[:, :1])
            and np.all(np.diff(batch_offsets, axis=1) == -1)
        ),
    }
    return arrays, report


def write_compact_neighbor_plan(
    neighbor_map: np.ndarray,
    output_dir: Path,
    *,
    force: bool,
) -> dict[str, Any]:
    arrays, report = build_compact_neighbor_plan(neighbor_map)
    filenames = {
        "group_unique_offsets": "pcg_group_unique_offsets.u32",
        "group_unique_local_address": "pcg_group_unique_local_address.u32",
        "neighbor_group_local_index": "pcg_neighbor_group_local_index.u16",
        "group_page_offsets": "pcg_group_page_offsets.u32",
        "group_vector_page_local_index": "pcg_group_vector_page_local_index.u32",
        "neighbor_group_page_lane": "pcg_neighbor_group_page_lane.u16",
        "device_group_vector_page_table": "pcg_device_group_vector_page_table.u8",
        "halo_offsets": "pcg_halo_offsets.u32",
        "halo_packed_position": "pcg_halo_packed_position.u32",
    }
    report["files"] = {
        name: _write_array(output_dir / filenames[name], array, force=force)
        for name, array in arrays.items()
    }
    return report


def _logical_vector_from_preshifted(
    packed_bits: np.ndarray,
) -> np.ndarray:
    """Extract ``[component][canonical_position]`` from center-offset tiles."""

    bits = np.asarray(packed_bits)
    if bits.ndim != 4 or bits.shape[2:] != (OFFSETS * 3, LANES):
        raise ValueError(
            "packed shifted vector must have shape [chip, group, 81, 1024]"
        )
    center_term = (OFFSETS // 2) * 3
    return np.stack(
        [
            np.asarray(bits[:, :, center_term + component, :]).reshape(-1)
            for component in range(3)
        ]
    ).astype("<u2", copy=False)


def build_page_reader_vector_split(
    plan_arrays: dict[str, np.ndarray],
    logical_vector: np.ndarray,
    *,
    chips: int,
    groups_per_chip: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build one BF16 split in the standalone page-reader device layout.

    Each chip shard is ``[component][page][lane]``.  Its first
    ``groups_per_chip`` pages are the chip's canonical vector slice.  The
    destination-specific compact halo follows immediately, and the tail of
    the final page plus shorter chip shards are deterministically zero padded.
    """

    logical = np.asarray(logical_vector, dtype="<u2")
    positions_per_chip = groups_per_chip * LANES
    if logical.shape != (3, chips * positions_per_chip):
        raise ValueError("logical vector does not match chip/group dimensions")
    halo_offsets = np.asarray(plan_arrays["halo_offsets"], dtype="<u4")
    halo_positions = np.asarray(
        plan_arrays["halo_packed_position"], dtype="<u4"
    )
    if halo_offsets.shape != (chips + 1,):
        raise ValueError("halo offsets do not match chip count")
    if int(halo_offsets[0]) != 0 or int(halo_offsets[-1]) != halo_positions.size:
        raise ValueError("halo offsets do not span the halo position array")
    if np.any(np.diff(halo_offsets.astype(np.int64)) < 0):
        raise ValueError("halo offsets must be monotone")
    if halo_positions.size and int(np.max(halo_positions)) >= logical.shape[1]:
        raise ValueError("halo position is outside the canonical vector")

    halo_counts = np.diff(halo_offsets.astype(np.int64))
    combined_lanes = positions_per_chip + halo_counts
    pages_per_chip = (combined_lanes + LANES - 1) // LANES
    max_pages = int(np.max(pages_per_chip, initial=groups_per_chip))
    shards = np.zeros((chips, 3, max_pages, LANES), dtype="<u2")
    for chip in range(chips):
        destination = shards[chip].reshape(3, -1)
        local_start = chip * positions_per_chip
        local_stop = local_start + positions_per_chip
        destination[:, :positions_per_chip] = logical[:, local_start:local_stop]
        halo_start = int(halo_offsets[chip])
        halo_stop = int(halo_offsets[chip + 1])
        if halo_stop > halo_start:
            destination[
                :, positions_per_chip : positions_per_chip + halo_stop - halo_start
            ] = logical[:, halo_positions[halo_start:halo_stop]]

    return shards, {
        "schema": "tt_gmg_canonical_pcg_page_reader_vector_split_v1",
        "layout": "[chip][component][combined_local_then_halo_page][lane]",
        "chips": chips,
        "components": 3,
        "groups_per_chip": groups_per_chip,
        "positions_per_chip": positions_per_chip,
        "halo_lanes_per_chip": [int(value) for value in halo_counts],
        "combined_lanes_per_chip": [int(value) for value in combined_lanes],
        "combined_pages_per_chip": [int(value) for value in pages_per_chip],
        "padded_pages_per_chip": max_pages,
        "padded_lanes_per_chip": max_pages * LANES,
        "bytes": int(shards.nbytes),
    }


def verify_page_reader_vector_split(
    plan_arrays: dict[str, np.ndarray],
    page_vector: np.ndarray,
    packed_bits: np.ndarray,
) -> dict[str, Any]:
    """Replay page DMA plus local lane selection against fixed shifted B."""

    pages = np.asarray(page_vector, dtype="<u2")
    expected_bits = np.asarray(packed_bits)
    if expected_bits.ndim != 4 or expected_bits.shape[2:] != (
        OFFSETS * 3,
        LANES,
    ):
        raise ValueError("packed shifted vector has an unexpected shape")
    chips, groups = expected_bits.shape[:2]
    if pages.ndim != 4 or pages.shape[:2] != (chips, 3) or pages.shape[3] != LANES:
        raise ValueError("page vector must have shape [chip, 3, page, 1024]")

    device_page_table = np.asarray(
        plan_arrays["device_group_vector_page_table"], dtype=np.uint8
    )
    lane_index = np.asarray(
        plan_arrays["neighbor_group_page_lane"], dtype="<u2"
    )
    if device_page_table.shape != (chips, groups, 64):
        raise ValueError("device page table does not match vector dimensions")
    if lane_index.shape != (chips, groups, OFFSETS, LANES):
        raise ValueError("page/lane index does not match vector dimensions")
    sentinel = np.iinfo(np.uint16).max
    mismatches = 0
    verified_words = 0
    first: dict[str, int] | None = None
    for chip in range(chips):
        for group in range(groups):
            page_count = int(device_page_table[chip, group, 63])
            if page_count > 62:
                raise ValueError("device page table count exceeds its contract")
            selected_pages = device_page_table[
                chip, group, :page_count
            ].astype("<u4")
            if selected_pages.size and int(np.max(selected_pages)) >= pages.shape[2]:
                raise ValueError("page plan references beyond the padded vector shard")
            staged = np.take(pages[chip], selected_pages, axis=1).reshape(3, -1)
            indices = lane_index[chip, group]
            valid = indices != sentinel
            if np.any(valid) and int(np.max(indices[valid])) >= staged.shape[1]:
                raise ValueError("page/lane plan references beyond staged pages")
            reconstructed = np.zeros((OFFSETS, 3, LANES), dtype="<u2")
            for component in range(3):
                reconstructed[:, component, :][valid] = staged[component, indices[valid]]
            actual = expected_bits[chip, group].reshape(OFFSETS, 3, LANES)
            mismatch = actual != reconstructed
            count = int(np.count_nonzero(mismatch))
            mismatches += count
            verified_words += int(actual.size)
            if count and first is None:
                offset, component, lane = np.argwhere(mismatch)[0]
                first = {
                    "chip": chip,
                    "group": group,
                    "offset": int(offset),
                    "component": int(component),
                    "lane": int(lane),
                    "actual_u16": int(actual[offset, component, lane]),
                    "reconstructed_u16": int(
                        reconstructed[offset, component, lane]
                    ),
                }
    return {
        "schema": "tt_gmg_canonical_pcg_page_reader_replay_split_v1",
        "verified_bf16_words": verified_words,
        "bit_mismatches": mismatches,
        "first_mismatch": first,
        "passing": mismatches == 0,
    }


def write_page_reader_vector_bundle(
    plan_arrays: dict[str, np.ndarray],
    vector_manifest_path: Path,
    output_dir: Path,
    *,
    force: bool,
) -> dict[str, Any]:
    """Persist and exhaustively replay the standalone changing-x input bundle."""

    started = time.time()
    vector = json.loads(vector_manifest_path.read_text(encoding="utf-8"))
    shape = tuple(vector["files"]["packed_b0"]["shape"])
    if len(shape) != 4 or shape[2:] != (OFFSETS * 3, LANES):
        raise ValueError("vector manifest packed-B shape is not canonical")
    chips, groups = shape[:2]
    output_dir.mkdir(parents=True, exist_ok=True)
    file_records: dict[str, Any] = {}
    split_reports: list[dict[str, Any]] = []
    layout_report: dict[str, Any] | None = None
    verified_words = 0
    for split in range(3):
        record = vector["files"][f"packed_b{split}"]
        if tuple(record["shape"]) != shape:
            raise ValueError("packed-B split shapes differ")
        packed = np.memmap(record["path"], dtype="<u2", mode="r", shape=shape)
        logical = _logical_vector_from_preshifted(packed)
        shards, current_layout = build_page_reader_vector_split(
            plan_arrays,
            logical,
            chips=chips,
            groups_per_chip=groups,
        )
        if layout_report is None:
            layout_report = current_layout
        elif current_layout != layout_report:
            raise AssertionError("page-reader split layouts differ")
        replay = verify_page_reader_vector_split(plan_arrays, shards, packed)
        verified_words += int(replay["verified_bf16_words"])
        split_reports.append(replay)
        file_records[f"page_vector_b{split}"] = _write_array(
            output_dir / f"pcg_page_vector_b{split}.bf16",
            shards,
            force=force,
        )
        del packed, logical, shards

    mismatch_counts = [int(item["bit_mismatches"]) for item in split_reports]
    return {
        "schema": "tt_gmg_canonical_pcg_page_reader_vector_bundle_v1",
        "host_only": True,
        "vector_manifest": {
            "path": str(vector_manifest_path.resolve()),
            "sha256": _sha256(vector_manifest_path),
            "vector_tag": vector.get("vector_tag"),
        },
        "layout": layout_report,
        "files": file_records,
        "verified_bf16_words": verified_words,
        "bit_mismatches_by_split": mismatch_counts,
        "split_replay": split_reports,
        "passing": sum(mismatch_counts) == 0,
        "build_and_verify_seconds": time.time() - started,
    }


def build_and_analyze(
    brick_path: Path,
    layout_manifest_path: Path,
    output_dir: Path,
    *,
    write_map: bool,
    force: bool,
    vector_manifest_path: Path | None = None,
    write_compact_plan: bool = False,
    write_page_reader_vector: bool = False,
) -> dict[str, Any]:
    started = time.time()
    layout = json.loads(layout_manifest_path.read_text(encoding="utf-8"))
    files = layout["files"]
    packed_record = files["packed_to_source"]
    inverse_record = files["source_to_packed"]
    class_record = files["group_class"]
    packed = np.memmap(
        packed_record["path"],
        dtype="<i4",
        mode="r",
        shape=tuple(packed_record["shape"]),
    )
    inverse = np.memmap(
        inverse_record["path"],
        dtype="<i4",
        mode="r",
        shape=tuple(inverse_record["shape"]),
    )
    roles = np.memmap(
        class_record["path"],
        dtype=np.uint8,
        mode="r",
        shape=tuple(class_record["shape"]),
    )
    operator = BrickOperator(brick_path)
    mapping = build_neighbor_map(
        packed,
        inverse,
        operator.section("brick_nbr"),
        operator.section("slot_to_node"),
    )
    analysis = analyze_neighbor_map(mapping, roles)
    verification = (
        verify_neighbor_map_against_preshifted(mapping, vector_manifest_path)
        if vector_manifest_path is not None
        else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    compact_arrays = None
    compact_plan = None
    if write_compact_plan or write_page_reader_vector:
        compact_arrays, compact_plan = build_compact_neighbor_plan(mapping)
        if write_compact_plan:
            filenames = {
                "group_unique_offsets": "pcg_group_unique_offsets.u32",
                "group_unique_local_address": "pcg_group_unique_local_address.u32",
                "neighbor_group_local_index": "pcg_neighbor_group_local_index.u16",
                "group_page_offsets": "pcg_group_page_offsets.u32",
                "group_vector_page_local_index": "pcg_group_vector_page_local_index.u32",
                "neighbor_group_page_lane": "pcg_neighbor_group_page_lane.u16",
                "device_group_vector_page_table": "pcg_device_group_vector_page_table.u8",
                "halo_offsets": "pcg_halo_offsets.u32",
                "halo_packed_position": "pcg_halo_packed_position.u32",
            }
            compact_plan["files"] = {
                name: _write_array(
                    output_dir / filenames[name], array, force=force
                )
                for name, array in compact_arrays.items()
            }

    page_reader_vector = None
    if write_page_reader_vector:
        if vector_manifest_path is None:
            raise ValueError(
                "--write-page-reader-vector requires --vector-manifest"
            )
        if compact_arrays is None:
            raise AssertionError("compact plan arrays were not built")
        page_reader_vector = write_page_reader_vector_bundle(
            compact_arrays,
            vector_manifest_path,
            output_dir,
            force=force,
        )

    map_record = None
    if write_map:
        map_path = output_dir / "packed_neighbor_position.i32"
        if map_path.exists() and not force:
            raise FileExistsError(f"{map_path} exists; pass --force to replace it")
        partial = map_path.with_name(map_path.name + ".partial")
        if partial.exists():
            partial.unlink()
        mapping.tofile(partial)
        os.replace(partial, map_path)
        map_record = {
            "path": str(map_path.resolve()),
            "bytes": map_path.stat().st_size,
            "sha256": _sha256(map_path),
            "dtype": "<i4",
            "shape": list(mapping.shape),
            "order": "[output_chip][output_group][offset][output_lane]",
        }

    report = {
        **analysis,
        "created_unix_s": time.time(),
        "host_only": True,
        "brick_operator": {
            "path": str(brick_path.resolve()),
            "sha256": _sha256(brick_path),
        },
        "canonical_layout": {
            "path": str(layout_manifest_path.resolve()),
            "sha256": _sha256(layout_manifest_path),
        },
        "neighbor_map": map_record,
        "compact_gather_plan": compact_plan,
        "page_reader_vector_bundle": page_reader_vector,
        "preshifted_vector_verification": verification,
        "build_seconds": time.time() - started,
    }
    report_path = output_dir / "canonical_pcg_gather_analysis.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brick", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--write-map", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--vector-manifest", type=Path)
    parser.add_argument("--write-compact-plan", action="store_true")
    parser.add_argument("--write-page-reader-vector", action="store_true")
    args = parser.parse_args()
    report = build_and_analyze(
        args.brick,
        args.layout,
        args.output_dir,
        write_map=args.write_map,
        force=args.force,
        vector_manifest_path=args.vector_manifest,
        write_compact_plan=args.write_compact_plan,
        write_page_reader_vector=args.write_page_reader_vector,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    verification = report["preshifted_vector_verification"]
    compact = report["compact_gather_plan"]
    page_vector = report["page_reader_vector_bundle"]
    passing = (verification is None or verification["passing"]) and (
        compact is None or compact["passing"]
    ) and (
        page_vector is None or page_vector["passing"]
    )
    return 0 if passing else 1


if __name__ == "__main__":
    raise SystemExit(main())
