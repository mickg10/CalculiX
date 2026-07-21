#!/usr/bin/env python3
"""Derive the exact 8-chip/56-core TT layout from a brick-format-v1 operator.

The source format is audit-friendly and component-major. The device consumes three raw bf16
coefficient files in chip/local-group/term/output-component/lane order, one sharded-L1 halo
file per test vector, and an fp32 reference in device output order.

Each 4^3 output brick stores a 6^3 resident x halo. A fixed offset therefore becomes sixteen
4x4x4 sub-cubes per output tile, with no brick dictionary lookup in the timed kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from brick_format import (
    OFFSETS,
    TILE_ELEMENTS,
    BrickOperator,
    _bf16_values,
    _vectors,
    bcsr_apply,
    brick_apply,
    open_fine_dump,
    split_bf16x3,
)


HALO_SIDE = 6
HALO_RAW_VALUES_PER_BRICK = HALO_SIDE**3
# Store the three four-value k windows explicitly. This doubles the compact 6^3 halo but makes every
# reader copy an aligned uint64 rather than four scalar/possibly-unaligned bf16 loads.
HALO_VALUES_PER_BRICK = HALO_SIDE * HALO_SIDE * 3 * 4
HALO_VALUES_PER_GROUP = 16 * HALO_VALUES_PER_BRICK
HALO_PLANE_VALUES = 8192
TERMS = 81
OUTPUT_COMPONENTS = 3
A_SPLITS = 3
X_SPLITS = 3


def _sha256(path: Path, chunk_bytes: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


@dataclass(frozen=True)
class DevicePlan:
    source_groups: int
    chips: int = 8
    grid_x: int = 8
    grid_y: int = 7

    @property
    def cores_per_chip(self) -> int:
        return self.grid_x * self.grid_y

    @property
    def groups_per_chip(self) -> int:
        return math.ceil(self.source_groups / self.chips)

    @property
    def padded_groups(self) -> int:
        return self.groups_per_chip * self.chips

    @property
    def core_group_counts(self) -> tuple[int, ...]:
        base, remainder = divmod(self.groups_per_chip, self.cores_per_chip)
        return tuple(base + (index < remainder) for index in range(self.cores_per_chip))

    @property
    def core_group_starts(self) -> tuple[int, ...]:
        result = []
        cursor = 0
        for count in self.core_group_counts:
            result.append(cursor)
            cursor += count
        return tuple(result)

    @property
    def max_groups_per_core(self) -> int:
        return max(self.core_group_counts)

    def local_group_owner(self) -> tuple[np.ndarray, np.ndarray]:
        owner = np.empty(self.groups_per_chip, dtype=np.int32)
        slot = np.empty(self.groups_per_chip, dtype=np.int32)
        for core, (start, count) in enumerate(zip(self.core_group_starts, self.core_group_counts)):
            owner[start : start + count] = core
            slot[start : start + count] = np.arange(count, dtype=np.int32)
        return owner, slot

    def report(self) -> dict[str, Any]:
        a_bytes_per_chip = self.groups_per_chip * TERMS * OUTPUT_COMPONENTS * A_SPLITS * TILE_ELEMENTS * 2
        halo_bytes_per_core = self.max_groups_per_core * OUTPUT_COMPONENTS * X_SPLITS * HALO_PLANE_VALUES * 2
        return {
            "chips": self.chips,
            "worker_grid": [self.grid_x, self.grid_y],
            "cores_per_chip": self.cores_per_chip,
            "source_groups": self.source_groups,
            "groups_per_chip": self.groups_per_chip,
            "padded_groups": self.padded_groups,
            "padding_groups": self.padded_groups - self.source_groups,
            "core_group_counts": list(self.core_group_counts),
            "core_group_starts": list(self.core_group_starts),
            "max_groups_per_core": self.max_groups_per_core,
            "coefficient_bytes_per_chip": a_bytes_per_chip,
            "resident_halo_bytes_per_core": halo_bytes_per_core,
            "resident_halo_kib_per_core": halo_bytes_per_core / 1024.0,
        }


def pack_operator(
    brick_path: Path,
    output_dir: Path,
    *,
    chips: int = 8,
    grid_x: int = 8,
    grid_y: int = 7,
    force: bool = False,
    hash_files: bool = True,
) -> dict[str, Any]:
    started = time.time()
    operator = BrickOperator(brick_path)
    plan = DevicePlan(operator.ngroup, chips, grid_x, grid_y)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    coefficients = operator.section("coefficients")
    shape = (chips, plan.groups_per_chip, TERMS, OUTPUT_COMPONENTS, TILE_ELEMENTS)
    files = []
    for split in range(A_SPLITS):
        final = output_dir / f"operator_a{split}.bf16"
        partial = final.with_name(final.name + ".partial")
        if final.exists() and not force:
            raise FileExistsError(f"{final} exists; pass --force to replace it")
        if partial.exists():
            partial.unlink()
        destination = np.memmap(partial, dtype="<u2", mode="w+", shape=shape)
        destination[:] = 0
        for chip in range(chips):
            group0 = chip * plan.groups_per_chip
            group1 = min(group0 + plan.groups_per_chip, operator.ngroup)
            if group0 >= group1:
                continue
            source_components = [
                np.asarray(
                    coefficients[
                        split,
                        component * operator.ngroup + group0 : component * operator.ngroup + group1,
                    ],
                    dtype="<u2",
                )
                for component in range(OUTPUT_COMPONENTS)
            ]
            destination[chip, : group1 - group0] = np.stack(source_components, axis=2)
        destination.flush()
        del destination
        os.replace(partial, final)
        files.append(
            {
                "split": split,
                "path": str(final),
                "bytes": final.stat().st_size,
                "sha256": _sha256(final) if hash_files else None,
            }
        )
    manifest = {
        "schema": "tt_gmg_device_layout_v1",
        "created_unix_s": time.time(),
        "source_operator": str(operator.path),
        "source_operator_sha256": _sha256(operator.path) if hash_files else None,
        "source": {
            "active_nodes": operator.nb,
            "bricks": operator.nbrick,
            "brick_groups": operator.ngroup,
            "terms": operator.nterms,
        },
        "partition": plan.report(),
        "coefficient_layout": {
            "dtype": "<u2",
            "shape": list(shape),
            "order": "[chip][local_group][term][out_component][lane]",
            "files": files,
        },
        "resident_halo_layout": {
            "dtype": "<u2",
            "shape": [
                chips,
                plan.cores_per_chip,
                plan.max_groups_per_core,
                OUTPUT_COMPONENTS * X_SPLITS,
                HALO_PLANE_VALUES,
            ],
            "order": "[chip][core][group_slot][component*3+split][halo_lane]",
            "active_values_per_plane": HALO_VALUES_PER_GROUP,
            "plane_values": HALO_PLANE_VALUES,
            "per_brick_shape": [HALO_SIDE, HALO_SIDE, 3, 4],
            "k_window_starts": [-1, 0, 1],
        },
        "output_layout": {
            "dtype": "<f4",
            "shape": [chips, plan.groups_per_chip, OUTPUT_COMPONENTS, TILE_ELEMENTS],
            "order": "[chip][local_group][out_component][lane]",
        },
        "pack_seconds": time.time() - started,
    }
    manifest_path = output_dir / "device_layout.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _brick_halos(operator: BrickOperator, vector: np.ndarray, chunk_bricks: int = 512):
    brick_coords = np.asarray(operator.section("brick_coords")[: operator.nbrick], dtype=np.int64)
    slot_to_node = operator.section("slot_to_node")
    grid_shape = (
        math.ceil(operator.nx / operator.brick_size),
        math.ceil(operator.ny / operator.brick_size),
        math.ceil(operator.nz / operator.brick_size),
    )
    brick_grid = np.full(grid_shape, -1, dtype=np.int32)
    brick_grid[tuple(brick_coords.T)] = np.arange(operator.nbrick, dtype=np.int32)
    halo_local = np.asarray(
        [(i, j, k) for i in range(-1, 5) for j in range(-1, 5) for k in range(-1, 5)],
        dtype=np.int64,
    )
    x = np.asarray(vector, dtype=np.float32).reshape(operator.nb, OUTPUT_COMPONENTS)
    for brick0 in range(0, operator.nbrick, chunk_bricks):
        brick1 = min(brick0 + chunk_bricks, operator.nbrick)
        global_coord = brick_coords[brick0:brick1, None, :] * operator.brick_size + halo_local[None, :, :]
        source_coord = np.floor_divide(global_coord, operator.brick_size)
        source_local = global_coord - source_coord * operator.brick_size
        valid = np.all(source_coord >= 0, axis=2)
        for axis, extent in enumerate(grid_shape):
            valid &= source_coord[:, :, axis] < extent
        source_brick = np.full(valid.shape, -1, dtype=np.int32)
        source_brick[valid] = brick_grid[tuple(source_coord[valid].T)]
        valid &= source_brick >= 0
        source_slot = (
            (source_local[:, :, 0] * operator.brick_size + source_local[:, :, 1])
            * operator.brick_size
            + source_local[:, :, 2]
        ).astype(np.int64)
        source_node = np.full(valid.shape, -1, dtype=np.int32)
        source_node[valid] = slot_to_node[source_brick[valid], source_slot[valid]]
        valid &= source_node >= 0
        halo = np.zeros((brick1 - brick0, HALO_RAW_VALUES_PER_BRICK, OUTPUT_COMPONENTS), dtype=np.float32)
        halo[valid] = x[source_node[valid]]
        yield brick0, brick1, halo


def _vector_by_name(dump, name: str, seed: int) -> np.ndarray:
    return next(vector for vector_name, vector in _vectors(dump, (name,), seed) if vector_name == name)


def pack_vector(
    fine_path: Path,
    brick_path: Path,
    layout_manifest_path: Path,
    output_dir: Path,
    *,
    vector_name: str,
    seed: int = 351,
    force: bool = False,
    hash_files: bool = True,
) -> dict[str, Any]:
    started = time.time()
    dump = open_fine_dump(fine_path)
    operator = BrickOperator(brick_path)
    layout = json.loads(layout_manifest_path.read_text(encoding="utf-8"))
    partition = layout["partition"]
    plan = DevicePlan(
        operator.ngroup,
        int(partition["chips"]),
        int(partition["worker_grid"][0]),
        int(partition["worker_grid"][1]),
    )
    if plan.report() != partition:
        raise ValueError("layout manifest partition does not match the operator/device plan")
    vector = _vector_by_name(dump, vector_name, seed)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    halo_path = output_dir / f"halo_{vector_name}_seed{seed}.bf16"
    reference_path = output_dir / f"reference_{vector_name}_seed{seed}.fp32"
    for path in (halo_path, reference_path):
        if path.exists() and not force:
            raise FileExistsError(f"{path} exists; pass --force to replace it")

    halo_shape = (
        plan.chips,
        plan.cores_per_chip,
        plan.max_groups_per_core,
        OUTPUT_COMPONENTS * X_SPLITS,
        HALO_PLANE_VALUES,
    )
    halo_partial = halo_path.with_name(halo_path.name + ".partial")
    if halo_partial.exists():
        halo_partial.unlink()
    halo_out = np.memmap(halo_partial, dtype="<u2", mode="w+", shape=halo_shape)
    halo_out[:] = 0
    owner, group_slot = plan.local_group_owner()
    pending = np.zeros(
        (operator.nbrick_pad, HALO_VALUES_PER_BRICK, OUTPUT_COMPONENTS),
        dtype=np.float32,
    )
    for brick0, brick1, halo in _brick_halos(operator, vector):
        raw = halo.reshape(brick1 - brick0, HALO_SIDE, HALO_SIDE, HALO_SIDE, OUTPUT_COMPONENTS)
        pending[brick0:brick1] = np.stack(
            [raw[:, :, :, start : start + 4, :] for start in range(3)],
            axis=3,
        ).reshape(brick1 - brick0, HALO_VALUES_PER_BRICK, OUTPUT_COMPONENTS)
    for group in range(operator.ngroup):
        chip, local_group = divmod(group, plan.groups_per_chip)
        core = int(owner[local_group])
        slot = int(group_slot[local_group])
        group_values = pending[group * 16 : (group + 1) * 16].reshape(
            HALO_VALUES_PER_GROUP,
            OUTPUT_COMPONENTS,
        )
        hi, mid, lo = split_bf16x3(group_values)
        for component in range(OUTPUT_COMPONENTS):
            for split, bits in enumerate((hi, mid, lo)):
                halo_out[
                    chip,
                    core,
                    slot,
                    component * X_SPLITS + split,
                    :HALO_VALUES_PER_GROUP,
                ] = bits[:, component]
    halo_out.flush()
    del halo_out, pending
    os.replace(halo_partial, halo_path)

    reference = brick_apply(operator, vector, b_splits=3)
    bcsr_reference = bcsr_apply(dump, vector)
    reference_error = reference.astype(np.float64) - bcsr_reference
    node_to_pos = np.asarray(operator.section("node_to_pos"), dtype=np.int64)
    reference_box = np.zeros(
        (plan.padded_groups * TILE_ELEMENTS, OUTPUT_COMPONENTS),
        dtype="<f4",
    )
    reference_box[node_to_pos] = reference
    reference_device = reference_box.reshape(
        plan.padded_groups,
        TILE_ELEMENTS,
        OUTPUT_COMPONENTS,
    ).transpose(0, 2, 1)
    reference_device = reference_device.reshape(
        plan.chips,
        plan.groups_per_chip,
        OUTPUT_COMPONENTS,
        TILE_ELEMENTS,
    )
    reference_partial = reference_path.with_name(reference_path.name + ".partial")
    reference_device.tofile(reference_partial)
    os.replace(reference_partial, reference_path)

    report = {
        "schema": "tt_gmg_device_vector_v1",
        "created_unix_s": time.time(),
        "vector": {"name": vector_name, "seed": seed},
        "fine_dump": str(dump.path),
        "brick_operator": str(operator.path),
        "layout_manifest": str(layout_manifest_path.resolve()),
        "halo": {
            "path": str(halo_path),
            "bytes": halo_path.stat().st_size,
            "sha256": _sha256(halo_path) if hash_files else None,
            "shape": list(halo_shape),
        },
        "reference": {
            "path": str(reference_path),
            "bytes": reference_path.stat().st_size,
            "sha256": _sha256(reference_path) if hash_files else None,
            "shape": list(reference_device.shape),
            "l2_relative_vs_bcsr": float(
                np.linalg.norm(reference_error.ravel())
                / max(np.linalg.norm(bcsr_reference.ravel()), 1e-300)
            ),
            "max_relative_vs_bcsr": float(
                np.max(np.abs(reference_error))
                / max(np.max(np.abs(bcsr_reference)), 1e-300)
            ),
        },
        "pack_seconds": time.time() - started,
    }
    report_path = output_dir / f"vector_{vector_name}_seed{seed}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def halo_b_tile(
    halo_group: np.ndarray,
    offset_id: int,
    component: int,
    split: int,
) -> np.ndarray:
    """Materialize one raw 1024-lane b tile exactly as the TT reader will."""

    if not 0 <= offset_id < 27 or not 0 <= component < 3 or not 0 <= split < 3:
        raise ValueError("offset/component/split outside the device ABI")
    di, dj, dk = (int(value) for value in OFFSETS[offset_id])
    plane = np.asarray(
        halo_group[component * X_SPLITS + split, :HALO_VALUES_PER_GROUP],
        dtype="<u2",
    )
    bricks = plane.reshape(16, HALO_SIDE, HALO_SIDE, 3, 4)
    return np.ascontiguousarray(
        bricks[:, di + 1 : di + 5, dj + 1 : dj + 5, dk + 1, :]
    ).reshape(TILE_ELEMENTS)


def verify_device_pack(
    brick_path: Path,
    layout_manifest_path: Path,
    vector_report_path: Path,
) -> dict[str, Any]:
    """Apply the derived files on the host using the exact device page and halo ABI."""

    started = time.time()
    operator = BrickOperator(brick_path)
    layout = json.loads(layout_manifest_path.read_text(encoding="utf-8"))
    vector_report = json.loads(vector_report_path.read_text(encoding="utf-8"))
    partition = layout["partition"]
    plan = DevicePlan(
        operator.ngroup,
        int(partition["chips"]),
        int(partition["worker_grid"][0]),
        int(partition["worker_grid"][1]),
    )
    a_shape = tuple(layout["coefficient_layout"]["shape"])
    a_files = [
        np.memmap(item["path"], dtype="<u2", mode="r", shape=a_shape)
        for item in sorted(
            layout["coefficient_layout"]["files"],
            key=lambda item: item["split"],
        )
    ]
    halo_shape = tuple(vector_report["halo"]["shape"])
    halo = np.memmap(
        vector_report["halo"]["path"],
        dtype="<u2",
        mode="r",
        shape=halo_shape,
    )
    reference_shape = tuple(vector_report["reference"]["shape"])
    reference = np.memmap(
        vector_report["reference"]["path"],
        dtype="<f4",
        mode="r",
        shape=reference_shape,
    )
    owner, group_slot = plan.local_group_owner()
    candidate = np.zeros(reference_shape, dtype=np.float32)
    products = ((0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0))
    for group in range(operator.ngroup):
        chip, local_group = divmod(group, plan.groups_per_chip)
        core = int(owner[local_group])
        slot = int(group_slot[local_group])
        halo_group = halo[chip, core, slot]
        accum = np.zeros((OUTPUT_COMPONENTS, TILE_ELEMENTS), dtype=np.float32)
        for offset_id in range(27):
            for input_component in range(OUTPUT_COMPONENTS):
                term = offset_id * 3 + input_component
                b = [
                    _bf16_values(
                        halo_b_tile(
                            halo_group,
                            offset_id,
                            input_component,
                            split,
                        )
                    )
                    for split in range(X_SPLITS)
                ]
                for output_component in range(OUTPUT_COMPONENTS):
                    a = [
                        _bf16_values(
                            a_files[split][
                                chip,
                                local_group,
                                term,
                                output_component,
                            ]
                        )
                        for split in range(A_SPLITS)
                    ]
                    for a_level, b_level in products:
                        accum[output_component] += a[a_level] * b[b_level]
        candidate[chip, local_group] = accum
    difference = candidate - np.asarray(reference, dtype=np.float32)
    ref_l2 = float(np.linalg.norm(np.asarray(reference, dtype=np.float64).ravel()))
    ref_max = float(np.max(np.abs(reference)))
    bitwise = bool(
        np.array_equal(
            candidate.view(np.uint32),
            np.asarray(reference).view(np.uint32),
        )
    )
    return {
        "schema": "tt_gmg_device_pack_verification_v1",
        "brick_operator": str(operator.path),
        "layout_manifest": str(layout_manifest_path.resolve()),
        "vector_report": str(vector_report_path.resolve()),
        "l2_relative": float(
            np.linalg.norm(difference.astype(np.float64).ravel())
            / max(ref_l2, 1e-300)
        ),
        "max_relative": float(
            np.max(np.abs(difference)) / max(ref_max, 1e-300)
        ),
        "max_absolute": float(np.max(np.abs(difference))),
        "pass": bitwise,
        "bitwise_equal_to_reference": bitwise,
        "verification_seconds": time.time() - started,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build and verify the TT brick-stencil device layout"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    operator_parser = sub.add_parser("pack-operator")
    operator_parser.add_argument("--brick", type=Path, required=True)
    operator_parser.add_argument("--output-dir", type=Path, required=True)
    operator_parser.add_argument("--chips", type=int, default=8)
    operator_parser.add_argument("--grid-x", type=int, default=8)
    operator_parser.add_argument("--grid-y", type=int, default=7)
    operator_parser.add_argument("--force", action="store_true")
    operator_parser.add_argument("--skip-hash", action="store_true")

    vector_parser = sub.add_parser("pack-vector")
    vector_parser.add_argument("--fine", type=Path, required=True)
    vector_parser.add_argument("--brick", type=Path, required=True)
    vector_parser.add_argument("--layout", type=Path, required=True)
    vector_parser.add_argument("--output-dir", type=Path, required=True)
    vector_parser.add_argument(
        "--vector",
        choices=("random", "smooth", "boundary", "constant"),
        required=True,
    )
    vector_parser.add_argument("--seed", type=int, default=351)
    vector_parser.add_argument("--force", action="store_true")
    vector_parser.add_argument("--skip-hash", action="store_true")

    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--brick", type=Path, required=True)
    verify_parser.add_argument("--layout", type=Path, required=True)
    verify_parser.add_argument("--vector-report", type=Path, required=True)
    verify_parser.add_argument("--report", type=Path)

    args = parser.parse_args(argv)
    if args.command == "pack-operator":
        report = pack_operator(
            args.brick,
            args.output_dir,
            chips=args.chips,
            grid_x=args.grid_x,
            grid_y=args.grid_y,
            force=args.force,
            hash_files=not args.skip_hash,
        )
    elif args.command == "pack-vector":
        report = pack_vector(
            args.fine,
            args.brick,
            args.layout,
            args.output_dir,
            vector_name=args.vector,
            seed=args.seed,
            force=args.force,
            hash_files=not args.skip_hash,
        )
    else:
        report = verify_device_pack(
            args.brick,
            args.layout,
            args.vector_report,
        )
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("pass", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
