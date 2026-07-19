#!/usr/bin/env python3
"""Host arithmetic oracle for the canonical-base plus dense-fallback layout.

This replays the qualifying run52/run64 BF16x3 product order on the packed
files.  It also computes the base path a second time without skipping zero
coefficient planes, proving whether the sparse schedule changes any fp32 bit.
It never opens a Tenstorrent device.
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

from brick_format import _bf16_values
from canonical_packed_layout import (
    GROUP_BASE,
    GROUP_FALLBACK,
    LANES,
    OUTPUT_COMPONENTS,
    TERMS,
)


A_SPLITS = 3
PRODUCTS_WITH_BLOW = ((0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0))
PRODUCTS_WITHOUT_BLOW = ((0, 0), (0, 1), (1, 0), (2, 0), (1, 1))


def _sha256(path: Path, chunk_bytes: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def retains_blow(term: int) -> bool:
    offset = term // 3
    di = offset // 9
    dj = (offset // 3) % 3
    dk = offset % 3
    manhattan = abs(di - 1) + abs(dj - 1) + abs(dk - 1)
    return manhattan <= 2 or offset in (6, 20)


def _accumulate_base(
    b_splits: tuple[np.ndarray, ...],
    palette_splits: tuple[np.ndarray, ...],
    schedule: np.ndarray,
    zero_index: int,
    *,
    skip_zero_planes: bool,
) -> np.ndarray:
    groups = b_splits[0].shape[0]
    accumulator = np.zeros(
        (groups, OUTPUT_COMPONENTS, LANES),
        dtype="<f4",
    )
    for term in reversed(range(TERMS)):
        products = PRODUCTS_WITH_BLOW if retains_blow(term) else PRODUCTS_WITHOUT_BLOW
        for output in range(OUTPUT_COMPONENTS):
            palette_index = int(schedule[term, output])
            if skip_zero_planes and palette_index == zero_index:
                continue
            for a_level, b_level in products:
                product = (
                    palette_splits[a_level][palette_index]
                    * b_splits[b_level][:, term, :]
                )
                accumulator[:, output, :] = accumulator[:, output, :] + product
    return accumulator


def _accumulate_fallback(
    a_splits: tuple[np.ndarray, ...],
    b_splits: tuple[np.ndarray, ...],
) -> np.ndarray:
    groups = b_splits[0].shape[0]
    accumulator = np.zeros(
        (groups, OUTPUT_COMPONENTS, LANES),
        dtype="<f4",
    )
    for term in reversed(range(TERMS)):
        products = PRODUCTS_WITH_BLOW if retains_blow(term) else PRODUCTS_WITHOUT_BLOW
        for output in range(OUTPUT_COMPONENTS):
            for a_level, b_level in products:
                product = (
                    a_splits[a_level][:, term, output, :]
                    * b_splits[b_level][:, term, :]
                )
                accumulator[:, output, :] = accumulator[:, output, :] + product
    return accumulator


def run_oracle(
    layout_manifest_path: Path,
    vector_manifest_path: Path,
    report_path: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    started = time.time()
    layout = _load_json(layout_manifest_path)
    vector = _load_json(vector_manifest_path)
    plan = layout["plan"]
    chips = int(plan["chips"])
    packed_groups = int(plan["groups_per_chip"])
    b_shape = tuple(vector["files"]["packed_b0"]["shape"])
    b_bits = tuple(
        np.memmap(
            vector["files"][f"packed_b{split}"]["path"],
            dtype="<u2",
            mode="r",
            shape=b_shape,
        )
        for split in range(A_SPLITS)
    )
    b_values = tuple(_bf16_values(bits) for bits in b_bits)

    fallback_shape = tuple(layout["fallback_operator_layout"]["shape"])
    fallback_bits = tuple(
        np.memmap(
            layout["files"][f"fallback_a{split}"]["path"],
            dtype="<u2",
            mode="r",
            shape=fallback_shape,
        )
        for split in range(A_SPLITS)
    )
    fallback_values = tuple(_bf16_values(bits) for bits in fallback_bits)

    palette_shape = tuple(layout["files"]["palette_a0"]["shape"])
    palette_splits = tuple(
        _bf16_values(
            np.memmap(
                layout["files"][f"palette_a{split}"]["path"],
                dtype="<u2",
                mode="r",
                shape=palette_shape,
            )[:, 0]
        )
        for split in range(A_SPLITS)
    )
    schedule = np.memmap(
        layout["files"]["palette_schedule"]["path"],
        dtype=np.uint8,
        mode="r",
        shape=(TERMS, OUTPUT_COMPONENTS),
    )
    zero_index = int(layout["palette"]["zero_index"])
    group_class = np.memmap(
        layout["files"]["group_class"]["path"],
        dtype=np.uint8,
        mode="r",
        shape=(chips, packed_groups),
    )

    candidate = np.zeros(
        (chips, packed_groups, OUTPUT_COMPONENTS, LANES),
        dtype="<f4",
    )
    dense_base_bit_mismatches = 0
    dense_base_numeric_mismatches = 0
    for chip in range(chips):
        base_groups = int(plan["base_groups_per_chip"][chip])
        fallback_groups = int(plan["fallback_groups_per_chip"][chip])
        if not np.all(group_class[chip, :base_groups] == GROUP_BASE):
            raise ValueError("layout base-group roles disagree with plan")
        if not np.all(
            group_class[chip, base_groups : base_groups + fallback_groups]
            == GROUP_FALLBACK
        ):
            raise ValueError("layout fallback-group roles disagree with plan")

        chip_base_b = tuple(
            values[chip, :base_groups] for values in b_values
        )
        sparse_base = _accumulate_base(
            chip_base_b,
            palette_splits,
            schedule,
            zero_index,
            skip_zero_planes=True,
        )
        dense_base = _accumulate_base(
            chip_base_b,
            palette_splits,
            schedule,
            zero_index,
            skip_zero_planes=False,
        )
        dense_base_bit_mismatches += int(
            np.count_nonzero(
                sparse_base.view("<u4") != dense_base.view("<u4")
            )
        )
        dense_base_numeric_mismatches += int(
            np.count_nonzero(sparse_base != dense_base)
        )
        candidate[chip, :base_groups] = sparse_base

        if fallback_groups:
            chip_fallback_a = tuple(
                values[chip, :fallback_groups] for values in fallback_values
            )
            chip_fallback_b = tuple(
                values[
                    chip,
                    base_groups : base_groups + fallback_groups,
                ]
                for values in b_values
            )
            candidate[
                chip,
                base_groups : base_groups + fallback_groups,
            ] = _accumulate_fallback(chip_fallback_a, chip_fallback_b)

    reference_shape = tuple(vector["files"]["packed_reference"]["shape"])
    reference = np.memmap(
        vector["files"]["packed_reference"]["path"],
        dtype="<f4",
        mode="r",
        shape=reference_shape,
    )
    error = candidate.astype(np.float64) - np.asarray(reference, dtype=np.float64)
    reference64 = np.asarray(reference, dtype=np.float64)
    l2_relative = float(
        np.linalg.norm(error.ravel())
        / max(np.linalg.norm(reference64.ravel()), 1.0e-300)
    )
    max_relative = float(
        np.max(np.abs(error))
        / max(np.max(np.abs(reference64)), 1.0e-300)
    )
    per_component = {}
    for component in range(OUTPUT_COMPONENTS):
        component_error = error[:, :, component, :]
        component_reference = reference64[:, :, component, :]
        per_component[str(component)] = {
            "l2_relative": float(
                np.linalg.norm(component_error.ravel())
                / max(np.linalg.norm(component_reference.ravel()), 1.0e-300)
            ),
            "max_relative": float(
                np.max(np.abs(component_error))
                / max(np.max(np.abs(component_reference)), 1.0e-300)
            ),
        }

    output_record = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial = output_path.with_name(output_path.name + ".partial")
        if partial.exists():
            partial.unlink()
        candidate.tofile(partial)
        os.replace(partial, output_path)
        output_record = {
            "path": str(output_path.resolve()),
            "bytes": output_path.stat().st_size,
            "sha256": _sha256(output_path),
            "dtype": "<f4",
            "shape": list(candidate.shape),
        }

    checks = {
        "sparse_zero_skip_numerically_identical_to_dense_base": (
            dense_base_numeric_mismatches == 0
        ),
        "sparse_zero_skip_bitwise_identical_to_dense_base": (
            dense_base_bit_mismatches == 0
        ),
        "l2_relative_le_2e_5": l2_relative <= 2.0e-5,
        "max_relative_le_5e_5": max_relative <= 5.0e-5,
    }
    report = {
        "schema": "tt_gmg_canonical_packed_host_oracle_v1",
        "created_unix_s": time.time(),
        "host_only": True,
        "vector_tag": vector["vector_tag"],
        "term_order": "reverse",
        "arithmetic": "run52_bf16x3_selective_blow",
        "base_coefficient_quantum": 1.0e-7,
        "fallback_coefficients": "exact_source_bf16x3",
        "thresholds": {
            "l2_relative": 2.0e-5,
            "max_relative": 5.0e-5,
            "basis": "established_four_vector_brick_host_oracle",
        },
        "checks": checks,
        "passing": all(checks.values()),
        "l2_relative": l2_relative,
        "max_relative": max_relative,
        "per_component": per_component,
        "dense_base_numeric_mismatches": dense_base_numeric_mismatches,
        "dense_base_bit_mismatches": dense_base_bit_mismatches,
        "layout_manifest": {
            "path": str(layout_manifest_path.resolve()),
            "sha256": _sha256(layout_manifest_path),
        },
        "vector_manifest": {
            "path": str(vector_manifest_path.resolve()),
            "sha256": _sha256(vector_manifest_path),
        },
        "candidate_output": output_record,
        "oracle_seconds": time.time() - started,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-manifest", type=Path, required=True)
    parser.add_argument("--vector-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_oracle(
        args.layout_manifest,
        args.vector_manifest,
        args.report,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
