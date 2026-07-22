#!/usr/bin/env python3
"""Explain one device-output lane from the packed Row236 brick-stencil files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CHIPS = 8
GROUPS_PER_CHIP = 198
TERMS = 81
COMPONENTS = 3
LANES = 1024
CORES = 56
MAX_GROUPS_PER_CORE = 4
HALO_PLANES = 9
HALO_LANES = 8192
PRODUCTS = ((0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0))


def bf16(bits: np.uint16) -> np.float32:
    word = np.asarray([bits], dtype=np.uint16).astype(np.uint32) << 16
    return word.view(np.float32)[0]


def owner_and_slot(local_group: int) -> tuple[int, int]:
    base, remainder = divmod(GROUPS_PER_CHIP, CORES)
    cursor = 0
    for core in range(CORES):
        count = base + (core < remainder)
        if cursor <= local_group < cursor + count:
            return core, local_group - cursor
        cursor += count
    raise ValueError("local group is outside the device partition")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("chip", type=int)
    parser.add_argument("group", type=int)
    parser.add_argument("component", type=int)
    parser.add_argument("lane", type=int)
    parser.add_argument("--vector-tag", default="random_seed351")
    parser.add_argument("--hardware", type=float)
    args = parser.parse_args()

    a_shape = (CHIPS, GROUPS_PER_CHIP, TERMS, COMPONENTS, LANES)
    halo_shape = (CHIPS, CORES, MAX_GROUPS_PER_CORE, HALO_PLANES, HALO_LANES)
    reference_shape = (CHIPS, GROUPS_PER_CHIP, COMPONENTS, LANES)
    a = [
        np.memmap(args.root / f"operator_a{split_id}.bf16", dtype="<u2", mode="r", shape=a_shape)
        for split_id in range(3)
    ]
    halo = np.memmap(
        args.root / f"halo_{args.vector_tag}.bf16",
        dtype="<u2",
        mode="r",
        shape=halo_shape,
    )
    reference = np.memmap(
        args.root / f"reference_{args.vector_tag}.fp32",
        dtype="<f4",
        mode="r",
        shape=reference_shape,
    )

    core, slot = owner_and_slot(args.group)
    brick, in_brick = divmod(args.lane, 64)
    i, in_brick = divmod(in_brick, 16)
    j, k = divmod(in_brick, 4)
    accumulator = np.float32(0.0)
    accumulator_double = 0.0
    chunk_accumulators = [np.float32(0.0) for _ in range(3)]
    terms = []
    for offset_id in range(27):
        di, remainder = divmod(offset_id, 9)
        dj, dk = divmod(remainder, 3)
        for input_component in range(3):
            term = offset_id * 3 + input_component
            aval = [
                bf16(a[split_id][args.chip, args.group, term, args.component, args.lane])
                for split_id in range(3)
            ]
            plane_bits = halo[
                args.chip,
                core,
                slot,
                input_component * 3 : input_component * 3 + 3,
                : 16 * 6 * 6 * 3 * 4,
            ].reshape(3, 16, 6, 6, 3, 4)
            bval = [
                bf16(plane_bits[split_id, brick, i + di, j + dj, dk, k])
                for split_id in range(3)
            ]
            term_sum = np.float32(0.0)
            products = []
            for a_level, b_level in PRODUCTS:
                product = np.float32(aval[a_level] * bval[b_level])
                accumulator = np.float32(accumulator + product)
                chunk_id = term // (TERMS // len(chunk_accumulators))
                chunk_accumulators[chunk_id] = np.float32(
                    chunk_accumulators[chunk_id] + product
                )
                accumulator_double += float(product)
                term_sum = np.float32(term_sum + product)
                products.append(float(product))
            terms.append(
                {
                    "offset_id": offset_id,
                    "input_component": input_component,
                    "a": [float(value) for value in aval],
                    "b": [float(value) for value in bval],
                    "products": products,
                    "term_sum_fp32": float(term_sum),
                    "accumulator_fp32": float(accumulator),
                }
            )

    expected = float(reference[args.chip, args.group, args.component, args.lane])
    chunk_reduced = np.float32(
        np.float32(chunk_accumulators[0] + chunk_accumulators[1])
        + chunk_accumulators[2]
    )
    report = {
        "schema": "tt_gmg_device_lane_diagnostic_v1",
        "location": {
            "chip": args.chip,
            "local_group": args.group,
            "component": args.component,
            "lane": args.lane,
            "core": core,
            "group_slot": slot,
            "brick_in_group": brick,
            "local_ijk": [i, j, k],
        },
        "strict_fp32_accumulator": float(accumulator),
        "strict_fp32_chunk_accumulators": [
            float(value) for value in chunk_accumulators
        ],
        "strict_fp32_chunk_reduced": float(chunk_reduced),
        "sum_products_fp64": accumulator_double,
        "reference": expected,
        "strict_minus_reference": float(accumulator) - expected,
        "hardware": args.hardware,
        "hardware_minus_reference": None if args.hardware is None else args.hardware - expected,
        "first_chunk_terms": terms[: TERMS // len(chunk_accumulators)],
        "largest_term_sums": sorted(
            terms,
            key=lambda item: abs(item["term_sum_fp32"]),
            reverse=True,
        )[:16],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
