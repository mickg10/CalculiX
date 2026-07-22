#!/usr/bin/env python3
"""Search symmetric corner A-low masks against the exact brick host oracle.

Run52 already omits x-low for most corner terms.  This tool asks the separate
question of whether selected ``a_low * x_hi`` corner products can also be
omitted without crossing the host-oracle tolerance.  Opposite geometric
offsets and a fixed input component form one pair, leaving twelve binary
choices.  The search therefore preserves inversion symmetry and is small
enough to enumerate exactly.

The first pass measures the exact single-pair error deltas and builds their
Gram matrix.  It ranks all 2**12 masks from that quadratic model, then runs the
full ordered-fp32 brick oracle for the best masks.  The exact validation is the
authority; the Gram model is only a search accelerator.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from brick_format import (
    OFFSETS,
    BrickOperator,
    _vectors,
    bcsr_apply,
    brick_apply,
    open_fine_dump,
)


def _term_ids(offsets: Iterable[int]) -> tuple[int, ...]:
    return tuple(3 * offset + component for offset in offsets for component in range(3))


def _noncorner_terms() -> tuple[int, ...]:
    return tuple(
        term
        for term in range(81)
        if sum(abs(int(value)) for value in OFFSETS[term // 3]) <= 2
    )


def _opposite_corner_pairs() -> tuple[tuple[int, int], ...]:
    corners = tuple(
        offset
        for offset, delta in enumerate(OFFSETS)
        if sum(abs(int(value)) for value in delta) == 3
    )
    pairs: list[tuple[int, int]] = []
    for left in corners:
        right = 26 - left
        if left >= right:
            continue
        for component in range(3):
            pairs.append((3 * left + component, 3 * right + component))
    if len(pairs) != 12:
        raise AssertionError(f"expected 12 opposite-corner term pairs, found {len(pairs)}")
    return tuple(pairs)


def _metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    difference = candidate.astype(np.float64, copy=False) - reference
    return {
        "l2_relative": float(
            np.linalg.norm(difference.ravel())
            / max(np.linalg.norm(reference.ravel()), 1.0e-300)
        ),
        "max_relative": float(
            np.max(np.abs(difference))
            / max(np.max(np.abs(reference)), 1.0e-300)
        ),
        "max_absolute": float(np.max(np.abs(difference))),
    }


def _emit(record: dict[str, Any]) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)


def search(
    fine_path: Path,
    brick_path: Path,
    *,
    seed: int,
    term_order: str,
    retained_b_low_corner_offsets: Sequence[int],
    l2_limit: float,
    validate_top: int,
    group_chunk: int,
) -> list[dict[str, Any]]:
    dump = open_fine_dump(fine_path)
    operator = BrickOperator(brick_path)
    vector = next(vector for name, vector in _vectors(dump, ("smooth",), seed))

    started = time.time()
    reference = bcsr_apply(dump, vector)
    _emit({"stage": "reference", "seconds": time.time() - started})

    noncorner = _noncorner_terms()
    b_low_terms = tuple(
        sorted(set(noncorner) | set(_term_ids(retained_b_low_corner_offsets)))
    )
    all_a_low_terms = tuple(range(81))
    started = time.time()
    base = brick_apply(
        operator,
        vector,
        group_chunk=group_chunk,
        b_low_terms=b_low_terms,
        a_low_terms=all_a_low_terms,
        term_order=term_order,
    ).astype(np.float64)
    base_error = base - reference
    base_metrics = _metrics(base, reference)
    _emit(
        {
            "stage": "base",
            "seconds": time.time() - started,
            "b_low_terms": len(b_low_terms),
            **base_metrics,
        }
    )

    pairs = _opposite_corner_pairs()
    deltas: list[np.ndarray] = []
    for index, pair in enumerate(pairs):
        retained = tuple(term for term in all_a_low_terms if term not in pair)
        started = time.time()
        candidate = brick_apply(
            operator,
            vector,
            group_chunk=group_chunk,
            b_low_terms=b_low_terms,
            a_low_terms=retained,
            term_order=term_order,
        ).astype(np.float64)
        delta = (candidate - base).astype(np.float32)
        deltas.append(delta)
        _emit(
            {
                "stage": "single_pair",
                "pair_index": index,
                "omitted_terms": list(pair),
                "seconds": time.time() - started,
                **_metrics(candidate, reference),
            }
        )

    reference_norm = float(np.linalg.norm(reference.ravel()))
    base_flat = base_error.ravel()
    gram = np.empty((len(pairs), len(pairs)), dtype=np.float64)
    base_dot = np.empty(len(pairs), dtype=np.float64)
    for left, delta_left in enumerate(deltas):
        left_flat = delta_left.ravel().astype(np.float64, copy=False)
        base_dot[left] = np.dot(base_flat, left_flat)
        for right in range(left + 1):
            value = np.dot(
                left_flat,
                deltas[right].ravel().astype(np.float64, copy=False),
            )
            gram[left, right] = gram[right, left] = value

    base_norm_sq = float(np.dot(base_flat, base_flat))
    ranked: list[tuple[int, float, tuple[int, ...]]] = []
    for mask in range(1 << len(pairs)):
        chosen = tuple(index for index in range(len(pairs)) if mask & (1 << index))
        norm_sq = base_norm_sq
        if chosen:
            selected = np.asarray(chosen, dtype=np.int64)
            norm_sq += 2.0 * float(base_dot[selected].sum())
            norm_sq += float(gram[np.ix_(selected, selected)].sum())
        predicted = float(np.sqrt(max(norm_sq, 0.0)) / max(reference_norm, 1.0e-300))
        if predicted <= l2_limit:
            ranked.append((len(chosen), predicted, chosen))
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    _emit(
        {
            "stage": "quadratic_search",
            "mask_count": 1 << len(pairs),
            "within_l2_limit": len(ranked),
            "l2_limit": l2_limit,
            "maximum_omitted_pairs": ranked[0][0] if ranked else None,
        }
    )

    validated: list[dict[str, Any]] = []
    seen_term_sets: set[tuple[int, ...]] = set()
    for omitted_pair_count, predicted, chosen in ranked:
        omitted = tuple(sorted(itertools.chain.from_iterable(pairs[index] for index in chosen)))
        if omitted in seen_term_sets:
            continue
        seen_term_sets.add(omitted)
        retained = tuple(term for term in all_a_low_terms if term not in omitted)
        started = time.time()
        candidate = brick_apply(
            operator,
            vector,
            group_chunk=group_chunk,
            b_low_terms=b_low_terms,
            a_low_terms=retained,
            term_order=term_order,
        ).astype(np.float64)
        metrics = _metrics(candidate, reference)
        record = {
            "stage": "exact_validation",
            "rank": len(validated),
            "omitted_pair_count": omitted_pair_count,
            "omitted_term_count": len(omitted),
            "omitted_terms": list(omitted),
            "retained_terms": list(retained),
            "predicted_l2_relative": predicted,
            "seconds": time.time() - started,
            "pass_l2_limit": metrics["l2_relative"] <= l2_limit,
            **metrics,
        }
        validated.append(record)
        _emit(record)
        if len(validated) >= validate_top:
            break
    return validated


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine", type=Path, required=True)
    parser.add_argument("--brick", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=351)
    parser.add_argument(
        "--term-order",
        choices=("natural", "center_first", "reverse"),
        default="reverse",
    )
    parser.add_argument(
        "--retained-b-low-corner-offsets",
        default="6,20",
        help="corner offsets added to the noncorner b-low mask",
    )
    parser.add_argument("--l2-limit", type=float, default=1.8e-5)
    parser.add_argument("--validate-top", type=int, default=8)
    parser.add_argument("--group-chunk", type=int, default=8)
    args = parser.parse_args(argv)
    offsets = tuple(
        int(item.strip())
        for item in args.retained_b_low_corner_offsets.split(",")
        if item.strip()
    )
    validated = search(
        args.fine,
        args.brick,
        seed=args.seed,
        term_order=args.term_order,
        retained_b_low_corner_offsets=offsets,
        l2_limit=args.l2_limit,
        validate_top=args.validate_top,
        group_chunk=args.group_chunk,
    )
    return 0 if any(record["pass_l2_limit"] for record in validated) else 1


if __name__ == "__main__":
    raise SystemExit(main())
