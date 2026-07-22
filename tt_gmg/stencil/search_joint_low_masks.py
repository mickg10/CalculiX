#!/usr/bin/env python3
"""Search inversion-symmetric joint A-low/B-low product masks.

Each of the 27 stencil offsets belongs to one of fourteen inversion groups
(thirteen opposite-offset pairs plus the self-opposite center).  For every
group the search may retain both low cross products, omit ``A_low * B_high``,
omit ``A_high * B_low``, or omit both.  A complete smooth-vector oracle builds
the 28 single-action error deltas and their Gram model.  A bounded beam then
ranks joint masks; the exact ordered-FP32 oracle over all requested vectors is
the final authority.

The quadratic model is only a search accelerator.  A mask is never reported
as passing unless the exact brick oracle satisfies both correctness limits on
every validation vector.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from brick_format import BrickOperator, _vectors, bcsr_apply, brick_apply, open_fine_dump


@dataclass(frozen=True)
class State:
    action_mask: int
    omitted_terms: int
    predicted_norm_sq: float


def inversion_groups() -> tuple[tuple[int, ...], ...]:
    groups: list[tuple[int, ...]] = []
    for offset in range(27):
        opposite = 26 - offset
        if offset > opposite:
            continue
        offsets = (offset,) if offset == opposite else (offset, opposite)
        groups.append(
            tuple(3 * member + component for member in offsets for component in range(3))
        )
    if len(groups) != 14 or sum(len(group) for group in groups) != 81:
        raise AssertionError("unexpected inversion grouping")
    return tuple(groups)


def metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
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


def emit(record: dict[str, Any]) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)


def add_action(
    state: State,
    action: int,
    base_dot: np.ndarray,
    gram: np.ndarray,
    action_term_counts: Sequence[int],
) -> State:
    if state.action_mask & (1 << action):
        return state
    selected = [
        index
        for index in range(len(base_dot))
        if state.action_mask & (1 << index)
    ]
    cross = float(gram[selected, action].sum()) if selected else 0.0
    norm_sq = (
        state.predicted_norm_sq
        + 2.0 * float(base_dot[action])
        + 2.0 * cross
        + float(gram[action, action])
    )
    return State(
        action_mask=state.action_mask | (1 << action),
        omitted_terms=state.omitted_terms + int(action_term_counts[action]),
        predicted_norm_sq=norm_sq,
    )


def beam_search(
    base_norm_sq: float,
    base_dot: np.ndarray,
    gram: np.ndarray,
    groups: Sequence[Sequence[int]],
    *,
    keep_per_omission_count: int,
    relaxed_relative_limit: float,
    reference_norm: float,
) -> list[State]:
    action_term_counts = [len(groups[index // 2]) for index in range(2 * len(groups))]
    states = [State(0, 0, base_norm_sq)]
    relaxed_norm_sq = (relaxed_relative_limit * reference_norm) ** 2
    for group_index in range(len(groups)):
        a_action = 2 * group_index
        b_action = a_action + 1
        expanded: dict[int, State] = {}
        for state in states:
            options = [
                state,
                add_action(state, a_action, base_dot, gram, action_term_counts),
                add_action(state, b_action, base_dot, gram, action_term_counts),
            ]
            options.append(
                add_action(options[1], b_action, base_dot, gram, action_term_counts)
            )
            for option in options:
                if option.predicted_norm_sq > relaxed_norm_sq:
                    continue
                previous = expanded.get(option.action_mask)
                if previous is None or option.predicted_norm_sq < previous.predicted_norm_sq:
                    expanded[option.action_mask] = option

        by_count: dict[int, list[State]] = {}
        for state in expanded.values():
            by_count.setdefault(state.omitted_terms, []).append(state)
        states = []
        for omitted_terms, bucket in by_count.items():
            del omitted_terms
            bucket.sort(key=lambda state: (state.predicted_norm_sq, state.action_mask))
            states.extend(bucket[:keep_per_omission_count])
        emit(
            {
                "stage": "beam",
                "group_index": group_index,
                "state_count": len(states),
                "max_omitted_terms": max(state.omitted_terms for state in states),
            }
        )
    return states


def retained_terms(groups: Sequence[Sequence[int]], action_mask: int, parity: int) -> tuple[int, ...]:
    omitted: set[int] = set()
    for group_index, group in enumerate(groups):
        if action_mask & (1 << (2 * group_index + parity)):
            omitted.update(group)
    return tuple(term for term in range(81) if term not in omitted)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine", type=Path, required=True)
    parser.add_argument("--brick", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=351)
    parser.add_argument("--term-order", default="reverse", choices=("natural", "center_first", "reverse"))
    parser.add_argument("--search-vector", default="smooth")
    parser.add_argument("--validate-vectors", default="random,smooth,boundary,constant")
    parser.add_argument("--l2-limit", type=float, default=2.0e-5)
    parser.add_argument("--max-limit", type=float, default=5.0e-5)
    parser.add_argument("--relaxed-factor", type=float, default=4.0)
    parser.add_argument("--keep-per-omission-count", type=int, default=2000)
    parser.add_argument("--validate-top", type=int, default=24)
    parser.add_argument("--group-chunk", type=int, default=8)
    args = parser.parse_args(argv)

    dump = open_fine_dump(args.fine)
    operator = BrickOperator(args.brick)
    groups = inversion_groups()
    vector = next(
        value
        for name, value in _vectors(dump, (args.search_vector,), args.seed)
        if name == args.search_vector
    )
    started = time.time()
    reference = bcsr_apply(dump, vector)
    emit({"stage": "reference", "seconds": time.time() - started})

    all_terms = tuple(range(81))
    started = time.time()
    base = brick_apply(
        operator,
        vector,
        group_chunk=args.group_chunk,
        a_low_terms=all_terms,
        b_low_terms=all_terms,
        term_order=args.term_order,
    ).astype(np.float64)
    base_error = (base - reference).astype(np.float64)
    emit({"stage": "base", "seconds": time.time() - started, **metrics(base, reference)})

    deltas: list[np.ndarray] = []
    for group_index, group in enumerate(groups):
        for parity, label in ((0, "a_low"), (1, "b_low")):
            retained = tuple(term for term in all_terms if term not in group)
            started = time.time()
            candidate = brick_apply(
                operator,
                vector,
                group_chunk=args.group_chunk,
                a_low_terms=retained if parity == 0 else all_terms,
                b_low_terms=retained if parity == 1 else all_terms,
                term_order=args.term_order,
            ).astype(np.float64)
            deltas.append((candidate - base).astype(np.float32))
            emit(
                {
                    "stage": "single_action",
                    "group_index": group_index,
                    "kind": label,
                    "omitted_terms": list(group),
                    "seconds": time.time() - started,
                    **metrics(candidate, reference),
                }
            )

    started = time.time()
    delta_matrix = np.stack([delta.ravel() for delta in deltas]).astype(np.float64)
    base_flat = base_error.ravel()
    base_dot = delta_matrix @ base_flat
    gram = delta_matrix @ delta_matrix.T
    reference_norm = float(np.linalg.norm(reference.ravel()))
    base_norm_sq = float(np.dot(base_flat, base_flat))
    emit({"stage": "gram", "seconds": time.time() - started})

    states = beam_search(
        base_norm_sq,
        base_dot,
        gram,
        groups,
        keep_per_omission_count=args.keep_per_omission_count,
        relaxed_relative_limit=args.l2_limit * args.relaxed_factor,
        reference_norm=reference_norm,
    )
    eligible = [
        state
        for state in states
        if np.sqrt(max(state.predicted_norm_sq, 0.0)) / reference_norm <= args.l2_limit
    ]
    eligible.sort(key=lambda state: (-state.omitted_terms, state.predicted_norm_sq, state.action_mask))
    emit(
        {
            "stage": "ranked",
            "eligible_count": len(eligible),
            "max_omitted_terms": eligible[0].omitted_terms if eligible else None,
        }
    )

    validation_names = tuple(
        item.strip() for item in args.validate_vectors.split(",") if item.strip()
    )
    validation_vectors = dict(_vectors(dump, validation_names, args.seed))
    validation_references = {
        name: bcsr_apply(dump, validation_vectors[name]) for name in validation_names
    }
    any_pass = False
    for rank, state in enumerate(eligible[: args.validate_top]):
        a_terms = retained_terms(groups, state.action_mask, 0)
        b_terms = retained_terms(groups, state.action_mask, 1)
        vector_results: dict[str, Any] = {}
        passed = True
        started = time.time()
        for name in validation_names:
            candidate = brick_apply(
                operator,
                validation_vectors[name],
                group_chunk=args.group_chunk,
                a_low_terms=a_terms,
                b_low_terms=b_terms,
                term_order=args.term_order,
            )
            result = metrics(candidate, validation_references[name])
            result["pass"] = (
                result["l2_relative"] <= args.l2_limit
                and result["max_relative"] <= args.max_limit
            )
            passed = passed and bool(result["pass"])
            vector_results[name] = result
        any_pass = any_pass or passed
        emit(
            {
                "stage": "exact_validation",
                "rank": rank,
                "pass": passed,
                "seconds": time.time() - started,
                "predicted_l2_relative": float(
                    np.sqrt(max(state.predicted_norm_sq, 0.0)) / reference_norm
                ),
                "omitted_product_terms": state.omitted_terms,
                "retained_a_low_count": len(a_terms),
                "retained_b_low_count": len(b_terms),
                "retained_a_low_terms": list(a_terms),
                "retained_b_low_terms": list(b_terms),
                "vectors": vector_results,
            }
        )
    return 0 if any_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
