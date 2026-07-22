from __future__ import annotations

import re
import unittest
from pathlib import Path

import numpy as np

from canonical_packed_layout import _build_palette


STENCIL_DIR = Path(__file__).resolve().parent
FORK_DIR = STENCIL_DIR.parents[1]
SIGNATURE = (
    FORK_DIR
    / "tt_gmg/evidence/device_v1/canonical_base_v1"
    / "canonical_base_signature_q1e7.i32"
)
SCHEDULE_HEADER = STENCIL_DIR / "kernels/canonical_base_palette_schedule.hpp"


def _header_schedule() -> np.ndarray:
    text = SCHEDULE_HEADER.read_text(encoding="utf-8")
    rows: list[tuple[int, list[int]]] = []
    pattern = re.compile(
        r"\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\},\s*// term (\d+)"
    )
    for match in pattern.finditer(text):
        rows.append(
            (
                int(match.group(4)),
                [int(match.group(1)), int(match.group(2)), int(match.group(3))],
            )
        )
    if [term for term, _ in rows] != list(range(81)):
        raise AssertionError("header does not contain one ordered row per term")
    return np.asarray([row for _, row in rows], dtype=np.uint8)


def _retains_blow(term: int) -> bool:
    offset = term // 3
    di, remainder = divmod(offset, 9)
    dj, dk = divmod(remainder, 3)
    manhattan = abs(di - 1) + abs(dj - 1) + abs(dk - 1)
    return manhattan <= 2 or offset in (6, 20)


class CanonicalPackedKernelContractTests(unittest.TestCase):
    def test_header_is_exact_signature_schedule(self) -> None:
        signature = np.fromfile(SIGNATURE, dtype="<i4")
        palette, expected, _ = _build_palette(signature)
        actual = _header_schedule()
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(palette.tolist(), [
            -59656973,
            -22371364,
            -15361670,
            -5592841,
            -3952274,
            0,
            5592841,
            7009694,
            22371364,
            29828486,
            126472778,
        ])

    def test_sparse_product_and_core_budget(self) -> None:
        schedule = _header_schedule()
        nonzero = int(np.count_nonzero(schedule != 5))
        products = sum(
            0 if int(schedule[term, output]) == 5
            else 6 if _retains_blow(term)
            else 5
            for term in range(81)
            for output in range(3)
        )
        self.assertEqual(nonzero, 153)
        self.assertEqual(products, 864)

        base_tiles = 1064
        fallback_tiles = 197
        dense_products = 8 * 198 * 1404
        packed_products = base_tiles * products + fallback_tiles * 1404
        self.assertEqual(dense_products, 2_223_936)
        self.assertEqual(packed_products, 1_195_884)
        self.assertGreater(1.0 - packed_products / dense_products, 0.46)

        # A 42/14 split maps cleanly onto the 8x7 Wormhole worker grid.
        self.assertEqual((133 + 42 - 1) // 42, 4)
        self.assertEqual((25 + 14 - 1) // 14, 2)
        self.assertEqual(max(4 * products, 2 * 1404), 3456)


if __name__ == "__main__":
    unittest.main()
