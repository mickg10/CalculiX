import tempfile
import unittest
from pathlib import Path

import numpy as np

from brick_format import (
    BrickOperator,
    OFFSETS,
    _bf16_fidelity_phases,
    _bf16_hifi2_exact_product_phases,
    _bf16_bits,
    _bf16_values,
    open_fine_dump,
    serialize,
    verify,
)


def write_synthetic_dump(path: Path) -> None:
    coords = np.asarray([(i, j, k) for i in range(5) for j in range(3) for k in range(2)], dtype="<i8")
    nb = len(coords)
    lookup = {tuple(int(v) for v in coord): index for index, coord in enumerate(coords)}
    ptr = [0]
    cols = []
    blocks = []
    for row, coord in enumerate(coords):
        for oid, offset in enumerate(OFFSETS):
            target = tuple(int(v) for v in coord + offset)
            if target not in lookup:
                continue
            col = lookup[target]
            cols.append(col)
            block = np.empty((3, 3), dtype=np.float64)
            for r in range(3):
                for c in range(3):
                    block[r, c] = (1.0 if row == col and r == c else 0.0) + (oid + 1) * 1.0e-3 + r * 1.0e-4 + c * 1.0e-5
            blocks.append(block.reshape(-1))
        ptr.append(len(cols))
    header = np.asarray((nb, len(cols), 5, 3, 2), dtype="<i8")
    with path.open("wb") as handle:
        header.tofile(handle)
        np.asarray(ptr, dtype="<i8").tofile(handle)
        np.asarray(cols, dtype="<i4").tofile(handle)
        np.asarray(blocks, dtype="<f8").tofile(handle)
        coords.tofile(handle)


class BrickFormatTest(unittest.TestCase):
    def test_bf16_fidelity_phase_decomposition(self):
        a = np.array([1.9921875, -0.3359375], dtype=np.float32)
        b = np.array([1.0078125, 0.6796875], dtype=np.float32)
        hifi4 = sum(_bf16_fidelity_phases(a, b, 4))
        hifi3 = sum(_bf16_fidelity_phases(a, b, 3))
        hifi2 = sum(_bf16_fidelity_phases(a, b, 2))
        np.testing.assert_array_equal(hifi4, a * b)
        self.assertGreater(float(np.max(np.abs(hifi3 - hifi4))), 0.0)
        self.assertGreaterEqual(
            float(np.max(np.abs(hifi2 - hifi4))),
            float(np.max(np.abs(hifi3 - hifi4))),
        )

    def test_fixed_hifi2_reconstructs_exact_bf16_product(self):
        rng = np.random.default_rng(351)
        a = _bf16_values(_bf16_bits(rng.standard_normal(100_000).astype(np.float32)))
        b = _bf16_values(_bf16_bits(rng.standard_normal(100_000).astype(np.float32)))
        reconstructed = np.zeros_like(a)
        for phase in _bf16_hifi2_exact_product_phases(a, b):
            reconstructed = reconstructed + phase
        np.testing.assert_array_equal(reconstructed, a * b)

    def test_roundtrip_and_oracle_cross_brick_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fine = root / "fine.bin"
            brick = root / "brick.bin"
            manifest = root / "brick.json"
            write_synthetic_dump(fine)
            result = serialize(fine, brick, manifest, hash_files=True)
            self.assertEqual(result["brick"]["count"], 2)
            self.assertEqual(result["brick"]["padded_count"], 16)
            self.assertTrue(brick.exists())
            operator = BrickOperator(brick)
            self.assertEqual(operator.nbrick, 2)
            self.assertEqual(operator.nbrick_pad, 16)
            self.assertEqual(operator.section("coefficients").shape, (3, 3, 81, 1024))
            report = verify(fine, brick, vector_names=("random", "smooth", "boundary", "constant"), group_chunk=1)
            self.assertTrue(report["metadata"]["pass"], report)
            self.assertTrue(report["pass"], report)
            report_x2 = verify(
                fine,
                brick,
                vector_names=("random", "smooth", "boundary", "constant"),
                group_chunk=1,
                b_splits=2,
            )
            self.assertEqual(report_x2["device_mac"]["cross_terms"], 5)
            self.assertTrue(report_x2["pass"], report_x2)
            report_mixed = verify(
                fine,
                brick,
                vector_names=("random", "smooth", "boundary", "constant"),
                group_chunk=1,
                b_low_terms=tuple(range(81)),
                term_order="reverse",
            )
            self.assertEqual(report_mixed["device_mac"]["b_low_terms"], list(range(81)))
            self.assertIsNone(report_mixed["device_mac"]["a_low_terms"])
            self.assertEqual(report_mixed["device_mac"]["cross_terms"], 6)
            self.assertTrue(report_mixed["pass"], report_mixed)
            report_mixed_a = verify(
                fine,
                brick,
                vector_names=("random", "smooth", "boundary", "constant"),
                group_chunk=1,
                a_low_terms=tuple(range(81)),
                term_order="reverse",
            )
            self.assertEqual(
                report_mixed_a["device_mac"]["a_low_terms"],
                list(range(81)),
            )
            self.assertIsNone(report_mixed_a["device_mac"]["b_low_terms"])
            self.assertEqual(report_mixed_a["device_mac"]["cross_terms"], 6)
            self.assertTrue(report_mixed_a["pass"], report_mixed_a)
            noncorner_terms = tuple(
                3 * oid + component
                for oid, offset in enumerate(OFFSETS)
                if sum(abs(int(value)) for value in offset) <= 2
                for component in range(3)
            )
            report_affine_fold = verify(
                fine,
                brick,
                vector_names=("random", "smooth", "boundary", "constant"),
                group_chunk=1,
                a_low_terms=noncorner_terms,
                a_low_affine_corner_fold=True,
                term_order="reverse",
            )
            self.assertTrue(
                report_affine_fold["device_mac"]["a_low_affine_corner_fold"]
            )
            self.assertEqual(
                report_affine_fold["device_mac"]["cross_terms"],
                "mixed_low_products",
            )
            self.assertTrue(report_affine_fold["pass"], report_affine_fold)
            report_fidelity_batch = verify(
                fine,
                brick,
                vector_names=("random", "smooth", "boundary", "constant"),
                group_chunk=1,
                term_order="reverse",
                hifi2_products=((0, 2), (1, 1), (2, 0)),
                fidelity_batch_terms=3,
            )
            self.assertEqual(
                report_fidelity_batch["device_mac"]["fidelity_batch_terms"],
                3,
            )
            self.assertTrue(report_fidelity_batch["pass"], report_fidelity_batch)
            report_fixed_hifi2 = verify(
                fine,
                brick,
                vector_names=("random", "smooth", "boundary", "constant"),
                group_chunk=1,
                term_order="reverse",
                fixed_hifi2_exact_leading=True,
            )
            self.assertTrue(
                report_fixed_hifi2["device_mac"]["fixed_hifi2_exact_leading"]
            )
            self.assertEqual(
                report_fixed_hifi2["device_mac"][
                    "fixed_hifi2_fidelity_phases_per_compensated_term"
                ],
                18,
            )
            self.assertTrue(report_fixed_hifi2["pass"], report_fixed_hifi2)
            report_center_first = verify(
                fine,
                brick,
                vector_names=("random", "smooth", "boundary", "constant"),
                group_chunk=1,
                term_order="center_first",
            )
            self.assertEqual(
                report_center_first["device_mac"]["term_order"],
                "center_first",
            )
            self.assertTrue(report_center_first["pass"], report_center_first)
            report_reordered_products = verify(
                fine,
                brick,
                vector_names=("random", "smooth", "boundary", "constant"),
                group_chunk=1,
                term_order="reverse",
                compensated_product_order=(
                    (0, 0),
                    (1, 0),
                    (2, 0),
                    (0, 1),
                    (1, 1),
                    (0, 2),
                ),
                compensated_product_order_no_blow=(
                    (0, 0),
                    (1, 0),
                    (2, 0),
                    (0, 1),
                    (1, 1),
                ),
            )
            self.assertTrue(
                report_reordered_products["pass"],
                report_reordered_products,
            )
            report_quantized_coefficients = verify(
                fine,
                brick,
                vector_names=("random", "smooth", "boundary", "constant"),
                group_chunk=1,
                term_order="reverse",
                coefficient_quantum=1.0e-7,
            )
            self.assertEqual(
                report_quantized_coefficients["device_mac"][
                    "coefficient_quantum"
                ],
                1.0e-7,
            )
            self.assertTrue(
                report_quantized_coefficients["pass"],
                report_quantized_coefficients,
            )
            report_biased = verify(
                fine,
                brick,
                vector_names=("random", "smooth", "boundary", "constant"),
                group_chunk=1,
                accumulator_bias=16.0,
            )
            self.assertEqual(report_biased["device_mac"]["accumulator_bias"], 16.0)
            self.assertTrue(report_biased["pass"], report_biased)
            self.assertTrue(
                all(
                    item["accumulator_prefix_extrema"]["minimum"] > 0.0
                    for item in report_biased["vectors"]
                ),
                report_biased,
            )
            report_chunked = verify(
                fine,
                brick,
                vector_names=("random", "smooth", "boundary", "constant"),
                group_chunk=1,
                accumulator_bias=4.0,
                accumulator_chunk_terms=9,
            )
            self.assertEqual(report_chunked["device_mac"]["accumulator_bias"], 4.0)
            self.assertEqual(report_chunked["device_mac"]["accumulator_chunk_terms"], 9)
            self.assertTrue(report_chunked["pass"], report_chunked)
            self.assertTrue(
                all(
                    item["accumulator_prefix_extrema"]["minimum"] > 0.0
                    for item in report_chunked["vectors"]
                ),
                report_chunked,
            )
            report_scaled_chunked = verify(
                fine,
                brick,
                vector_names=("random", "smooth", "boundary", "constant"),
                group_chunk=1,
                accumulator_bias_scale=4.0,
                accumulator_chunk_terms=9,
            )
            self.assertEqual(
                report_scaled_chunked["device_mac"]["accumulator_bias_scale"],
                4.0,
            )
            self.assertTrue(report_scaled_chunked["pass"], report_scaled_chunked)
            self.assertTrue(
                all(
                    item["accumulator_bias"] >= 4.0 * item["vector_max_absolute"]
                    for item in report_scaled_chunked["vectors"]
                ),
                report_scaled_chunked,
            )

    def test_rejects_non_geometric_block(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fine = root / "fine.bin"
            write_synthetic_dump(fine)
            dump = open_fine_dump(fine)
            # Change one column to a node more than one lattice step away.
            col_offset = 40 + 8 * (dump.nb + 1)
            col = np.memmap(fine, dtype="<i4", mode="r+", offset=col_offset, shape=(dump.nblk,))
            col[0] = dump.nb - 1
            col.flush()
            with self.assertRaisesRegex(ValueError, "non-27-point"):
                serialize(fine, root / "brick.bin", hash_files=False)


if __name__ == "__main__":
    unittest.main()
