import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from brick_format import OFFSETS
from canonical_packed_layout import GROUP_BASE, LANES
from canonical_pcg_layout import (
    analyze_neighbor_map,
    build_compact_neighbor_plan,
    build_neighbor_map,
    component_major_ordered_term,
    component_major_stream_order,
    verify_neighbor_map_against_preshifted,
    write_page_reader_vector_bundle,
)


class CanonicalPcgLayoutTest(unittest.TestCase):
    def test_component_major_device_order_is_batched_permutation(self):
        order = component_major_stream_order()
        self.assertEqual(order.tolist(), [
            component_major_ordered_term(term) for term in range(81)
        ])
        self.assertEqual(np.sort(order).tolist(), list(range(81)))
        for batch in order.reshape(-1, 3):
            self.assertTrue(np.all(batch % 3 == batch[0] % 3))
            self.assertEqual(
                np.diff((batch // 3).astype(np.int16)).tolist(), [-1, -1]
            )
        self.assertEqual(order[:3].tolist(), [78, 75, 72])
        self.assertEqual(order[27:30].tolist(), [79, 76, 73])
        self.assertEqual(order[-3:].tolist(), [8, 5, 2])
        with self.assertRaises(ValueError):
            component_major_ordered_term(-1)
        with self.assertRaises(ValueError):
            component_major_ordered_term(81)

    def test_single_full_brick_matches_direct_coordinate_shift(self):
        packed = np.full((1, 1, LANES), -1, dtype="<i4")
        packed[0, 0, :64] = np.arange(64, dtype="<i4")
        inverse = np.full(LANES, -1, dtype="<i4")
        inverse[:64] = np.arange(64, dtype="<i4")
        brick_nbr = np.full((1, 27), -1, dtype="<i4")
        brick_nbr[0, 13] = 0
        slot_to_node = np.arange(64, dtype="<i4").reshape(1, 64)

        mapping = build_neighbor_map(packed, inverse, brick_nbr, slot_to_node)
        for slot in range(64):
            coord = np.asarray((slot // 16, (slot // 4) % 4, slot % 4))
            for offset_id, offset in enumerate(OFFSETS):
                target = coord + offset
                expected = (
                    int((target[0] * 4 + target[1]) * 4 + target[2])
                    if np.all((target >= 0) & (target < 4))
                    else -1
                )
                self.assertEqual(int(mapping[0, 0, offset_id, slot]), expected)
        self.assertTrue(np.all(mapping[0, 0, :, 64:] == -1))

    def test_analysis_reports_local_identity_center_tile(self):
        mapping = np.full((1, 1, 27, LANES), -1, dtype="<i4")
        mapping[0, 0, 13] = np.arange(LANES, dtype="<i4")
        roles = np.asarray([[GROUP_BASE]], dtype=np.uint8)
        report = analyze_neighbor_map(mapping, roles)
        self.assertEqual(report["valid_neighbor_node_references"], LANES)
        self.assertEqual(report["same_chip_neighbor_fraction"], 1.0)
        self.assertEqual(report["valid_gather_tiles"], 1)
        self.assertEqual(report["single_page_affine_lane_tiles"], 1)
        self.assertEqual(report["input_pages_per_gather_tile"]["max"], 1.0)
        self.assertEqual(report["neighbor_map_bytes"], mapping.nbytes)
        self.assertEqual(report["logical_vector_active_lanes"], LANES)
        self.assertEqual(report["remote_halo_unique_input_positions_total"], 0)

    def test_preshifted_vector_bitwise_verification(self):
        mapping = np.full((1, 1, 27, LANES), -1, dtype="<i4")
        for offset in range(27):
            mapping[0, 0, offset] = (
                np.arange(LANES, dtype="<i4") + offset - 13
            ) % LANES

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {}
            for split in range(3):
                logical = np.stack(
                    [
                        np.arange(LANES, dtype=np.uint16)
                        + np.uint16(1000 * split + 100 * component)
                        for component in range(3)
                    ]
                )
                shifted = np.empty((1, 1, 81, LANES), dtype="<u2")
                for offset in range(27):
                    for component in range(3):
                        shifted[0, 0, offset * 3 + component] = logical[
                            component, mapping[0, 0, offset]
                        ]
                path = root / f"packed_b{split}.bf16"
                shifted.tofile(path)
                files[f"packed_b{split}"] = {
                    "path": str(path),
                    "shape": list(shifted.shape),
                }
            manifest = root / "vector.json"
            manifest.write_text(
                json.dumps({"files": files, "vector_tag": "synthetic"}),
                encoding="utf-8",
            )
            report = verify_neighbor_map_against_preshifted(mapping, manifest)
            self.assertTrue(report["passing"])
            self.assertEqual(report["bit_mismatches_by_split"], [0, 0, 0])

    def test_compact_plan_reconstructs_local_and_remote_positions(self):
        mapping = np.full((2, 1, 27, LANES), -1, dtype="<i4")
        mapping[0, 0, 13, :3] = [0, 1, LANES]
        mapping[1, 0, 13, :3] = [LANES, LANES + 1, 0]
        arrays, report = build_compact_neighbor_plan(mapping)
        self.assertTrue(report["passing"])
        json.dumps(report)
        self.assertEqual(report["reconstruction_mismatches"], 0)
        self.assertEqual(report["invalid_index_mismatches"], 0)
        self.assertEqual(report["page_reconstruction_mismatches"], 0)
        self.assertEqual(report["invalid_page_index_mismatches"], 0)
        self.assertEqual(report["device_page_table_mismatches"], 0)
        self.assertTrue(report["device_stream_order"]["is_permutation"])
        self.assertTrue(
            report["device_stream_order"][
                "each_publish_batch_single_component"
            ]
        )
        self.assertTrue(
            report["device_stream_order"][
                "each_publish_batch_descending_offsets"
            ]
        )
        self.assertEqual(report["halo_unique_input_positions_total"], 2)
        self.assertLess(report["compact_plan_bytes"], report["raw_neighbor_map_bytes"])
        self.assertLess(
            report["page_gather_plan_bytes"], report["raw_neighbor_map_bytes"]
        )
        self.assertEqual(report["max_group_vector_pages"], 2)
        self.assertEqual(arrays["halo_offsets"].tolist(), [0, 1, 2])
        self.assertEqual(arrays["halo_packed_position"].tolist(), [LANES, 0])
        self.assertEqual(
            arrays["device_group_vector_page_table"][:, :, 63].tolist(),
            [[2], [2]],
        )
        self.assertEqual(
            arrays["device_group_vector_page_table"][0, 0, :2].tolist(),
            [0, 1],
        )

    def test_empty_group_has_zero_device_page_count(self):
        mapping = np.full((1, 2, 27, LANES), -1, dtype="<i4")
        mapping[0, 0, 13, :2] = [0, 1]
        arrays, report = build_compact_neighbor_plan(mapping)
        self.assertTrue(report["passing"])
        table = arrays["device_group_vector_page_table"]
        self.assertEqual(int(table[0, 1, 63]), 0)
        self.assertTrue(np.all(table[0, 1, :63] == 255))

    def test_page_reader_bundle_replays_local_remote_and_absent_words(self):
        mapping = np.full((2, 1, 27, LANES), -1, dtype="<i4")
        mapping[0, 0, 13, :3] = [0, 1, 2]
        mapping[1, 0, 13, :3] = [LANES, LANES + 1, LANES + 2]
        mapping[0, 0, 12, :3] = [LANES, LANES + 1, 2]
        mapping[1, 0, 14, :3] = [0, LANES + 1, 1]
        arrays, plan_report = build_compact_neighbor_plan(mapping)
        self.assertTrue(plan_report["passing"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {}
            logical_position = np.arange(2 * LANES, dtype=np.uint32)
            for split in range(3):
                logical = np.stack(
                    [
                        (
                            logical_position
                            + np.uint32(1000 * split + 100 * component + 1)
                        ).astype("<u2")
                        for component in range(3)
                    ]
                )
                shifted = np.zeros((2, 1, 81, LANES), dtype="<u2")
                for chip in range(2):
                    for offset in range(27):
                        valid = mapping[chip, 0, offset] >= 0
                        for component in range(3):
                            shifted[chip, 0, offset * 3 + component, valid] = (
                                logical[
                                    component,
                                    mapping[chip, 0, offset, valid],
                                ]
                            )
                path = root / f"packed_b{split}.bf16"
                shifted.tofile(path)
                files[f"packed_b{split}"] = {
                    "path": str(path),
                    "shape": list(shifted.shape),
                }
            manifest = root / "vector.json"
            manifest.write_text(
                json.dumps({"files": files, "vector_tag": "page_synthetic"}),
                encoding="utf-8",
            )
            report = write_page_reader_vector_bundle(
                arrays,
                manifest,
                root / "page_bundle",
                force=False,
            )
            self.assertTrue(report["passing"])
            self.assertEqual(report["bit_mismatches_by_split"], [0, 0, 0])
            self.assertEqual(
                report["verified_bf16_words"],
                3 * 2 * 81 * LANES,
            )
            self.assertEqual(
                report["layout"]["layout"],
                "[chip][component][combined_local_then_halo_page][lane]",
            )
            for split in range(3):
                record = report["files"][f"page_vector_b{split}"]
                self.assertEqual(record["shape"], [2, 3, 2, LANES])
                self.assertTrue(Path(record["path"]).is_file())


if __name__ == "__main__":
    unittest.main()
