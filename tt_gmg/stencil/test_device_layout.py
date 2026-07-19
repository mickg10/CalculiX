import json
import tempfile
import unittest
from pathlib import Path

from brick_format import BrickOperator, serialize
from device_layout import DevicePlan, pack_operator, pack_vector, verify_device_pack
from test_brick_format import write_synthetic_dump


class DeviceLayoutTest(unittest.TestCase):
    def test_partition_is_balanced_and_row236_sized(self):
        plan = DevicePlan(1581, 8, 8, 7)
        self.assertEqual(plan.groups_per_chip, 198)
        self.assertEqual(plan.padded_groups, 1584)
        self.assertEqual(plan.core_group_counts.count(4), 30)
        self.assertEqual(plan.core_group_counts.count(3), 26)
        self.assertEqual(sum(plan.core_group_counts), 198)
        self.assertEqual(plan.report()["resident_halo_kib_per_core"], 576.0)

    def test_synthetic_device_pack_is_bitwise_equal_to_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fine = root / "fine.bin"
            brick = root / "brick.bin"
            device = root / "device"
            write_synthetic_dump(fine)
            serialize(fine, brick, hash_files=False)
            layout = pack_operator(
                brick,
                device,
                chips=2,
                grid_x=1,
                grid_y=1,
                hash_files=False,
            )
            self.assertEqual(layout["partition"]["padded_groups"], 2)
            vector = pack_vector(
                fine,
                brick,
                device / "device_layout.json",
                device,
                vector_name="random",
                seed=351,
                hash_files=False,
            )
            report = verify_device_pack(
                brick,
                device / "device_layout.json",
                device / "vector_random_seed351.json",
            )
            self.assertTrue(report["bitwise_equal_to_reference"], report)
            self.assertTrue(report["pass"], report)
            self.assertLess(vector["reference"]["l2_relative_vs_bcsr"], 2.0e-5)
            operator = BrickOperator(brick)
            self.assertEqual(operator.ngroup, 1)
            stored = json.loads((device / "device_layout.json").read_text())
            self.assertEqual(stored["schema"], "tt_gmg_device_layout_v1")


if __name__ == "__main__":
    unittest.main()
