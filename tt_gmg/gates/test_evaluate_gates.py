import copy
import json
import tempfile
import unittest
from pathlib import Path

from evaluate_gates import evaluate


HERE = Path(__file__).resolve().parent


class EvaluateGatesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads((HERE / "gate_contract.json").read_text(encoding="utf-8"))
        cls.evidence = json.loads((HERE / "current_evidence.json").read_text(encoding="utf-8"))

    def test_current_status_is_fail_closed(self):
        report = evaluate(self.contract, self.evidence)
        self.assertEqual(report["performance_score"], {"green": 2, "total": 8})
        by_id = {item["id"]: item for item in report["performance_gates"]}
        self.assertEqual(by_id["G1"]["status"], "red")
        self.assertEqual(by_id["G2"]["status"], "red")
        self.assertEqual(by_id["G5"]["status"], "green")
        self.assertEqual(by_id["G3"]["status"], "green")
        self.assertAlmostEqual(by_id["G3"]["median"], 0.001858679)
        self.assertFalse(report["acceptance_complete"])

    def test_projection_cannot_close_g3(self):
        evidence = copy.deepcopy(self.evidence)
        evidence["measurements"]["G3"] = {
            "samples": [0.0028, 0.0028, 0.0028],
            "tags": ["real_row236", "real_8x_wormhole", "measured", "durable"],
            "evidence": "diagnostic.log",
        }
        report = evaluate(self.contract, evidence)
        g3 = next(item for item in report["performance_gates"] if item["id"] == "G3")
        self.assertEqual(g3["status"], "open")
        self.assertTrue(any("missing evidence tags" in blocker for blocker in g3["blockers"]))

    def test_all_performance_gates_need_full_acceptance_invariants(self):
        evidence = copy.deepcopy(self.evidence)
        for gate in self.contract["performance_gates"]:
            evidence["measurements"][gate["id"]] = {
                "samples": [0.0] * gate["minimum_samples"],
                "tags": gate["required_tags"],
                "evidence": "measured.json",
            }
        report = evaluate(self.contract, evidence)
        self.assertEqual(report["performance_score"], {"green": 8, "total": 8})
        self.assertFalse(report["acceptance_complete"])
        evidence["correctness"]["full_ccx"] = {"status": "green"}
        evidence["correctness"]["safety_path"] = {"status": "green"}
        self.assertTrue(evaluate(self.contract, evidence)["acceptance_complete"])


if __name__ == "__main__":
    unittest.main()
