#!/usr/bin/env python3
"""Static regression checks for the guarded TT hardware runners."""

from __future__ import annotations

import pathlib
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parent
RUNNERS = (
    ROOT / "run_brick_hw.sh",
    ROOT / "run_canonical_packed_hw.sh",
)


class RunnerSafetyContractTests(unittest.TestCase):
    def test_shell_syntax_and_global_idle_contract(self) -> None:
        for runner in RUNNERS:
            with self.subTest(runner=runner.name):
                subprocess.run(["bash", "-n", str(runner)], check=True)
                text = runner.read_text()

                self.assertIn("global_portal_state()", text)
                self.assertIn("controller_runs_nonterminal", text)
                self.assertIn("controller_jobs_nonterminal", text)
                self.assertIn("portal_submissions_nonterminal", text)
                self.assertIn("pre_stop_global_portal_state", text)
                self.assertIn("post_stop_global_portal_state", text)

                # Session-scoped /api/jobs remains a useful availability check,
                # but it must never be the only proof that the shared portal is idle.
                self.assertIn('"${PORTAL_URL}/api/jobs"', text)
                self.assertIn("TT_FOLD_CONTROLLER_DB", text)
                self.assertIn("TT_FOLD_PORTAL_DB", text)

    def test_service_restore_requires_workers_and_real_holders(self) -> None:
        for runner in RUNNERS:
            with self.subTest(runner=runner.name):
                text = runner.read_text()
                self.assertIn("TT_FOLD_EXPECTED_ONLINE_WORKERS", text)
                self.assertIn("TT_FOLD_EXPECTED_DEVICE_HOLDERS", text)
                self.assertIn("online_workers=${online}", text)
                self.assertIn("device_holders=${holders}", text)
                self.assertIn("restoration incomplete", text)


if __name__ == "__main__":
    unittest.main()
