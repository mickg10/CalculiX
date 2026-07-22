import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
KERNELS = ROOT / "kernels"


class CanonicalDynamicSourceContractTest(unittest.TestCase):
    def test_common_header_pins_page_and_publish_contract(self):
        source = (KERNELS / "canonical_dynamic_page_common.hpp").read_text(
            encoding="utf-8"
        )
        expected_constants = {
            "kOffsets": "27",
            "kComponents": "3",
            "kTermsPerPublish": "3",
            "kTileBytes": "2048",
            "kVectorPagesPerComponent": "180",
            "kMaxGroupPages": "62",
            "kPageTableBytes": "64",
            "kPageCountByte": "63",
            "kLaneSentinel": "65535",
        }
        for name, value in expected_constants.items():
            self.assertRegex(
                source,
                rf"constexpr\s+uint(?:16|32)_t\s+{name}\s*=\s*{value}\s*;",
            )
        self.assertIn("stream_term / kOffsets", source)
        self.assertIn("kOffsets - 1 - stream_term % kOffsets", source)
        self.assertIn("reverse_offset * kComponents + component", source)
        self.assertIn("low_publish_count", source)
        self.assertIn("low_count == 0 ? 0 : kTermsPerPublish", source)

    def test_dynamic_reader_files_use_the_same_staging_contract(self):
        paths = [
            KERNELS / "brick_reader_canonical_dynamic_base_bhi.cpp",
            KERNELS
            / "brick_reader_canonical_dynamic_base_bmid_blow_writer.cpp",
            KERNELS / "brick_reader_canonical_dynamic_fallback_bhi.cpp",
            KERNELS
            / "brick_reader_canonical_dynamic_fallback_bmid_blow_writer.cpp",
        ]
        for path in paths:
            source = path.read_text(encoding="utf-8")
            self.assertIn("canonical_dynamic_page_common.hpp", source)
            self.assertIn("kPageCountByte", source)
            self.assertIn("kVectorPagesPerComponent", source)
            self.assertIn("ordered_term(stream_term)", source)
            self.assertIn("gather_b_tile(", source)
            # Run67 proved that a staging CB cannot be used as a variable-count
            # ring: push/pop sequences that do not complete an exact CB-size
            # cycle eventually cross the physical FIFO end.  TT-Metal requires
            # get_write_ptr to follow a reservation, so each dedicated staging
            # allocation is reserved once at full capacity and never pushed.
            stage_cbs = re.findall(
                r"constexpr\s+uint32_t\s+(cb_stage(?:_mid|_low)?)\s*=",
                source,
            )
            self.assertGreaterEqual(len(stage_cbs), 1)
            for stage_cb in stage_cbs:
                full_reservation = (
                    f"cb_reserve_back({stage_cb}, "
                    "canonical_dynamic_page::kMaxGroupPages);"
                )
                self.assertEqual(source.count(full_reservation), 1)
                self.assertLess(
                    source.index(full_reservation),
                    source.index(f"get_write_ptr({stage_cb})"),
                )
            self.assertNotRegex(
                source,
                r"cb_(?:push_back|wait_front|pop_front)\(\s*"
                r"cb_stage(?:_mid|_low)?\b",
            )
            self.assertRegex(
                source,
                r"(?:stage|mid_stage|low_stage)_l1\s*=\s*get_write_ptr\(cb_stage",
            )

        base_reader = paths[0].read_text(encoding="utf-8")
        base_writer = paths[1].read_text(encoding="utf-8")
        fallback_reader = paths[2].read_text(encoding="utf-8")
        fallback_writer = paths[3].read_text(encoding="utf-8")
        self.assertIn("TensorAccessorArgs<5>()", base_reader)
        self.assertIn("TensorAccessorArgs<7>()", base_writer)
        self.assertIn("TensorAccessorArgs<5>()", fallback_reader)
        self.assertIn("TensorAccessorArgs<8>()", fallback_writer)
        self.assertIn("constexpr uint32_t cb_a_hi_mid", fallback_reader)
        self.assertIn("constexpr uint32_t cb_a_low", fallback_writer)
        for source in (base_writer, fallback_writer):
            self.assertIn("low_publish_count(low_count)", source)
            self.assertIn("cb_push_back(cb_blow, low_publish_count)", source)

    def test_mode3_low_split_fifo_cycles_are_full_pages(self):
        def retains_blow(offset):
            di = offset // 9
            dj = (offset // 3) % 3
            dk = offset % 3
            manhattan = abs(di - 1) + abs(dj - 1) + abs(dk - 1)
            return manhattan <= 2 or offset in (6, 20)

        compact_counts = []
        for _component in range(3):
            for offset_batch in range(0, 27, 3):
                offsets = [26 - offset_batch - lane for lane in range(3)]
                compact_counts.append(sum(retains_blow(offset) for offset in offsets))
        self.assertEqual(compact_counts[:9], [1, 3, 2, 3, 3, 3, 2, 3, 1])

        cursor = 0
        compact_crossings = 0
        for count in compact_counts:
            compact_crossings += int(cursor + count > 3)
            cursor = (cursor + count) % 3
        self.assertGreater(compact_crossings, 0)

        padded_counts = [0 if count == 0 else 3 for count in compact_counts]
        cursor = 0
        for count in padded_counts:
            self.assertLessEqual(cursor + count, 3)
            cursor = (cursor + count) % 3
        self.assertEqual(cursor, 0)

    def test_compute_sources_accept_component_major_runtime_mode(self):
        for filename in (
            "brick_compute_canonical_packed_base.cpp",
            "brick_compute_bf16x3_preexpanded_b.cpp",
        ):
            source = (KERNELS / filename).read_text(encoding="utf-8")
            self.assertIn("if (order_mode == 3)", source)
            self.assertIn("stream_term / 27", source)
            self.assertIn("26 - stream_term % 27", source)
            self.assertIn("reverse_offset * 3 + component", source)

    def test_offline_compiler_exposes_both_dynamic_role_sets(self):
        source = (ROOT / "offline_compile_affine_roles.cpp").read_text(
            encoding="utf-8"
        )
        for mode in (
            "--canonical-dynamic-base",
            "--canonical-dynamic-fallback",
        ):
            self.assertGreaterEqual(source.count(mode), 2)
        for stem in (
            "brick_reader_canonical_dynamic_base_bhi_offline/",
            "brick_reader_canonical_dynamic_base_bmid_blow_writer_offline/",
            "brick_compute_canonical_dynamic_base_offline/",
            "brick_reader_canonical_dynamic_fallback_bhi_offline/",
            "brick_reader_canonical_dynamic_fallback_bmid_blow_writer_offline/",
            "brick_compute_canonical_dynamic_fallback_offline/",
        ):
            self.assertIn(stem, source)
        self.assertRegex(
            source,
            re.escape("canonical_packed_fallback || canonical_dynamic_fallback"),
        )

    def test_standalone_harness_is_default_off_and_fail_closed(self):
        source = (ROOT / "canonical_dynamic_spmv.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("CANONICAL_DYNAMIC_HOST_PREFLIGHT_ONLY", source)
        self.assertLess(
            source.index("CANONICAL_DYNAMIC_HOST_PREFLIGHT_ONLY"),
            source.index("MeshDevice::create"),
        )
        self.assertIn("audit_device_plan(", source)
        self.assertIn("kExpectedValidLaneReferences = 33'553'154", source)
        self.assertIn("kExpectedAbsentLaneReferences = 1'393'918", source)
        self.assertIn("kExpectedGroupVectorPages = 18'673", source)
        self.assertIn("kTermOrderMode = 3", source)
        self.assertIn("base_cb_bytes()", source)
        self.assertIn("fallback_cb_bytes()", source)
        self.assertIn("CANONICAL_DYNAMIC_RING_CONTRACT", source)
        self.assertIn("stage_full_reservation_pages=%u", source)
        self.assertIn("stage_fifo_pushes=0", source)
        self.assertIn("mode3_fifo_wrap_safe=1", source)
        cmake = (ROOT / "tt_metal_CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("metal_example_canonical_dynamic_spmv", cmake)

    def test_hardware_runner_preserves_global_and_boot_guards(self):
        source = (ROOT / "run_canonical_dynamic_hw.sh").read_text(
            encoding="utf-8"
        )
        for required in (
            "global_portal_state()",
            "controller_runs_nonterminal",
            "portal_submissions_nonterminal",
            ".last_brick_workload_boot_id",
            "fresh_boot_required",
            "pre_stop_interactive_users",
            "post_settle_device_holders",
            "trap restart_fold EXIT",
            "CANONICAL_DYNAMIC_HOST_PREFLIGHT_ONLY=1",
            "metal_example_canonical_dynamic_spmv",
        ):
            self.assertIn(required, source)
        self.assertNotIn("tt-smi", source)
        self.assertLess(
            source.index("CANONICAL_DYNAMIC_HOST_PREFLIGHT_ONLY=1"),
            source.index("sudo -n systemctl stop tt-fold"),
        )

    def test_bmc_cycle_script_is_strict_and_single_path(self):
        source = (ROOT / "bmc_cycle_if_strict_idle.sh").read_text(
            encoding="utf-8"
        )
        for required in (
            "global_portal_state()",
            "session_jobs_empty()",
            "systemctl list-jobs",
            "interactive_users",
            "dstate_tasks",
            "conflicting_workload",
            "holder_ownership=tt-fold-only",
            "/system.slice/tt-fold.service",
            "post_stop_holders",
            "cycle_not_acknowledged restarting_tt_fold=1",
            "sudo -n ipmitool chassis power cycle",
            "command_acknowledged=1",
        ):
            self.assertIn(required, source)
        self.assertNotIn("tt-smi", source)
        self.assertEqual(source.count("ipmitool chassis power cycle"), 1)
        self.assertLess(
            source.index("post_stop_holders"),
            source.index("ipmitool chassis power cycle"),
        )


if __name__ == "__main__":
    unittest.main()
