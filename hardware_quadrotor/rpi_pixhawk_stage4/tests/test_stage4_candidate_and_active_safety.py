from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path, PurePosixPath, PureWindowsPath
from unittest import mock

from test_stage4_safety import healthy


class Stage4CandidateCompatibilityTests(unittest.TestCase):
    def test_default_candidate_is_self_contained_deployment_config(self):
        module = importlib.import_module("rpi_pixhawk_stage4.controller.candidate_adapter")
        resolved = module.resolve_candidate_path()
        expected = (
            module.PROJECT_ROOT
            / "rpi_pixhawk_stage4"
            / "config"
            / "best_candidate_config.json"
        ).resolve()
        self.assertEqual(resolved, expected)
        config = module.load_candidate_config()
        self.assertEqual(config["candidate_id"], "candidate_097")
        self.assertEqual(
            config["interface_defaults"]["interface_version"],
            "candidate097_outer_loop_behavioral_equivalence_v1",
        )

    def test_self_contained_config_loads_without_stage1_5_tree(self):
        module = importlib.import_module("rpi_pixhawk_stage4.controller.candidate_adapter")
        source = (
            module.PROJECT_ROOT
            / "rpi_pixhawk_stage4"
            / "config"
            / "best_candidate_config.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            isolated_root = Path(directory)
            destination = (
                isolated_root
                / "rpi_pixhawk_stage4"
                / "config"
                / "best_candidate_config.json"
            )
            destination.parent.mkdir(parents=True)
            destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            with mock.patch.object(module, "PROJECT_ROOT", isolated_root):
                config = module.load_candidate_config()
        self.assertEqual(config["candidate_id"], "candidate_097")
        self.assertEqual(
            config["interface_defaults_source"],
            "sitl_outer_loop_stage/python_controller/controller_interface_defaults.json",
        )
    def test_legacy_windows_and_wsl_paths_migrate_to_deployment_config(self):
        module = importlib.import_module("rpi_pixhawk_stage4.controller.candidate_adapter")
        expected = (
            module.PROJECT_ROOT
            / "rpi_pixhawk_stage4"
            / "config"
            / "best_candidate_config.json"
        ).resolve()
        legacy_windows = (
            r"C:\Users\different-user\Desktop\TinyMPC-style lightweight linear MPC"
            r"\TinyMPC-style lightweight linear MPC"
            + "\u65b0\u5efa\u6587\u4ef6\u5939"
            + r"\results_f450_auto_tuning\best_candidate_config.json"
        )
        legacy_wsl = (
            "/mnt/c/Users/different-user/Desktop/TinyMPC-style lightweight linear MPC/"
            "TinyMPC-style lightweight linear MPC/"
            "\u65b0\u5efa\u6587\u4ef6\u5939"
            "/results_f450_auto_tuning/best_candidate_config.json"
        )
        self.assertEqual(module.resolve_candidate_path(legacy_windows), expected)
        self.assertEqual(module.resolve_candidate_path(legacy_wsl), expected)

    def test_linux_project_relative_deployment_path_loads(self):
        module = importlib.import_module("rpi_pixhawk_stage4.controller.candidate_adapter")
        config = module.load_candidate_config(
            "rpi_pixhawk_stage4/config/best_candidate_config.json"
        )
        self.assertEqual(config["candidate_id"], "candidate_097")
        self.assertEqual(config["k_i_z"], 1.5)
        self.assertEqual(config["hover_bias_abs_limit"], 6)
        self.assertEqual(config["k_i_xy"], 0.1)
        self.assertEqual(config["xy_bias_abs_limit"], 0.2)

    def test_incomplete_interface_defaults_raise_explicit_schema_error(self):
        module = importlib.import_module("rpi_pixhawk_stage4.controller.candidate_adapter")
        payload = {
            "candidate_id": "candidate_097",
            "k_i_z": 1.5,
            "hover_bias_abs_limit": 6,
            "k_i_xy": 0.1,
            "xy_bias_abs_limit": 0.2,
            "interface_defaults": {
                "interface_version": "candidate097_outer_loop_behavioral_equivalence_v1",
                "horizontal_position_gain_per_s": 0.6,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "candidate_schema_error:missing_interface_defaults_fields:",
            ):
                module.load_candidate_config(str(path))

    def test_sparse_candidate_merges_validated_authoritative_defaults(self):
        module = importlib.import_module("rpi_pixhawk_stage4.controller.candidate_adapter")
        payload = {
            "candidate_id": "candidate_097",
            "k_i_z": 1.5,
            "hover_bias_abs_limit": 6,
            "k_i_xy": 0.1,
            "xy_bias_abs_limit": 0.2,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            config = module.load_candidate_config(str(path))
        defaults_path = (
            module.PROJECT_ROOT
            / "sitl_outer_loop_stage"
            / "python_controller"
            / "controller_interface_defaults.json"
        )
        expected = json.loads(defaults_path.read_text(encoding="utf-8"))
        self.assertEqual(config["interface_defaults"], expected)
        self.assertEqual(
            config["interface_defaults_source"],
            "sitl_outer_loop_stage/python_controller/controller_interface_defaults.json",
        )

    def test_missing_candidate_field_raises_explicit_schema_error(self):
        module = importlib.import_module("rpi_pixhawk_stage4.controller.candidate_adapter")
        payload = {
            "candidate_id": "candidate_097",
            "k_i_z": 1.5,
            "hover_bias_abs_limit": 6,
            "k_i_xy": 0.1,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "candidate_schema_error:missing_candidate_fields:xy_bias_abs_limit",
            ):
                module.load_candidate_config(str(path))
    def test_stage4_candidate_matches_frozen_stage2_behavior(self):
        stage4 = importlib.import_module("rpi_pixhawk_stage4.controller.candidate_adapter")
        frozen = importlib.import_module("sitl_guided_stage2.controller.candidate097_velocity")
        adapter = stage4.Candidate097Adapter()
        reference = {"north_m": 1.08, "east_m": 1.96, "down_m": -0.8, "yaw_rad": 0.05}
        state = {"north_m": 1.0, "east_m": 2.0, "down_m": -0.75, "yaw_rad": 0.01}
        expected = frozen.Candidate097VelocityController(adapter.config).step(0.0, state, reference, 0.1)
        actual = adapter.step(state, reference, 0.1)
        for field in ("vx_north_mps", "vy_east_mps", "vz_down_mps", "yaw_rate_radps"):
            self.assertAlmostEqual(actual[field], expected[field], places=12)

    def test_candidate_paths_map_between_windows_wsl_and_relative(self):
        module = importlib.import_module("rpi_pixhawk_stage4.controller.candidate_adapter")
        windows = r"C:\Users\zhang\Desktop\project\candidate.json"
        wsl = "/mnt/c/Users/zhang/Desktop/project/candidate.json"
        self.assertEqual(str(module.portable_path_for_host(windows, "posix")), wsl)
        self.assertEqual(str(module.portable_path_for_host(wsl, "nt")), windows)
        resolved = module.resolve_candidate_path(
            "新建文件夹/results_f450_auto_tuning/best_candidate_config.json"
        )
        self.assertTrue(resolved.is_file())


class Stage4ActiveSafetyTests(unittest.TestCase):
    def test_active_state_rejects_disarm_missing_local_rc_and_stale_rc(self):
        safety = importlib.import_module("rpi_pixhawk_stage4.safety.preflight")
        origin = (1.0, 2.0, -0.8)
        cases = (
            (healthy(armed=False), "vehicle_disarmed"),
            (healthy(local_position_valid=False), "local_position_missing"),
            (healthy(rc_valid=False), "rc_data_missing"),
            (healthy(rc_age_s=0.6), "stale_telemetry"),
        )
        for state, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(safety.validate_active_state(state, origin, None).reason, expected)


if __name__ == "__main__":
    unittest.main()
