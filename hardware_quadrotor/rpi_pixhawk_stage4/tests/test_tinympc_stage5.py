"""Offline numerical and integration tests for the Stage 5 TinyMPC source."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np

from rpi_pixhawk_stage4.controller.tinympc_solver import (
    CondensedTinyMPCSolver,
    TinyMPCSolverError,
)
from rpi_pixhawk_stage4.controller.tinympc_source import (
    TinyMPCSafetyError,
    TinyMPCZeroMotionSource,
)
from rpi_pixhawk_stage4.flight_handover.state_v2 import (
    FlightHandoverV2,
    StaticHoldSource,
)
from rpi_pixhawk_stage4.flight_handover.v2_safe import load_v2_config


ROOT = Path(__file__).resolve().parents[1]
STAGE5_CONFIG = ROOT / "config" / "alpha_stage5_tinympc_zero_motion.json"


def controller_config():
    return {
        "mode": "tinympc_zero_motion",
        "dt_seconds": 0.05,
        "horizon_steps": 10,
        "state_weights": [15.0, 15.0, 25.0, 4.0, 4.0, 6.0],
        "input_weights": [2.0, 2.0, 3.0],
        "acceleration_limits_mps2": {"x": 0.20, "y": 0.20, "z": 0.15},
        "max_iterations": 40,
        "residual_tolerance": 1.0e-6,
        "solve_time_limit_seconds": 0.020,
        "safety": {
            "max_position_error_m": {"x": 0.20, "y": 0.20, "z": 0.15},
            "max_horizontal_velocity_mps": 0.20,
            "max_vertical_velocity_mps": 0.15,
        },
    }


class TestTinyMPCNumerics(unittest.TestCase):
    def setUp(self):
        self.solver = CondensedTinyMPCSolver(controller_config())

    def test_double_integrator_matrices_are_exact(self):
        dt = 0.05
        expected_a = np.block(
            [[np.eye(3), dt * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]]
        )
        expected_b = np.vstack((0.5 * dt * dt * np.eye(3), dt * np.eye(3)))
        np.testing.assert_allclose(self.solver.A, expected_a, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(self.solver.B, expected_b, rtol=0.0, atol=0.0)

    def test_condensed_prediction_matches_direct_rollout(self):
        x0 = np.array([0.1, -0.08, 0.04, 0.02, -0.01, 0.03])
        controls = np.linspace(-0.1, 0.1, 30)
        predicted = self.solver.predict_states(x0, controls)
        direct = []
        state = x0.copy()
        for step in range(self.solver.horizon_steps):
            state = self.solver.A.dot(state) + self.solver.B.dot(
                controls[3 * step : 3 * step + 3]
            )
            direct.append(state.copy())
        np.testing.assert_allclose(predicted, np.concatenate(direct), atol=1.0e-12)

    def test_condensed_hessian_is_positive_definite(self):
        eigenvalues = np.linalg.eigvalsh(self.solver.H)
        self.assertGreater(float(eigenvalues.min()), 0.0)

    def test_zero_state_produces_zero_first_input(self):
        result = self.solver.solve(np.zeros(6))
        self.assertTrue(result["success"], result)
        np.testing.assert_allclose(result["control"], np.zeros(3), atol=1.0e-10)

    def test_x_position_error_has_restoring_sign(self):
        positive = self.solver.solve(np.array([0.10, 0, 0, 0, 0, 0]))
        self.solver.reset_warm_start()
        negative = self.solver.solve(np.array([-0.10, 0, 0, 0, 0, 0]))
        self.assertLess(positive["control"][0], 0.0)
        self.assertGreater(negative["control"][0], 0.0)

    def test_z_position_error_has_restoring_sign(self):
        positive = self.solver.solve(np.array([0, 0, 0.10, 0, 0, 0]))
        self.solver.reset_warm_start()
        negative = self.solver.solve(np.array([0, 0, -0.10, 0, 0, 0]))
        self.assertLess(positive["control"][2], 0.0)
        self.assertGreater(negative["control"][2], 0.0)

    def test_acceleration_bounds_are_never_exceeded(self):
        result = self.solver.solve(np.array([0.20, -0.20, 0.15, 0.2, -0.2, 0.15]))
        self.assertTrue(result["success"], result)
        self.assertLessEqual(abs(result["control"][0]), 0.20 + 1.0e-12)
        self.assertLessEqual(abs(result["control"][1]), 0.20 + 1.0e-12)
        self.assertLessEqual(abs(result["control"][2]), 0.15 + 1.0e-12)

    def test_nonfinite_state_fails_closed(self):
        with self.assertRaisesRegex(TinyMPCSolverError, "nonfinite"):
            self.solver.solve(np.array([math.nan, 0, 0, 0, 0, 0]))


class TestTinyMPCSource(unittest.TestCase):
    def test_static_hold_output_is_unchanged(self):
        source = StaticHoldSource()
        target = {"position": {"x": 1.0, "y": 2.0, "z": 3.0}, "yaw": 0.4}
        command = source.compute(2.0, {"position": {"x": 99.0}}, target)
        self.assertEqual(command["position"], target["position"])
        self.assertEqual(command["velocity"], {"x": 0.0, "y": 0.0, "z": 0.0})
        self.assertEqual(command["acceleration"], {"x": 0.0, "y": 0.0, "z": 0.0})
        self.assertEqual(command["yaw"], 0.4)

    def test_source_emits_current_state_and_mpc_acceleration(self):
        source = TinyMPCZeroMotionSource(controller_config())
        odom = {
            "position": {"x": 1.05, "y": 1.98, "z": 3.02},
            "velocity": {"x": 0.01, "y": -0.02, "z": 0.01},
        }
        target = {"position": {"x": 1.0, "y": 2.0, "z": 3.0}, "yaw": 0.4}
        command = source.compute(5.0, odom, target)
        self.assertEqual(command["position"], odom["position"])
        self.assertEqual(command["velocity"], odom["velocity"])
        self.assertEqual(command["jerk"], {"x": 0.0, "y": 0.0, "z": 0.0})
        self.assertEqual(command["yaw"], 0.4)
        self.assertEqual(command["yaw_dot"], 0.0)
        self.assertLess(command["acceleration"]["x"], 0.0)
        self.assertGreater(command["acceleration"]["y"], 0.0)

    def test_source_rejects_out_of_envelope_state(self):
        source = TinyMPCZeroMotionSource(controller_config())
        odom = {
            "position": {"x": 1.21, "y": 2.0, "z": 3.0},
            "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        }
        target = {"position": {"x": 1.0, "y": 2.0, "z": 3.0}, "yaw": 0.0}
        with self.assertRaisesRegex(TinyMPCSafetyError, "position_error_x"):
            source.compute(0.0, odom, target)

    def test_source_does_not_reuse_a_command_after_solver_nonconvergence(self):
        config = controller_config()
        config["max_iterations"] = 1
        config["residual_tolerance"] = 1.0e-12
        source = TinyMPCZeroMotionSource(config)
        odom = {
            "position": {"x": 0.10, "y": 0.0, "z": 0.0},
            "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        }
        target = {"position": {"x": 0.0, "y": 0.0, "z": 0.0}, "yaw": 0.0}
        with self.assertRaisesRegex(TinyMPCSafetyError, "not_converged"):
            source.compute(0.0, odom, target)

    def test_source_rejects_solver_time_over_twenty_milliseconds(self):
        source = TinyMPCZeroMotionSource(controller_config())

        class SlowSuccessfulSolver:
            axis_limits = np.array([0.20, 0.20, 0.15])

            @staticmethod
            def solve(_state):
                return {
                    "success": True,
                    "control": np.zeros(3),
                    "iterations": 1,
                    "residual": 0.0,
                    "solve_time_seconds": 0.021,
                }

        source.solver = SlowSuccessfulSolver()
        odom = {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        }
        target = {"position": {"x": 0.0, "y": 0.0, "z": 0.0}, "yaw": 0.0}
        with self.assertRaisesRegex(TinyMPCSafetyError, "solve_time_exceeded"):
            source.compute(0.0, odom, target)

    def test_stage5_config_selects_tinympc_and_three_second_window(self):
        config = load_v2_config(str(STAGE5_CONFIG))
        machine = FlightHandoverV2(config)
        self.assertEqual(machine.source.name, "TinyMPCZeroMotionSource")
        self.assertEqual(config["timing"]["active_command_window_seconds"], 3.0)
        summary = machine.summary({})
        self.assertEqual(summary["selected_controller"], "TinyMPCZeroMotionSource")
        self.assertFalse(summary["candidate_097_used"])

    def test_stage5_config_only_changes_controller_and_active_window(self):
        baseline = json.loads(
            (ROOT / "config" / "alpha_stage4_flight_handover_v2_prop_off.json").read_text(
                encoding="utf-8-sig"
            )
        )
        stage5 = json.loads(STAGE5_CONFIG.read_text(encoding="utf-8-sig"))
        controller = stage5.pop("controller")
        self.assertEqual(controller["mode"], "tinympc_zero_motion")
        stage5["timing"]["active_command_window_seconds"] = 5.0
        self.assertEqual(stage5, baseline)


class TestFinalPassStateMachineRegression(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            (ROOT / "config" / "alpha_stage4_flight_handover_v2_prop_off.json").read_text(
                encoding="utf-8-sig"
            )
        )

    @staticmethod
    def snapshot(**changes):
        value = {
            "mavros_connected": True,
            "armed": True,
            "px4_mode": "OFFBOARD",
            "offboard_enabled": True,
            "px4ctrl_state": "AUTO_HOVER",
            "ch5_semantic": "ONBOARD",
            "ch6_semantic": "HOVER",
            "ch7_semantic": "ARM",
            "ch8_semantic": "INIT",
            "state_fresh": True,
            "rc_fresh": True,
            "battery_fresh": True,
            "odom_fresh": True,
            "cmd_fresh": True,
            "subscriber_count": 1,
            "subscriber_nodes": ["/px4ctrl"],
            "ch5_onboard_edge": False,
            "ch5_optical_flow_edge": False,
            "ch6_tracking_edge": False,
            "ch6_hover_edge": False,
        }
        value.update(changes)
        return value

    def test_ch5_onboard_uses_entry_semantic_not_instantaneous_edge_flag(self):
        machine = FlightHandoverV2(self.config)
        machine.state = "WAIT_CH5_ONBOARD_EDGE"
        machine.wait_ch5_entered_semantic = "OPTICAL_FLOW"
        machine.step(self.snapshot(ch5_semantic="ONBOARD", ch5_onboard_edge=False), 1.0)
        self.assertEqual(machine.state, "VERIFY_AUTO_HOVER")

    def test_controller_mode_selects_tinympc_without_changing_state_machine(self):
        config = dict(self.config)
        config["controller"] = controller_config()
        machine = FlightHandoverV2(config)
        self.assertEqual(machine.source.name, "TinyMPCZeroMotionSource")

    def test_ch6_tracking_uses_entry_semantic_not_instantaneous_edge_flag(self):
        machine = FlightHandoverV2(self.config)
        machine.state = "WAIT_CH6_TRACKING_EDGE"
        machine.wait_ch6_entered_semantic = "HOVER"
        actions = machine.step(
            self.snapshot(ch6_semantic="TRACKING", ch6_tracking_edge=False), 1.0
        )
        self.assertEqual(machine.state, "VERIFY_CMD_CTRL")
        self.assertIn("PUBLISH_STATIC_HOLD", actions)

    def test_post_active_wait_states_do_not_publish(self):
        machine = FlightHandoverV2(self.config)
        machine.state = "WAIT_CH6_HOVER"
        machine.wait_ch6_return_entered_semantic = "TRACKING"
        actions = machine.step(self.snapshot(ch6_semantic="TRACKING"), 1.0)
        self.assertFalse(any(action.startswith("PUBLISH") for action in actions))
        actions = machine.step(self.snapshot(ch6_semantic="HOVER"), 1.1)
        self.assertEqual(machine.state, "VERIFY_AUTO_HOVER_RETURN")
        self.assertFalse(any(action.startswith("PUBLISH") for action in actions))
        actions = machine.step(self.snapshot(px4ctrl_state="AUTO_HOVER"), 1.2)
        self.assertFalse(any(action.startswith("PUBLISH") for action in actions))

    def test_ch6_return_uses_semantic_transition_window(self):
        machine = FlightHandoverV2(self.config)
        machine.state = "WAIT_CH6_HOVER"
        machine.wait_ch6_return_entered_semantic = "TRACKING"
        machine.step(self.snapshot(ch6_semantic="HOVER", ch6_hover_edge=False), 1.0)
        self.assertEqual(machine.state, "VERIFY_AUTO_HOVER_RETURN")

    def test_ch5_return_uses_semantic_transition_window(self):
        machine = FlightHandoverV2(self.config)
        machine.state = "WAIT_CH5_OPTICAL_FLOW"
        machine.wait_ch5_return_entered_semantic = "ONBOARD"
        machine.step(
            self.snapshot(ch5_semantic="OPTICAL_FLOW", ch5_optical_flow_edge=False),
            1.0,
        )
        self.assertEqual(machine.state, "VERIFY_POSCTL")


if __name__ == "__main__":
    unittest.main()
