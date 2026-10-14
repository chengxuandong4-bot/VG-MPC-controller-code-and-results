"""Minimal mathematical and transactional tests for Stage 5 augmented MPC."""

from __future__ import annotations

import unittest

import numpy as np

from rpi_pixhawk_stage4.controller.tinympc_solver import CondensedTinyMPCSolver
from rpi_pixhawk_stage4.controller.tinympc_source import (
    TinyMPCSafetyError,
    TinyMPCZeroMotionSource,
)


def augmented_config():
    return {
        "mode": "tinympc_zero_motion",
        "dt_seconds": 0.05,
        "horizon_steps": 20,
        "state_weights": [15.0, 15.0, 25.0, 4.0, 4.0, 6.0, 0.5, 8.0, 12.0],
        "input_weights": [2.0, 2.0, 3.0],
        "acceleration_limits_mps2": {"x": 0.20, "y": 0.20, "z": 0.15},
        "max_iterations": 60,
        "residual_tolerance": 1.0e-6,
        "solve_time_limit_seconds": 0.020,
        "safety": {
            "max_position_error_m": {"x": 0.20, "y": 0.20, "z": 0.15},
            "max_horizontal_velocity_mps": 0.20,
            "max_vertical_velocity_mps": 0.15,
        },
        "target_offset_m": {"x": 0.05, "y": 0.0, "z": 0.0},
        "augmented_integral": {
            "enabled": True,
            "state_limit_m_s": {"x": 0.10, "y": 0.15, "z": 0.12},
        },
    }


def odom(error=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0)):
    # target position is zero plus the configured +5 cm X offset.
    return {
        "position": {
            "x": 0.05 + error[0],
            "y": error[1],
            "z": error[2],
        },
        "velocity": {"x": velocity[0], "y": velocity[1], "z": velocity[2]},
    }


TARGET = {"position": {"x": 0.0, "y": 0.0, "z": 0.0}, "yaw": 0.0}


class AugmentedSolverTests(unittest.TestCase):
    def setUp(self):
        self.solver = CondensedTinyMPCSolver(augmented_config())

    def test_augmented_a_and_b_are_exact(self):
        dt = 0.05
        expected_a = np.block(
            [
                [np.eye(3), dt * np.eye(3), np.zeros((3, 3))],
                [np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))],
                [dt * np.eye(3), np.zeros((3, 3)), np.eye(3)],
            ]
        )
        expected_b = np.vstack(
            (0.5 * dt * dt * np.eye(3), dt * np.eye(3), np.zeros((3, 3)))
        )
        self.assertEqual(self.solver.state_size, 9)
        self.assertEqual(self.solver.A.shape, (9, 9))
        self.assertEqual(self.solver.B.shape, (9, 3))
        np.testing.assert_allclose(self.solver.A, expected_a, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(self.solver.B, expected_b, rtol=0.0, atol=0.0)

    def test_augmented_condensed_prediction_matches_direct_rollout(self):
        state = np.array([0.02, -0.03, 0.01, 0.01, 0.0, -0.01, 0.04, -0.02, 0.03])
        controls = np.linspace(-0.1, 0.1, 60)
        predicted = self.solver.predict_states(state, controls)
        direct = []
        current = state.copy()
        for step in range(self.solver.horizon_steps):
            current = self.solver.A.dot(current) + self.solver.B.dot(
                controls[3 * step : 3 * step + 3]
            )
            direct.append(current.copy())
        np.testing.assert_allclose(predicted, np.concatenate(direct), atol=1.0e-12)

    def test_zero_state_has_zero_control(self):
        result = self.solver.solve(np.zeros(9))
        self.assertTrue(result["success"], result)
        np.testing.assert_allclose(result["control"], np.zeros(3), atol=1.0e-10)

    def test_negative_x_error_commands_positive_x(self):
        result = self.solver.solve(np.array([-0.05, 0, 0, 0, 0, 0, 0, 0, 0]))
        self.assertGreater(result["control"][0], 0.0)

    def test_positive_y_integral_commands_negative_y(self):
        result = self.solver.solve(np.array([0, 0, 0, 0, 0, 0, 0, 0.05, 0]))
        self.assertLess(result["control"][1], 0.0)

    def test_negative_z_integral_commands_positive_z(self):
        result = self.solver.solve(np.array([0, 0, 0, 0, 0, 0, 0, 0, -0.05]))
        self.assertGreater(result["control"][2], 0.0)


class AugmentedSourceTests(unittest.TestCase):
    def test_prestream_does_not_accumulate_integral(self):
        source = TinyMPCZeroMotionSource(augmented_config())
        source.compute(0.0, odom(error=(0.01, -0.02, 0.01)), TARGET)
        np.testing.assert_array_equal(source._aug_integral_state, np.zeros(3))
        self.assertFalse(source._aug_integrator_active)

    def test_active_reset_zeros_and_enables_integrator(self):
        source = TinyMPCZeroMotionSource(augmented_config())
        source._aug_integral_state[:] = [0.02, -0.03, 0.04]
        source.reset_integrator()
        np.testing.assert_array_equal(source._aug_integral_state, np.zeros(3))
        self.assertTrue(source._aug_integrator_active)

    def test_successful_solve_commits_integral_after_solve(self):
        source = TinyMPCZeroMotionSource(augmented_config())
        source.reset_integrator()
        seen_states = []
        original_solve = source.solver.solve

        def recording_solve(state):
            seen_states.append(np.asarray(state).copy())
            return original_solve(state)

        source.solver.solve = recording_solve
        source.compute(0.0, odom(error=(0.01, -0.02, 0.01)), TARGET)
        np.testing.assert_array_equal(seen_states[0][6:9], np.zeros(3))
        np.testing.assert_allclose(
            source._aug_integral_state, [0.0005, -0.001, 0.0005], atol=1.0e-15
        )

    def test_failed_or_timeout_solve_does_not_commit_integral(self):
        for result in (
            {
                "success": False,
                "control": np.zeros(3),
                "iterations": 60,
                "residual": 1.0,
                "solve_time_seconds": 0.001,
            },
            {
                "success": True,
                "control": np.zeros(3),
                "iterations": 1,
                "residual": 0.0,
                "solve_time_seconds": 0.021,
            },
            {
                "success": True,
                "control": np.array([np.inf, 0.0, 0.0]),
                "iterations": 1,
                "residual": 0.0,
                "solve_time_seconds": 0.001,
            },
        ):
            source = TinyMPCZeroMotionSource(augmented_config())
            source.reset_integrator()
            before = source._aug_integral_state.copy()
            source.solver.solve = lambda _state, value=result: value
            with self.assertRaises(TinyMPCSafetyError):
                source.compute(0.0, odom(error=(0.01, 0.01, 0.01)), TARGET)
            np.testing.assert_array_equal(source._aug_integral_state, before)

    def test_anti_windup_clips_integral_state(self):
        source = TinyMPCZeroMotionSource(augmented_config())
        source.reset_integrator()
        source._aug_integral_state[:] = [0.0999, 0.1499, -0.1199]
        source.compute(0.0, odom(error=(0.10, 0.10, -0.10)), TARGET)
        np.testing.assert_allclose(source._aug_integral_state, [0.10, 0.15, -0.12])

    def test_legacy_and_augmented_integral_conflict_fails_closed(self):
        for state_weights in (
            augmented_config()["state_weights"],
            augmented_config()["state_weights"][:6],
        ):
            config = augmented_config()
            config["state_weights"] = state_weights
            config["integral_compensation"] = {"enabled": True}
            with self.assertRaisesRegex(
                TinyMPCSafetyError, "tinympc_integral_modes_conflict"
            ):
                TinyMPCZeroMotionSource(config)


if __name__ == "__main__":
    unittest.main()
