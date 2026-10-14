"""Tests for the offline-only 200-candidate iteration harness."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import numpy as np

from rpi_pixhawk_stage4.flight_handover.v2_safe import load_v2_config
from rpi_pixhawk_stage4.iteration.run_tinympc_200_iterations import (
    DEFAULT_CONFIG,
    HORIZON_BOUNDS,
    _assert_frozen,
    _candidate_config,
    _controller_parameters,
    _parameters_from_unit,
    _reference_arrays,
    _scenario_metrics,
    SCENARIOS,
)
from rpi_pixhawk_stage4.iteration.paper_q_score import (
    PAPER_NORMALIZERS,
    aggregate_paper_q,
    trajectory_paper_q,
)


class TinyMPCIterationHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = json.loads(Path(DEFAULT_CONFIG).read_text(encoding="utf-8-sig"))

    def test_reference_grid_is_full_146_seconds_at_20_hz(self):
        times, position, velocity = _reference_arrays(self.base)
        self.assertEqual(len(times), 2920)
        self.assertEqual(position.shape, (2920, 3))
        self.assertEqual(velocity.shape, (2920, 3))
        np.testing.assert_array_equal(position[0], np.zeros(3))
        self.assertLess(times[-1], 146.0)

    def test_candidate_changes_only_declared_tunables(self):
        params = _controller_parameters(self.base["controller"])
        params["q_pos_y"] *= 1.01
        params["horizon_steps"] = HORIZON_BOUNDS[0]
        candidate = _candidate_config(self.base, params)
        _assert_frozen(self.base, candidate)
        self.assertNotEqual(
            candidate["controller"]["state_weights"],
            self.base["controller"]["state_weights"],
        )

    def test_frozen_field_change_is_rejected(self):
        changed = copy.deepcopy(self.base)
        changed["controller"]["acceleration_limits_mps2"]["y"] = 0.21
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            _assert_frozen(self.base, changed)

    def test_parameter_decoding_respects_bounds(self):
        params = _parameters_from_unit(np.full(15, 2.0), 999)
        self.assertEqual(params["horizon_steps"], HORIZON_BOUNDS[1])
        self.assertGreater(params["q_pos_y"], 0.0)

    def test_nominal_baseline_completes(self):
        references = _reference_arrays(self.base)
        result = _scenario_metrics(self.base["controller"], references, SCENARIOS[0])
        self.assertTrue(result["completed"], result)
        self.assertIsNone(result["failure"])
        self.assertGreater(result["paper_q_compat"]["q_score"], 0.0)

    def test_paper_q_keeps_legacy_weights_and_normalizers(self):
        result = aggregate_paper_q(PAPER_NORMALIZERS)
        self.assertAlmostEqual(result["normalized_cost"], 1.0)
        self.assertAlmostEqual(result["s_score"], 0.5)
        self.assertAlmostEqual(result["q_score"], 0.5)

    def test_paper_q_integrals_are_duration_normalized_to_ten_seconds(self):
        def score(step_count):
            return trajectory_paper_q(
                position_errors_m=np.tile([[0.1, 0.0, 0.0]], (step_count, 1)),
                achieved_acceleration_mps2=np.zeros((step_count, 3)),
                commanded_acceleration_mps2=np.zeros((step_count, 3)),
                solve_times_seconds=np.full(step_count, 0.001),
                dt_seconds=0.1,
                acceleration_limits_mps2=[0.2, 0.2, 0.15],
            )

        ten_seconds = score(100)
        twenty_seconds = score(200)
        for metric in ("IAE_pos", "IAE_att", "control_energy"):
            self.assertAlmostEqual(
                ten_seconds["metrics"][metric],
                twenty_seconds["metrics"][metric],
            )

    def test_paper_q_saturation_uses_old_per_step_any_semantics(self):
        result = trajectory_paper_q(
            position_errors_m=np.zeros((2, 3)),
            achieved_acceleration_mps2=np.zeros((2, 3)),
            commanded_acceleration_mps2=[[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]],
            solve_times_seconds=[0.001, 0.001],
            dt_seconds=0.05,
            acceleration_limits_mps2=[0.2, 0.2, 0.15],
        )
        self.assertAlmostEqual(result["metrics"]["saturation_rate"], 0.5)

    def test_delivered_candidate_is_fail_closed_for_offline_review(self):
        path = (
            Path(DEFAULT_CONFIG).parent
            / "alpha_stage5_tinympc_xy_figure8_r1m_146s_augint_iter200_offline.json"
        )
        config = load_v2_config(str(path))
        self.assertTrue(config["audit_only"])
        self.assertFalse(config["actual_publish_enabled"])
        self.assertFalse(config["actual_publish_ready"])
        self.assertEqual(config["controller"]["horizon_steps"], 14)


if __name__ == "__main__":
    unittest.main()
