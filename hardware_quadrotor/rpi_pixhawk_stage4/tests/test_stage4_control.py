from __future__ import annotations

import importlib
import math
import unittest


class Stage4ControlTests(unittest.TestCase):
    def test_stage4_limits_are_stric_and_finite(self):
        try:
            limits = importlib.import_module("rpi_pixhawk_stage4.controller.limits")
        except ModuleNotFoundError as exc:
            self.fail("missing Stage 4 limits: %s" % exc)
        target = limits.limit_target(1.0, 1.0, 1.0, 1.0)
        self.assertLessEqual(math.hypot(target["vx_north_mps"], target["vy_east_mps"]), 0.10)
        self.assertLessEqual(abs(target["vz_down_mps"]), 0.10)
        self.assertLessEqual(abs(target["yaw_rate_radps"]), 0.08)
        self.assertTrue(target["saturated"])
        invalid = limits.limit_target(math.nan, 0.0, 0.0, 0.0)
        self.assertTrue(invalid["invalid_input"])

    def test_reference_holds_two_seconds_then_smoothly_moves_point_one_meter(self):
        try:
            reference = importlib.import_module("rpi_pixhawk_stage4.controller.reference")
        except ModuleNotFoundError as exc:
            self.fail("missing Stage 4 reference: %s" % exc)
        profile = reference.FirstFlightReference((1.0, 2.0, -0.8), 0.3)
        self.assertEqual(profile.at(0.0)["north_m"], 1.0)
        self.assertEqual(profile.at(2.0)["north_m"], 1.0)
        self.assertAlmostEqual(profile.at(4.0)["north_m"], 1.1)
        self.assertEqual(profile.at(5.0)["east_m"], 2.0)
        self.assertEqual(profile.at(5.0)["down_m"], -0.8)
        self.assertEqual(profile.at(5.0)["yaw_rad"], 0.3)

    def test_state_machine_contains_required_states(self):
        try:
            machine = importlib.import_module("rpi_pixhawk_stage4.safety.state_machine")
        except ModuleNotFoundError as exc:
            self.fail("missing state machine: %s" % exc)
        expected = {
            "DISABLED", "PREFLIGHT_CHECK", "WAITING_FOR_GUIDED", "WAITING_FOR_ARMED",
            "REFERENCE_LOCK", "ACTIVE_CONTROL", "PILOT_OVERRIDE", "SAFETY_STOP", "COMPLETE",
        }
        self.assertTrue(expected.issubset({item.value for item in machine.FlightState}))


if __name__ == "__main__":
    unittest.main()
