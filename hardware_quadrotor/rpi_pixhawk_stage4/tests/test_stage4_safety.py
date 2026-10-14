from __future__ import annotations

import importlib
import math
import unittest


def healthy(**changes):
    value = {
        "autopilot": 3, "vehicle_type": 2, "flight_mode": "GUIDED", "armed": True,
        "gps_fix_type": 3, "ekf_ok": True, "local_position_valid": True,
        "rc_valid": True, "telemetry_age_s": 0.1, "rc_age_s": 0.1,
        "received_monotonic_s": 10.0, "time_boot_ms": 1000,
        "north_m": 1.0, "east_m": 2.0, "down_m": -0.8,
        "vn_mps": 0.0, "ve_mps": 0.0, "vd_mps": 0.0,
        "roll_rad": 0.0, "pitch_rad": 0.0, "yaw_rad": 0.0,
        "rc_channels": {7: 1200},
    }
    value.update(changes)
    return value


class Stage4SafetyTests(unittest.TestCase):
    def _module(self):
        try:
            return importlib.import_module("rpi_pixhawk_stage4.safety.preflight")
        except ModuleNotFoundError as exc:
            self.fail("missing preflight module: %s" % exc)

    def test_flight_requires_explicit_switch_phrase_and_verified_command_link(self):
        safety = self._module()
        state = healthy()
        report = {"passed": True, "command_endpoint": "udpout:127.0.0.1:16042"}
        decision = safety.authorize_flight(False, safety.FLIGHT_CONFIRMATION, "udpout:127.0.0.1:16042", report, state, False)
        self.assertEqual(decision.reason, "real_hardware_transmit_not_enabled")
        decision = safety.authorize_flight(True, "wrong", "udpout:127.0.0.1:16042", report, state, False)
        self.assertEqual(decision.reason, "confirmation_phrase_mismatch")
        decision = safety.authorize_flight(True, safety.FLIGHT_CONFIRMATION, "udpout:127.0.0.1:16042", {"passed": False}, state, False)
        self.assertEqual(decision.reason, "command_link_not_verified")

    def test_flight_rejects_disarmed_non_guided_gps_ekf_local_rc_and_stale(self):
        safety = self._module()
        report = {"passed": True, "command_endpoint": "udpout:127.0.0.1:16042"}
        cases = (
            ({"armed": False}, "vehicle_not_armed"), ({"flight_mode": "LOITER"}, "vehicle_not_guided"),
            ({"gps_fix_type": 2}, "gps_fix_insufficient"), ({"ekf_ok": False}, "ekf_unhealthy"),
            ({"local_position_valid": False}, "local_position_missing"), ({"rc_valid": False}, "rc_data_missing"),
            ({"telemetry_age_s": 0.6}, "stale_telemetry"), ({"autopilot": 0}, "not_ardupilot_copter"),
        )
        for changes, expected in cases:
            with self.subTest(expected=expected):
                decision = safety.authorize_flight(True, safety.FLIGHT_CONFIRMATION, "udpout:127.0.0.1:16042", report, healthy(**changes), False)
                self.assertEqual(decision.reason, expected)

    def test_emergency_stop_has_priority(self):
        safety = self._module()
        report = {"passed": True, "command_endpoint": "udpout:127.0.0.1:16042"}
        decision = safety.authorize_flight(True, safety.FLIGHT_CONFIRMATION, "udpout:127.0.0.1:16042", report, healthy(), True)
        self.assertEqual(decision.reason, "emergency_stop")

    def test_state_validation_rejects_nonfinite_regression_and_bounds(self):
        safety = self._module()
        origin = (1.0, 2.0, -0.8)
        cases = (
            (healthy(north_m=math.nan), None, "nonfinite_telemetry"),
            (healthy(time_boot_ms=999), 1000, "timestamp_regression"),
            (healthy(north_m=1.6), None, "horizontal_radius_exceeded"),
            (healthy(down_m=-2.31), None, "relative_height_exceeded"),
        )
        for state, previous, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(safety.validate_active_state(state, origin, previous).reason, expected)

    def test_rc_override_is_explicit_and_directional(self):
        safety = self._module()
        self.assertTrue(safety.pilot_override_active(healthy(rc_channels={7: 1800}), 7, 1700, "high"))
        self.assertFalse(safety.pilot_override_active(healthy(rc_channels={7: 1200}), 7, 1700, "high"))
        self.assertTrue(safety.pilot_override_active(healthy(rc_channels={7: 1000}), 7, 1200, "low"))


if __name__ == "__main__":
    unittest.main()
