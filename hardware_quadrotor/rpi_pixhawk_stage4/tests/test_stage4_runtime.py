from __future__ import annotations

import importlib
import tempfile
import threading
import unittest

from test_stage4_safety import healthy


class FakeTelemetry:
    def __init__(self, states, interrupt=False):
        self.states = iter(states)
        self.interrupt = interrupt
        self.closed = False
    def next_state(self, timeout_seconds):
        if self.interrupt:
            raise KeyboardInterrupt()
        return next(self.states, None)
    def close(self):
        self.closed = True


class FakeTransport:
    def __init__(self):
        self.targets = []
        self.zeros = []
        self.closed = False
    def send_velocity(self, target):
        self.targets.append(dict(target))
        return True
    def send_zero_velocity(self, reason):
        self.zeros.append(reason)
        return True
    def close(self):
        self.closed = True


class Controller:
    def step(self, **kwargs):
        return {"vx_north_mps": 0.04, "vy_east_mps": 0.0, "vz_down_mps": 0.0, "yaw_rate_radps": 0.0, "invalid_input": False}


class Stop:
    def __init__(self, active=False):
        self.active = active
    def requested(self):
        return self.active


class Stage4RuntimeTests(unittest.TestCase):
    def _run(self, states, stop=None, interrupt=False):
        try:
            runtime = importlib.import_module("rpi_pixhawk_stage4.scripts.flight_session")
        except ModuleNotFoundError as exc:
            self.fail("missing flight session: %s" % exc)
        telemetry = FakeTelemetry(states, interrupt=interrupt)
        transport = FakeTransport()
        before = threading.active_count()
        with tempfile.TemporaryDirectory() as tmp:
            result = runtime.run_flight_session(
                Controller(), telemetry, transport, tmp, maximum_runtime_seconds=0.03,
                emergency_stop=stop or Stop(), rc_override_channel=7,
                rc_override_threshold=1700, rc_override_direction="high",
            )
        self.assertEqual(threading.active_count(), before)
        self.assertTrue(telemetry.closed)
        self.assertTrue(transport.closed)
        return result, transport

    def test_mode_departure_immediately_stops_without_zero(self):
        result, transport = self._run([healthy(), healthy(flight_mode="LOITER")])
        self.assertEqual(result.exit_reason, "pilot_override_mode_departure")
        self.assertEqual(transport.zeros, [])

    def test_rc_override_immediately_stops_without_zero(self):
        result, transport = self._run([healthy(rc_channels={7: 1800})])
        self.assertEqual(result.exit_reason, "pilot_override_rc")
        self.assertEqual(transport.targets, [])
        self.assertEqual(transport.zeros, [])

    def test_stale_telemetry_attempts_one_zero_then_stops(self):
        result, transport = self._run([healthy(telemetry_age_s=0.6)])
        self.assertEqual(result.exit_reason, "stale_telemetry")
        self.assertEqual(len(transport.zeros), 1)

    def test_nan_regression_ekf_and_gps_faults_stop_without_recovery(self):
        cases = (
            (healthy(north_m=float("nan")), "nonfinite_telemetry"),
            (healthy(ekf_ok=False), "ekf_unhealthy"),
            (healthy(gps_fix_type=2), "gps_fix_insufficient"),
        )
        for state, expected in cases:
            with self.subTest(expected=expected):
                result, transport = self._run([state])
                self.assertEqual(result.exit_reason, expected)
                self.assertLessEqual(len(transport.zeros), 1)

    def test_emergency_stop_and_keyboard_interrupt_close_synchronously(self):
        result, transport = self._run([healthy()], stop=Stop(True))
        self.assertEqual(result.exit_reason, "emergency_stop")
        self.assertLessEqual(len(transport.zeros), 1)
        result, transport = self._run([], interrupt=True)
        self.assertEqual(result.exit_reason, "keyboard_interrupt")
        self.assertLessEqual(len(transport.zeros), 1)


if __name__ == "__main__":
    unittest.main()
