from __future__ import annotations

import importlib
import tempfile
import unittest

from test_stage4_runtime import Controller, FakeTelemetry, FakeTransport, Stop
from test_stage4_safety import healthy


class Stage4CommandIdentityTests(unittest.TestCase):
    def test_boundary_requires_verified_mavlink_target_before_velocity(self):
        module = importlib.import_module("rpi_pixhawk_stage4.transport.pymavlink_boundary")
        boundary = module.PymavlinkCommandBoundary("udpout:127.0.0.1:16042")
        boundary.open = lambda: None
        with self.assertRaises(RuntimeError):
            boundary.send_velocity(type("Target", (), {
                "vx_north_mps": 0.0, "vy_east_mps": 0.0,
                "vz_down_mps": 0.0, "yaw_rate_radps": 0.0,
            })())

    def test_missing_selected_rc_override_channel_stops(self):
        runtime = importlib.import_module("rpi_pixhawk_stage4.scripts.flight_session")
        telemetry = FakeTelemetry([healthy(rc_channels={6: 1200})])
        transport = FakeTransport()
        with tempfile.TemporaryDirectory() as tmp:
            result = runtime.run_flight_session(
                Controller(), telemetry, transport, tmp,
                maximum_runtime_seconds=0.03, emergency_stop=Stop(),
                rc_override_channel=7, rc_override_threshold=1700,
                rc_override_direction="high",
            )
        self.assertEqual(result.exit_reason, "rc_override_channel_missing")
        self.assertEqual(len(transport.zeros), 1)


if __name__ == "__main__":
    unittest.main()
