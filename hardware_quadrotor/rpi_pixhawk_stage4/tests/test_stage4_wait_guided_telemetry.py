from __future__ import annotations

import math
import unittest


class FakeMavutil:
    @staticmethod
    def mode_string_v10(msg):
        return getattr(msg, "mode", "UNKNOWN")


class Message:
    def __init__(self, kind, system_id=1, component_id=1, **fields):
        self.kind = kind
        self.system_id = system_id
        self.component_id = component_id
        for key, value in fields.items():
            setattr(self, key, value)

    def get_type(self):
        return self.kind

    def get_srcSystem(self):
        return self.system_id

    def get_srcComponent(self):
        return self.component_id


class WaitGuidedTelemetryTests(unittest.TestCase):
    def source(self):
        from rpi_pixhawk_stage4.telemetry.readonly_source import ReadonlyHardwareTelemetry
        source = ReadonlyHardwareTelemetry(
            "udpin:127.0.0.1:14556",
            heartbeat_timeout_s=1.5,
            mavutil_module=FakeMavutil(),
            monotonic=lambda: 0.0,
        )
        return source

    def test_only_target_copter_heartbeat_refreshes_heartbeat_age(self):
        source = self.source()
        source._update(Message(
            "HEARTBEAT", autopilot=3, type=2, base_mode=128, mode="LOITER"
        ), 10.0)
        source._update(Message("ATTITUDE", time_boot_ms=1, roll=0, pitch=0, yaw=0), 11.0)
        source._refresh_ages(11.0)
        self.assertEqual(source._state.last_heartbeat_monotonic_s, 10.0)
        self.assertAlmostEqual(source._state.heartbeat_age_s, 1.0)
        self.assertTrue(source._state.heartbeat_fresh)
        source._refresh_ages(11.6)
        self.assertFalse(source._state.heartbeat_fresh)

    def test_wrong_identity_heartbeat_does_not_replace_locked_target(self):
        source = self.source()
        source._update(Message(
            "HEARTBEAT", system_id=1, component_id=1,
            autopilot=3, type=2, base_mode=128, mode="LOITER"
        ), 10.0)
        source._update(Message(
            "HEARTBEAT", system_id=2, component_id=1,
            autopilot=3, type=2, base_mode=0, mode="GUIDED"
        ), 10.5)
        self.assertEqual((source._state.system_id, source._state.component_id), (1, 1))
        self.assertEqual(source._state.flight_mode, "LOITER")
        self.assertTrue(source._state.armed)
        self.assertEqual(source._state.last_heartbeat_monotonic_s, 10.0)

    def test_relative_altitude_and_speeds_are_normalized(self):
        source = self.source()
        source._update(Message(
            "HEARTBEAT", autopilot=3, type=2, base_mode=0, mode="LOITER"
        ), 1.0)
        source._update(Message(
            "GLOBAL_POSITION_INT", relative_alt=1234
        ), 1.1)
        source._update(Message(
            "LOCAL_POSITION_NED", time_boot_ms=2, x=0, y=0, z=-1.2,
            vx=0.3, vy=0.4, vz=-0.2
        ), 1.2)
        self.assertTrue(source._state.relative_alt_valid)
        self.assertAlmostEqual(source._state.relative_alt_m, 1.234)
        self.assertAlmostEqual(source._state.horizontal_speed_m_s, 0.5)
        self.assertAlmostEqual(source._state.vertical_speed_m_s, 0.2)
        self.assertTrue(math.isfinite(source._state.relative_alt_m))


if __name__ == "__main__":
    unittest.main()
