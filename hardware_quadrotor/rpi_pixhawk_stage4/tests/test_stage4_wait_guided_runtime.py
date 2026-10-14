from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rpi_pixhawk_stage4.tests.test_stage4_wait_guided_state_machine import state


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0.001, float(seconds))


class FakeTelemetry:
    def __init__(self, states, clock):
        self.states = list(states)
        self.last = self.states[-1] if self.states else None
        self.clock = clock
        self.closed = False

    def next_state(self, timeout):
        self.clock.now += max(0.05, float(timeout))
        if self.states:
            self.last = self.states.pop(0)
        return self.last

    def close(self):
        self.closed = True


class FakeTransport:
    def __init__(self, closed=False):
        self.targets = []
        self.zeros = []
        self.closed = closed

    def send_velocity(self, target):
        self.targets.append(target)

    def send_zero_velocity(self, reason):
        self.zeros.append(reason)

    def close(self):
        self.closed = True


class Controller:
    def step(self, state, reference, dt_seconds):
        return {
            "vx_north_mps": 0.02,
            "vy_east_mps": 0.0,
            "vz_down_mps": 0.0,
            "yaw_rate_radps": 0.0,
            "invalid_input": False,
        }


def active(**changes):
    values = {
        "armed": True,
        "flight_mode": "GUIDED",
        "relative_alt_m": 1.1,
        "heartbeat_fresh": True,
        "heartbeat_age_s": 0.1,
    }
    values.update(changes)
    return state(**values)


class WaitGuidedRuntimeTests(unittest.TestCase):
    def config(self, **changes):
        from rpi_pixhawk_stage4.safety.wait_guided import WaitGuidedConfig
        values = {
            "wait_timeout_s": 5.0,
            "heartbeat_timeout_s": 1.5,
            "min_airborne_alt_m": 1.0,
            "min_abort_alt_m": 0.7,
            "stable_hover_duration_s": 0.2,
            "max_horizontal_speed_m_s": 0.5,
            "max_vertical_speed_m_s": 0.3,
            "max_roll_pitch_deg": 10.0,
            "guided_settle_s": 0.1,
            "max_control_duration_s": 0.5,
            "zero_command_count": 3,
            "zero_command_period_s": 0.01,
            "one_shot": True,
        }
        values.update(changes)
        return WaitGuidedConfig(**values)

    def success_states(self):
        return [
            state(), state(), state(),
            state(armed=True, relative_alt_m=0.2),
            state(armed=True, relative_alt_m=1.1),
            state(armed=True, relative_alt_m=1.1),
            state(armed=True, relative_alt_m=1.1),
            state(armed=True, relative_alt_m=1.1),
            active(), active(), active(),
        ]

    def test_audit_only_runs_full_state_machine_without_transport(self):
        from rpi_pixhawk_stage4.scripts.wait_guided_session import run_wait_guided_session
        clock = FakeClock()
        telemetry = FakeTelemetry(self.success_states(), clock)
        created = []
        with tempfile.TemporaryDirectory() as tmp:
            result = run_wait_guided_session(
                telemetry=telemetry,
                results_dir=tmp,
                config=self.config(),
                audit_only=True,
                transport_factory=lambda: created.append(True),
                controller_factory=lambda: Controller(),
                monotonic=clock.monotonic,
                sleeper=clock.sleep,
            )
            summary = json.loads(Path(tmp, "wait_guided_summary.json").read_text())
            header = Path(tmp, "wait_guided_state_log.csv").read_text().splitlines()[0]
        self.assertEqual(created, [])
        self.assertEqual(result.command_count, 0)
        self.assertTrue(result.would_execute)
        self.assertEqual(result.final_state, "COMPLETED")
        self.assertTrue(summary["audit_only"])
        for field in (
            "previous_mode", "guided_edge_detected", "guided_trigger_consumed",
            "heartbeat_age_s", "relative_alt_m", "abort_reason",
        ):
            self.assertIn(field, header)

    def test_wait_timeout_aborts_without_transport(self):
        from rpi_pixhawk_stage4.scripts.wait_guided_session import run_wait_guided_session
        clock = FakeClock()
        telemetry = FakeTelemetry([state()], clock)
        with tempfile.TemporaryDirectory() as tmp:
            result = run_wait_guided_session(
                telemetry, tmp, self.config(wait_timeout_s=0.3), True,
                monotonic=clock.monotonic, sleeper=clock.sleep,
            )
        self.assertEqual(result.final_state, "ABORTED")
        self.assertEqual(result.abort_reason, "wait_timeout")
        self.assertEqual(result.command_count, 0)

    def test_cli_defaults_wait_guided_to_audit_only(self):
        from rpi_pixhawk_stage4.scripts import run_stage4
        args = run_stage4.build_parser().parse_args(["--mode", "flight-wait-guided"])
        self.assertTrue(run_stage4.wait_guided_is_audit_only(args))
        self.assertEqual(args.heartbeat_timeout_s, 1.5)
        self.assertEqual(args.wait_timeout_s, 600.0)

    def test_explicit_audit_overrides_transmit_flag_fail_safe(self):
        from rpi_pixhawk_stage4.scripts import run_stage4
        args = run_stage4.build_parser().parse_args([
            "--mode", "flight-wait-guided", "--audit-only",
            "--enable-real-hardware-transmit",
            "--link-profile", "rpanion-v07-dynamic-peer",
        ])
        self.assertTrue(run_stage4.wait_guided_is_audit_only(args))
        run_stage4.validate_profile_arguments(args)

    def test_dynamic_peer_wait_guided_live_remains_rejected(self):
        from rpi_pixhawk_stage4.scripts import run_stage4
        args = run_stage4.build_parser().parse_args([
            "--mode", "flight-wait-guided",
            "--link-profile", "rpanion-v07-dynamic-peer",
            "--enable-real-hardware-transmit",
        ])
        with self.assertRaisesRegex(ValueError, "dynamic_peer_flight_live_rejected"):
            run_stage4.validate_profile_arguments(args)

    def test_fixed_live_missing_endpoint_or_authorization_is_rejected(self):
        from rpi_pixhawk_stage4.scripts import run_stage4
        args = run_stage4.build_parser().parse_args([
            "--mode", "flight-wait-guided",
            "--enable-real-hardware-transmit",
        ])
        with self.assertRaises(ValueError):
            run_stage4.validate_wait_guided_live_arguments(args)

    def test_fixed_live_rejects_reused_telemetry_port(self):
        from rpi_pixhawk_stage4.safety.preflight import FLIGHT_CONFIRMATION
        from rpi_pixhawk_stage4.scripts import run_stage4
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp, "report.json")
            report.write_text("{}", encoding="utf-8")
            args = run_stage4.build_parser().parse_args([
                "--mode", "flight-wait-guided",
                "--enable-real-hardware-transmit",
                "--command-endpoint", "udpout:127.0.0.1:14556",
                "--confirmation", FLIGHT_CONFIRMATION,
                "--command-link-verification-report", str(report),
            ])
            with self.assertRaisesRegex(
                    ValueError, "command_and_telemetry_endpoints_must_differ"):
                run_stage4.validate_wait_guided_live_arguments(args)

    def test_active_heartbeat_timeout_stops_even_when_other_telemetry_is_fresh(self):
        from rpi_pixhawk_stage4.scripts.flight_session import (
            ActiveControlSafety, run_flight_session,
        )
        clock = FakeClock()
        telemetry = FakeTelemetry([
            active(),
            active(heartbeat_fresh=False, heartbeat_age_s=1.6, telemetry_age_s=0.1),
        ], clock)
        transport = FakeTransport()
        safety = ActiveControlSafety.wait_guided(
            min_abort_alt_m=0.7,
            guided_trigger_consumed=True,
            send_authorized=True,
            safety_zero_count=3,
            zero_command_period_s=0.01,
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = run_flight_session(
                Controller(), telemetry, transport, tmp, 0.5,
                safety=safety, monotonic=clock.monotonic, sleeper=clock.sleep,
            )
        self.assertEqual(result.exit_reason, "heartbeat_stale")
        self.assertEqual(len(transport.targets), 1)
        self.assertEqual(transport.zeros, [])

    def test_active_rc_override_and_mode_exit_send_no_zero(self):
        from rpi_pixhawk_stage4.scripts.flight_session import (
            ActiveControlSafety, run_flight_session,
        )
        for fault, expected in (
            ({"rc_channels": {7: 1800}}, "pilot_override_rc"),
            ({"flight_mode": "LOITER"}, "pilot_override_mode_departure"),
        ):
            with self.subTest(expected=expected):
                clock = FakeClock()
                telemetry = FakeTelemetry([active(**fault)], clock)
                transport = FakeTransport()
                with tempfile.TemporaryDirectory() as tmp:
                    result = run_flight_session(
                        Controller(), telemetry, transport, tmp, 0.5,
                        rc_override_channel=7,
                        safety=ActiveControlSafety.wait_guided(),
                        monotonic=clock.monotonic,
                        sleeper=clock.sleep,
                    )
                self.assertEqual(result.exit_reason, expected)
                self.assertEqual(transport.targets, [])
                self.assertEqual(transport.zeros, [])

    def test_active_low_altitude_uses_three_safe_zeros(self):
        from rpi_pixhawk_stage4.scripts.flight_session import (
            ActiveControlSafety, run_flight_session,
        )
        clock = FakeClock()
        telemetry = FakeTelemetry([active(relative_alt_m=0.6)], clock)
        transport = FakeTransport()
        with tempfile.TemporaryDirectory() as tmp:
            result = run_flight_session(
                Controller(), telemetry, transport, tmp, 0.5,
                safety=ActiveControlSafety.wait_guided(
                    min_abort_alt_m=0.7,
                    safety_zero_count=3,
                    zero_command_period_s=0.01,
                ),
                monotonic=clock.monotonic,
                sleeper=clock.sleep,
            )
        self.assertEqual(result.exit_reason, "minimum_abort_altitude")
        self.assertEqual(transport.targets, [])
        self.assertEqual(len(transport.zeros), 3)

    def test_normal_completion_sends_one_zero_and_does_not_repeat(self):
        from rpi_pixhawk_stage4.scripts.flight_session import (
            ActiveControlSafety, run_flight_session,
        )
        clock = FakeClock()
        telemetry = FakeTelemetry([active()], clock)
        transport = FakeTransport()
        with tempfile.TemporaryDirectory() as tmp:
            result = run_flight_session(
                Controller(), telemetry, transport, tmp, 0.3,
                safety=ActiveControlSafety.wait_guided(),
                monotonic=clock.monotonic,
                sleeper=clock.sleep,
            )
        self.assertEqual(result.exit_reason, "maximum_runtime")
        self.assertGreater(result.transmitted_command_count, 0)
        self.assertEqual(len(transport.zeros), 1)


if __name__ == "__main__":
    unittest.main()
