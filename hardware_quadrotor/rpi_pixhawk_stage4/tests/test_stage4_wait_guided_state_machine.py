from __future__ import annotations

import unittest


def state(**changes):
    value = {
        "system_id": 1,
        "component_id": 1,
        "autopilot": 3,
        "vehicle_type": 2,
        "flight_mode": "LOITER",
        "armed": False,
        "heartbeat_fresh": True,
        "heartbeat_age_s": 0.1,
        "telemetry_age_s": 0.1,
        "rc_age_s": 0.1,
        "gps_fix_type": 3,
        "ekf_ok": True,
        "local_position_valid": True,
        "rc_valid": True,
        "relative_alt_valid": True,
        "relative_alt_m": 0.0,
        "north_m": 0.0,
        "east_m": 0.0,
        "down_m": 0.0,
        "horizontal_speed_m_s": 0.0,
        "vertical_speed_m_s": 0.0,
        "roll_rad": 0.0,
        "pitch_rad": 0.0,
        "yaw_rad": 0.0,
        "time_boot_ms": 1000,
        "rc_channels": {7: 1200},
    }
    value.update(changes)
    return value


class WaitGuidedStateMachineTests(unittest.TestCase):
    def setUp(self):
        from rpi_pixhawk_stage4.safety.wait_guided import WaitGuidedConfig, WaitGuidedMachine
        self.config = WaitGuidedConfig(
            wait_timeout_s=5.0,
            heartbeat_timeout_s=1.5,
            min_airborne_alt_m=1.0,
            min_abort_alt_m=0.7,
            stable_hover_duration_s=0.2,
            max_horizontal_speed_m_s=0.5,
            max_vertical_speed_m_s=0.3,
            max_roll_pitch_deg=10.0,
            guided_settle_s=0.1,
            max_control_duration_s=0.5,
            zero_command_count=3,
            zero_command_period_s=0.01,
            one_shot=True,
        )
        self.machine = WaitGuidedMachine(self.config, started_monotonic=0.0)

    def ground_baseline(self):
        self.machine.update(state(), 0.00)
        self.machine.update(state(), 0.01)
        self.machine.update(state(), 0.02)
        self.assertEqual(self.machine.state.value, "WAIT_ARMED")

    def airborne(self, now=0.10, **changes):
        value = state(armed=True, relative_alt_m=1.1, down_m=-1.1, **changes)
        self.machine.update(value, now)
        if self.machine.state.value == "WAIT_AIRBORNE":
            self.machine.update(value, now + 0.01)
        return value

    def test_ground_disarmed_non_guided_waits_without_trigger(self):
        self.ground_baseline()
        self.assertFalse(self.machine.guided_edge_detected)
        self.assertFalse(self.machine.guided_trigger_consumed)
        self.assertFalse(self.machine.ready_to_authorize)

    def test_armed_on_ground_does_not_execute(self):
        self.ground_baseline()
        self.machine.update(state(armed=True, relative_alt_m=0.2), 0.1)
        self.machine.update(state(armed=True, relative_alt_m=0.2), 0.2)
        self.assertEqual(self.machine.state.value, "WAIT_AIRBORNE")
        self.assertFalse(self.machine.ready_to_authorize)

    def test_airborne_requires_minimum_altitude(self):
        self.ground_baseline()
        self.machine.update(state(armed=True, relative_alt_m=0.99), 0.1)
        self.machine.update(state(armed=True, relative_alt_m=0.99), 0.2)
        self.assertNotEqual(self.machine.state.value, "WAIT_STABLE_HOVER")

    def test_airborne_requires_existing_position_safety(self):
        self.ground_baseline()
        unsafe = state(armed=True, relative_alt_m=1.2, ekf_ok=False)
        self.machine.update(unsafe, 0.1)
        self.machine.update(unsafe, 0.2)
        self.assertEqual(self.machine.state.value, "WAIT_AIRBORNE")
        self.assertFalse(self.machine.ready_to_authorize)

    def test_starting_guided_and_armed_cannot_trigger(self):
        guided = state(armed=True, flight_mode="GUIDED", relative_alt_m=1.2)
        self.machine.update(guided, 0.0)
        self.machine.update(guided, 0.1)
        self.machine.update(guided, 0.2)
        self.assertFalse(self.machine.seen_non_guided_since_start)
        self.assertFalse(self.machine.guided_edge_detected)
        self.assertFalse(self.machine.ready_to_authorize)

    def test_local_ned_fallback_uses_ground_minus_current_down(self):
        baseline = state(relative_alt_valid=False, relative_alt_m=None, down_m=0.4)
        self.machine.update(baseline, 0.0)
        self.machine.update(baseline, 0.01)
        self.machine.update(baseline, 0.02)
        flying = state(
            armed=True, relative_alt_valid=False, relative_alt_m=None, down_m=-0.7
        )
        self.assertAlmostEqual(self.machine.relative_altitude(flying), 1.1)

    def test_unstable_hover_resets_continuous_timer(self):
        self.ground_baseline()
        stable = self.airborne()
        self.machine.update(stable, 0.20)
        self.machine.update(dict(stable, horizontal_speed_m_s=0.8), 0.35)
        self.machine.update(stable, 0.40)
        self.machine.update(stable, 0.55)
        self.assertEqual(self.machine.state.value, "WAIT_STABLE_HOVER")
        self.machine.update(stable, 0.61)
        self.assertEqual(self.machine.state.value, "WAIT_GUIDED_EDGE")

    def test_stable_hover_then_guided_edge_and_settle_authorizes(self):
        self.ground_baseline()
        stable = self.airborne()
        self.machine.update(stable, 0.20)
        self.machine.update(stable, 0.41)
        self.assertEqual(self.machine.state.value, "WAIT_GUIDED_EDGE")
        guided = dict(stable, flight_mode="GUIDED")
        self.machine.update(guided, 0.42)
        self.assertEqual(self.machine.state.value, "GUIDED_SETTLE")
        self.assertTrue(self.machine.guided_edge_detected)
        self.assertTrue(self.machine.guided_trigger_consumed)
        self.machine.update(guided, 0.53)
        self.assertEqual(self.machine.state.value, "AUTHORIZE")
        self.assertTrue(self.machine.ready_to_authorize)

    def test_guided_settle_preflight_failure_aborts(self):
        self.ground_baseline()
        stable = self.airborne()
        self.machine.update(stable, 0.20)
        self.machine.update(stable, 0.41)
        guided = dict(stable, flight_mode="GUIDED")
        self.machine.update(guided, 0.42)
        self.machine.update(dict(guided, ekf_ok=False), 0.43)
        self.assertEqual(self.machine.state.value, "ABORTED")
        self.assertEqual(self.machine.abort_reason, "guided_settle_preflight_failed")

    def test_early_guided_requires_return_and_new_stable_hover(self):
        self.ground_baseline()
        flying = self.airborne()
        early = dict(flying, flight_mode="GUIDED")
        self.machine.update(early, 0.20)
        self.machine.update(early, 0.50)
        self.assertFalse(self.machine.stable_hover_confirmed)
        self.assertFalse(self.machine.guided_edge_detected)
        self.machine.update(flying, 0.60)
        self.machine.update(flying, 0.81)
        self.assertEqual(self.machine.state.value, "WAIT_GUIDED_EDGE")
        self.machine.update(early, 0.82)
        self.assertTrue(self.machine.guided_edge_detected)

    def test_consumed_trigger_cannot_repeat(self):
        self.test_stable_hover_then_guided_edge_and_settle_authorizes()
        self.machine.mark_active()
        self.machine.mark_completed()
        for now in (0.7, 0.8, 0.9):
            self.machine.update(state(armed=True, flight_mode="GUIDED", relative_alt_m=1.1), now)
        self.assertEqual(self.machine.state.value, "COMPLETED")
        self.assertTrue(self.machine.guided_trigger_consumed)


if __name__ == "__main__":
    unittest.main()
