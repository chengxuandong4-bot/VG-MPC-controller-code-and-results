import unittest

from rpi_pixhawk_stage4.flight_handover.rc_semantics import RcSemanticTracker
from rpi_pixhawk_stage4.flight_handover.state_v2 import FlightHandoverV2
from rpi_pixhawk_stage4.flight_handover.v2_safe import load_v2_config
from rpi_pixhawk_stage4.tests.flight_v2_helpers import V2_CONFIG, config_fixture, snapshot


class V2SafetyTests(unittest.TestCase):
    def test_unfilled_pwm_calibration_blocks_actual(self):
        config = load_v2_config(str(V2_CONFIG))
        self.assertFalse(config["actual_publish_ready"])
        self.assertIn("rc_pwm_calibration_missing", config["configuration_errors"])

    def test_starting_onboard_has_no_entry_edge_and_transition_is_debounced_once(self):
        tracker = RcSemanticTracker(config_fixture()["rc"])
        channels = [1500, 1500, 1500, 1500, 2000, 1000, 2000, 1000]
        first = [tracker.update(channels, index * 0.07) for index in range(6)]
        self.assertFalse(any(item["ch5_onboard_edge"] for item in first))
        channels[4] = 1000
        for index in range(6, 12): tracker.update(channels, index * 0.07)
        channels[4] = 2000
        edges = [tracker.update(channels, index * 0.07) for index in range(12, 18)]
        self.assertEqual(sum(item["ch5_onboard_edge"] for item in edges), 1)

    def test_active_faults_fail_closed(self):
        for changes in ({"px4_mode": "STABILIZED"}, {"odom_fresh": False}, {"rc_fresh": False}, {"ch8_semantic": "EXECUTE"}):
            machine = FlightHandoverV2(config_fixture()); machine.state = "ACTIVE_CONTROL"; machine.active_started_at = 0.0
            base = dict(px4_mode="OFFBOARD", offboard_enabled=True, px4ctrl_state="CMD_CTRL", ch5_semantic="ONBOARD", ch6_semantic="TRACKING", cmd_fresh=True); base.update(changes)
            self.assertIn("UNREGISTER_PUBLISHER", machine.step(snapshot(**base), 0.1))
            self.assertEqual(machine.state, "ABORT_ACTIVE_CONTROL")

    def test_candidate097_is_not_selected(self):
        machine = FlightHandoverV2(config_fixture())
        self.assertFalse(machine.summary(snapshot())["candidate_097_used"])


if __name__ == "__main__":
    unittest.main()
