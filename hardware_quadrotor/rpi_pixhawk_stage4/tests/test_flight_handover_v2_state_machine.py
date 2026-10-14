import unittest

from rpi_pixhawk_stage4.flight_handover.state_v2 import FlightHandoverV2
from rpi_pixhawk_stage4.tests.flight_v2_helpers import config_fixture, snapshot


class V2StateMachineTests(unittest.TestCase):
    def machine(self):
        value = FlightHandoverV2(config_fixture())
        value.set_parameter_audit({"parameter_audit_passed": True, "parameter_conflicts": []})
        return value

    def test_no_publisher_before_auto_hover_or_offboard(self):
        for state in ("BOOT", "PARAMETER_AUDIT", "WAIT_CH8_INIT", "WAIT_OPTICAL_FLOW_MODE", "WAIT_ARM_AND_STABLE", "WAIT_CH5_ONBOARD_EDGE", "VERIFY_AUTO_HOVER", "VERIFY_OFFBOARD"):
            machine = self.machine(); machine.state = state; machine.state_entered_at = 0.0
            self.assertNotIn("CREATE_PUBLISHER", machine.step(snapshot(ch5_onboard_edge=True), 0.1), state)
            self.assertEqual(machine.published_before_offboard_count, 0)

    def test_wait_arm_and_stable_requires_armed_and_ch7_unlocked(self):
        cases = (
            (snapshot(armed=False, ch7_semantic="ARM"), "WAIT_ARM_AND_STABLE"),
            (snapshot(armed=True, ch7_semantic="DISARM"), "WAIT_ARM_AND_STABLE"),
        )
        for current, expected in cases:
            with self.subTest(armed=current["armed"], ch7=current["ch7_semantic"]):
                machine = self.machine()
                machine.state = "WAIT_ARM_AND_STABLE"
                machine.state_entered_at = 0.0
                machine.step(current, 0.0)
                machine.step(current, 1.1)
                self.assertEqual(machine.state, expected)

    def test_wait_arm_and_stable_advances_only_when_armed_and_ch7_unlocked(self):
        machine = self.machine()
        machine.state = "WAIT_ARM_AND_STABLE"
        machine.state_entered_at = 0.0
        ready = snapshot(armed=True, ch7_semantic="ARM")
        machine.step(ready, 0.0)
        machine.step(ready, 1.1)
        self.assertEqual(machine.state, "WAIT_CH5_ONBOARD_EDGE")
    def test_auto_hover_and_offboard_are_both_required(self):
        machine = self.machine(); machine.state = "VERIFY_AUTO_HOVER"
        ready = snapshot(px4_mode="OFFBOARD", offboard_enabled=True, px4ctrl_state="AUTO_HOVER", ch5_semantic="ONBOARD")
        self.assertNotIn("CREATE_PUBLISHER", machine.step(ready, 0.9))
        self.assertNotIn("CREATE_PUBLISHER", machine.step(ready, 1.0))
        self.assertIn("CREATE_PUBLISHER", machine.step(ready, 1.1))
        self.assertTrue(machine.summary(ready)["publisher_created_after_auto_hover"])
        self.assertTrue(machine.summary(ready)["publisher_created_after_offboard"])

    def test_cmd_ctrl_required_for_active(self):
        machine = self.machine(); machine.state = "VERIFY_CMD_CTRL"; machine.state_entered_at = 0.0
        common = dict(px4_mode="OFFBOARD", offboard_enabled=True, ch5_semantic="ONBOARD", ch6_semantic="TRACKING", cmd_fresh=True)
        machine.step(snapshot(px4ctrl_state="AUTO_HOVER", **common), 0.1)
        self.assertNotEqual(machine.state, "ACTIVE_CONTROL")
        machine.step(snapshot(px4ctrl_state="CMD_CTRL", **common), 0.2)
        self.assertEqual(machine.state, "ACTIVE_CONTROL")

    def test_active_hard_limit_and_strict_return_order(self):
        machine = self.machine(); machine.state = "ACTIVE_CONTROL"; machine.active_started_at = 0.0
        active = snapshot(px4_mode="OFFBOARD", offboard_enabled=True, px4ctrl_state="CMD_CTRL", ch5_semantic="ONBOARD", ch6_semantic="TRACKING", cmd_fresh=True)
        machine.step(active, 5.001)
        seen = [machine.state]
        machine.step(snapshot(px4_mode="OFFBOARD", offboard_enabled=True, px4ctrl_state="CMD_CTRL", ch5_semantic="ONBOARD", ch6_semantic="HOVER", ch6_hover_edge=True, cmd_fresh=True), 5.1); seen.append(machine.state)
        machine.step(snapshot(px4_mode="OFFBOARD", offboard_enabled=True, px4ctrl_state="AUTO_HOVER", ch5_semantic="ONBOARD", ch6_semantic="HOVER", cmd_fresh=True), 5.2); seen.append(machine.state)
        machine.step(snapshot(px4_mode="OFFBOARD", offboard_enabled=True, px4ctrl_state="AUTO_HOVER", ch5_semantic="ONBOARD", ch6_semantic="HOVER", cmd_fresh=False), 6.0); seen.append(machine.state)
        machine.step(snapshot(ch5_semantic="OPTICAL_FLOW", ch5_optical_flow_edge=True), 6.1); seen.append(machine.state)
        machine.step(snapshot(cmd_fresh=False), 6.2); seen.append(machine.state)
        self.assertEqual(seen, ["WAIT_CH6_HOVER", "VERIFY_AUTO_HOVER_RETURN", "STOP_AND_DRAIN_COMMAND", "WAIT_CH5_OPTICAL_FLOW", "VERIFY_POSCTL", "COMPLETE"])
        self.assertLessEqual(machine.active_duration_seconds, 5.1)


if __name__ == "__main__":
    unittest.main()
