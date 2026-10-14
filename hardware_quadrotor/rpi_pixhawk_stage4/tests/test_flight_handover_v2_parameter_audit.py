import unittest

from rpi_pixhawk_stage4.flight_handover.parameter_audit import ParameterAudit
from rpi_pixhawk_stage4.tests.flight_v2_helpers import config_fixture, parameters


class V2ParameterAuditTests(unittest.TestCase):
    def test_valid_mapping_passes(self):
        config = config_fixture()
        config["ch7_policy"] = "px4_native_kill"
        result = ParameterAudit(config).evaluate(parameters())
        self.assertTrue(result["parameter_audit_passed"])

    def test_native_kill_policy_allows_ch7_kill_without_arm_mapping(self):
        config = config_fixture()
        config["ch7_policy"] = "px4_native_kill"
        values = parameters()
        values.update(RC_MAP_KILL_SW=7, RC_MAP_ARM_SW=0)
        self.assertTrue(ParameterAudit(config).evaluate(values)["parameter_audit_passed"])

    def test_ch7_kill_mapping_without_explicit_native_policy_fails_closed(self):
        config = config_fixture()
        config.pop("ch7_policy", None)
        values = parameters()
        values.update(RC_MAP_KILL_SW=7, RC_MAP_ARM_SW=0)
        result = ParameterAudit(config).evaluate(values)
        self.assertFalse(result["parameter_audit_passed"])
        self.assertIn("ch7_policy", str(result))

    def test_ch7_arm_mapping_always_conflicts_with_native_kill_policy(self):
        config = config_fixture()
        config["ch7_policy"] = "px4_native_kill"
        values = parameters()
        values.update(RC_MAP_KILL_SW=7, RC_MAP_ARM_SW=7)
        result = ParameterAudit(config).evaluate(values)
        self.assertFalse(result["parameter_audit_passed"])
        self.assertIn("RC_MAP_ARM_SW", str(result))
    def test_onboard_slot_not_position_fails(self):
        values = parameters(); values["COM_FLTMODE3"] = 8
        result = ParameterAudit(config_fixture()).evaluate(values)
        self.assertFalse(result["parameter_audit_passed"])

    def test_offboard_switch_conflict_fails(self):
        values = parameters(); values["RC_MAP_OFFB_SW"] = 6
        self.assertFalse(ParameterAudit(config_fixture()).evaluate(values)["parameter_audit_passed"])
    def test_unreadable_parameter_fails_closed(self):
        values = parameters(); del values["COM_RC_IN_MODE"]
        result = ParameterAudit(config_fixture()).evaluate(values)
        self.assertFalse(result["parameter_audit_passed"])
        self.assertIn("parameter_unreadable", str(result))


if __name__ == "__main__":
    unittest.main()
