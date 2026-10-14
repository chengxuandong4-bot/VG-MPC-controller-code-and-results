from __future__ import annotations

import importlib
import unittest


class Stage4CrossLinkIdentityTests(unittest.TestCase):
    def test_telemetry_and_command_links_must_identify_same_vehicle(self):
        safety = importlib.import_module("rpi_pixhawk_stage4.safety.preflight")
        self.assertTrue(safety.command_identity_matches(
            {"system_id": 1, "component_id": 1},
            {"target_system": 1, "target_component": 1},
        ).safe)
        self.assertEqual(safety.command_identity_matches(
            {"system_id": 1, "component_id": 1},
            {"target_system": 2, "target_component": 1},
        ).reason, "command_target_mismatch")
        self.assertEqual(safety.command_identity_matches(
            {"system_id": 0, "component_id": 0},
            {"target_system": 1, "target_component": 1},
        ).reason, "telemetry_target_identity_missing")


if __name__ == "__main__":
    unittest.main()
