from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


STAGE_DIR = Path(__file__).resolve().parents[1]


class Stage4ContractTests(unittest.TestCase):
    def test_delivery_structure_exists(self):
        required = [
            "config/stage4_defaults.json", "scripts/run_stage4.py", "scripts/self_check_stage4.py",
            "scripts/scan_stage4_safety.py", "transport/pymavlink_boundary.py",
            "docs/tomorrow_field_sequence.md", "requirements-rpi.txt",
        ]
        self.assertEqual([name for name in required if not (STAGE_DIR / name).is_file()], [])

    def test_defaults_are_fail_closed_and_have_no_command_port(self):
        path = STAGE_DIR / "config" / "stage4_defaults.json"
        self.assertTrue(path.is_file())
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["default_mode"], "audit-only")
        self.assertIsNone(data["command_endpoint"])
        self.assertFalse(data["enable_real_hardware_transmit"])
        self.assertEqual(data["maximum_runtime_seconds"], 5.0)

    def test_cli_defaults_to_audit_without_endpoint_or_transmit(self):
        try:
            module = importlib.import_module("rpi_pixhawk_stage4.scripts.run_stage4")
        except ModuleNotFoundError as exc:
            self.fail("missing Stage 4 CLI: %s" % exc)
        args = module.build_parser().parse_args([])
        self.assertEqual(args.mode, "audit-only")
        self.assertIsNone(args.command_endpoint)
        self.assertFalse(args.enable_real_hardware_transmit)

    def test_audit_does_not_import_transport_or_pymavlink(self):
        try:
            module = importlib.import_module("rpi_pixhawk_stage4.scripts.run_stage4")
        except ModuleNotFoundError as exc:
            self.fail("missing Stage 4 CLI: %s" % exc)
        sys.modules.pop("rpi_pixhawk_stage4.transport.pymavlink_boundary", None)
        sys.modules.pop("pymavlink", None)
        with tempfile.TemporaryDirectory() as tmp:
            code = module.main(["--mode", "audit-only", "--results-dir", tmp])
        self.assertEqual(code, 0)
        self.assertNotIn("rpi_pixhawk_stage4.transport.pymavlink_boundary", sys.modules)
        self.assertNotIn("pymavlink", sys.modules)


if __name__ == "__main__":
    unittest.main()
