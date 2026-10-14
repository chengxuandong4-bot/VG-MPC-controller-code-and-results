from __future__ import annotations

import ast
import importlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


STAGE_DIR = Path(__file__).resolve().parents[1]


class Stage4DeliveryTests(unittest.TestCase):
    def test_static_scan_confines_all_transmit_calls_to_one_boundary(self):
        try:
            scanner = importlib.import_module("rpi_pixhawk_stage4.scripts.scan_stage4_safety")
        except ModuleNotFoundError as exc:
            self.fail("missing Stage 4 scanner: %s" % exc)
        report = scanner.scan_stage4_sources(STAGE_DIR)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["transmit_boundary_files"], ["transport/pymavlink_boundary.py"])
        self.assertEqual(report["violations"], [])

    def test_scanner_detects_forbidden_arm_mode_param_and_outside_send(self):
        try:
            scanner = importlib.import_module("rpi_pixhawk_stage4.scripts.scan_stage4_safety")
        except ModuleNotFoundError as exc:
            self.fail("missing Stage 4 scanner: %s" % exc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bad.py").write_text("x.arducopter_arm()\nx.set_mode(4)\nx.param_set_send(1)\nx.ping_send(1)\n", encoding="utf-8")
            report = scanner.scan_stage4_sources(root)
        self.assertFalse(report["ok"])
        self.assertGreaterEqual(len(report["violations"]), 4)

    def test_python38_grammar_compatibility(self):
        for path in STAGE_DIR.rglob("*.py"):
            with self.subTest(path=path):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))

    def test_direct_script_help_exposes_wait_guided(self):
        script = STAGE_DIR / "scripts" / "run_stage4.py"
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(STAGE_DIR.parent),
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("flight-wait-guided", completed.stdout)
        self.assertIn("--heartbeat-timeout-s", completed.stdout)

    def test_field_document_has_seven_separate_sections(self):
        path = STAGE_DIR / "docs" / "tomorrow_field_sequence.md"
        self.assertTrue(path.is_file())
        text = path.read_text(encoding="utf-8")
        for heading in (
            "A. Propeller-off GPS/EKF and Stage 3", "B. Propeller-off command-link bench-disarmed",
            "C. Propeller-off motors, failsafes and pilot override", "D. First native ArduPilot flight",
            "E. Post-landing log review", "F. Separate second TinyMPC flight", "G. Emergency stop and shutdown",
        ):
            self.assertIn(heading, text)
        self.assertIn("Do not switch to TinyMPC during the first native ArduPilot flight", text)


if __name__ == "__main__":
    unittest.main()
