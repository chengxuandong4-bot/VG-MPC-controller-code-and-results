import importlib
import tempfile
import unittest


class LegacyV1BlockedTests(unittest.TestCase):
    def test_explicit_v1_actual_publish_is_rejected_before_ros(self):
        runner = importlib.import_module("rpi_pixhawk_stage4.run_stage4_alpha")
        with self.assertRaisesRegex(RuntimeError, "legacy_v1_actual_publish_forbidden"):
            runner.main(["--flight-handover-hold", "--flight-handover-version", "v1", "--enable-actual-publish", "--actual-publish-confirmation", "ALPHA_FLIGHT_HANDOVER_HOLD"])

    def test_explicit_v1_audit_creates_no_publisher(self):
        runner = importlib.import_module("rpi_pixhawk_stage4.run_stage4_alpha")
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(runner.main(["--flight-handover-hold", "--flight-handover-version", "v1", "--results-dir", directory]), 0)


if __name__ == "__main__":
    unittest.main()
