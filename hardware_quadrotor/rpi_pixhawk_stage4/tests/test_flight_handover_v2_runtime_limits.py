import unittest
from unittest import mock

import rpi_pixhawk_stage4.run_stage4_alpha as runner


class V2SupervisorRuntimeTests(unittest.TestCase):
    def _argv(self, seconds):
        return [
            "--flight-handover-hold",
            "--flight-handover-version", "v2",
            "--enable-actual-publish",
            "--actual-publish-confirmation", "ALPHA_FLIGHT_HANDOVER_HOLD",
            "--maximum-runtime-seconds", str(seconds),
        ]

    def test_supervisor_accepts_thirty_and_sixty_seconds(self):
        with mock.patch.object(runner, "run_v2", return_value=0) as run_v2:
            for seconds in (30.0, 60.0):
                with self.subTest(seconds=seconds):
                    self.assertEqual(runner.main(self._argv(seconds)), 0)
                    self.assertEqual(run_v2.call_args.args[2], seconds)

    def test_supervisor_rejects_values_outside_one_to_sixty_seconds(self):
        with mock.patch.object(runner, "run_v2", return_value=0):
            for seconds in (0.99, 60.01):
                with self.subTest(seconds=seconds):
                    with self.assertRaisesRegex(ValueError, "flight_handover_runtime_exceeds_hard_limit"):
                        runner.main(self._argv(seconds))


if __name__ == "__main__":
    unittest.main()