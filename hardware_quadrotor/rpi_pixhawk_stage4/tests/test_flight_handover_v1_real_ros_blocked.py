import importlib
import types
import unittest
from pathlib import Path
from unittest import mock


STAGE4_DIR = Path(__file__).resolve().parents[1]
V1_CONFIG = STAGE4_DIR / "config" / "alpha_candidate_001_flight_hold.yaml"


class LegacyV1RealRosBlockedTests(unittest.TestCase):
    def test_implicit_v1_rejects_real_rospy_before_node_or_publisher(self):
        runner = importlib.import_module("rpi_pixhawk_stage4.run_stage4_alpha")
        real_shaped_rospy = types.ModuleType("rospy")
        real_shaped_rospy.__file__ = "/opt/ros/noetic/lib/python3/dist-packages/rospy/__init__.py"
        with mock.patch.object(
            runner,
            "_load_ros_bench",
            return_value=(real_shaped_rospy, object(), object(), object()),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "legacy_v1_actual_publish_forbidden"
            ):
                runner.main(
                    [
                        "--flight-handover-hold",
                        "--enable-actual-publish",
                        "--actual-publish-confirmation",
                        "ALPHA_FLIGHT_HANDOVER_HOLD",
                        "--config",
                        str(V1_CONFIG),
                    ]
                )
        self.assertFalse(hasattr(real_shaped_rospy, "Publisher"))


if __name__ == "__main__":
    unittest.main()
