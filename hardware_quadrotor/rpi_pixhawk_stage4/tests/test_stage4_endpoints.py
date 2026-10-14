from __future__ import annotations

import importlib
import unittest


class Stage4EndpointTests(unittest.TestCase):
    def _module(self):
        try:
            return importlib.import_module("rpi_pixhawk_stage4.safety.endpoint_guard")
        except ModuleNotFoundError as exc:
            self.fail("missing Stage 4 endpoint guard: %s" % exc)

    def test_user_port_loopback_udpout_is_accepted(self):
        guard = self._module()
        self.assertEqual(guard.validate_command_endpoint("udpout:127.0.0.1:16042"), "udpout:127.0.0.1:16042")
        self.assertEqual(guard.validate_command_endpoint("udpout:localhost:17000"), "udpout:localhost:17000")

    def test_command_port_is_required_and_not_hardcoded(self):
        guard = self._module()
        for value in (None, "", "udpout:127.0.0.1:<port>"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                guard.validate_command_endpoint(value)

    def test_non_loopback_and_devices_are_rejected(self):
        guard = self._module()
        bad = ("udpout:192.168.1.2:14550", "serial:/dev/serial0:921600", "/dev/ttyAMA0", "COM4", "USB0")
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                guard.validate_command_endpoint(value)

    def test_telemetry_requires_loopback_udpin(self):
        guard = self._module()
        self.assertEqual(guard.validate_telemetry_endpoint("udpin:127.0.0.1:14556"), "udpin:127.0.0.1:14556")
        for value in ("udpout:127.0.0.1:14556", "udpin:10.0.0.2:14556", "/dev/serial0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                guard.validate_telemetry_endpoint(value)


if __name__ == "__main__":
    unittest.main()
