from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path


class FakeTransport:
    def __init__(self):
        self.probes = 0
        self.zeros = 0
        self.closed = False
    def probe_command_link(self):
        self.probes += 1
        return {"passed": True, "ping_response": True, "timesync_response": True}
    def send_zero_velocity(self, reason):
        self.zeros += 1
        return True
    def close(self):
        self.closed = True


class Stage4BenchTests(unittest.TestCase):
    def _module(self):
        try:
            return importlib.import_module("rpi_pixhawk_stage4.scripts.bench_session")
        except ModuleNotFoundError as exc:
            self.fail("missing bench session: %s" % exc)

    def test_disarmed_probe_uses_only_ping_timesync_by_default(self):
        bench = self._module()
        transport = FakeTransport()
        with tempfile.TemporaryDirectory() as tmp:
            result = bench.run_bench_session(
                {"armed": False}, transport, "udpout:127.0.0.1:16042", tmp,
                enable_probe=True, confirmation=bench.BENCH_CONFIRMATION,
                enable_zero_setpoint=False,
            )
        self.assertTrue(result["passed"])
        self.assertEqual(transport.probes, 1)
        self.assertEqual(transport.zeros, 0)
        self.assertTrue(transport.closed)

    def test_armed_bench_rejects_before_probe(self):
        bench = self._module()
        transport = FakeTransport()
        with tempfile.TemporaryDirectory() as tmp:
            result = bench.run_bench_session(
                {"armed": True}, transport, "udpout:127.0.0.1:16042", tmp,
                enable_probe=True, confirmation=bench.BENCH_CONFIRMATION,
                enable_zero_setpoint=False,
            )
        self.assertEqual(result["reason"], "hardware_armed_detected")
        self.assertEqual(transport.probes, 0)
        self.assertEqual(transport.zeros, 0)

    def test_zero_setpoint_requires_separate_explicit_switch(self):
        bench = self._module()
        transport = FakeTransport()
        with tempfile.TemporaryDirectory() as tmp:
            result = bench.run_bench_session(
                {"armed": False}, transport, "udpout:127.0.0.1:16042", tmp,
                enable_probe=True, confirmation=bench.BENCH_CONFIRMATION,
                enable_zero_setpoint=True,
            )
            report = Path(tmp) / "command_link_verification.json"
            self.assertTrue(report.is_file())
        self.assertEqual(transport.zeros, 1)
        self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
