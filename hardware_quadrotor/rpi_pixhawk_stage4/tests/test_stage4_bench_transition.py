from __future__ import annotations

import importlib
import tempfile
import unittest


class Transport:
    def __init__(self):
        self.probes = 0
        self.zeros = 0
        self.closed = False
    def probe_command_link(self):
        self.probes += 1
        return {"passed": True}
    def send_zero_velocity(self, reason):
        self.zeros += 1
    def close(self):
        self.closed = True


class Stage4BenchTransitionTests(unittest.TestCase):
    def test_armed_transition_after_probe_blocks_optional_zero(self):
        bench = importlib.import_module("rpi_pixhawk_stage4.scripts.bench_session")
        transport = Transport()
        with tempfile.TemporaryDirectory() as tmp:
            result = bench.run_bench_session(
                {"armed": False}, transport, "udpout:127.0.0.1:16042", tmp,
                enable_probe=True, confirmation=bench.BENCH_CONFIRMATION,
                enable_zero_setpoint=True, state_provider=lambda: {"armed": True},
            )
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "hardware_armed_detected")
        self.assertEqual(transport.zeros, 0)
        self.assertTrue(transport.closed)


if __name__ == "__main__":
    unittest.main()
