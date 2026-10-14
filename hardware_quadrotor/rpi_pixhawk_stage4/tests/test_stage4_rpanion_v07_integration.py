from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

from rpi_pixhawk_stage4.tests.test_stage4_rpanion_v07_dynamic_peer import (
    FakeClock, FakeCodec, FakeSocket, LOOPBACK_PEER, message, valid_mapping,
)


class RpanionV07IntegrationTests(unittest.TestCase):
    def test_probe_response_identity_mismatch_stops_before_timesync(self):
        module = importlib.import_module("rpi_pixhawk_stage4.transport.rpanion_v07_dynamic_peer")
        mapping = valid_mapping()
        mapping[b"bad_response"] = [
            message("PING", system_id=2, component_id=1, sequence=7, time_usec=123456)
        ]
        sock = FakeSocket([
            (item, LOOPBACK_PEER) for item in (b"h", b"a", b"s", b"bad_response")
        ])
        with tempfile.TemporaryDirectory() as tmp:
            report = module.run_dynamic_peer_probe(
                sock, FakeCodec(mapping), tmp, True, module.BENCH_CONFIRMATION,
                timeout_seconds=1.0, monotonic=FakeClock().monotonic,
                ping_sequence=7, ping_time_usec=123456, timesync_token=987654321,
            )
        self.assertEqual(report["reason"], "dynamic_peer_identity_changed")
        self.assertEqual(len(sock.sent), 1)

    def test_authorization_failures_happen_before_socket_creation(self):
        module = importlib.import_module("rpi_pixhawk_stage4.transport.rpanion_v07_dynamic_peer")
        original = module.socket.socket
        calls = []

        def forbidden_socket(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("socket_must_not_be_created")

        module.socket.socket = forbidden_socket
        try:
            with tempfile.TemporaryDirectory() as tmp:
                report = module.run_dynamic_peer_endpoint(
                    module.REQUIRED_TELEMETRY_ENDPOINT, tmp, False, ""
                )
                self.assertEqual(report["reason"], "bench_probe_not_enabled")
                report = module.run_dynamic_peer_endpoint(
                    module.REQUIRED_TELEMETRY_ENDPOINT, tmp, True, "WRONG"
                )
                self.assertEqual(report["reason"], "bench_confirmation_phrase_mismatch")
        finally:
            module.socket.socket = original
        self.assertEqual(calls, [])

    def test_main_stage4_scan_includes_dynamic_profile_proof(self):
        scanner = importlib.import_module("rpi_pixhawk_stage4.scripts.scan_stage4_safety")
        stage_dir = Path(__file__).resolve().parents[1]
        report = scanner.scan_stage4_sources(stage_dir)
        self.assertTrue(report["ok"], report)
        self.assertTrue(report["dynamic_profile_scan"]["ok"])

    def test_field_document_contains_exact_rpanion_v07_command(self):
        path = Path(__file__).resolve().parents[1] / "docs" / "rpanion_v07_bench.md"
        self.assertTrue(path.is_file())
        text = path.read_text(encoding="utf-8")
        self.assertIn("--link-profile rpanion-v07-dynamic-peer", text)
        self.assertIn("--telemetry-endpoint udpin:127.0.0.1:14556", text)
        self.assertIn("--enable-bench-probe", text)
        self.assertIn("I_CONFIRM_PROPELLERS_REMOVED_AND_PIXHAWK_DISARMED", text)
        self.assertNotIn("--command-endpoint", text)


if __name__ == "__main__":
    unittest.main()
