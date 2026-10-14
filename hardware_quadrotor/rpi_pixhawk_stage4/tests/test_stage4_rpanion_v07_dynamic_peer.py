from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path


LOOPBACK_PEER = ("127.0.0.1", 42123)
OTHER_PEER = ("127.0.0.1", 42124)


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def monotonic(self) -> float:
        self.now += 0.05
        return self.now


class FakeSocket:
    def __init__(self, packets) -> None:
        self.packets = list(packets)
        self.sent = []
        self.closed = False
        self.timeout = None

    def settimeout(self, value) -> None:
        self.timeout = value

    def recvfrom(self, size):
        if not self.packets:
            raise TimeoutError("fake timeout")
        return self.packets.pop(0)

    def sendto(self, payload, peer):
        self.sent.append((payload, peer))
        return len(payload)

    def close(self) -> None:
        self.closed = True


class FakeCodec:
    source_system = 245
    source_component = 190

    def __init__(self, mapping) -> None:
        self.mapping = mapping
        self.encoded = []

    def decode_datagram(self, payload):
        return list(self.mapping.get(payload, []))

    def encode_ping(self, sequence, time_usec):
        self.encoded.append(("PING", sequence, time_usec, 0, 0))
        return ("PING:{0}:{1}".format(sequence, time_usec)).encode("ascii")

    def encode_timesync(self, token):
        self.encoded.append(("TIMESYNC", token))
        return ("TIMESYNC:{0}".format(token)).encode("ascii")

    def inspect_outbound(self, payload):
        fields = payload.decode("ascii").split(":")
        if fields[0] == "PING":
            return {
                "message_type": "PING", "system_id": 245, "component_id": 190,
                "sequence": int(fields[1]), "time_usec": int(fields[2]),
                "target_system": 0, "target_component": 0,
            }
        return {
            "message_type": "TIMESYNC", "system_id": 245, "component_id": 190,
            "tc1": 0, "ts1": int(fields[1]),
        }


def message(kind, system_id=1, component_id=1, **fields):
    data = {
        "message_type": kind,
        "system_id": system_id,
        "component_id": component_id,
        "autopilot": -1,
        "vehicle_type": -1,
        "armed": False,
    }
    data.update(fields)
    return data


def valid_mapping():
    return {
        b"h": [message("HEARTBEAT", autopilot=3, vehicle_type=2, armed=False)],
        b"a": [message("ATTITUDE")],
        b"s": [message("SYS_STATUS")],
        b"pr": [message("PING", sequence=7, time_usec=123456, target_system=245, target_component=190)],
        b"tr": [message("TIMESYNC", tc1=777, ts1=987654321)],
        b"tr2": [message("TIMESYNC", tc1=778, ts1=987654322)],
        b"tr3": [message("TIMESYNC", tc1=779, ts1=987654323)],
    }


class RpanionV07DynamicPeerTests(unittest.TestCase):
    def _module(self):
        return importlib.import_module("rpi_pixhawk_stage4.transport.rpanion_v07_dynamic_peer")

    def _run(self, packets, mapping=None, **kwargs):
        module = self._module()
        sock = FakeSocket(packets)
        codec = FakeCodec(mapping or valid_mapping())
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as tmp:
            report = module.run_dynamic_peer_probe(
                telemetry_sock=sock,
                codec=codec,
                results_dir=tmp,
                enable_probe=True,
                confirmation=module.BENCH_CONFIRMATION,
                timeout_seconds=1.0,
                monotonic=clock.monotonic,
                ping_sequence=7,
                ping_time_usec=123456,
                timesync_token=987654321,
                **kwargs
            )
            saved = Path(tmp, "command_link_verification_report.json")
            self.assertTrue(saved.is_file())
        return report, sock, codec

    def test_no_telemetry_means_no_transmit(self):
        report, sock, _ = self._run([])
        self.assertFalse(report["passed"])
        self.assertEqual(report["reason"], "dynamic_peer_discovery_timeout")
        self.assertEqual(sock.sent, [])

    def test_non_loopback_source_is_rejected_before_transmit(self):
        report, sock, _ = self._run([(b"h", ("192.168.1.9", 42123))])
        self.assertEqual(report["reason"], "wrong_udp_source")
        self.assertEqual(sock.sent, [])

    def test_missing_or_non_copter_heartbeat_is_rejected(self):
        report, sock, _ = self._run([(b"a", LOOPBACK_PEER), (b"s", LOOPBACK_PEER)])
        self.assertEqual(report["reason"], "dynamic_peer_discovery_timeout")
        self.assertEqual(sock.sent, [])
        mapping = valid_mapping()
        mapping[b"h"] = [message("HEARTBEAT", autopilot=12, vehicle_type=2)]
        report, sock, _ = self._run([(b"h", LOOPBACK_PEER)], mapping)
        self.assertEqual(report["reason"], "not_ardupilot_copter")
        self.assertEqual(sock.sent, [])

    def test_system_component_identity_mismatch_fails_closed(self):
        mapping = valid_mapping()
        mapping[b"a"] = [message("ATTITUDE", system_id=2, component_id=1)]
        report, sock, _ = self._run(
            [(b"h", LOOPBACK_PEER), (b"a", LOOPBACK_PEER)], mapping
        )
        self.assertEqual(report["reason"], "dynamic_peer_identity_changed")
        self.assertEqual(sock.sent, [])

    def test_dynamic_peer_is_learned_and_only_timesync_is_sent(self):
        packets = [
            (item, LOOPBACK_PEER)
            for item in (b"h", b"a", b"s", b"tr", b"tr2", b"tr3")
        ]
        report, sock, codec = self._run(packets)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["locked_peer"], [LOOPBACK_PEER[0], LOOPBACK_PEER[1]])
        self.assertEqual(report["target_system"], 1)
        self.assertEqual(report["target_component"], 1)
        self.assertEqual([item[0] for item in codec.encoded], [
            "PING", "TIMESYNC", "TIMESYNC", "TIMESYNC",
        ])
        self.assertEqual([peer for _, peer in sock.sent], [LOOPBACK_PEER] * 3)
        self.assertEqual(report["sent_message_types"], ["TIMESYNC"] * 3)
        self.assertFalse(report["ping_probe_required"])
        self.assertEqual(report["timesync_responses_matched"], 3)

    def test_peer_port_change_immediately_exits(self):
        packets = [(b"h", LOOPBACK_PEER), (b"a", OTHER_PEER)]
        report, sock, _ = self._run(packets)
        self.assertEqual(report["reason"], "dynamic_peer_changed")
        self.assertEqual(sock.sent, [])

    def test_armed_vehicle_exits_without_transmit(self):
        mapping = valid_mapping()
        mapping[b"h"] = [message("HEARTBEAT", autopilot=3, vehicle_type=2, armed=True)]
        report, sock, _ = self._run([(b"h", LOOPBACK_PEER)], mapping)
        self.assertEqual(report["reason"], "hardware_armed_detected")
        self.assertEqual(sock.sent, [])

    def test_ping_is_not_required_and_timesync_quorum_is_matched(self):
        packets = [
            (item, LOOPBACK_PEER)
            for item in (b"h", b"a", b"s", b"tr", b"tr2", b"tr3")
        ]
        report, _, _ = self._run(packets)
        self.assertFalse(report["ping_probe_required"])
        self.assertFalse(report["ping_response_matched"])
        self.assertEqual(report["timesync_responses_matched"], 3)
        self.assertTrue(report["timesync_response_matched"])

    def test_probe_response_timeout_fails_closed(self):
        packets = [(item, LOOPBACK_PEER) for item in (b"h", b"a", b"s")]
        report, sock, _ = self._run(packets)
        self.assertEqual(report["reason"], "insufficient_timesync_responses")
        self.assertEqual(len(sock.sent), 3)

    def test_keyboard_interrupt_closes_socket_and_writes_report(self):
        class InterruptSocket(FakeSocket):
            def recvfrom(self, size):
                raise KeyboardInterrupt()

        module = self._module()
        sock = InterruptSocket([])
        with tempfile.TemporaryDirectory() as tmp:
            report = module.run_dynamic_peer_probe(
                sock, FakeCodec({}), tmp, True, module.BENCH_CONFIRMATION,
                timeout_seconds=1.0, monotonic=FakeClock().monotonic,
            )
            self.assertTrue(Path(tmp, "command_link_verification_report.json").is_file())
        self.assertEqual(report["reason"], "keyboard_interrupt")
        self.assertTrue(sock.closed)

    def test_dynamic_profile_cli_is_bench_only_and_has_no_fixed_command_endpoint(self):
        cli = importlib.import_module("rpi_pixhawk_stage4.scripts.run_stage4")
        args = cli.build_parser().parse_args([
            "--mode", "bench-disarmed",
            "--link-profile", "rpanion-v07-dynamic-peer",
        ])
        self.assertEqual(args.link_profile, "rpanion-v07-dynamic-peer")
        self.assertIsNone(args.command_endpoint)
        with self.assertRaisesRegex(ValueError, "dynamic_peer_flight_live_rejected"):
            cli.validate_profile_arguments(cli.build_parser().parse_args([
                "--mode", "flight-live",
                "--link-profile", "rpanion-v07-dynamic-peer",
            ]))
        with self.assertRaisesRegex(ValueError, "dynamic_peer_command_endpoint_rejected"):
            cli.validate_profile_arguments(cli.build_parser().parse_args([
                "--mode", "bench-disarmed",
                "--link-profile", "rpanion-v07-dynamic-peer",
                "--command-endpoint", "udpout:127.0.0.1:16042",
            ]))
        with self.assertRaisesRegex(ValueError, "dynamic_peer_zero_setpoint_rejected"):
            cli.validate_profile_arguments(cli.build_parser().parse_args([
                "--mode", "bench-disarmed",
                "--link-profile", "rpanion-v07-dynamic-peer",
                "--enable-zero-setpoint-test",
            ]))

    def test_dynamic_profile_requires_exact_telemetry_endpoint(self):
        module = self._module()
        self.assertEqual(
            module.validate_dynamic_telemetry_endpoint("udpin:127.0.0.1:14556"),
            "udpin:127.0.0.1:14556",
        )
        for endpoint in (
            "udpin:localhost:14556", "udpin:127.0.0.1:14555",
            "udpout:127.0.0.1:14556", "serial:/dev/serial0:921600",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                module.validate_dynamic_telemetry_endpoint(endpoint)

    def test_dynamic_source_has_no_vehicle_command_or_setpoint_boundary(self):
        scanner = importlib.import_module("rpi_pixhawk_stage4.scripts.scan_stage4_safety")
        stage_dir = Path(__file__).resolve().parents[1]
        report = scanner.scan_rpanion_v07_sources(stage_dir)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["outbound_message_types"], ["PING", "TIMESYNC"])
        self.assertEqual(report["violations"], [])


if __name__ == "__main__":
    unittest.main()
