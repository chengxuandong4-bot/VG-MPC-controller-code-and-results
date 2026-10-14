from __future__ import annotations

import importlib
import importlib.util
import tempfile
import unittest


PEER = ("127.0.0.1", 35883)
CHANGED_PEER = ("127.0.0.1", 35884)
LOCAL_SOURCE = (245, 190)
TOKEN_BASE = 987654321


class FakeClock:
    def __init__(self) -> None:
        self.now = 10.0

    def monotonic(self) -> float:
        self.now += 0.05
        return self.now


class FakeSocket:
    def __init__(self, name, local_address, packets=()) -> None:
        self.name = name
        self.local_address = local_address
        self.packets = list(packets)
        self.sent = []
        self.closed = False
        self.on_dispatch = None

    def settimeout(self, value) -> None:
        del value

    def recvfrom(self, size):
        del size
        if not self.packets:
            raise TimeoutError("empty fake socket")
        return self.packets.pop(0)

    def sendto(self, payload, peer):
        self.sent.append((payload, peer))
        if self.on_dispatch is not None:
            self.on_dispatch(len(self.sent) - 1)
        return len(payload)

    def getsockname(self):
        return self.local_address

    def close(self) -> None:
        self.closed = True


class FakePoller:
    def __call__(self, sockets, timeout):
        del timeout
        return [sock for sock in sockets if sock.packets]


def message(kind, system_id=1, component_id=1, **fields):
    value = {
        "message_type": kind,
        "system_id": system_id,
        "component_id": component_id,
        "autopilot": -1,
        "vehicle_type": -1,
        "armed": False,
    }
    value.update(fields)
    return value


class FakeCodec:
    source_system = LOCAL_SOURCE[0]
    source_component = LOCAL_SOURCE[1]

    def __init__(self, mapping):
        self.mapping = mapping

    def decode_datagram(self, payload):
        value = self.mapping.get(payload, [])
        if isinstance(value, Exception):
            raise value
        return list(value)

    def encode_ping(self, sequence, time_usec):
        return ("PING:{0}:{1}".format(sequence, time_usec)).encode("ascii")

    def encode_timesync(self, token):
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


def base_mapping(overrides=None):
    value = {
        b"h": [message("HEARTBEAT", autopilot=3, vehicle_type=2, armed=False)],
        b"a": [message("ATTITUDE")],
        b"s": [message("SYS_STATUS")],
        b"t1": [message("TIMESYNC", tc1=111, ts1=TOKEN_BASE)],
        b"t2": [message("TIMESYNC", tc1=222, ts1=TOKEN_BASE + 1)],
        b"t3": [message("TIMESYNC", tc1=333, ts1=TOKEN_BASE + 2)],
    }
    value.update(overrides or {})
    return value


class RpanionV07TimesyncProbeTests(unittest.TestCase):
    def _module(self):
        return importlib.import_module("rpi_pixhawk_stage4.transport.rpanion_v07_dynamic_peer")

    def _run(self, responses_by_attempt=None, codec_mapping=None, armed=False):
        module = self._module()
        mapping = codec_mapping or base_mapping()
        if armed:
            mapping = dict(mapping)
            mapping[b"h"] = [message("HEARTBEAT", autopilot=3, vehicle_type=2, armed=True)]
        telemetry = FakeSocket(
            "telemetry_14556", ("127.0.0.1", 14556),
            [(b"h", PEER), (b"a", PEER), (b"s", PEER)],
        )
        sender = FakeSocket("dynamic_send_ephemeral", ("127.0.0.1", 49152))
        schedule = responses_by_attempt or [[], [], []]

        def dispatch(attempt):
            if attempt >= len(schedule):
                return
            for socket_name, payload, peer in schedule[attempt]:
                target = telemetry if socket_name == "telemetry_14556" else sender
                target.packets.append((payload, peer))

        sender.on_dispatch = dispatch
        with tempfile.TemporaryDirectory() as tmp:
            report = module.run_dynamic_peer_probe(
                telemetry_sock=telemetry,
                send_sock=sender,
                codec=FakeCodec(mapping),
                results_dir=tmp,
                enable_probe=True,
                confirmation=module.BENCH_CONFIRMATION,
                timeout_seconds=0.5,
                monotonic=FakeClock().monotonic,
                timesync_token=TOKEN_BASE,
                poller=FakePoller(),
            )
        return report, telemetry, sender

    @staticmethod
    def _three_good(socket_name="dynamic_send_ephemeral"):
        return [[(socket_name, b"t1", PEER)], [(socket_name, b"t2", PEER)], [(socket_name, b"t3", PEER)]]

    def test_ping_nonresponse_does_not_block_timesync(self):
        report, _, sender = self._run(self._three_good())
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["timesync_responses_matched"], 3)
        self.assertNotIn("PING", report["sent_message_types"])
        self.assertEqual(len(sender.sent), 3)

    def test_default_bench_does_not_send_ping(self):
        report, _, sender = self._run(self._three_good())
        self.assertFalse(report["ping_probe_required"])
        self.assertEqual(report["ping_probe_reason"], "unsupported_by_arducopter")
        self.assertTrue(all(not payload.startswith(b"PING:") for payload, _ in sender.sent))

    def test_strict_timesync_response_matches(self):
        report, _, _ = self._run(self._three_good())
        self.assertEqual(report["timesync_requests_sent"], 3)
        self.assertEqual(report["timesync_responses_matched"], 3)
        self.assertEqual(report["timesync_required_matches"], 2)
        self.assertEqual([row["ts1"] for row in report["timesync_request_records"]], [TOKEN_BASE, TOKEN_BASE + 1, TOKEN_BASE + 2])

    def test_unsolicited_broadcast_timesync_is_not_a_response(self):
        mapping = base_mapping({b"broadcast": [message("TIMESYNC", tc1=0, ts1=42)]})
        schedule = [[("dynamic_send_ephemeral", b"broadcast", PEER), ("dynamic_send_ephemeral", b"t1", PEER)], [("dynamic_send_ephemeral", b"t2", PEER)], []]
        report, _, _ = self._run(schedule, mapping)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["timesync_responses_matched"], 2)
        self.assertIn("tc1_zero", report["timesync_rejected_candidates"][0]["rejection_reasons"])

    def test_wrong_ts1_is_rejected(self):
        mapping = base_mapping({b"wrong": [message("TIMESYNC", tc1=12, ts1=99)]})
        report, _, _ = self._run([[('dynamic_send_ephemeral', b"wrong", PEER)], [], []], mapping)
        self.assertFalse(report["passed"])
        self.assertIn("wrong_ts1", report["timesync_rejected_candidates"][0]["rejection_reasons"])

    def test_tc1_zero_is_rejected(self):
        mapping = base_mapping({b"zero": [message("TIMESYNC", tc1=0, ts1=TOKEN_BASE)]})
        report, _, _ = self._run([[('dynamic_send_ephemeral', b"zero", PEER)], [], []], mapping)
        self.assertFalse(report["passed"])
        self.assertIn("tc1_zero", report["timesync_rejected_candidates"][0]["rejection_reasons"])

    def test_wrong_header_is_rejected(self):
        mapping = base_mapping({b"wrong": [message("TIMESYNC", system_id=2, component_id=3, tc1=1, ts1=TOKEN_BASE)]})
        schedule = [[("dynamic_send_ephemeral", b"wrong", PEER), ("dynamic_send_ephemeral", b"t1", PEER)], [("dynamic_send_ephemeral", b"t2", PEER)], []]
        report, _, _ = self._run(schedule, mapping)
        self.assertTrue(report["passed"], report)
        reasons = report["timesync_rejected_candidates"][0]["rejection_reasons"]
        self.assertIn("wrong_header_system", reasons)
        self.assertIn("wrong_header_component", reasons)

    def test_telemetry_socket_responses_pass(self):
        report, _, _ = self._run(self._three_good("telemetry_14556"))
        self.assertTrue(report["passed"], report)
        self.assertEqual(set(report["timesync_response_socket_sources"]), {"telemetry_14556"})

    def test_ephemeral_socket_responses_pass(self):
        report, _, _ = self._run(self._three_good())
        self.assertTrue(report["passed"], report)
        self.assertEqual(set(report["timesync_response_socket_sources"]), {"dynamic_send_ephemeral"})

    def test_two_of_three_matches_pass(self):
        schedule = [[("dynamic_send_ephemeral", b"t1", PEER)], [], [("dynamic_send_ephemeral", b"t3", PEER)]]
        report, _, _ = self._run(schedule)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["timesync_responses_matched"], 2)

    def test_fewer_than_two_matches_fail(self):
        report, _, _ = self._run([[('dynamic_send_ephemeral', b"t1", PEER)], [], []])
        self.assertFalse(report["passed"])
        self.assertEqual(report["reason"], "insufficient_timesync_responses")
        self.assertEqual(report["timesync_responses_matched"], 1)

    def test_peer_change_fails_closed(self):
        report, _, _ = self._run([[('dynamic_send_ephemeral', b"t1", CHANGED_PEER)], [], []])
        self.assertFalse(report["passed"])
        self.assertEqual(report["reason"], "dynamic_peer_changed")
        self.assertFalse(report["dynamic_peer_stable"])

    def test_armed_discovery_sends_nothing(self):
        report, _, sender = self._run(self._three_good(), armed=True)
        self.assertEqual(report["reason"], "hardware_armed_detected")
        self.assertEqual(sender.sent, [])

    def test_flight_live_remains_rejected(self):
        cli = importlib.import_module("rpi_pixhawk_stage4.scripts.run_stage4")
        args = cli.build_parser().parse_args(["--mode", "flight-live", "--link-profile", "rpanion-v07-dynamic-peer"])
        with self.assertRaisesRegex(ValueError, "dynamic_peer_flight_live_rejected"):
            cli.validate_profile_arguments(args)

    def test_dynamic_profile_static_scan_has_no_control_send(self):
        scanner = importlib.import_module("rpi_pixhawk_stage4.scripts.scan_stage4_safety")
        stage_dir = importlib.import_module("pathlib").Path(__file__).resolve().parents[1]
        report = scanner.scan_rpanion_v07_sources(stage_dir)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["violations"], [])

    @unittest.skipUnless(importlib.util.find_spec("pymavlink"), "pymavlink not installed")
    def test_real_pymavlink_ping_encoding_is_retained(self):
        module = self._module()
        codec = module.PymavlinkDatagramCodec()
        decoded = codec.inspect_outbound(codec.encode_ping(7, 123456))
        self.assertEqual((decoded["target_system"], decoded["target_component"]), (0, 0))
        self.assertEqual((decoded["system_id"], decoded["component_id"]), (245, 190))


if __name__ == "__main__":
    unittest.main()