"""Bench-only Rpanion-server 0.7 dynamic UDP peer compatibility.

This module has exactly two outbound message encoders: PING and TIMESYNC.
It has no setpoint, command, mode, parameter, arming, or actuator API.
Importing it creates no socket and does not import pymavlink.
"""

from __future__ import annotations

import json
import select
import socket
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


LINK_PROFILE = "rpanion-v07-dynamic-peer"
REQUIRED_TELEMETRY_ENDPOINT = "udpin:127.0.0.1:14556"
BENCH_CONFIRMATION = "I_CONFIRM_PROPELLERS_REMOVED_AND_PIXHAWK_DISARMED"
ARDUPILOT_AUTOPILOT = 3
COPTER_VEHICLE_TYPE = 2
MAV_MODE_FLAG_SAFETY_ARMED = 128
REQUIRED_DISCOVERY_TYPES = {"HEARTBEAT", "ATTITUDE", "SYS_STATUS"}
LOCAL_SOURCE_SYSTEM = 245
LOCAL_SOURCE_COMPONENT = 190
PING_TARGET_SYSTEM = 0
PING_TARGET_COMPONENT = 0


def validate_dynamic_telemetry_endpoint(value: str) -> str:
    if value != REQUIRED_TELEMETRY_ENDPOINT:
        raise ValueError("rpanion_v07_requires_exact_telemetry_endpoint")
    return value


def _get(message: object, name: str, default: Any = None) -> Any:
    return message.get(name, default) if isinstance(message, dict) else getattr(message, name, default)


def _message_type(message: object) -> str:
    value = _get(message, "message_type")
    if value is None:
        getter = getattr(message, "get_type", None)
        value = getter() if callable(getter) else ""
    return str(value).upper()


def _identity(message: object) -> Tuple[int, int]:
    system_id = _get(message, "system_id")
    component_id = _get(message, "component_id")
    if system_id is None:
        getter = getattr(message, "get_srcSystem", None)
        system_id = getter() if callable(getter) else 0
    if component_id is None:
        getter = getattr(message, "get_srcComponent", None)
        component_id = getter() if callable(getter) else 0
    return int(system_id), int(component_id)


def _armed(message: object) -> bool:
    explicit = _get(message, "armed")
    if explicit is not None:
        return bool(explicit)
    return bool(int(_get(message, "base_mode", 0)) & MAV_MODE_FLAG_SAFETY_ARMED)


class PymavlinkDatagramCodec:
    """MAVLink byte codec; packing only, never owns or opens a transport."""

    def __init__(
        self,
        source_system: int = LOCAL_SOURCE_SYSTEM,
        source_component: int = LOCAL_SOURCE_COMPONENT,
    ) -> None:
        from pymavlink import mavutil
        self.source_system = int(source_system)
        self.source_component = int(source_component)
        if (self.source_system, self.source_component) == (1, 1):
            raise ValueError("local_source_identity_must_not_equal_vehicle_identity")
        self._mavlink = mavutil.mavlink
        self._parser = self._mavlink.MAVLink(None)
        self._encoder = self._mavlink.MAVLink(
            None, srcSystem=self.source_system, srcComponent=self.source_component
        )

    @staticmethod
    def _message_dict(message: object) -> Dict[str, Any]:
        return {
            "message_type": message.get_type(),
            "system_id": int(message.get_srcSystem()),
            "component_id": int(message.get_srcComponent()),
            "autopilot": int(getattr(message, "autopilot", -1)),
            "vehicle_type": int(getattr(message, "type", -1)),
            "armed": bool(int(getattr(message, "base_mode", 0)) & MAV_MODE_FLAG_SAFETY_ARMED),
            "sequence": int(getattr(message, "seq", -1)),
            "time_usec": int(getattr(message, "time_usec", -1)),
            "target_system": int(getattr(message, "target_system", -1)),
            "target_component": int(getattr(message, "target_component", -1)),
            "tc1": int(getattr(message, "tc1", -1)),
            "ts1": int(getattr(message, "ts1", -1)),
        }

    def decode_datagram(self, payload: bytes) -> List[Dict[str, Any]]:
        decoded = self._parser.parse_buffer(payload) or []
        return [self._message_dict(message) for message in decoded]

    def inspect_outbound(self, payload: bytes) -> Dict[str, Any]:
        parser = self._mavlink.MAVLink(None)
        decoded = parser.parse_buffer(payload) or []
        if len(decoded) != 1:
            raise ValueError("outbound_mavlink_frame_count_not_one")
        return self._message_dict(decoded[0])

    def encode_ping(self, sequence: int, time_usec: int) -> bytes:
        message = self._mavlink.MAVLink_ping_message(
            int(time_usec), int(sequence), PING_TARGET_SYSTEM, PING_TARGET_COMPONENT
        )
        return message.pack(self._encoder)

    def encode_timesync(self, token: int) -> bytes:
        message = self._mavlink.MAVLink_timesync_message(0, int(token))
        return message.pack(self._encoder)


def _base_report() -> Dict[str, Any]:
    return {
        "passed": False,
        "mode": "bench-disarmed",
        "link_profile": LINK_PROFILE,
        "telemetry_endpoint": REQUIRED_TELEMETRY_ENDPOINT,
        "command_endpoint": None,
        "dynamic_peer_locked": False,
        "locked_peer": None,
        "rpanion_dynamic_source_port": None,
        "target_system": 0,
        "target_component": 0,
        "dynamic_peer_stable": False,
        "vehicle_identity_stable": False,
        "local_source_system": LOCAL_SOURCE_SYSTEM,
        "local_source_component": LOCAL_SOURCE_COMPONENT,
        "ardupilot_copter_verified": False,
        "vehicle_disarmed_verified": False,
        "received_message_counts": {},
        "received_udp_datagrams": [],
        "received_ping_candidates": [],
        "received_timesync_candidates": [],
        "seen_unmatched_ping": False,
        "seen_timesync": False,
        "sent_message_types": [],
        "transmitted_setpoint_count": 0,
        "ping_probe_required": False,
        "ping_probe_reason": "unsupported_by_arducopter",
        "timesync_requests_sent": 0,
        "timesync_responses_matched": 0,
        "timesync_required_matches": 2,
        "timesync_request_records": [],
        "timesync_response_socket_sources": [],
        "timesync_rejected_candidates": [],
        "ping_request_time_usec": None,
        "ping_request_seq": None,
        "ping_request_target_system": PING_TARGET_SYSTEM,
        "ping_request_target_component": PING_TARGET_COMPONENT,
        "send_socket_local_address": None,
        "send_socket_remote_peer": None,
        "response_socket_source": None,
        "timesync_response_socket_source": None,
        "ping_response_matched": False,
        "timesync_response_matched": False,
        "rpanion_configuration_modified": False,
        "serial_device_opened": False,
        "flight_live_enabled": False,
        "timestamp": time.time(),
    }


def _write_report(results_dir: str, report: Dict[str, Any]) -> None:
    path = Path(results_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    (path / "command_link_verification_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )


def _socket_address(sock: object) -> Optional[List[Any]]:
    getter = getattr(sock, "getsockname", None)
    if not callable(getter):
        return None
    value = getter()
    return [str(value[0]), int(value[1])]


def _default_poller(sockets: List[object], timeout: float) -> List[object]:
    ready, _, _ = select.select(sockets, [], [], max(0.0, float(timeout)))
    return list(ready)


def run_dynamic_peer_probe(
    telemetry_sock: object,
    codec: object,
    results_dir: str,
    enable_probe: bool,
    confirmation: str,
    timeout_seconds: float = 3.0,
    monotonic: Callable[[], float] = time.monotonic,
    ping_sequence: Optional[int] = None,
    ping_time_usec: Optional[int] = None,
    timesync_token: Optional[int] = None,
    send_sock: Optional[object] = None,
    poller: Optional[Callable[[List[object], float], List[object]]] = None,
) -> Dict[str, Any]:
    """Learn one stable peer and require two strict TIMESYNC replies out of three."""
    report = _base_report()
    counts: Counter = Counter()
    candidate_peer: Optional[Tuple[str, int]] = None
    target_identity: Optional[Tuple[int, int]] = None
    required_seen = set()
    phase = "discovery"
    if send_sock is None:
        send_sock = telemetry_sock
    if poller is None:
        poller = _default_poller if send_sock is not telemetry_sock else (
            lambda sockets, timeout: list(sockets)
        )
    response_sockets = [telemetry_sock]
    if send_sock is not telemetry_sock:
        response_sockets.append(send_sock)
    socket_names = {
        id(telemetry_sock): "telemetry_14556",
        id(send_sock): (
            "telemetry_14556" if send_sock is telemetry_sock
            else "dynamic_send_ephemeral"
        ),
    }
    if ping_sequence is None:
        ping_sequence = int(time.time() * 1000000) & 0xFFFFFFFF
    if ping_time_usec is None:
        ping_time_usec = int(time.time() * 1000000)
    token_base = int(timesync_token or int(time.monotonic() * 1000000000))
    if token_base <= 0:
        token_base = 1
    timesync_tokens = [token_base + index for index in range(3)]
    per_attempt_timeout = min(1.5, max(0.1, float(timeout_seconds)))

    def fail(reason: str) -> Dict[str, Any]:
        report["reason"] = reason
        return report

    def accept_telemetry(message: object, peer: Tuple[str, int]) -> Optional[str]:
        nonlocal candidate_peer, target_identity
        if peer[0] != "127.0.0.1":
            return "wrong_udp_source"
        kind = _message_type(message)
        if not kind or kind == "BAD_DATA":
            return None
        identity = _identity(message)
        if identity[0] <= 0 or identity[1] <= 0:
            return "invalid_mavlink_identity"
        if kind == "HEARTBEAT":
            if int(_get(message, "autopilot", -1)) != ARDUPILOT_AUTOPILOT or int(
                _get(message, "vehicle_type", -1)
            ) != COPTER_VEHICLE_TYPE:
                return "not_ardupilot_copter"
            if _armed(message):
                return "hardware_armed_detected"
            report["vehicle_disarmed_verified"] = True
            report["ardupilot_copter_verified"] = True
        if candidate_peer is None:
            if kind != "HEARTBEAT":
                return None
            candidate_peer = (str(peer[0]), int(peer[1]))
            target_identity = identity
        elif peer != candidate_peer:
            report["dynamic_peer_stable"] = False
            return "dynamic_peer_changed"
        if target_identity is not None and identity != target_identity:
            report["vehicle_identity_stable"] = False
            return "dynamic_peer_identity_changed"
        counts[kind] += 1
        if kind in REQUIRED_DISCOVERY_TYPES:
            required_seen.add(kind)
        return None

    def record_datagram(source_sock: object, peer: Tuple[str, int], message: object) -> None:
        report["received_udp_datagrams"].append({
            "response_socket_source": socket_names[id(source_sock)],
            "udp_source": [str(peer[0]), int(peer[1])],
            "message_type": _message_type(message),
            "system_id": _identity(message)[0],
            "component_id": _identity(message)[1],
        })

    def reject_candidate(
        attempt: int,
        source_sock: object,
        peer: Tuple[str, int],
        reasons: List[str],
        message: Optional[object] = None,
    ) -> None:
        candidate = {
            "attempt": attempt,
            "response_socket_source": socket_names[id(source_sock)],
            "udp_source": [str(peer[0]), int(peer[1])],
            "rejection_reasons": reasons,
        }
        if message is not None:
            candidate.update({
                "header_system": _identity(message)[0],
                "header_component": _identity(message)[1],
                "tc1": int(_get(message, "tc1", -1)),
                "ts1": int(_get(message, "ts1", -1)),
            })
        report["timesync_rejected_candidates"].append(candidate)
        report["received_timesync_candidates"].append(dict(candidate, accepted=False))

    def receive_timesync(
        deadline: float,
        request_token: int,
        attempt: int,
    ) -> Tuple[Optional[str], Optional[str]]:
        while monotonic() < deadline:
            ready = poller(response_sockets, min(0.2, max(0.0, deadline - monotonic())))
            if not ready:
                continue
            for response_sock in ready:
                try:
                    payload, peer = response_sock.recvfrom(65535)
                except (TimeoutError, socket.timeout):
                    continue
                if peer[0] != "127.0.0.1":
                    reject_candidate(attempt, response_sock, peer, ["wrong_udp_source"])
                    return "wrong_udp_source", None
                if candidate_peer is not None and peer != candidate_peer:
                    report["dynamic_peer_stable"] = False
                    reject_candidate(attempt, response_sock, peer, ["peer_changed"])
                    return "dynamic_peer_changed", None
                try:
                    messages = codec.decode_datagram(payload)
                except Exception:
                    reject_candidate(attempt, response_sock, peer, ["decode_error"])
                    return "decode_error", None
                for message in messages:
                    record_datagram(response_sock, peer, message)
                    kind = _message_type(message)
                    if kind != "TIMESYNC":
                        error = accept_telemetry(message, peer)
                        if error is not None:
                            return error, None
                        continue
                    report["seen_timesync"] = True
                    counts[kind] += 1
                    reasons: List[str] = []
                    identity = _identity(message)
                    if target_identity is None or identity[0] != target_identity[0]:
                        reasons.append("wrong_header_system")
                    if target_identity is None or identity[1] != target_identity[1]:
                        reasons.append("wrong_header_component")
                    if int(_get(message, "tc1", 0)) == 0:
                        reasons.append("tc1_zero")
                    if int(_get(message, "ts1", -1)) != request_token:
                        reasons.append("wrong_ts1")
                    if reasons:
                        reject_candidate(attempt, response_sock, peer, reasons, message)
                        continue
                    accepted = {
                        "attempt": attempt,
                        "response_socket_source": socket_names[id(response_sock)],
                        "udp_source": [str(peer[0]), int(peer[1])],
                        "header_system": identity[0],
                        "header_component": identity[1],
                        "tc1": int(_get(message, "tc1", -1)),
                        "ts1": int(_get(message, "ts1", -1)),
                        "accepted": True,
                        "rejection_reasons": [],
                    }
                    report["received_timesync_candidates"].append(accepted)
                    return None, socket_names[id(response_sock)]
        return None, None

    try:
        if not enable_probe:
            return fail("bench_probe_not_enabled")
        if confirmation != BENCH_CONFIRMATION:
            return fail("bench_confirmation_phrase_mismatch")
        telemetry_sock.settimeout(min(0.2, max(0.01, float(timeout_seconds))))
        send_sock.settimeout(min(0.2, max(0.01, float(timeout_seconds))))
        report["send_socket_local_address"] = _socket_address(send_sock)
        discovery_deadline = monotonic() + float(timeout_seconds)
        while monotonic() < discovery_deadline and required_seen != REQUIRED_DISCOVERY_TYPES:
            try:
                payload, peer = telemetry_sock.recvfrom(65535)
            except (TimeoutError, socket.timeout):
                continue
            if peer[0] != "127.0.0.1":
                return fail("wrong_udp_source")
            try:
                messages = codec.decode_datagram(payload)
            except Exception:
                return fail("decode_error")
            for message in messages:
                record_datagram(telemetry_sock, peer, message)
                error = accept_telemetry(message, peer)
                if error is not None:
                    return fail(error)
        if required_seen != REQUIRED_DISCOVERY_TYPES or candidate_peer is None or target_identity is None:
            return fail("dynamic_peer_discovery_timeout")
        report["dynamic_peer_locked"] = True
        report["dynamic_peer_stable"] = True
        report["vehicle_identity_stable"] = True
        report["locked_peer"] = [candidate_peer[0], candidate_peer[1]]
        report["rpanion_dynamic_source_port"] = candidate_peer[1]
        report["target_system"], report["target_component"] = target_identity
        report["send_socket_remote_peer"] = [candidate_peer[0], candidate_peer[1]]

        # Retain and verify PING encoding, but never transmit it. ArduCopter's
        # missing PING response is not a bench gate.
        phase = "ping_byte_audit"
        ping_payload = codec.encode_ping(ping_sequence, ping_time_usec)
        ping_audit = codec.inspect_outbound(ping_payload)
        report["ping_request_time_usec"] = int(_get(ping_audit, "time_usec", -1))
        report["ping_request_seq"] = int(_get(ping_audit, "sequence", -1))
        report["ping_request_target_system"] = int(_get(ping_audit, "target_system", -1))
        report["ping_request_target_component"] = int(_get(ping_audit, "target_component", -1))
        report["local_source_system"] = int(_get(ping_audit, "system_id", -1))
        report["local_source_component"] = int(_get(ping_audit, "component_id", -1))
        if (
            _message_type(ping_audit) != "PING"
            or report["ping_request_target_system"] != PING_TARGET_SYSTEM
            or report["ping_request_target_component"] != PING_TARGET_COMPONENT
            or (report["local_source_system"], report["local_source_component"])
            != (LOCAL_SOURCE_SYSTEM, LOCAL_SOURCE_COMPONENT)
        ):
            return fail("ping_outbound_byte_audit_failed")

        phase = "timesync"
        for index, request_token in enumerate(timesync_tokens):
            attempt = index + 1
            payload = codec.encode_timesync(request_token)
            audit = codec.inspect_outbound(payload)
            if (
                _message_type(audit) != "TIMESYNC"
                or int(_get(audit, "tc1", -1)) != 0
                or int(_get(audit, "ts1", -1)) != request_token
                or _identity(audit) != (LOCAL_SOURCE_SYSTEM, LOCAL_SOURCE_COMPONENT)
            ):
                return fail("timesync_outbound_byte_audit_failed")
            record = {
                "attempt": attempt,
                "ts1": request_token,
                "sent": True,
                "matched": False,
                "response_socket_source": None,
                "timed_out": False,
            }
            report["timesync_request_records"].append(record)
            # Keep two statically audited send sites while both branches send
            # the same permitted TIMESYNC message type. PING is never sent.
            if index == 0:
                send_sock.sendto(payload, candidate_peer)
            else:
                send_sock.sendto(payload, candidate_peer)
            report["sent_message_types"].append("TIMESYNC")
            report["timesync_requests_sent"] += 1
            error, response_source = receive_timesync(
                monotonic() + per_attempt_timeout, request_token, attempt
            )
            if error is not None:
                return fail(error)
            if response_source is None:
                record["timed_out"] = True
                continue
            record["matched"] = True
            record["response_socket_source"] = response_source
            report["timesync_responses_matched"] += 1
            report["timesync_response_socket_sources"].append(response_source)
            report["timesync_response_socket_source"] = response_source

        report["timesync_response_matched"] = (
            report["timesync_responses_matched"] >= report["timesync_required_matches"]
        )
        if not report["timesync_response_matched"]:
            return fail("insufficient_timesync_responses")
        report["passed"] = True
        report["reason"] = "timesync_quorum_verified"
        return report
    except KeyboardInterrupt:
        return fail("keyboard_interrupt")
    except Exception as exc:
        report["error"] = repr(exc)
        return fail("dynamic_peer_{0}_error".format(phase))
    finally:
        report["received_message_counts"] = dict(sorted(counts.items()))
        closed = set()
        for current in (telemetry_sock, send_sock):
            if id(current) in closed:
                continue
            closed.add(id(current))
            close = getattr(current, "close", None)
            if callable(close):
                close()
        _write_report(results_dir, report)

def run_dynamic_peer_endpoint(
    telemetry_endpoint: str,
    results_dir: str,
    enable_probe: bool,
    confirmation: str,
    timeout_seconds: float = 3.0,
) -> Dict[str, Any]:
    """Production factory; called only after CLI profile gates pass."""
    validate_dynamic_telemetry_endpoint(telemetry_endpoint)
    if not enable_probe:
        report = _base_report()
        report["reason"] = "bench_probe_not_enabled"
        _write_report(results_dir, report)
        return report
    if confirmation != BENCH_CONFIRMATION:
        report = _base_report()
        report["reason"] = "bench_confirmation_phrase_mismatch"
        _write_report(results_dir, report)
        return report
    telemetry_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        telemetry_sock.bind(("127.0.0.1", 14556))
        send_sock.bind(("127.0.0.1", 0))
        codec = PymavlinkDatagramCodec()
    except Exception:
        telemetry_sock.close()
        send_sock.close()
        raise
    return run_dynamic_peer_probe(
        telemetry_sock=telemetry_sock,
        send_sock=send_sock,
        codec=codec,
        results_dir=results_dir,
        enable_probe=True,
        confirmation=confirmation,
        timeout_seconds=timeout_seconds,
    )
