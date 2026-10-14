"""Strict endpoint validation for the two independent Stage 4 links."""

from __future__ import annotations

import re
from typing import Tuple


_ENDPOINT = re.compile(r"^(?P<scheme>[a-z]+):(?P<host>[^:]+):(?P<port>[0-9]+)$", re.I)
_BLOCKED_MARKERS = ("com", "tty", "/dev", "serial", "usb")
_LOOPBACK = ("127.0.0.1", "localhost")


def _parse(value: str, required_scheme: str) -> Tuple[str, int]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("endpoint_required")
    lowered = value.strip().lower()
    if any(marker in lowered for marker in _BLOCKED_MARKERS):
        raise ValueError("serial_or_usb_endpoint_rejected")
    match = _ENDPOINT.fullmatch(lowered)
    if match is None:
        raise ValueError("malformed_endpoint")
    if match.group("scheme") != required_scheme:
        raise ValueError("wrong_endpoint_mode")
    host = match.group("host")
    if host not in _LOOPBACK:
        raise ValueError("non_localhost_endpoint_rejected")
    port = int(match.group("port"))
    if not 1 <= port <= 65535:
        raise ValueError("invalid_port")
    return host, port


def validate_command_endpoint(value: str) -> str:
    """Accept only an explicit loopback udpout endpoint."""
    host, port = _parse(value, "udpout")
    return "udpout:{0}:{1}".format(host, port)


def validate_telemetry_endpoint(value: str) -> str:
    """Accept only a loopback receive-only udpin endpoint."""
    host, port = _parse(value, "udpin")
    return "udpin:{0}:{1}".format(host, port)
