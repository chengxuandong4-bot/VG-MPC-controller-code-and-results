"""THE ONLY Stage 4 file allowed to transmit MAVLink messages.

Importing this module does not open a connection. Construction requires an
explicit, validated localhost udpout endpoint. It contains no arming, takeoff,
mode, landing, parameter, motor-test, force, moment, or PWM operation.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from rpi_pixhawk_stage4.safety.endpoint_guard import validate_command_endpoint


class PymavlinkCommandBoundary:
    def __init__(self, endpoint: str, target_system: int = 0, target_component: int = 0) -> None:
        self.endpoint = validate_command_endpoint(endpoint)
        self._connection = None
        self.target_system = int(target_system)
        self.target_component = int(target_component)
        self.closed = False

    def open(self) -> None:
        if self._connection is not None:
            return
        from pymavlink import mavutil
        self._mavutil = mavutil
        self._connection = mavutil.mavlink_connection(self.endpoint)

    def probe_command_link(self, timeout_seconds: float = 3.0) -> Dict[str, Any]:
        self.open()
        sequence = int(time.time() * 1000000) & 0xFFFFFFFF
        sent_at = time.monotonic()
        self._connection.mav.ping_send(sequence, 0, 0, 0)
        self._connection.mav.timesync_send(0, int(time.time() * 1e9))
        deadline = time.monotonic() + float(timeout_seconds)
        response_types = []
        while time.monotonic() < deadline:
            msg = self._connection.recv_match(blocking=True, timeout=0.2)
            if msg is None:
                continue
            kind = msg.get_type()
            get_system = getattr(msg, "get_srcSystem", None)
            get_component = getattr(msg, "get_srcComponent", None)
            if callable(get_system) and int(get_system()) > 0:
                self.target_system = int(get_system())
                self.target_component = int(get_component()) if callable(get_component) else 0
            if kind in ("PING", "TIMESYNC", "HEARTBEAT"):
                response_types.append(kind)
            if kind in ("PING", "TIMESYNC") and self.target_system > 0 and self.target_component > 0:
                return {
                    "passed": True,
                    "endpoint": self.endpoint,
                    "response_types": response_types,
                    "target_system": self.target_system,
                    "target_component": self.target_component,
                    "round_trip_s": time.monotonic() - sent_at,
                }
        return {
            "passed": False,
            "endpoint": self.endpoint,
            "response_types": response_types,
            "target_system": self.target_system,
            "target_component": self.target_component,
            "reason": "no_bidirectional_probe_response",
        }

    def send_velocity(self, target: object) -> None:
        self.open()
        if self.target_system <= 0 or self.target_component <= 0:
            raise RuntimeError("command_target_not_verified")
        ignore_position_accel_yaw = 0b0000010111000111
        self._connection.mav.set_position_target_local_ned_send(
            0,
            self.target_system,
            self.target_component,
            self._mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            ignore_position_accel_yaw,
            0.0, 0.0, 0.0,
            float(getattr(target, "vx_north_mps")),
            float(getattr(target, "vy_east_mps")),
            float(getattr(target, "vz_down_mps")),
            0.0, 0.0, 0.0,
            0.0,
            float(getattr(target, "yaw_rate_radps")),
        )

    def send_zero_velocity(self, reason: str) -> None:
        from rpi_pixhawk_stage4.controller.limits import LimitedTarget
        self.send_velocity(LimitedTarget(0.0, 0.0, 0.0, 0.0, False, False))

    def close(self) -> None:
        if self._connection is not None:
            close = getattr(self._connection, "close", None)
            if callable(close):
                close()
        self._connection = None
        self.closed = True
