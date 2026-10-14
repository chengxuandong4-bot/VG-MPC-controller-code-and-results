"""Standalone receive-only telemetry source for the Rpanion UDP output."""

from __future__ import annotations

import math
import time
from collections import Counter
from typing import Optional

from rpi_pixhawk_stage4.safety.endpoint_guard import validate_telemetry_endpoint
from .state import HardwareState


class ReadonlyHardwareTelemetry:
    """Never owns a transmitter and never calls a MAVLink send method."""

    TARGET_TYPES = {
        "HEARTBEAT", "ATTITUDE", "LOCAL_POSITION_NED", "GLOBAL_POSITION_INT",
        "GPS_RAW_INT", "SYS_STATUS", "BATTERY_STATUS", "RC_CHANNELS",
        "EKF_STATUS_REPORT",
    }

    def __init__(self, endpoint: str, heartbeat_timeout_s: float = 1.5,
                 mavutil_module: object = None, monotonic: object = time.monotonic,
                 connection: object = None) -> None:
        self.endpoint = validate_telemetry_endpoint(endpoint)
        self._connection = connection
        self._mavutil = mavutil_module
        self._monotonic = monotonic
        self._heartbeat_timeout_s = float(heartbeat_timeout_s)
        self._state = HardwareState()
        self._last_any = 0.0
        self._last_rc = 0.0
        self._target_identity = None
        self.message_counts = Counter()

    def open(self) -> None:
        if self._mavutil is None:
            from pymavlink import mavutil
            self._mavutil = mavutil
        self._connection = self._mavutil.mavlink_connection(self.endpoint)

    @staticmethod
    def _source_identity(msg: object) -> object:
        get_system = getattr(msg, "get_srcSystem", None)
        get_component = getattr(msg, "get_srcComponent", None)
        return (
            int(get_system()) if callable(get_system) else 0,
            int(get_component()) if callable(get_component) else 0,
        )

    def _accept_identity(self, msg: object, kind: str) -> bool:
        source_identity = self._source_identity(msg)
        if kind == "HEARTBEAT":
            if int(getattr(msg, "autopilot", -1)) != 3:
                return False
            if int(getattr(msg, "type", -1)) != 2:
                return False
            if self._target_identity is None:
                self._target_identity = source_identity
            return source_identity == self._target_identity
        return self._target_identity is not None and source_identity == self._target_identity

    def _update(self, msg: object, now: float) -> None:
        kind = msg.get_type()
        if kind not in self.TARGET_TYPES or not self._accept_identity(msg, kind):
            return
        self.message_counts[kind] += 1
        self._last_any = now
        state = self._state
        if kind == "HEARTBEAT":
            state.system_id, state.component_id = self._source_identity(msg)
            state.autopilot = int(getattr(msg, "autopilot", -1))
            state.vehicle_type = int(getattr(msg, "type", -1))
            state.armed = bool(int(getattr(msg, "base_mode", 0)) & 128)
            state.flight_mode = str(self._mavutil.mode_string_v10(msg)).upper()
            state.last_heartbeat_monotonic_s = now
        elif kind == "ATTITUDE":
            state.time_boot_ms = int(getattr(msg, "time_boot_ms", state.time_boot_ms))
            state.roll_rad = float(getattr(msg, "roll", state.roll_rad))
            state.pitch_rad = float(getattr(msg, "pitch", state.pitch_rad))
            state.yaw_rad = float(getattr(msg, "yaw", state.yaw_rad))
        elif kind == "LOCAL_POSITION_NED":
            state.time_boot_ms = int(getattr(msg, "time_boot_ms", state.time_boot_ms))
            state.north_m = float(getattr(msg, "x", state.north_m))
            state.east_m = float(getattr(msg, "y", state.east_m))
            state.down_m = float(getattr(msg, "z", state.down_m))
            state.vn_mps = float(getattr(msg, "vx", state.vn_mps))
            state.ve_mps = float(getattr(msg, "vy", state.ve_mps))
            state.vd_mps = float(getattr(msg, "vz", state.vd_mps))
            state.local_position_valid = all(math.isfinite(value) for value in (
                state.north_m, state.east_m, state.down_m,
                state.vn_mps, state.ve_mps, state.vd_mps,
            ))
            state.horizontal_speed_m_s = math.hypot(state.vn_mps, state.ve_mps)
            state.vertical_speed_m_s = abs(state.vd_mps)
        elif kind == "GLOBAL_POSITION_INT":
            relative_alt = float(getattr(msg, "relative_alt", math.nan)) / 1000.0
            state.relative_alt_valid = math.isfinite(relative_alt)
            if state.relative_alt_valid:
                state.relative_alt_m = relative_alt
        elif kind == "GPS_RAW_INT":
            state.gps_fix_type = int(getattr(msg, "fix_type", 0))
        elif kind in ("SYS_STATUS", "BATTERY_STATUS"):
            state.battery_remaining = int(
                getattr(msg, "battery_remaining", state.battery_remaining)
            )
        elif kind == "RC_CHANNELS":
            count = int(getattr(msg, "chancount", 18))
            channels = {}
            for index in range(1, min(count, 18) + 1):
                value = int(getattr(msg, "chan{0}_raw".format(index), 0))
                if value not in (0, 65535):
                    channels[index] = value
            state.rc_channels = channels
            state.rc_valid = bool(channels)
            self._last_rc = now
        elif kind == "EKF_STATUS_REPORT":
            flags = int(getattr(msg, "flags", 0))
            core = (flags & 0x07) == 0x07
            horizontal_position = bool(flags & (0x08 | 0x10))
            vertical_position = bool(flags & (0x20 | 0x40))
            state.ekf_ok = bool(core and horizontal_position and vertical_position)
        state.received_monotonic_s = now

    def _refresh_ages(self, now: float) -> None:
        self._state.telemetry_age_s = now - self._last_any if self._last_any else 99.0
        self._state.rc_age_s = now - self._last_rc if self._last_rc else 99.0
        heartbeat = self._state.last_heartbeat_monotonic_s
        self._state.heartbeat_age_s = now - heartbeat if heartbeat else 99.0
        self._state.heartbeat_fresh = bool(
            heartbeat and self._state.heartbeat_age_s <= self._heartbeat_timeout_s
        )

    def next_state(self, timeout: float = 0.1) -> Optional[HardwareState]:
        if self._connection is None:
            self.open()
        deadline = self._monotonic() + max(0.0, float(timeout))
        received = False
        while self._monotonic() <= deadline:
            remaining = max(0.0, deadline - self._monotonic())
            msg = self._connection.recv_match(
                blocking=True, timeout=min(remaining, 0.05)
            )
            if msg is None:
                break
            received = True
            self._update(msg, self._monotonic())
        now = self._monotonic()
        self._refresh_ages(now)
        return self._state if received or self._last_any else None

    def close(self) -> None:
        if self._connection is not None:
            close = getattr(self._connection, "close", None)
            if callable(close):
                close()
            self._connection = None
