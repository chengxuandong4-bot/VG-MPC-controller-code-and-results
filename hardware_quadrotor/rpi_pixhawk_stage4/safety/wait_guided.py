"""Pure one-shot state machine for ground-started GUIDED-edge activation.

This module owns no transport and cannot transmit MAVLink messages. It turns
normalized telemetry into an explicit authorization decision which the caller
may use only after completing the separate command-link authorization gates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from .state_machine import FlightState


def _get(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


@dataclass(frozen=True)
class WaitGuidedConfig:
    wait_timeout_s: float = 600.0
    heartbeat_timeout_s: float = 1.5
    min_airborne_alt_m: float = 1.0
    min_abort_alt_m: float = 0.70
    stable_hover_duration_s: float = 2.0
    max_horizontal_speed_m_s: float = 0.50
    max_vertical_speed_m_s: float = 0.30
    max_roll_pitch_deg: float = 10.0
    guided_settle_s: float = 0.50
    max_control_duration_s: float = 5.0
    zero_command_count: int = 3
    zero_command_period_s: float = 0.10
    one_shot: bool = True


class WaitGuidedMachine:
    """Require a real non-GUIDED -> GUIDED edge after stable airborne hover."""

    def __init__(self, config: WaitGuidedConfig,
                 started_monotonic: float) -> None:
        self.config = config
        self.started_monotonic = float(started_monotonic)
        self.state = FlightState.INIT
        self.previous_mode: Optional[str] = None
        self.seen_non_guided_since_start = False
        self.guided_edge_detected = False
        self.guided_trigger_consumed = False
        self.stable_hover_confirmed = False
        self.ready_to_authorize = False
        self.abort_reason = ""
        self._baseline_samples = 0
        self._ground_relative_alt_m: Optional[float] = None
        self._ground_down_m: Optional[float] = None
        self._hover_started: Optional[float] = None
        self._settle_started: Optional[float] = None

    def _abort(self, reason: str) -> FlightState:
        self.abort_reason = reason
        self.ready_to_authorize = False
        self.state = FlightState.ABORTED
        return self.state

    def _capture_ground(self, value: object) -> None:
        relative = _get(value, "relative_alt_m")
        if bool(_get(value, "relative_alt_valid", False)):
            try:
                relative_f = float(relative)
                if math.isfinite(relative_f):
                    self._ground_relative_alt_m = relative_f
            except (TypeError, ValueError):
                pass
        try:
            down = float(_get(value, "down_m"))
            if math.isfinite(down):
                self._ground_down_m = down
        except (TypeError, ValueError):
            pass

    def relative_altitude(self, value: object) -> Optional[float]:
        relative = _get(value, "relative_alt_m")
        if bool(_get(value, "relative_alt_valid", False)):
            try:
                relative_f = float(relative)
                if math.isfinite(relative_f):
                    baseline = self._ground_relative_alt_m or 0.0
                    return relative_f - baseline
            except (TypeError, ValueError):
                return None
        if self._ground_down_m is None:
            return None
        try:
            down = float(_get(value, "down_m"))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(down):
            return None
        return self._ground_down_m - down

    def _heartbeat_ok(self, value: object) -> bool:
        age = _get(value, "heartbeat_age_s", 99.0)
        try:
            return (bool(_get(value, "heartbeat_fresh", False))
                    and math.isfinite(float(age))
                    and float(age) <= self.config.heartbeat_timeout_s)
        except (TypeError, ValueError):
            return False

    def _position_safety_ok(self, value: object) -> bool:
        try:
            return bool(
                int(_get(value, "gps_fix_type", 0)) >= 3
                and bool(_get(value, "ekf_ok", False))
                and bool(_get(value, "local_position_valid", False))
                and bool(_get(value, "rc_valid", False))
                and float(_get(value, "telemetry_age_s", 99.0)) <= 0.5
                and float(_get(value, "rc_age_s", 99.0)) <= 0.5
            )
        except (TypeError, ValueError):
            return False

    def _stable(self, value: object) -> bool:
        if str(_get(value, "flight_mode", "")).upper() == "GUIDED":
            return False
        if (not bool(_get(value, "armed", False)) or not self._heartbeat_ok(value)
                or not self._position_safety_ok(value)):
            return False
        altitude = self.relative_altitude(value)
        if altitude is None or altitude < self.config.min_airborne_alt_m:
            return False
        try:
            horizontal = float(_get(value, "horizontal_speed_m_s", math.inf))
            vertical = float(_get(value, "vertical_speed_m_s", math.inf))
            roll = abs(math.degrees(float(_get(value, "roll_rad", math.inf))))
            pitch = abs(math.degrees(float(_get(value, "pitch_rad", math.inf))))
        except (TypeError, ValueError):
            return False
        return bool(
            all(math.isfinite(item) for item in (horizontal, vertical, roll, pitch))
            and horizontal <= self.config.max_horizontal_speed_m_s
            and vertical <= self.config.max_vertical_speed_m_s
            and max(roll, pitch) <= self.config.max_roll_pitch_deg
        )

    def update(self, value: object, now: float) -> FlightState:
        now = float(now)
        if self.state in (FlightState.COMPLETED, FlightState.ABORTED):
            return self.state
        if now - self.started_monotonic > self.config.wait_timeout_s:
            return self._abort("wait_timeout")

        mode = str(_get(value, "flight_mode", "UNKNOWN")).upper()
        armed = bool(_get(value, "armed", False))
        heartbeat_ok = self._heartbeat_ok(value)
        previous_mode = self.previous_mode
        if mode != "GUIDED":
            self.seen_non_guided_since_start = True

        if not heartbeat_ok:
            if armed:
                return self._abort("heartbeat_stale")
            self.state = FlightState.WAIT_HEARTBEAT
            self.previous_mode = mode
            return self.state

        if self.state in (FlightState.INIT, FlightState.WAIT_HEARTBEAT,
                          FlightState.WAIT_DISARMED_BASELINE):
            if not armed and mode != "GUIDED":
                self._capture_ground(value)
                self._baseline_samples += 1
                self.state = (FlightState.WAIT_ARMED if self._baseline_samples >= 2
                              else FlightState.WAIT_DISARMED_BASELINE)
            else:
                self.state = FlightState.WAIT_DISARMED_BASELINE

        elif self.state == FlightState.WAIT_ARMED:
            if armed:
                self.state = FlightState.WAIT_AIRBORNE

        elif self.state == FlightState.WAIT_AIRBORNE:
            if not armed:
                self.state = FlightState.WAIT_ARMED
            else:
                altitude = self.relative_altitude(value)
                if (altitude is not None
                        and altitude >= self.config.min_airborne_alt_m
                        and self._position_safety_ok(value)):
                    self.state = FlightState.WAIT_STABLE_HOVER
                    self._hover_started = None

        elif self.state == FlightState.WAIT_STABLE_HOVER:
            if not armed:
                return self._abort("vehicle_disarmed_after_takeoff")
            if self._stable(value):
                if self._hover_started is None:
                    self._hover_started = now
                elif now - self._hover_started + 1e-9 >= self.config.stable_hover_duration_s:
                    self.stable_hover_confirmed = True
                    self.state = FlightState.WAIT_GUIDED_EDGE
            else:
                self._hover_started = None
                self.stable_hover_confirmed = False

        elif self.state == FlightState.WAIT_GUIDED_EDGE:
            if not armed:
                return self._abort("vehicle_disarmed_after_takeoff")
            if mode == "GUIDED":
                if (previous_mode is not None and previous_mode != "GUIDED"
                        and self.seen_non_guided_since_start
                        and self.stable_hover_confirmed
                        and not self.guided_trigger_consumed):
                    self.guided_edge_detected = True
                    self.guided_trigger_consumed = True
                    self._settle_started = now
                    self.state = FlightState.GUIDED_SETTLE
            elif not self._stable(value):
                self.stable_hover_confirmed = False
                self._hover_started = None
                self.state = FlightState.WAIT_STABLE_HOVER

        elif self.state == FlightState.GUIDED_SETTLE:
            altitude = self.relative_altitude(value)
            if not armed:
                return self._abort("vehicle_disarmed_during_guided_settle")
            if mode != "GUIDED":
                return self._abort("guided_departure_during_settle")
            if altitude is None or altitude < self.config.min_abort_alt_m:
                return self._abort("minimum_abort_altitude")
            if not self._position_safety_ok(value):
                return self._abort("guided_settle_preflight_failed")
            if (self._settle_started is not None
                    and now - self._settle_started + 1e-9 >= self.config.guided_settle_s):
                self.ready_to_authorize = True
                self.state = FlightState.AUTHORIZE

        self.previous_mode = mode
        return self.state

    def stable_hover_elapsed(self, now: float) -> float:
        if self._hover_started is None:
            return 0.0
        return max(0.0, float(now) - self._hover_started)

    def mark_active(self) -> None:
        if self.state != FlightState.AUTHORIZE or not self.guided_trigger_consumed:
            raise RuntimeError("wait_guided_not_authorized")
        self.ready_to_authorize = False
        self.state = FlightState.ACTIVE_CONTROL

    def mark_completed(self) -> None:
        if self.state not in (FlightState.AUTHORIZE, FlightState.ACTIVE_CONTROL):
            raise RuntimeError("wait_guided_not_active")
        self.ready_to_authorize = False
        self.state = FlightState.COMPLETED
