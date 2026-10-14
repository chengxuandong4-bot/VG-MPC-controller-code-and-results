"""Pure safety decisions shared by offline tests and the live wrapper."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .endpoint_guard import validate_command_endpoint


FLIGHT_CONFIRMATION = "I_CONFIRM_SECOND_FLIGHT_REAL_HARDWARE_PILOT_READY"


@dataclass(frozen=True)
class SafetyDecision:
    safe: bool
    reason: str


def _get(state: object, name: str, default: Any = None) -> Any:
    if isinstance(state, dict):
        return state.get(name, default)
    return getattr(state, name, default)


def _stop_triggered(emergency_stop: object) -> bool:
    if emergency_stop is None:
        return False
    callback = getattr(emergency_stop, "triggered", None)
    return bool(callback()) if callable(callback) else bool(emergency_stop)


def authorize_flight(enable: bool, confirmation: str, endpoint: str,
                     command_link_report: Optional[Dict[str, Any]], state: object,
                     emergency_stop: object = None) -> SafetyDecision:
    if _stop_triggered(emergency_stop):
        return SafetyDecision(False, "emergency_stop")
    if not enable:
        return SafetyDecision(False, "real_hardware_transmit_not_enabled")
    if confirmation != FLIGHT_CONFIRMATION:
        return SafetyDecision(False, "confirmation_phrase_mismatch")
    try:
        normalized = validate_command_endpoint(endpoint)
    except (TypeError, ValueError):
        return SafetyDecision(False, "command_link_not_verified")
    if not command_link_report or not command_link_report.get("passed"):
        return SafetyDecision(False, "command_link_not_verified")
    report_endpoint = command_link_report.get("command_endpoint", command_link_report.get("endpoint"))
    if report_endpoint != normalized:
        return SafetyDecision(False, "command_link_not_verified")
    if int(_get(state, "autopilot", -1)) != 3 or int(_get(state, "vehicle_type", -1)) != 2:
        return SafetyDecision(False, "not_ardupilot_copter")
    if str(_get(state, "flight_mode", "")).upper() != "GUIDED":
        return SafetyDecision(False, "vehicle_not_guided")
    if not bool(_get(state, "armed", False)):
        return SafetyDecision(False, "vehicle_not_armed")
    if int(_get(state, "gps_fix_type", 0)) < 3:
        return SafetyDecision(False, "gps_fix_insufficient")
    if not bool(_get(state, "ekf_ok", False)):
        return SafetyDecision(False, "ekf_unhealthy")
    if not bool(_get(state, "local_position_valid", False)):
        return SafetyDecision(False, "local_position_missing")
    if not bool(_get(state, "rc_valid", False)):
        return SafetyDecision(False, "rc_data_missing")
    if float(_get(state, "telemetry_age_s", 99.0)) > 0.5 or float(_get(state, "rc_age_s", 99.0)) > 0.5:
        return SafetyDecision(False, "stale_telemetry")
    return SafetyDecision(True, "authorized")


def pilot_override_active(state: object, channel: int, threshold: int,
                          direction: str = "high") -> bool:
    channels = _get(state, "rc_channels", {}) or {}
    value = channels.get(int(channel))
    if value is None:
        return False
    if direction == "low":
        return int(value) <= int(threshold)
    return int(value) >= int(threshold)


def validate_active_state(state: object, origin: object,
                          previous_time_boot_ms: Optional[int]) -> SafetyDecision:
    numeric_names = (
        "north_m", "east_m", "down_m", "vn_mps", "ve_mps", "vd_mps",
        "roll_rad", "pitch_rad", "yaw_rad", "telemetry_age_s",
    )
    for name in numeric_names:
        value = _get(state, name, _get(state, name.replace("_m", ""), 0.0))
        try:
            if not math.isfinite(float(value)):
                return SafetyDecision(False, "nonfinite_telemetry")
        except (TypeError, ValueError):
            return SafetyDecision(False, "nonfinite_telemetry")
    current_boot = int(_get(state, "time_boot_ms", 0))
    if previous_time_boot_ms is not None and current_boot < int(previous_time_boot_ms):
        return SafetyDecision(False, "timestamp_regression")
    if not bool(_get(state, "armed", False)):
        return SafetyDecision(False, "vehicle_disarmed")
    if not bool(_get(state, "local_position_valid", False)):
        return SafetyDecision(False, "local_position_missing")
    if not bool(_get(state, "rc_valid", False)):
        return SafetyDecision(False, "rc_data_missing")
    if float(_get(state, "telemetry_age_s", 99.0)) > 0.5 or float(_get(state, "rc_age_s", 99.0)) > 0.5:
        return SafetyDecision(False, "stale_telemetry")
    if int(_get(state, "gps_fix_type", 0)) < 3:
        return SafetyDecision(False, "gps_fix_insufficient")
    if not bool(_get(state, "ekf_ok", False)):
        return SafetyDecision(False, "ekf_unhealthy")

    north = float(_get(state, "north_m", _get(state, "north", 0.0)))
    east = float(_get(state, "east_m", _get(state, "east", 0.0)))
    down = float(_get(state, "down_m", _get(state, "down", 0.0)))
    if isinstance(origin, (tuple, list)):
        origin_n, origin_e, origin_d = map(float, origin[:3])
    else:
        origin_n = float(_get(origin, "north_m", _get(origin, "north", 0.0)))
        origin_e = float(_get(origin, "east_m", _get(origin, "east", 0.0)))
        origin_d = float(_get(origin, "down_m", _get(origin, "down", 0.0)))
    if math.hypot(north - origin_n, east - origin_e) > 0.5:
        return SafetyDecision(False, "horizontal_radius_exceeded")
    if abs(down - origin_d) > 1.5:
        return SafetyDecision(False, "relative_height_exceeded")
    return SafetyDecision(True, "active_state_valid")


def command_identity_matches(state: object, command_link_report: Dict[str, Any]) -> SafetyDecision:
    telemetry_system = int(_get(state, "system_id", 0))
    telemetry_component = int(_get(state, "component_id", 0))
    target_system = int(command_link_report.get("target_system", 0))
    target_component = int(command_link_report.get("target_component", 0))
    if telemetry_system <= 0 or telemetry_component <= 0:
        return SafetyDecision(False, "telemetry_target_identity_missing")
    if target_system <= 0 or target_component <= 0:
        return SafetyDecision(False, "command_target_not_verified")
    if telemetry_system != target_system:
        return SafetyDecision(False, "command_target_mismatch")
    return SafetyDecision(True, "command_target_identity_matched")


def load_command_link_report(path: str) -> Dict[str, Any]:
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))
