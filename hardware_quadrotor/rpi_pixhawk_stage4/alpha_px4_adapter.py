"""PX4/ROS Noetic audit-only sidecar for the frozen F450 Stage 4 output.

The configuration file is JSON-compatible YAML, so it can be parsed by the
Python standard library on Ubuntu 20.04/Python 3.8 without adding PyYAML.
No ROS publisher exists in this module.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rpi_pixhawk_stage4.controller.limits import (
    MAX_HORIZONTAL_SPEED_MPS,
    MAX_VERTICAL_SPEED_MPS,
    MAX_YAW_RATE_RADPS,
    limit_target,
)


STAGE4_DIR = Path(__file__).resolve().parent
DEFAULT_ALPHA_CONFIG = STAGE4_DIR / "config" / "alpha_candidate_001.yaml"
_MISSING = object()


class AlphaConfigError(ValueError):
    """Raised when the fail-closed Alpha audit configuration is invalid."""


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _require_finite(config: Dict[str, Any], names: Sequence[str]) -> None:
    for name in names:
        if name not in config or not _finite_number(config[name]):
            raise AlphaConfigError("alpha_config_invalid_numeric:{0}".format(name))


def _require_range(value: Any, label: str) -> List[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise AlphaConfigError("alpha_config_invalid_range:{0}".format(label))
    if not all(_finite_number(item) for item in value):
        raise AlphaConfigError("alpha_config_invalid_range:{0}".format(label))
    result = [float(value[0]), float(value[1])]
    if result[0] > result[1]:
        raise AlphaConfigError("alpha_config_invalid_range:{0}".format(label))
    return result


def load_alpha_config(
    path: Optional[str] = None,
    allow_bench_publish: bool = False,
    allow_flight_handover_hold: bool = False,
) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve() if path else DEFAULT_ALPHA_CONFIG
    try:
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AlphaConfigError("alpha_config_unreadable:{0}".format(exc))
    if not isinstance(config, dict):
        raise AlphaConfigError("alpha_config_must_be_object")

    required_strings = (
        "platform", "backend", "state_topic", "rc_topic", "battery_topic",
        "odom_topic", "command_topic",
    )
    missing = [name for name in required_strings if not isinstance(config.get(name), str)]
    if missing:
        raise AlphaConfigError(
            "alpha_config_missing_string_fields:{0}".format(",".join(missing))
        )
    if config["platform"] != "alpha_v1" or config["backend"] != "ros_px4ctrl":
        raise AlphaConfigError("alpha_config_platform_backend_mismatch")

    bench_publish_enabled = config.get("bench_publish_enabled") is True
    flight_handover_hold_enabled = config.get("flight_handover_hold_enabled") is True
    if bench_publish_enabled and flight_handover_hold_enabled:
        raise AlphaConfigError("alpha_config_publish_mode_conflict")
    if bench_publish_enabled:
        if not allow_bench_publish:
            raise AlphaConfigError("alpha_config_bench_publish_not_authorized")
        if config.get("audit_only") is not False:
            raise AlphaConfigError("alpha_bench_config_audit_only_must_be_false")
        if config.get("actual_publish_enabled") is not True:
            raise AlphaConfigError("alpha_bench_config_actual_publish_must_be_true")
        if config["command_topic"] != "/setpoints_cmd":
            raise AlphaConfigError("alpha_bench_command_topic_mismatch")
    elif flight_handover_hold_enabled:
        if not allow_flight_handover_hold:
            raise AlphaConfigError("alpha_config_flight_handover_hold_not_authorized")
        if config.get("audit_only") is not False:
            raise AlphaConfigError("alpha_flight_config_audit_only_must_be_false")
        if config.get("actual_publish_enabled") is not True:
            raise AlphaConfigError("alpha_flight_config_actual_publish_must_be_true")
        if config["command_topic"] != "/setpoints_cmd":
            raise AlphaConfigError("alpha_flight_command_topic_mismatch")
        if not isinstance(config.get("px4ctrl_node_name"), str) or not config["px4ctrl_node_name"]:
            raise AlphaConfigError("alpha_flight_px4ctrl_node_name_missing")
    else:
        if config.get("audit_only") is not True:
            raise AlphaConfigError("alpha_config_audit_only_must_be_true")
        if config.get("actual_publish_enabled") is not False:
            raise AlphaConfigError("alpha_config_actual_publish_must_be_false")

    numeric_names = (
        "max_session_duration_s", "max_horizontal_velocity_mps",
        "max_vertical_velocity_mps", "max_yaw_rate_radps",
        "state_timeout_s", "rc_timeout_s", "odom_timeout_s",
        "battery_timeout_s",
    )
    if flight_handover_hold_enabled:
        numeric_names = numeric_names + (
            "validation_duration_seconds", "publish_rate_hz",
        )
    _require_finite(config, numeric_names)
    max_duration = 60.0 if flight_handover_hold_enabled else 5.0
    if not 0.0 < float(config["max_session_duration_s"]) <= max_duration:
        raise AlphaConfigError("alpha_config_session_duration_exceeds_hard_limit")
    if flight_handover_hold_enabled:
        if float(config["validation_duration_seconds"]) <= 0.0:
            raise AlphaConfigError("alpha_flight_validation_duration_not_positive")
        if float(config["publish_rate_hz"]) <= 0.0:
            raise AlphaConfigError("alpha_flight_publish_rate_not_positive")
        raw_handover = config.get("flight_handover", {})
        if not isinstance(raw_handover, dict):
            raise AlphaConfigError("alpha_flight_handover_config_must_be_object")
        handover = {
            "prestream_rate_hz": 10.0,
            "prestream_min_messages": 20,
            "prestream_min_duration_s": 2.0,
            "status_print_interval_s": 0.5,
            "return_to_manual_timeout_s": 5.0,
        }
        handover.update(raw_handover)
        for name in (
            "prestream_rate_hz",
            "prestream_min_duration_s",
            "status_print_interval_s",
            "return_to_manual_timeout_s",
        ):
            if not _finite_number(handover.get(name)):
                raise AlphaConfigError(
                    "alpha_flight_handover_invalid_numeric:{0}".format(name)
                )
        message_count = handover.get("prestream_min_messages")
        if (
            isinstance(message_count, bool)
            or not isinstance(message_count, int)
            or not 1 <= message_count <= 500
        ):
            raise AlphaConfigError(
                "alpha_flight_handover_invalid_message_count"
            )
        numeric_ranges = {
            "prestream_rate_hz": (1.0, 50.0),
            "prestream_min_duration_s": (0.1, 10.0),
            "status_print_interval_s": (0.1, 5.0),
            "return_to_manual_timeout_s": (0.1, 10.0),
        }
        for name, bounds in numeric_ranges.items():
            value = float(handover[name])
            if not bounds[0] <= value <= bounds[1]:
                raise AlphaConfigError(
                    "alpha_flight_handover_out_of_range:{0}".format(name)
                )
            handover[name] = value
        config = dict(config)
        config["flight_handover"] = handover
    expected_limits = (
        ("max_horizontal_velocity_mps", MAX_HORIZONTAL_SPEED_MPS),
        ("max_vertical_velocity_mps", MAX_VERTICAL_SPEED_MPS),
        ("max_yaw_rate_radps", MAX_YAW_RATE_RADPS),
    )
    for name, expected in expected_limits:
        if float(config[name]) != float(expected):
            raise AlphaConfigError("alpha_config_frozen_limit_mismatch:{0}".format(name))
    for name in ("state_timeout_s", "rc_timeout_s", "odom_timeout_s", "battery_timeout_s"):
        if float(config[name]) <= 0.0:
            raise AlphaConfigError("alpha_config_timeout_not_positive:{0}".format(name))

    rc = config.get("rc")
    if not isinstance(rc, dict):
        raise AlphaConfigError("alpha_config_rc_must_be_object")
    expected_indexes = {"ch5": 4, "ch6": 5, "ch7": 6, "ch8": 7}
    for channel, expected_index in expected_indexes.items():
        item = rc.get(channel)
        if not isinstance(item, dict) or item.get("index") != expected_index:
            raise AlphaConfigError("alpha_config_rc_index_mismatch:{0}".format(channel))
    if flight_handover_hold_enabled:
        _require_range(rc["ch5"].get("onboard_range"), "ch5_onboard")
        _require_range(rc["ch6"].get("tracking_range"), "ch6_tracking")
        _require_range(rc["ch6"].get("onboard_hover_range"), "ch6_onboard_hover")
        _require_range(rc["ch7"].get("unlocked_range"), "ch7_unlocked")
    else:
        _require_range(rc["ch5"].get("authorization_range"), "ch5")
        _require_range(rc["ch6"].get("authorization_range"), "ch6")
        _require_range(rc["ch7"].get("locked_range"), "ch7")
    bands = rc["ch8"].get("bands")
    if not isinstance(bands, dict):
        raise AlphaConfigError("alpha_config_ch8_bands_missing")
    for band in ("low", "middle", "high"):
        _require_range(bands.get(band), "ch8_{0}".format(band))

    command = config.get("position_command")
    if not isinstance(command, dict):
        raise AlphaConfigError("alpha_config_position_command_missing")
    for name in ("kx", "kv"):
        values = command.get(name)
        if (
            not isinstance(values, list)
            or len(values) != 3
            or not all(_finite_number(item) for item in values)
        ):
            raise AlphaConfigError("alpha_config_invalid_position_command:{0}".format(name))
        if flight_handover_hold_enabled and any(float(item) != 0.0 for item in values):
            raise AlphaConfigError("alpha_flight_position_command_must_be_zero:{0}".format(name))
    if command.get("trajectory_flag_ready") != 1:
        raise AlphaConfigError("alpha_config_ready_flag_must_be_one")
    result = dict(config)
    result["resolved_path"] = str(config_path)
    return result
def _read_path(source: Any, path: str, default: Any = _MISSING) -> Any:
    current = source
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                if default is _MISSING:
                    raise KeyError(path)
                return default
            current = current[part]
        else:
            if not hasattr(current, part):
                if default is _MISSING:
                    raise AttributeError(path)
                return default
            current = getattr(current, part)
    return current


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _header_stamp_for_diagnostics(message: Any) -> Optional[float]:
    """Extract a ROS header stamp for logs only, never for freshness gates."""
    stamp = _read_path(message, "header.stamp", None)
    if stamp is None:
        return None
    try:
        if hasattr(stamp, "to_sec"):
            return float(stamp.to_sec())
        seconds = _read_path(stamp, "secs", None)
        nanoseconds = _read_path(stamp, "nsecs", None)
        if seconds is not None:
            return float(seconds) + float(nanoseconds or 0.0) * 1.0e-9
        return float(stamp)
    except (AttributeError, TypeError, ValueError):
        return None


def _quaternion_yaw(x: float, y: float, z: float, w: float) -> float:
    values = (x, y, z, w)
    if not all(math.isfinite(item) for item in values):
        return float("nan")
    norm = math.sqrt(sum(item * item for item in values))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        return float("nan")
    x, y, z, w = (item / norm for item in values)
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(sin_yaw, cos_yaw)

def _in_range(value: float, bounds: Sequence[float]) -> bool:
    return float(bounds[0]) <= float(value) <= float(bounds[1])


class AlphaPx4AuditAdapter:
    """Receive-only ROS snapshot gate and PositionCommand candidate builder."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        clock: Optional[Any] = None,
        bench_publish_mode: bool = False,
        flight_handover_hold_mode: bool = False,
    ) -> None:
        if bench_publish_mode and flight_handover_hold_mode:
            raise AlphaConfigError("alpha_adapter_publish_mode_conflict")
        self.config = (
            dict(config)
            if config is not None
            else load_alpha_config(
                config_path,
                allow_bench_publish=bench_publish_mode,
                allow_flight_handover_hold=flight_handover_hold_mode,
            )
        )
        self.bench_publish_mode = bool(bench_publish_mode)
        self.flight_handover_hold_mode = bool(flight_handover_hold_mode)
        if self.bench_publish_mode:
            if self.config.get("bench_publish_enabled") is not True:
                raise AlphaConfigError("alpha_adapter_bench_publish_not_enabled")
            if self.config.get("audit_only") is not False:
                raise AlphaConfigError("alpha_adapter_bench_requires_non_audit_config")
            if self.config.get("actual_publish_enabled") is not True:
                raise AlphaConfigError("alpha_adapter_bench_requires_publish_enabled")
        elif self.flight_handover_hold_mode:
            if self.config.get("flight_handover_hold_enabled") is not True:
                raise AlphaConfigError("alpha_adapter_flight_handover_hold_not_enabled")
            if self.config.get("audit_only") is not False:
                raise AlphaConfigError("alpha_adapter_flight_requires_non_audit_config")
            if self.config.get("actual_publish_enabled") is not True:
                raise AlphaConfigError("alpha_adapter_flight_requires_publish_enabled")
        else:
            if self.config.get("audit_only") is not True:
                raise AlphaConfigError("alpha_adapter_requires_audit_only")
            if self.config.get("actual_publish_enabled") is not False:
                raise AlphaConfigError("alpha_adapter_rejects_actual_publish")
        self.clock = clock or time.monotonic
        self.state_snapshot = None
        self.rc_snapshot = None
        self.battery_snapshot = None
        self.odom_snapshot = None
        self.state_received_at = None
        self.rc_received_at = None
        self.battery_received_at = None
        self.odom_received_at = None
        self.first_received_at = {
            "state": None,
            "rc": None,
            "battery": None,
            "odom": None,
        }
        self.header_stamps = {
            "state": None,
            "rc": None,
            "battery": None,
            "odom": None,
        }
        self.actual_publish_count = 0
        self._trajectory_id = 0
        self._ros_subscribers = ()
    def _record_receive(
        self, name: str, message: Any, received_at: Optional[float]
    ) -> float:
        # Callback arrival time is the sole freshness clock. ROS header time may
        # be epoch-based, reset to 1970, or originate from another clock domain.
        arrival = float(self.clock() if received_at is None else received_at)
        if self.first_received_at[name] is None:
            self.first_received_at[name] = arrival
        self.header_stamps[name] = _header_stamp_for_diagnostics(message)
        return arrival

    def receive_diagnostics(self, now: Optional[float] = None) -> Dict[str, Any]:
        current = float(self.clock() if now is None else now)
        freshness = self.telemetry_freshness(current)
        return {
            "first_message_received": {
                name: self.first_received_at[name] is not None
                for name in self.first_received_at
            },
            "first_receive_time_seconds": dict(self.first_received_at),
            "last_receive_age_seconds": {
                name: item["age_seconds"] for name, item in freshness.items()
            },
            "header_stamp": dict(self.header_stamps),
        }

    def update_state(self, message: Any, received_at: Optional[float] = None) -> None:
        self.state_snapshot = {
            "connected": bool(_read_path(message, "connected", False)),
            "armed": bool(_read_path(message, "armed", False)),
            "guided": bool(_read_path(message, "guided", False)),
            "mode": str(_read_path(message, "mode", "UNKNOWN")),
        }
        self.state_received_at = self._record_receive("state", message, received_at)

    def update_rc(self, message: Any, received_at: Optional[float] = None) -> None:
        channels = _read_path(message, "channels", [])
        self.rc_snapshot = {"channels": list(channels) if channels is not None else []}
        self.rc_received_at = self._record_receive("rc", message, received_at)

    def update_battery(self, message: Any, received_at: Optional[float] = None) -> None:
        self.battery_snapshot = {
            "voltage": _float_or_nan(_read_path(message, "voltage", float("nan"))),
            "current": _float_or_nan(_read_path(message, "current", float("nan"))),
            "percentage": _float_or_nan(_read_path(message, "percentage", float("nan"))),
        }
        self.battery_received_at = self._record_receive("battery", message, received_at)

    def update_odom(self, message: Any, received_at: Optional[float] = None) -> None:
        simple_position = _read_path(message, "position", None)
        if simple_position is not None:
            position = {
                "x": _float_or_nan(_read_path(simple_position, "x", float("nan"))),
                "y": _float_or_nan(_read_path(simple_position, "y", float("nan"))),
                "z": _float_or_nan(_read_path(simple_position, "z", float("nan"))),
            }
            simple_velocity = _read_path(message, "velocity", {})
            velocity = {
                "x": _float_or_nan(_read_path(simple_velocity, "x", float("nan"))),
                "y": _float_or_nan(_read_path(simple_velocity, "y", float("nan"))),
                "z": _float_or_nan(_read_path(simple_velocity, "z", float("nan"))),
            }
            yaw = _float_or_nan(_read_path(message, "yaw", float("nan")))
        else:
            position = {
                "x": _float_or_nan(_read_path(message, "pose.pose.position.x", float("nan"))),
                "y": _float_or_nan(_read_path(message, "pose.pose.position.y", float("nan"))),
                "z": _float_or_nan(_read_path(message, "pose.pose.position.z", float("nan"))),
            }
            velocity = {
                "x": _float_or_nan(_read_path(message, "twist.twist.linear.x", float("nan"))),
                "y": _float_or_nan(_read_path(message, "twist.twist.linear.y", float("nan"))),
                "z": _float_or_nan(_read_path(message, "twist.twist.linear.z", float("nan"))),
            }
            qx = _float_or_nan(_read_path(message, "pose.pose.orientation.x", float("nan")))
            qy = _float_or_nan(_read_path(message, "pose.pose.orientation.y", float("nan")))
            qz = _float_or_nan(_read_path(message, "pose.pose.orientation.z", float("nan")))
            qw = _float_or_nan(_read_path(message, "pose.pose.orientation.w", float("nan")))
            yaw = _quaternion_yaw(qx, qy, qz, qw)
        self.odom_snapshot = {"position": position, "velocity": velocity, "yaw": yaw}
        self.odom_received_at = self._record_receive("odom", message, received_at)

    def attach_ros_subscribers(self) -> Tuple[object, ...]:
        """Create subscribers only when explicitly called inside a ROS Noetic process."""
        import rospy
        from mavros_msgs.msg import RCIn, State
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import BatteryState

        self._ros_subscribers = (
            rospy.Subscriber(self.config["state_topic"], State, self.update_state, queue_size=10),
            rospy.Subscriber(self.config["rc_topic"], RCIn, self.update_rc, queue_size=10),
            rospy.Subscriber(
                self.config["battery_topic"], BatteryState, self.update_battery, queue_size=10
            ),
            rospy.Subscriber(self.config["odom_topic"], Odometry, self.update_odom, queue_size=10),
        )
        return self._ros_subscribers

    def telemetry_freshness(
        self, now_mono: Optional[float] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Return the one authoritative monotonic freshness snapshot."""
        current = float(self.clock() if now_mono is None else now_mono)
        definitions = (
            (
                "state",
                self.state_snapshot,
                self.state_received_at,
                "state_timeout_s",
                "STATE_MISSING",
                "STATE_STALE",
            ),
            (
                "rc",
                self.rc_snapshot,
                self.rc_received_at,
                "rc_timeout_s",
                "RC_MISSING",
                "RC_STALE",
            ),
            (
                "battery",
                self.battery_snapshot,
                self.battery_received_at,
                "battery_timeout_s",
                "BATTERY_MISSING",
                "BATTERY_STALE",
            ),
            (
                "odom",
                self.odom_snapshot,
                self.odom_received_at,
                "odom_timeout_s",
                "ODOM_MISSING",
                "ODOM_STALE",
            ),
        )
        result = {}
        for name, snapshot, received_at, timeout_name, missing, stale in definitions:
            timeout = float(self.config[timeout_name])
            received = snapshot is not None and received_at is not None
            last_receive = float(received_at) if received_at is not None else None
            age = None
            reason = missing
            fresh = False
            if received:
                raw_age = current - float(last_receive)
                if math.isfinite(current) and math.isfinite(raw_age):
                    # A subscriber callback can update after a loop captures
                    # now_mono. That represents age zero, never stale data.
                    age = max(0.0, raw_age)
                    fresh = age <= timeout
                reason = None if fresh else stale
            result[name] = {
                "received": received,
                "fresh": fresh,
                "age_seconds": age,
                "timeout_seconds": timeout,
                "now_monotonic": current,
                "last_receive_monotonic": last_receive,
                "reason": reason,
            }
        return result
    def _telemetry_is_finite(self) -> bool:
        if self.rc_snapshot is None or self.odom_snapshot is None or self.battery_snapshot is None:
            return False
        channels = self.rc_snapshot.get("channels", [])
        values = list(channels)
        for group in (
            self.odom_snapshot["position"],
            self.odom_snapshot["velocity"],
            self.battery_snapshot,
        ):
            values.extend(group.values())
        values.append(self.odom_snapshot["yaw"])
        return all(_finite_number(value) for value in values)

    def _rc_values(self) -> Dict[str, Optional[float]]:
        channels = [] if self.rc_snapshot is None else self.rc_snapshot.get("channels", [])
        result = {}
        for channel in ("ch5", "ch6", "ch7", "ch8"):
            index = int(self.config["rc"][channel]["index"])
            label = channel.upper()
            result[label] = float(channels[index]) if len(channels) > index else None
        return result

    def _ch8_band(self, value: Optional[float]) -> str:
        if value is None:
            return "unknown"
        bands = self.config["rc"]["ch8"]["bands"]
        for name in ("low", "middle", "high"):
            if _in_range(value, bands[name]):
                return name
        return "unknown"

    def audit_rejection_reasons(self, now: Optional[float] = None) -> List[str]:
        current = float(self.clock() if now is None else now)
        freshness = self.telemetry_freshness(current)
        reasons = [
            item["reason"]
            for item in freshness.values()
            if item["reason"] is not None
        ]

        if self.state_snapshot is not None:
            if not self.state_snapshot["connected"]:
                reasons.append("MAVROS_DISCONNECTED")
            if self.state_snapshot["armed"]:
                reasons.append("ARMED_DURING_AUDIT")

        channels = [] if self.rc_snapshot is None else self.rc_snapshot.get("channels", [])
        if self.rc_snapshot is not None and len(channels) < 8:
            reasons.append("RC_CHANNELS_INSUFFICIENT")
        if len(channels) >= 8 and all(_finite_number(value) for value in channels[:8]):
            rc = self.config["rc"]
            if not _in_range(channels[4], rc["ch5"]["authorization_range"]):
                reasons.append("CH5_NOT_ONBOARD")
            if not _in_range(channels[5], rc["ch6"]["authorization_range"]):
                reasons.append("CH6_TRACKING_NOT_ALLOWED")
            if _in_range(channels[6], rc["ch7"]["locked_range"]):
                reasons.append("CH7_LOCKED")

        if (
            self.rc_snapshot is not None
            and self.odom_snapshot is not None
            and self.battery_snapshot is not None
            and not self._telemetry_is_finite()
        ):
            reasons.append("NONFINITE_TELEMETRY")
        return reasons

    def flight_handover_safety_status(
        self,
        now_mono: Optional[float] = None,
        require_armed: bool = False,
        require_ch7_unlocked: bool = False,
        require_ch5_manual: bool = False,
        require_ch6_manual: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate one explicit flight-handover phase at one monotonic instant."""
        current = float(self.clock() if now_mono is None else now_mono)
        freshness = self.telemetry_freshness(current)
        reasons = [
            item["reason"] for item in freshness.values() if item["reason"] is not None
        ]
        if not self.flight_handover_hold_mode:
            reasons.append("FLIGHT_HANDOVER_HOLD_MODE_NOT_ENABLED")
        state = self.state_snapshot or {}
        if self.state_snapshot is not None:
            if not state.get("connected", False):
                reasons.append("MAVROS_DISCONNECTED")
            if require_armed and not state.get("armed", False):
                reasons.append("ARMED_REQUIRED")
        channels = [] if self.rc_snapshot is None else self.rc_snapshot.get("channels", [])
        if self.rc_snapshot is not None and len(channels) < 8:
            reasons.append("RC_CHANNELS_INSUFFICIENT")
        switches = self.flight_switch_state()
        if len(channels) >= 8 and all(_finite_number(value) for value in channels[:8]):
            if require_ch5_manual and switches["ch5_onboard"]:
                reasons.append("CH5_MUST_REMAIN_MANUAL")
            if require_ch6_manual and not switches["ch6_manual_hover"]:
                reasons.append("CH6_MUST_REMAIN_MANUAL")
            if require_ch7_unlocked and not switches["ch7_unlocked"]:
                reasons.append("CH7_NOT_UNLOCKED")
        if (
            self.rc_snapshot is not None
            and self.odom_snapshot is not None
            and self.battery_snapshot is not None
            and not self._telemetry_is_finite()
        ):
            reasons.append("NONFINITE_TELEMETRY")
        return {
            "now_monotonic": current,
            "telemetry_freshness": freshness,
            "switches": switches,
            "rejection_reasons": list(dict.fromkeys(reasons)),
        }

    def flight_switch_state(self) -> Dict[str, Any]:
        values = self._rc_values()
        rc = self.config["rc"]
        finite = all(_finite_number(values[name]) for name in ("CH5", "CH6", "CH7"))
        return {
            "values": values,
            "finite": finite,
            "ch5_onboard": finite and _in_range(values["CH5"], rc["ch5"]["onboard_range"]),
            "ch6_tracking": finite and _in_range(values["CH6"], rc["ch6"]["tracking_range"]),
            "ch6_manual_hover": finite and _in_range(values["CH6"], rc["ch6"]["onboard_hover_range"]),
            "ch7_unlocked": finite and _in_range(values["CH7"], rc["ch7"]["unlocked_range"]),
        }
    def flight_operator_handover_requested(self) -> bool:
        """Return true only for an intentional, finite RC handover indication."""
        values = self._rc_values()
        if not all(_finite_number(values[name]) for name in ("CH5", "CH6")):
            return False
        rc = self.config["rc"]
        return (
            _in_range(values["CH6"], rc["ch6"]["onboard_hover_range"])
            or not _in_range(values["CH5"], rc["ch5"]["onboard_range"])
        )

    def latch_flight_hold_target(
        self, odom_snapshot: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Copy one finite, low-speed odometry target; later odometry is ignored."""
        if not self.flight_handover_hold_mode:
            raise RuntimeError("flight_hold_target_requires_flight_mode")
        snapshot = self.odom_snapshot if odom_snapshot is None else odom_snapshot
        if not isinstance(snapshot, dict):
            raise RuntimeError("flight_hold_target_requires_finite_telemetry")
        try:
            position = snapshot["position"]
            velocity = snapshot["velocity"]
            yaw = snapshot["yaw"]
            snapshot_values = [position[axis] for axis in ("x", "y", "z")]
            snapshot_values.extend(velocity[axis] for axis in ("x", "y", "z"))
            snapshot_values.append(yaw)
        except (KeyError, TypeError):
            raise RuntimeError("flight_hold_target_requires_finite_telemetry")
        if not all(_finite_number(value) for value in snapshot_values):
            raise RuntimeError("flight_hold_target_requires_finite_telemetry")
        horizontal_speed = math.hypot(float(velocity["x"]), float(velocity["y"]))
        vertical_speed = abs(float(velocity["z"]))
        if (
            horizontal_speed > float(self.config["max_horizontal_velocity_mps"])
            or vertical_speed > float(self.config["max_vertical_velocity_mps"])
        ):
            raise RuntimeError("flight_hold_latch_velocity_exceeds_limit")
        return {
            "position": {
                "x": float(position["x"]),
                "y": float(position["y"]),
                "z": float(position["z"]),
            },
            "velocity": {
                "x": float(velocity["x"]),
                "y": float(velocity["y"]),
                "z": float(velocity["z"]),
            },
            "yaw": float(yaw),
        }
    def build_flight_hold_command(
        self, hold_target: Dict[str, Any], now: float
    ) -> Dict[str, Any]:
        """Build a position-only command from the immutable handover target."""
        if not self.flight_handover_hold_mode:
            raise RuntimeError("flight_hold_command_requires_flight_mode")
        position = hold_target.get("position", {})
        values = (position.get("x"), position.get("y"), position.get("z"), hold_target.get("yaw"))
        if not all(_finite_number(value) for value in values):
            raise RuntimeError("flight_hold_target_nonfinite")
        self._trajectory_id += 1
        command_config = self.config["position_command"]
        return {
            "header": {"stamp": float(now), "frame_id": command_config["frame_id"]},
            "position": {"x": float(position["x"]), "y": float(position["y"]), "z": float(position["z"])},
            "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "acceleration": {"x": 0.0, "y": 0.0, "z": 0.0},
            "jerk": {"x": 0.0, "y": 0.0, "z": 0.0},
            "yaw": float(hold_target["yaw"]),
            "yaw_dot": 0.0,
            "kx": [0.0, 0.0, 0.0],
            "kv": [0.0, 0.0, 0.0],
            "trajectory_id": self._trajectory_id,
            "trajectory_flag": int(command_config["trajectory_flag_ready"]),
        }

    def record_flight_hold_publish(self) -> None:
        if not self.flight_handover_hold_mode:
            raise RuntimeError("flight_hold_counter_requires_flight_mode")
        self.actual_publish_count += 1
    def bench_safety_status(
        self, now_mono: Optional[float] = None
    ) -> Dict[str, Any]:
        current = float(self.clock() if now_mono is None else now_mono)
        if not self.bench_publish_mode:
            return {
                "now_monotonic": current,
                "telemetry_freshness": self.telemetry_freshness(current),
                "rejection_reasons": ["BENCH_MODE_NOT_ENABLED"],
            }

        freshness = self.telemetry_freshness(current)
        reasons = [
            item["reason"]
            for item in freshness.values()
            if item["reason"] is not None
        ]
        if self.state_snapshot is not None:
            if not self.state_snapshot["connected"]:
                reasons.append("MAVROS_DISCONNECTED")
            if self.state_snapshot["armed"]:
                reasons.append("ARMED_DURING_BENCH")

        channels = [] if self.rc_snapshot is None else self.rc_snapshot.get("channels", [])
        if self.rc_snapshot is not None and len(channels) < 8:
            reasons.append("RC_CHANNELS_INSUFFICIENT")
        if len(channels) >= 8 and all(_finite_number(value) for value in channels[:8]):
            locked_range = self.config["rc"]["ch7"]["locked_range"]
            if not _in_range(channels[6], locked_range):
                reasons.append("CH7_NOT_LOCKED")

        if (
            self.rc_snapshot is not None
            and self.odom_snapshot is not None
            and self.battery_snapshot is not None
            and not self._telemetry_is_finite()
        ):
            reasons.append("NONFINITE_TELEMETRY")
        return {
            "now_monotonic": current,
            "telemetry_freshness": freshness,
            "rejection_reasons": list(dict.fromkeys(reasons)),
        }

    def bench_rejection_reasons(self, now: Optional[float] = None) -> List[str]:
        return list(self.bench_safety_status(now)["rejection_reasons"])

    def evaluate_bench_hold(
        self,
        now: Optional[float] = None,
        safety_status: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        current = float(self.clock() if now is None else now)
        status = safety_status or self.bench_safety_status(current)
        reasons = list(status["rejection_reasons"])
        limited = limit_target(0.0, 0.0, 0.0, 0.0)
        candidate = (
            self._build_position_command(limited, current) if not reasons else None
        )
        return {
            "timestamp": current,
            "event": "BENCH_READY" if not reasons else "BENCH_REJECTED",
            "gate_authorized": not reasons,
            "rejection_reasons": reasons,
            "telemetry_freshness": status["telemetry_freshness"],
            "candidate_command": candidate,
            "limit_result": limited.as_dict(),
            "bench_publish_enabled": True,
            "actual_publish_count": self.actual_publish_count,
        }
    def _build_position_command(self, limited: Any, now: float) -> Dict[str, Any]:
        self._trajectory_id += 1
        position = self.odom_snapshot["position"]
        command_config = self.config["position_command"]
        return {
            "header": {"stamp": float(now), "frame_id": command_config["frame_id"]},
            "position": {
                "x": float(position["x"]),
                "y": float(position["y"]),
                "z": float(position["z"]),
            },
            "velocity": {
                "x": float(limited["vx_north_mps"]),
                "y": float(limited["vy_east_mps"]),
                "z": -float(limited["vz_down_mps"]),
            },
            "acceleration": {"x": 0.0, "y": 0.0, "z": 0.0},
            "jerk": {"x": 0.0, "y": 0.0, "z": 0.0},
            "yaw": float(self.odom_snapshot["yaw"]),
            "yaw_dot": float(limited["yaw_rate_radps"]),
            "kx": [float(value) for value in command_config["kx"]],
            "kv": [float(value) for value in command_config["kv"]],
            "trajectory_id": self._trajectory_id,
            "trajectory_flag": int(command_config["trajectory_flag_ready"]),
        }

    def to_position_command_message(
        self,
        candidate: Dict[str, Any],
        position_command_type: Any,
        stamp: Any,
    ) -> Any:
        message = position_command_type()
        message.header.stamp = stamp
        message.header.frame_id = str(candidate["header"]["frame_id"])
        for field in ("position", "velocity", "acceleration", "jerk"):
            target = getattr(message, field)
            source = candidate[field]
            target.x = float(source["x"])
            target.y = float(source["y"])
            target.z = float(source["z"])
        message.yaw = float(candidate["yaw"])
        message.yaw_dot = float(candidate["yaw_dot"])
        message.kx = [float(value) for value in candidate["kx"]]
        message.kv = [float(value) for value in candidate["kv"]]
        message.trajectory_id = int(candidate["trajectory_id"])
        message.trajectory_flag = int(candidate["trajectory_flag"])
        return message

    def record_bench_publish(self) -> None:
        if not self.bench_publish_mode:
            raise RuntimeError("bench_publish_counter_requires_bench_mode")
        self.actual_publish_count += 1

    def evaluate(
        self, stage4_output: Optional[Dict[str, Any]] = None, now: Optional[float] = None
    ) -> Dict[str, Any]:
        current = float(self.clock() if now is None else now)
        output = stage4_output or {}
        limited = limit_target(
            output.get("vx_north_mps", 0.0),
            output.get("vy_east_mps", 0.0),
            output.get("vz_down_mps", 0.0),
            output.get("yaw_rate_radps", 0.0),
        )
        reasons = self.audit_rejection_reasons(current)
        if limited["invalid_input"]:
            reasons.append("NONFINITE_COMMAND")
        authorized = len(reasons) == 0
        command = self._build_position_command(limited, current) if authorized else None
        rc_values = self._rc_values()
        state = self.state_snapshot or {}
        return {
            "timestamp": current,
            "event": "WOULD_PUBLISH" if authorized else "AUDIT_REJECTED",
            "gate_authorized": authorized,
            "rejection_reasons": reasons,
            "rc_values": rc_values,
            "ch8_task_band": self._ch8_band(rc_values["CH8"]),
            "mode": state.get("mode", "UNKNOWN"),
            "guided_recorded_only": bool(state.get("guided", False)),
            "candidate_command": command,
            "limit_result": limited.as_dict(),
            "audit_only": True,
            "actual_publish_enabled": False,
            "actual_publish_count": self.actual_publish_count,
        }