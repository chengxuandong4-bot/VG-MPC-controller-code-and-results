

"""Configuration and ROS execution boundary for safe flight-handover V2."""

from __future__ import annotations

import json
import math
import os
import signal
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .parameter_audit import ParameterAudit, REQUIRED_PARAMETERS
from .rc_semantics import CHANNELS, RcSemanticTracker, valid_range
from .state_v2 import FlightHandoverV2


STAGE4_DIR = Path(__file__).resolve().parents[1]
DEFAULT_V2_CONFIG = STAGE4_DIR / "config" / "alpha_stage4_flight_handover_v2_prop_off.json"


class V2ConfigError(ValueError):
    pass


def load_v2_config(path: Optional[str] = None) -> Dict[str, Any]:
    resolved = Path(path).expanduser().resolve() if path else DEFAULT_V2_CONFIG
    try:
        config = json.loads(resolved.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V2ConfigError("flight_handover_v2_config_unreadable:{0}".format(exc))
    if not isinstance(config, dict) or config.get("schema_version") != "alpha_stage4_flight_handover_v2":
        raise V2ConfigError("flight_handover_v2_schema_mismatch")
    rc = config.get("rc")
    if not isinstance(rc, dict) or float(rc.get("debounce_seconds", 0.0)) < 0.25 or int(rc.get("debounce_min_messages", 0)) < 5:
        raise V2ConfigError("flight_handover_v2_rc_debounce_invalid")
    missing = []
    indexes = {"ch5": 4, "ch6": 5, "ch7": 6, "ch8": 7}
    for channel, states in CHANNELS.items():
        section = rc.get(channel)
        if not isinstance(section, dict) or section.get("index") != indexes[channel]:
            raise V2ConfigError("flight_handover_v2_rc_index_invalid:{0}".format(channel))
        for _, field in states:
            value = section.get(field)
            if value is None:
                missing.append("rc.{0}.{1}".format(channel, field))
            elif not valid_range(value):
                raise V2ConfigError("flight_handover_v2_rc_range_invalid:{0}.{1}".format(channel, field))
    slots = config.get("px4_mode_slots", {})
    for name in ("ch5_optical_flow_slot", "ch5_attitude_slot", "ch5_onboard_slot"):
        value = slots.get(name)
        if value is None:
            missing.append("px4_mode_slots.{0}".format(name))
        elif isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 6:
            raise V2ConfigError("flight_handover_v2_mode_slot_invalid:{0}".format(name))
    timing = config.get("timing", {})
    controller = config.get("controller") or {"mode": "static_hold"}
    controller_mode = controller.get("mode")
    expected_active_windows = {
        "static_hold": (5.0,),
        "tinympc_zero_motion": (3.0, 10.0, 15.0, 30.0, 90.0, 120.0, 146.0),
    }.get(controller_mode)
    if expected_active_windows is None:
        raise V2ConfigError("flight_handover_v2_controller_mode_invalid")
    active_window = float(timing.get("active_command_window_seconds", 0.0))
    if (
        float(timing.get("publish_rate_hz", 0.0)) != 20.0
        or active_window not in expected_active_windows
    ):
        raise V2ConfigError("flight_handover_v2_frozen_timing_mismatch")
    if float(timing.get("prestream_seconds", 0.0)) < 2.0 or int(timing.get("prestream_min_messages", 0)) < 30:
        raise V2ConfigError("flight_handover_v2_prestream_below_minimum")
    # Legacy template compatibility is explicit and deterministic; live deployment
    # uses session_hard_limit_seconds in the self-contained JSON config.
    if "session_hard_limit_seconds" not in timing:
        if "maximum_runtime_seconds" in timing:
            timing = dict(timing)
            timing["session_hard_limit_seconds"] = timing["maximum_runtime_seconds"]
            config = dict(config)
            config["timing"] = timing
        else:
            raise V2ConfigError("flight_handover_v2_missing_field:timing.session_hard_limit_seconds")
    session_limit = float(timing["session_hard_limit_seconds"])
    reference_profile = controller.get("reference_profile") or {}
    reference_active_window = reference_profile.get("active_window_seconds")
    extended_figure8_session = (
        controller_mode == "tinympc_zero_motion"
        and active_window == 120.0
        and isinstance(reference_profile, dict)
        and reference_profile.get("profile_id") == "xy_figure8_r1m_120s"
        and reference_profile.get("type") == "relative_waypoints_linear"
        and not isinstance(reference_active_window, bool)
        and isinstance(reference_active_window, (int, float))
        and float(reference_active_window) == 120.0
    )
    extended_figure8_146_session = (
        controller_mode == "tinympc_zero_motion"
        and active_window == 146.0
        and isinstance(reference_profile, dict)
        and reference_profile.get("profile_id") == "xy_figure8_r1m_146s"
        and reference_profile.get("type") == "relative_waypoints_linear"
        and not isinstance(reference_active_window, bool)
        and isinstance(reference_active_window, (int, float))
        and float(reference_active_window) == 146.0
    )
    if extended_figure8_146_session:
        session_ceiling = 270.0
    elif extended_figure8_session:
        session_ceiling = 240.0
    else:
        session_ceiling = 180.0
    if not 1.0 <= session_limit <= session_ceiling:
        raise V2ConfigError("flight_handover_v2_session_hard_limit_invalid")
    if config.get("ch7_policy") != "px4_native_kill":
        raise V2ConfigError("flight_handover_v2_ch7_policy_invalid")
    result = dict(config)
    result["resolved_path"] = str(resolved)
    result["configuration_errors"] = []
    if any(name.startswith("rc.") for name in missing):
        result["configuration_errors"].append("rc_pwm_calibration_missing")
    if any(name.startswith("px4_mode_slots.") for name in missing):
        result["configuration_errors"].append("px4_mode_slot_calibration_missing")
    result["calibration_missing_fields"] = missing
    result["actual_publish_ready"] = not missing and config.get("audit_only") is False and config.get("actual_publish_enabled") is True
    return result


def _path(value: Any, dotted: str, default: Any = None) -> Any:
    current = value
    for part in dotted.split("."):
        if isinstance(current, dict):
            current = current.get(part, default)
        else:
            current = getattr(current, part, default)
        if current is default:
            break
    return current


def _quaternion_rpy(message: Any):
    x = float(_path(message, "pose.pose.orientation.x", float("nan")))
    y = float(_path(message, "pose.pose.orientation.y", float("nan")))
    z = float(_path(message, "pose.pose.orientation.z", float("nan")))
    w = float(_path(message, "pose.pose.orientation.w", float("nan")))
    if not all(math.isfinite(item) for item in (x, y, z, w)):
        return (float("nan"),) * 3
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


class V2Adapter:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.rc_tracker = RcSemanticTracker(config["rc"])
        self.values = {name: None for name in ("state", "rc", "battery", "odom")}
        self.received = {name: None for name in self.values}
        self.rc_semantics: Dict[str, Any] = {}
        self.px4ctrl_state = None
        self.cmd_fresh = False
        self.actual_publish_count = 0
        self.trajectory_id = 0

    def update_state(self, message: Any) -> None:
        self.values["state"] = {"connected": bool(_path(message, "connected", False)), "armed": bool(_path(message, "armed", False)), "mode": str(_path(message, "mode", "UNKNOWN")).upper()}
        self.received["state"] = time.monotonic()

    def update_rc(self, message: Any) -> None:
        channels = list(_path(message, "channels", []) or [])
        now = time.monotonic(); self.values["rc"] = channels; self.received["rc"] = now
        self.rc_semantics = self.rc_tracker.update(channels, now)

    def update_battery(self, message: Any) -> None:
        self.values["battery"] = [float(_path(message, name, float("nan"))) for name in ("voltage", "current", "percentage")]
        self.received["battery"] = time.monotonic()

    def update_odom(self, message: Any) -> None:
        roll, pitch, yaw = _quaternion_rpy(message)
        self.values["odom"] = {
            "position": {axis: float(_path(message, "pose.pose.position." + axis, float("nan"))) for axis in ("x", "y", "z")},
            "velocity": {axis: float(_path(message, "twist.twist.linear." + axis, float("nan"))) for axis in ("x", "y", "z")},
            "roll": roll, "pitch": pitch, "yaw": yaw,
        }
        self.received["odom"] = time.monotonic()

    def update_fsm(self, message: Any) -> None:
        self.px4ctrl_state = str(_path(message, "data", message)).upper()

    def update_cmd_fresh(self, message: Any) -> None:
        self.cmd_fresh = bool(_path(message, "data", message))

    def snapshot(self, now: float, consumers: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for name in self.values:
            age = math.inf if self.received[name] is None else max(0.0, now - float(self.received[name]))
            result[name + "_age"] = age
            result[name + "_fresh"] = age <= float(self.config["freshness"][name + "_timeout_seconds"])
        state = self.values["state"] or {}; odom = self.values["odom"] or {}
        result.update({
            "mavros_connected": bool(state.get("connected")), "armed": bool(state.get("armed")),
            "px4_mode": state.get("mode", "UNKNOWN"), "offboard_enabled": state.get("mode") == "OFFBOARD",
            "px4ctrl_state": self.px4ctrl_state, "cmd_fresh": self.cmd_fresh,
            "position": odom.get("position"), "odom_velocity": odom.get("velocity"),
            "roll": odom.get("roll", float("nan")), "pitch": odom.get("pitch", float("nan")),
            "yaw": odom.get("yaw", float("nan")), "odom_reset_counter": None,
            "subscriber_nodes": list(consumers.get("subscriber_nodes", [])),
            "subscriber_count": int(consumers.get("subscriber_count", 0)),
        })
        result.update(self.rc_semantics)
        return result

    def message(self, command: Dict[str, Any], message_type: Any, stamp: Any) -> Any:
        self.trajectory_id += 1
        value = message_type(); value.header.stamp = stamp; value.header.frame_id = "world"
        for field in ("position", "velocity", "acceleration", "jerk"):
            for axis in ("x", "y", "z"):
                setattr(getattr(value, field), axis, float(command[field][axis]))
        value.yaw = float(command["yaw"]); value.yaw_dot = float(command["yaw_dot"])
        value.kx = [0.0, 0.0, 0.0]; value.kv = [0.0, 0.0, 0.0]
        value.trajectory_id = self.trajectory_id; value.trajectory_flag = 1
        return value


def build_wait_status(machine: FlightHandoverV2, snapshot: Dict[str, Any], now: float,
                      started: float, actual_publish_count: int) -> Dict[str, Any]:
    stability = machine.stability_status(snapshot, now)
    velocity = snapshot.get("odom_velocity") or {}
    roll = snapshot.get("roll")
    pitch = snapshot.get("pitch")
    roll_deg = math.degrees(float(roll)) if isinstance(roll, (int, float)) and math.isfinite(float(roll)) else None
    pitch_deg = math.degrees(float(pitch)) if isinstance(pitch, (int, float)) and math.isfinite(float(pitch)) else None
    return {
        "state": machine.state,
        "elapsed": max(0.0, float(now) - float(started)),
        "armed": bool(snapshot.get("armed")),
        "px4_mode": snapshot.get("px4_mode", "UNKNOWN"),
        "px4ctrl_state": snapshot.get("px4ctrl_state"),
        "ch5_pwm": snapshot.get("ch5_pwm"),
        "ch5_semantic": snapshot.get("ch5_semantic"),
        "ch6_pwm": snapshot.get("ch6_pwm"),
        "ch6_semantic": snapshot.get("ch6_semantic"),
        "ch7_pwm": snapshot.get("ch7_pwm"),
        "ch7_semantic": snapshot.get("ch7_semantic"),
        "ch8_pwm": snapshot.get("ch8_pwm"),
        "ch8_semantic": snapshot.get("ch8_semantic"),
        "state_age": snapshot.get("state_age", math.inf),
        "state_fresh": bool(snapshot.get("state_fresh")),
        "rc_age": snapshot.get("rc_age", math.inf),
        "rc_fresh": bool(snapshot.get("rc_fresh")),
        "battery_age": snapshot.get("battery_age", math.inf),
        "battery_fresh": bool(snapshot.get("battery_fresh")),
        "odom_age": snapshot.get("odom_age", math.inf),
        "odom_fresh": bool(snapshot.get("odom_fresh")),
        "cmd_fresh": bool(snapshot.get("cmd_fresh")),
        "vx": velocity.get("x"),
        "vy": velocity.get("y"),
        "vz": velocity.get("z"),
        "roll_deg": roll_deg,
        "pitch_deg": pitch_deg,
        "stable": stability["stable"],
        "stability_failure_fields": stability["failure_fields"],
        "stable_elapsed": stability["stable_elapsed"],
        "stable_required": stability["stable_required"],
        "publisher_created": machine.publisher_created_at is not None,
        "prestream_count": machine.prestream_message_count,
        "prestream_required": int(machine.timing["prestream_min_messages"]),
        "actual_publish_count": int(actual_publish_count),
        "wait_ch6_entered_semantic": machine.wait_ch6_entered_semantic,
        "ch6_edge_seen": bool(machine.ch6_edge_seen),
        "active_authorization": (
            "ACTIVE_AUTHORIZED" if machine.state == "ACTIVE_CONTROL"
            else "ACTIVE_NOT_AUTHORIZED"
        ),
        "permission_context": (
            "POST_AUTO_HOVER_PERMISSION"
            if machine.state in ("PRESTREAM_COMMAND", "WAIT_CH6_TRACKING_EDGE")
            else None
        ),
    }


def build_ready_ch6_status(machine: FlightHandoverV2,
                           snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Build the complete operator-facing CH6 authorization snapshot."""
    return {
        "state": machine.state,
        "action": "READY_CH6",
        "permission_context": "POST_AUTO_HOVER_PERMISSION",
        "ch6_pwm": snapshot.get("ch6_pwm"),
        "ch6_semantic": snapshot.get("ch6_semantic"),
        "ch6_initial_semantic": machine.wait_ch6_entered_semantic,
        "ch6_edge_seen": bool(machine.ch6_edge_seen),
        "px4ctrl_state": snapshot.get("px4ctrl_state"),
        "cmd_fresh": bool(snapshot.get("cmd_fresh")),
        "active_authorization": "ACTIVE_NOT_AUTHORIZED",
    }


class V2StatusReporter:
    """Synchronous, flushed operational logging with atomic live status."""

    def __init__(self, results_dir: Path, clock=time.monotonic) -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.console_path = self.results_dir / "console.log"
        self.status_path = self.results_dir / "live_status.json"
        self.jsonl_path = self.results_dir / "alpha_flight_handover_hold.jsonl"
        # console.log is owned exclusively by the Bash launcher via tee.
        self._jsonl = self.jsonl_path.open("a", encoding="utf-8")
        self._last_heartbeat = None
        self._live_status = {
            "event": "startup", "timestamp_monotonic": float(self.clock()),
            "state": "BOOT", "elapsed": 0.0, "armed": None,
            "px4_mode": "UNKNOWN", "px4ctrl_state": None,
            "ch5_pwm": None, "ch5_semantic": None,
            "ch6_pwm": None, "ch6_semantic": None,
            "ch7_pwm": None, "ch7_semantic": None,
            "ch8_pwm": None, "ch8_semantic": None,
            "state_age": None, "state_fresh": False,
            "rc_age": None, "rc_fresh": False,
            "battery_age": None, "battery_fresh": False,
            "odom_age": None, "odom_fresh": False,
            "cmd_fresh": False,
            "vx": None, "vy": None, "vz": None,
            "roll_deg": None, "pitch_deg": None,
            "stable": False, "stable_elapsed": 0.0,
            "stable_required": None,
            "stability_failure_fields": ["telemetry_not_received"],
            "publisher_created": False, "prestream_count": 0,
            "prestream_required": None, "actual_publish_count": 0,
            "wait_ch6_entered_semantic": None, "ch6_edge_seen": False,
            "active_authorization": "ACTIVE_NOT_AUTHORIZED",
            "permission_context": None,
        }
        self._write_status(self._live_status)

    def heartbeat_due(self, now: float) -> bool:
        return self._last_heartbeat is None or float(now) - self._last_heartbeat >= 0.5

    def note_heartbeat(self, now: float) -> None:
        self._last_heartbeat = float(now)

    def _write_status(self, value: Dict[str, Any]) -> None:
        temporary = self.status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(str(temporary), str(self.status_path))

    def emit(self, event: str, payload: Dict[str, Any], line: Optional[str] = None,
             update_live_status: bool = False) -> None:
        record = dict(payload)
        record["event"] = event
        record.setdefault("timestamp_monotonic", float(self.clock()))
        self._jsonl.write(json.dumps(record, sort_keys=True) + "\n")
        self._jsonl.flush()
        if update_live_status:
            self._live_status.update(record)
            self._write_status(self._live_status)
        text = line if line is not None else json.dumps(record, sort_keys=True)
        print(text, flush=True)

    def close(self) -> None:
        if not self._jsonl.closed:
            self._jsonl.flush()
            self._jsonl.close()


def _load_ros():
    try:
        import rosgraph, rosnode, rospy
        from mavros_msgs.msg import RCIn, State
        from mavros_msgs.srv import ParamGet
        from nav_msgs.msg import Odometry
        from quadrotor_msgs.msg import PositionCommand
        from sensor_msgs.msg import BatteryState
        from std_msgs.msg import Bool, String
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("flight_handover_v2_requires_ros_noetic_environment") from exc
    return locals()


def _consumers(ros: Dict[str, Any], topic: str) -> Dict[str, Any]:
    master = ros["rosgraph"].Master(ros["rospy"].get_name())
    _, subscribers, _ = master.getSystemState()
    nodes = next((list(names) for name, names in subscribers if name == topic), [])
    return {"subscriber_nodes": nodes, "subscriber_count": len(nodes)}


def _parameters(ros: Dict[str, Any]) -> Dict[str, Any]:
    rospy = ros["rospy"]
    rospy.wait_for_service("/mavros/param/get", timeout=3.0)
    getter = rospy.ServiceProxy("/mavros/param/get", ros["ParamGet"])
    values = {}
    for name in REQUIRED_PARAMETERS:
        response = getter(param_id=name)
        if not bool(getattr(response, "success", False)):
            values[name] = None
        else:
            integer = int(response.value.integer); real = float(response.value.real)
            values[name] = integer if integer != 0 or real == 0.0 else real
    return values


def run_v2(config: Dict[str, Any], results_dir: Path, maximum_runtime: float) -> int:
    """Run the V2 supervisor. All ROS transport remains behind explicit V2 gates."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    reporter = V2StatusReporter(results_dir)
    summary_path = results_dir / "run_summary.json"
    machine = FlightHandoverV2(config)
    adapter = V2Adapter(config)
    publisher = None
    publisher_created = False
    publisher_unregistered = False
    last: Dict[str, Any] = {}
    started = time.monotonic()
    exit_reason = "not_started"
    error = None
    exit_code = 1
    ros = None
    old_handlers = {}
    stop_requested = {"reason": None}

    def request_stop(signum, _frame):
        stop_requested["reason"] = "signal_{0}".format(signum)

    def finalize_summary(finished: float) -> Dict[str, Any]:
        summary = machine.summary(last)
        summary.update({
            "mode": "flight_handover_v2",
            "exit_reason": exit_reason or "unknown_exit",
            "error": error,
            "runtime_seconds": max(0.0, finished - started),
            "maximum_runtime_seconds": float(maximum_runtime),
            "publisher_created": publisher_created,
            "publisher_unregistered": publisher_unregistered,
            "actual_publish_count": adapter.actual_publish_count,
            "candidate_097_used": False,
            "safety_abort_reason": machine.abort_reason,
            "result_paths": {
                "console": str(reporter.console_path),
                "live_status": str(reporter.status_path),
                "events": str(reporter.jsonl_path),
                "summary": str(summary_path),
            },
        })
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        reporter.emit("exit", summary,
                      "[STAGE4][EXIT] exit_reason={0} safety_abort_reason={1} final_px4_mode={2} "
                      "final_px4ctrl_state={3} publisher_created={4} publisher_unregistered={5} "
                      "actual_publish_count={6} results_dir={7}".format(
                          summary["exit_reason"], summary["safety_abort_reason"],
                          summary.get("final_px4_mode"), summary.get("final_px4ctrl_state"),
                          summary["publisher_created"], summary["publisher_unregistered"],
                          summary["actual_publish_count"], results_dir))
        return summary

    try:
        session_limit = float(config["timing"].get("session_hard_limit_seconds", config["timing"].get("maximum_runtime_seconds", 0.0)))
        if not 1.0 <= float(maximum_runtime) <= session_limit:
            raise V2ConfigError("flight_handover_v2_supervisor_runtime_invalid")
        reporter.emit("startup", {
            "results_dir": str(results_dir),
            "config_resolved_path": config.get("resolved_path"),
            "schema_version": config.get("schema_version"),
            "session_hard_limit_seconds": session_limit,
            "active_hard_limit_seconds": float(config["timing"]["active_command_window_seconds"]),
            "next_action": "WAIT_FOR_TELEMETRY_AND_PARAMETER_AUDIT",
        }, "[STAGE4][START] results_dir={0} config={1} schema={2} session_limit={3:.1f}s active_limit={4:.1f}s next=WAIT_FOR_TELEMETRY".format(
            results_dir, config.get("resolved_path"), config.get("schema_version"), session_limit,
            float(config["timing"]["active_command_window_seconds"])))
        if not config.get("actual_publish_ready"):
            exit_reason = (config.get("configuration_errors") or ["v2_actual_publish_not_enabled"])[0]
            reporter.emit("configuration_rejected", {"state": machine.state, "reason": exit_reason},
                          "[STAGE4][EXIT] configuration_rejected reason={0}".format(exit_reason))
            return 1

        ros = _load_ros()
        rospy = ros["rospy"]
        rospy.init_node("stage4_alpha_flight_handover_v2", anonymous=False)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                old_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, request_stop)
            except (ValueError, AttributeError):
                pass
        rospy.Subscriber(config["state_topic"], ros["State"], adapter.update_state, queue_size=10)
        rospy.Subscriber(config["rc_topic"], ros["RCIn"], adapter.update_rc, queue_size=10)
        rospy.Subscriber(config["battery_topic"], ros["BatteryState"], adapter.update_battery, queue_size=10)
        rospy.Subscriber(config["odom_topic"], ros["Odometry"], adapter.update_odom, queue_size=50)
        rospy.Subscriber(config["px4ctrl_fsm_topic"], ros["String"], adapter.update_fsm, queue_size=20)
        rospy.Subscriber(config["px4ctrl_cmd_fresh_topic"], ros["Bool"], adapter.update_cmd_fresh, queue_size=20)
        try:
            machine.set_parameter_audit(ParameterAudit(config).evaluate(_parameters(ros)))
        except Exception as exc:
            machine.set_parameter_audit({
                "parameter_audit_passed": False,
                "parameter_conflicts": [{"parameter": "MAVROS_PARAM_GET", "reason": "parameter_unreadable", "actual": str(exc)}],
            })
        rate = rospy.Rate(float(config["timing"]["publish_rate_hz"]))
        while True:
            now = time.monotonic()
            if stop_requested["reason"]:
                exit_reason = stop_requested["reason"]
                break
            if rospy.is_shutdown():
                exit_reason = "ros_shutdown"
                break
            if now - started >= float(maximum_runtime):
                exit_reason = "maximum_runtime"
                break
            last = adapter.snapshot(now, _consumers(ros, config["command_topic"]))
            prior_state = machine.state
            actions = machine.step(last, now)
            if machine.state != prior_state:
                transition = machine.state_history[-1]
                reporter.emit("transition", transition,
                              "[STAGE4][STATE] {0} -> {1} reason={2}".format(prior_state, machine.state, transition["reason"]))
            for action in actions:
                if action == "ANNOUNCE_READY_CH5":
                    reporter.emit("operator_action", {"state": machine.state, "action": "READY_CH5"},
                                  "[STAGE4][ACTION] READY_CH5\nMove CH5 to ONBOARD. Keep CH6 at HOVER. Keep CH7 unlocked. Keep CH8 at INIT.")
                elif action == "ANNOUNCE_READY_CH6":
                    ready_ch6 = build_ready_ch6_status(machine, last)
                    reporter.emit(
                        "operator_action",
                        ready_ch6,
                        ("[STAGE4][ACTION] READY_CH6 permission={permission_context} "
                         "px4ctrl={px4ctrl_state} CH6={ch6_pwm}/{ch6_semantic} "
                         "initial={ch6_initial_semantic} ch6_edge_seen={ch6_edge_seen} "
                         "cmd_fresh={cmd_fresh} active_authorization={active_authorization}\n"
                         "Move CH6 to TRACKING. Keep CH5 at ONBOARD. Keep CH7 unlocked. "
                         "Keep CH8 at INIT.").format(**ready_ch6),
                    )
                elif action == "ANNOUNCE_CH6_EDGE_CONFIRMED":
                    reporter.emit(
                        "operator_action",
                        {"state": machine.state, "action": "HOVER_TO_TRACKING_EDGE_CONFIRMED",
                         "ch6_edge_seen": True},
                        "[STAGE4][ACTION] HOVER_TO_TRACKING_EDGE_CONFIRMED\n"
                        "CH6 authorization edge accepted; verifying CMD_CTRL and all ACTIVE prerequisites.",
                    )
                elif action == "ANNOUNCE_ACTIVE":
                    reset_integrator = getattr(machine.source, "reset_integrator", None)
                    if callable(reset_integrator):
                        reset_integrator()
                    active_limit = float(config["timing"]["active_command_window_seconds"])
                    reporter.emit("active_start", {"state": machine.state, "active_limit_seconds": active_limit},
                                  "[STAGE4][ACTIVE] started hard_limit={0:.1f} seconds".format(active_limit))
                elif action == "CREATE_PUBLISHER":
                    if machine.state != "PRESTREAM_COMMAND" or machine.auto_hover_confirmed_at is None or machine.offboard_confirmed_at is None or publisher is not None:
                        raise RuntimeError("publisher_creation_before_authority_forbidden")
                    publisher = rospy.Publisher(config["command_topic"], ros["PositionCommand"], queue_size=1)
                    publisher_created = True
                    machine.publisher_unregistered = False
                elif action in ("PUBLISH_STATIC_HOLD", "PUBLISH_LAST_FINITE_HOLD"):
                    if publisher is None or machine.target is None:
                        raise RuntimeError("publish_without_authorized_publisher")
                    command = machine.source.compute(now, adapter.values["odom"] or {}, machine.target)
                    publisher.publish(adapter.message(command, ros["PositionCommand"], rospy.Time.now()))
                    machine.record_publish(now, last)
                    adapter.actual_publish_count += 1
                elif action == "UNREGISTER_PUBLISHER" and publisher is not None:
                    publisher.unregister()
                    publisher = None
                    publisher_unregistered = True
                    machine.publisher_unregistered = True
            if reporter.heartbeat_due(now):
                status = build_wait_status(machine, last, now, started, adapter.actual_publish_count)
                line = ("[STAGE4][WAIT] state={state} elapsed={elapsed:.1f}s armed={armed} px4={px4_mode} "
                        "px4ctrl={px4ctrl_state} CH5={ch5_pwm}/{ch5_semantic} CH6={ch6_pwm}/{ch6_semantic} "
                        "CH7={ch7_pwm}/{ch7_semantic} CH8={ch8_pwm}/{ch8_semantic} "
                        "state_age={state_age:.3f} state_fresh={state_fresh} rc_age={rc_age:.3f} rc_fresh={rc_fresh} "
                        "battery_age={battery_age:.3f} battery_fresh={battery_fresh} "
                        "odom_age={odom_age:.3f} odom_fresh={odom_fresh} "
                        "cmd={cmd_fresh} velocity=({vx},{vy},{vz}) tilt_deg=({roll_deg},{pitch_deg}) "
                        "stable={stable} stable_t={stable_elapsed:.1f}/{stable_required:.1f} "
                        "failures={stability_failure_fields} publisher={publisher_created} "
                        "prestream={prestream_count}/{prestream_required} sent={actual_publish_count} "
                        "permission={permission_context} ch6_edge_seen={ch6_edge_seen} "
                        "active_authorization={active_authorization}").format(**status)
                reporter.emit("heartbeat", status, line, update_live_status=True)
                reporter.note_heartbeat(now)
            if machine.state == "ACTIVE_CONTROL" and machine.active_started_at is not None:
                active_limit = float(config["timing"]["active_command_window_seconds"])
                remaining = max(0.0, active_limit - (now - machine.active_started_at))
                reporter.emit("active_status", {"state": machine.state, "remaining_seconds": remaining},
                              "[STAGE4][ACTIVE] remaining={0:.2f}s".format(remaining))
            if machine.state == "COMPLETE":
                exit_reason = "complete"
                exit_code = 0
                break
            if machine.state in ("ABORT_PRE_TAKEOVER", "ABORT_ACTIVE_CONTROL"):
                exit_reason = machine.abort_reason or "safety_abort"
                break
            rate.sleep()
    except KeyboardInterrupt:
        exit_reason = "keyboard_interrupt"
        error = "KeyboardInterrupt"
    except Exception as exc:
        exit_reason = "exception"
        error = "{0}: {1}".format(type(exc).__name__, exc)
    finally:
        if publisher is not None:
            try:
                publisher.unregister()
                publisher_unregistered = True
                machine.publisher_unregistered = True
            except Exception as exc:
                if error is None:
                    error = "publisher_unregister_error:{0}: {1}".format(type(exc).__name__, exc)
            publisher = None
        for sig, handler in old_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, AttributeError):
                pass
        if exit_reason == "not_started":
            exit_reason = "exception" if error else "unknown_exit"
        finalize_summary(time.monotonic())
        reporter.close()
    return exit_code
