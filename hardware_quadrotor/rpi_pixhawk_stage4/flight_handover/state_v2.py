"""ROS-independent fail-closed Alpha flight-handover V2 state machine."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


STATES = (
    "BOOT", "PARAMETER_AUDIT", "WAIT_CH8_INIT", "WAIT_OPTICAL_FLOW_MODE",
    "WAIT_ARM_AND_STABLE", "WAIT_CH5_ONBOARD_EDGE", "VERIFY_AUTO_HOVER",
    "VERIFY_OFFBOARD", "LATCH_TARGET", "PRESTREAM_COMMAND",
    "WAIT_CH6_TRACKING_EDGE", "VERIFY_CMD_CTRL", "ACTIVE_CONTROL",
    "WAIT_CH6_HOVER", "VERIFY_AUTO_HOVER_RETURN", "STOP_AND_DRAIN_COMMAND",
    "WAIT_CH5_OPTICAL_FLOW", "VERIFY_POSCTL", "COMPLETE",
    "ABORT_PRE_TAKEOVER", "ABORT_ACTIVE_CONTROL",
)


class StaticHoldSource:
    name = "StaticHoldSource"

    def compute(self, timestamp: float, odom: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
        del odom
        return {
            "timestamp": float(timestamp),
            "position": dict(target["position"]),
            "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "acceleration": {"x": 0.0, "y": 0.0, "z": 0.0},
            "jerk": {"x": 0.0, "y": 0.0, "z": 0.0},
            "yaw": float(target["yaw"]),
            "yaw_dot": 0.0,
        }


class FlightHandoverV2:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.timing = config["timing"]
        self.safety = config["safety"]
        self.state = "BOOT"
        self.state_entered_at = 0.0
        self.state_history = [{"state": "BOOT", "entered_at": 0.0, "reason": "startup"}]
        self.events: List[Dict[str, Any]] = []
        self.parameter_audit = {"parameter_audit_passed": False, "parameter_conflicts": [{"reason": "not_run"}]}
        self.target: Optional[Dict[str, Any]] = None
        controller = config.get("controller") or {"mode": "static_hold"}
        mode = controller.get("mode")
        if mode == "static_hold":
            self.source = StaticHoldSource()
        elif mode == "tinympc_zero_motion":
            from rpi_pixhawk_stage4.controller.tinympc_source import (
                TinyMPCZeroMotionSource,
            )

            self.source = TinyMPCZeroMotionSource(controller)
        else:
            raise ValueError("unknown_stage4_controller_mode:{0}".format(mode))
        self.auto_hover_confirmed_at = None
        self.offboard_confirmed_at = None
        self.publisher_created_at = None
        self.publisher_unregistered = False
        self.published_count = 0
        self.published_before_offboard_count = 0
        self.prestream_started_at = None
        self.prestream_message_count = 0
        self.last_publish_at = None
        self.max_publish_gap_seconds = 0.0
        self.active_started_at = None
        self.active_ended_at = None
        self.active_duration_seconds = 0.0
        self.command_stream_stopped_at = None
        self.cmd_stale_confirmed = False
        self.abort_reason = None
        self.abort_context: Dict[str, Any] = {}
        self.stable_since: Optional[float] = None
        self.wait_ch5_entered_semantic: Optional[str] = None
        self.wait_ch5_return_entered_semantic: Optional[str] = None
        self.wait_ch6_entered_semantic: Optional[str] = None
        self.wait_ch6_return_entered_semantic: Optional[str] = None
        self.ch6_edge_seen = False

    def set_parameter_audit(self, result: Dict[str, Any]) -> None:
        self.parameter_audit = dict(result)

    def transition(self, state: str, now: float, reason: str) -> None:
        if state not in STATES:
            raise ValueError("unknown_handover_state:{0}".format(state))
        if state != self.state:
            self.state = state
            self.state_entered_at = float(now)
            self.state_history.append({"state": state, "entered_at": float(now), "reason": reason})

    @staticmethod
    def topology_valid(snapshot: Dict[str, Any]) -> bool:
        nodes = list(snapshot.get("subscriber_nodes") or [])
        return int(snapshot.get("subscriber_count", len(nodes))) == 1 and len(nodes) == 1 and "px4ctrl" in str(nodes[0]).lower()

    @staticmethod
    def fresh(snapshot: Dict[str, Any]) -> Optional[str]:
        if not snapshot.get("mavros_connected"):
            return "mavros_disconnected"
        for channel in ("state", "odom", "rc", "battery"):
            key = channel + "_fresh"
            if key in snapshot and not snapshot.get(key):
                return channel + "_stale"
        return None

    def stability_status(self, snapshot: Dict[str, Any], now: float) -> Dict[str, Any]:
        """Return deterministic stability evidence without mutating the machine."""
        failures: List[str] = []
        for channel in ("state", "rc", "battery", "odom"):
            key = channel + "_fresh"
            if key in snapshot and not snapshot.get(key):
                failures.append(channel + "_stale")
        checks = (
            (snapshot.get("armed") is True, "armed_not_true"),
            (snapshot.get("ch7_semantic") == "ARM", "ch7_not_unlocked"),
            (snapshot.get("px4_mode") == "POSCTL", "px4_mode_not_posctl"),
            (snapshot.get("px4ctrl_state") == "MANUAL_CTRL", "px4ctrl_not_manual"),
            (snapshot.get("ch8_semantic") == "INIT", "ch8_not_init"),
        )
        failures.extend(reason for valid, reason in checks if not valid)
        velocity = snapshot.get("odom_velocity") or {}
        numeric = {
            "velocity_x": velocity.get("x"), "velocity_y": velocity.get("y"),
            "velocity_z": velocity.get("z"), "roll": snapshot.get("roll"),
            "pitch": snapshot.get("pitch"),
        }
        finite: Dict[str, float] = {}
        for name, value in numeric.items():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                failures.append(name + "_nonfinite")
            else:
                finite[name] = float(value)
        horizontal_limit = float(self.safety["stable_horizontal_velocity_mps"])
        vertical_limit = float(self.safety["stable_vertical_velocity_mps"])
        tilt_limit = float(self.safety["stable_tilt_degrees"])
        if "velocity_x" in finite and abs(finite["velocity_x"]) >= horizontal_limit:
            failures.append("velocity_x_unstable")
        if "velocity_y" in finite and abs(finite["velocity_y"]) >= horizontal_limit:
            failures.append("velocity_y_unstable")
        if "velocity_z" in finite and abs(finite["velocity_z"]) >= vertical_limit:
            failures.append("velocity_z_unstable")
        if "roll" in finite and abs(math.degrees(finite["roll"])) >= tilt_limit:
            failures.append("roll_unstable")
        if "pitch" in finite and abs(math.degrees(finite["pitch"])) >= tilt_limit:
            failures.append("pitch_unstable")
        elapsed = 0.0
        if not failures and self.stable_since is not None:
            elapsed = max(0.0, float(now) - self.stable_since)
        return {
            "stable": not failures,
            "failure_fields": failures,
            "stable_elapsed": elapsed,
            "stable_required": float(self.timing["stable_hover_seconds"]),
        }
    def hover_permission(self, snapshot: Dict[str, Any]) -> Optional[str]:
        failure = self.fresh(snapshot)
        if failure:
            return failure
        conditions = (
            (snapshot.get("armed") is True, "arm_lost"),
            (snapshot.get("px4_mode") == "OFFBOARD", "unexpected_px4_mode"),
            (snapshot.get("offboard_enabled") is True, "offboard_not_enabled"),
            (snapshot.get("px4ctrl_state") == "AUTO_HOVER", "px4ctrl_not_auto_hover"),
            (snapshot.get("ch5_semantic") == "ONBOARD", "ch5_not_onboard"),
            (snapshot.get("ch6_semantic") == "HOVER", "ch6_not_hover"),
            (snapshot.get("ch7_semantic") == "ARM", "ch7_not_arm"),
            (snapshot.get("ch8_semantic") == "INIT", "ch8_not_init"),
            (self.topology_valid(snapshot), "setpoints_subscriber_invalid"),
        )
        return next((reason for allowed, reason in conditions if not allowed), None)

    def post_auto_hover_permission(self, snapshot: Dict[str, Any], allow_tracking: bool = False) -> Optional[str]:
        """Guard the legal post-OFFBOARD command stream without granting ACTIVE authority."""
        failure = self.fresh(snapshot)
        if failure:
            return failure
        ch6_allowed = snapshot.get("ch6_semantic") == "HOVER" or (
            allow_tracking and snapshot.get("ch6_semantic") == "TRACKING"
        )
        conditions = (
            (snapshot.get("armed") is True, "arm_lost"),
            (snapshot.get("px4_mode") == "OFFBOARD", "unexpected_px4_mode"),
            (snapshot.get("offboard_enabled") is True, "offboard_not_enabled"),
            (snapshot.get("px4ctrl_state") in ("AUTO_HOVER", "CMD_CTRL"), "px4ctrl_not_post_auto_hover"),
            (snapshot.get("ch5_semantic") == "ONBOARD", "ch5_not_onboard"),
            (ch6_allowed, "ch6_not_hover_or_tracking" if allow_tracking else "ch6_not_hover"),
            (snapshot.get("ch7_semantic") == "ARM", "ch7_not_arm"),
            (snapshot.get("ch8_semantic") == "INIT", "ch8_not_init"),
            (self.topology_valid(snapshot), "setpoints_subscriber_invalid"),
        )
        return next((reason for allowed, reason in conditions if not allowed), None)

    def active_permission(self, snapshot: Dict[str, Any]) -> Optional[str]:
        failure = self.fresh(snapshot)
        if failure:
            return failure
        conditions = (
            (snapshot.get("armed") is True, "arm_lost"),
            (snapshot.get("px4_mode") == "OFFBOARD", "unexpected_px4_mode"),
            (snapshot.get("offboard_enabled") is True, "offboard_not_enabled"),
            (snapshot.get("px4ctrl_state") == "CMD_CTRL", "px4ctrl_not_cmd_ctrl"),
            (snapshot.get("cmd_fresh") is True, "cmd_stale"),
            (snapshot.get("ch5_semantic") == "ONBOARD", "ch5_not_onboard"),
            (snapshot.get("ch6_semantic") == "TRACKING", "ch6_not_tracking"),
            (snapshot.get("ch7_semantic") == "ARM", "ch7_not_arm"),
            (snapshot.get("ch8_semantic") == "INIT", "ch8_not_init"),
            (self.topology_valid(snapshot), "setpoints_subscriber_invalid"),
        )
        return next((reason for allowed, reason in conditions if not allowed), None)

    def control_authority(self, snapshot: Dict[str, Any]) -> str:
        if self.active_permission(snapshot) is None:
            return "STAGE4_COMMAND"
        if self.hover_permission(snapshot) is None:
            return "PX4CTRL_HOVER"
        return "NONE"

    def abort(self, active: bool, reason: str, now: float) -> List[str]:
        self.abort_reason = reason
        self.transition("ABORT_ACTIVE_CONTROL" if active else "ABORT_PRE_TAKEOVER", now, reason)
        if active or (self.publisher_created_at is not None and not self.publisher_unregistered):
            return ["UNREGISTER_PUBLISHER"]
        return []

    def record_publish(self, now: float, snapshot: Dict[str, Any]) -> None:
        if self.offboard_confirmed_at is None or snapshot.get("px4_mode") != "OFFBOARD":
            self.published_before_offboard_count += 1
            raise RuntimeError("publish_before_offboard_forbidden")
        if self.last_publish_at is not None:
            gap = float(now) - float(self.last_publish_at)
            self.max_publish_gap_seconds = max(self.max_publish_gap_seconds, gap)
            if gap > float(self.timing["max_publish_gap_seconds"]):
                raise RuntimeError("publish_gap_exceeded")
        self.last_publish_at = float(now)
        self.published_count += 1
        if self.state == "PRESTREAM_COMMAND":
            self.prestream_message_count += 1

    def step(self, s: Dict[str, Any], now: float) -> List[str]:
        now = float(now)
        actions: List[str] = []
        state = self.state
        if state in ("COMPLETE", "ABORT_PRE_TAKEOVER", "ABORT_ACTIVE_CONTROL"):
            return actions
        if state in ("WAIT_CH8_INIT", "WAIT_OPTICAL_FLOW_MODE", "WAIT_ARM_AND_STABLE"):
            if s.get("ch5_semantic") == "ONBOARD":
                return self.abort(False, "premature_ch5_onboard", now)
            if s.get("ch6_semantic") == "TRACKING":
                return self.abort(False, "premature_ch6_tracking", now)
        if state == "BOOT":
            self.transition("PARAMETER_AUDIT", now, "boot_complete")
        elif state == "PARAMETER_AUDIT":
            if not self.parameter_audit.get("parameter_audit_passed"):
                return self.abort(False, "parameter_audit_failed", now)
            self.transition("WAIT_CH8_INIT", now, "parameter_audit_passed")
        elif state == "WAIT_CH8_INIT":
            if s.get("ch8_semantic") == "INIT":
                self.transition("WAIT_OPTICAL_FLOW_MODE", now, "ch8_init")
        elif state == "WAIT_OPTICAL_FLOW_MODE":
            if s.get("ch5_semantic") == "OPTICAL_FLOW" and s.get("ch6_semantic") == "HOVER" and s.get("px4_mode") == "POSCTL" and s.get("px4ctrl_state") == "MANUAL_CTRL" and s.get("ch8_semantic") == "INIT":
                self.transition("WAIT_ARM_AND_STABLE", now, "manual_posctl_ready")
        elif state == "WAIT_ARM_AND_STABLE":
            status = self.stability_status(s, now)
            if status["stable"]:
                if self.stable_since is None:
                    self.stable_since = now
                if now - self.stable_since >= float(self.timing["stable_hover_seconds"]):
                    self.wait_ch5_entered_semantic = s.get("ch5_semantic")
                    self.transition("WAIT_CH5_ONBOARD_EDGE", now, "stable_hover")
                    actions.append("ANNOUNCE_READY_CH5")
            else:
                self.stable_since = None
        elif state == "WAIT_CH5_ONBOARD_EDGE":
            checks = (
                (s.get("ch6_semantic") != "TRACKING", "premature_ch6_tracking"),
                (s.get("ch7_semantic") == "ARM", "ch7_kill_activated"),
                (s.get("armed") is True, "armed_lost_before_handover"),
                (s.get("ch8_semantic") == "INIT", "ch8_prerequisite_lost"),
            )
            failure = next((reason for valid, reason in checks if not valid), None)
            if failure:
                return self.abort(False, failure, now)
            if s.get("ch5_semantic") == "ONBOARD":
                valid_ch5_window = self.wait_ch5_entered_semantic == "OPTICAL_FLOW"
                if not valid_ch5_window:
                    return self.abort(False, "ch5_onboard_without_fresh_window", now)
                self.transition("VERIFY_AUTO_HOVER", now, "ch5_optical_to_onboard")
        elif state == "VERIFY_AUTO_HOVER":
            if s.get("px4ctrl_state") == "AUTO_HOVER":
                self.auto_hover_confirmed_at = now
                self.transition("VERIFY_OFFBOARD", now, "auto_hover_confirmed")
            elif now - self.state_entered_at > float(self.timing["auto_hover_timeout_seconds"]):
                self.abort_context = {"expected": {"px4ctrl_state": "AUTO_HOVER"}, "actual": {"px4ctrl_state": s.get("px4ctrl_state")}}
                return self.abort(False, "auto_hover_timeout", now)
        elif state == "VERIFY_OFFBOARD":
            if self.auto_hover_confirmed_at is not None and s.get("px4_mode") == "OFFBOARD" and s.get("offboard_enabled") is True and s.get("px4ctrl_state") == "AUTO_HOVER" and s.get("ch5_semantic") == "ONBOARD" and s.get("ch6_semantic") == "HOVER" and s.get("armed") is True and s.get("odom_fresh") and s.get("rc_fresh"):
                self.offboard_confirmed_at = now
                self.transition("LATCH_TARGET", now, "offboard_confirmed")
            elif now - self.state_entered_at > float(self.timing["offboard_timeout_seconds"]):
                self.abort_context = {"expected": {"px4_mode": "OFFBOARD", "px4ctrl_state": "AUTO_HOVER"}, "actual": {"px4_mode": s.get("px4_mode"), "px4ctrl_state": s.get("px4ctrl_state")}}
                return self.abort(False, "offboard_timeout", now)
        elif state == "LATCH_TARGET":
            position, yaw = s.get("position"), s.get("yaw")
            values = [] if not isinstance(position, dict) else [position.get(axis) for axis in ("x", "y", "z")]
            values.append(yaw)
            if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
                return self.abort(False, "target_nonfinite", now)
            self.target = {"position": {axis: float(position[axis]) for axis in ("x", "y", "z")}, "yaw": float(yaw), "odom_reset_counter": s.get("odom_reset_counter"), "source": "AUTO_HOVER_OFFBOARD_ODOM"}
            self.prestream_started_at = now
            self.publisher_created_at = now
            self.publisher_unregistered = False
            self.transition("PRESTREAM_COMMAND", now, "target_latched")
            actions.append("CREATE_PUBLISHER")
        elif state == "PRESTREAM_COMMAND":
            failure = self.post_auto_hover_permission(s)
            if failure:
                return self.abort(False, failure, now)
            actions.append("PUBLISH_STATIC_HOLD")
            if now - float(self.prestream_started_at) >= float(self.timing["prestream_seconds"]) and self.prestream_message_count >= int(self.timing["prestream_min_messages"]):
                self.wait_ch6_entered_semantic = s.get("ch6_semantic")
                self.ch6_edge_seen = False
                self.transition("WAIT_CH6_TRACKING_EDGE", now, "prestream_complete")
                actions.append("ANNOUNCE_READY_CH6")
        elif state == "WAIT_CH6_TRACKING_EDGE":
            failure = self.post_auto_hover_permission(s, allow_tracking=True)
            if failure:
                return self.abort(False, failure, now)
            if self.wait_ch6_entered_semantic is None:
                self.wait_ch6_entered_semantic = s.get("ch6_semantic")
            actions.append("PUBLISH_STATIC_HOLD")
            if s.get("ch6_semantic") == "TRACKING":
                valid_edge = self.wait_ch6_entered_semantic == "HOVER"
                if not valid_edge:
                    return self.abort(False, "ch6_tracking_without_fresh_edge", now)
                self.ch6_edge_seen = True
                self.transition("VERIFY_CMD_CTRL", now, "ch6_hover_to_tracking_edge_confirmed")
                actions.append("ANNOUNCE_CH6_EDGE_CONFIRMED")
        elif state == "VERIFY_CMD_CTRL":
            if self.active_permission(s) is None:
                self.active_started_at = now
                self.transition("ACTIVE_CONTROL", now, "cmd_ctrl_confirmed")
                actions.append("ANNOUNCE_ACTIVE")
            elif now - self.state_entered_at > float(self.timing["cmd_ctrl_timeout_seconds"]):
                return self.abort(False, "cmd_ctrl_timeout", now)
        elif state == "ACTIVE_CONTROL":
            elapsed = now - float(self.active_started_at)
            if elapsed >= float(self.timing["active_command_window_seconds"]):
                self.active_ended_at = now
                self.active_duration_seconds = min(elapsed, float(self.timing["active_command_window_seconds"]))
                self.wait_ch6_return_entered_semantic = s.get("ch6_semantic")
                self.transition("WAIT_CH6_HOVER", now, "active_window_complete")
                actions.append("PUBLISH_LAST_FINITE_HOLD")
            else:
                failure = self.active_permission(s)
                if failure:
                    return self.abort(True, failure, now)
                actions.append("PUBLISH_STATIC_HOLD")
        elif state == "WAIT_CH6_HOVER":
            if s.get("ch6_semantic") == "HOVER":
                valid_ch6_return = self.wait_ch6_return_entered_semantic == "TRACKING"
                if not valid_ch6_return:
                    return self.abort(True, "ch6_hover_without_fresh_return_window", now)
                self.transition("VERIFY_AUTO_HOVER_RETURN", now, "ch6_tracking_to_hover")
        elif state == "VERIFY_AUTO_HOVER_RETURN":
            if s.get("px4ctrl_state") == "AUTO_HOVER":
                self.command_stream_stopped_at = now
                self.publisher_unregistered = True
                self.transition("STOP_AND_DRAIN_COMMAND", now, "auto_hover_returned")
                actions.append("UNREGISTER_PUBLISHER")
            elif now - self.state_entered_at > float(self.timing["return_auto_hover_timeout_seconds"]):
                return self.abort(True, "return_auto_hover_timeout", now)
        elif state == "STOP_AND_DRAIN_COMMAND":
            if not s.get("cmd_fresh") or now - float(self.command_stream_stopped_at) >= float(self.timing["command_drain_seconds"]):
                self.cmd_stale_confirmed = not s.get("cmd_fresh")
                self.wait_ch5_return_entered_semantic = s.get("ch5_semantic")
                self.transition("WAIT_CH5_OPTICAL_FLOW", now, "cmd_stale")
        elif state == "WAIT_CH5_OPTICAL_FLOW":
            if s.get("ch5_semantic") == "OPTICAL_FLOW":
                valid_ch5_return = self.wait_ch5_return_entered_semantic == "ONBOARD"
                if not valid_ch5_return:
                    return self.abort(True, "ch5_optical_without_fresh_return_window", now)
                self.transition("VERIFY_POSCTL", now, "ch5_onboard_to_optical")
        elif state == "VERIFY_POSCTL":
            if s.get("px4ctrl_state") == "MANUAL_CTRL" and s.get("px4_mode") == "POSCTL" and not s.get("cmd_fresh") and s.get("ch5_semantic") == "OPTICAL_FLOW" and s.get("ch6_semantic") == "HOVER" and s.get("ch8_semantic") == "INIT" and self.publisher_unregistered:
                self.transition("COMPLETE", now, "manual_posctl_restored")
            elif now - self.state_entered_at > float(self.timing["posctl_return_timeout_seconds"]):
                return self.abort(True, "posctl_return_timeout", now)
        return actions

    def summary(self, final: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "state": self.state, "state_history": list(self.state_history),
            "parameter_audit_passed": bool(self.parameter_audit.get("parameter_audit_passed")),
            "parameter_conflicts": list(self.parameter_audit.get("parameter_conflicts", [])),
            "abort_context": dict(self.abort_context),
            "selected_setpoint_source": self.source.name, "selected_controller": self.source.name,
            "controller_module_path": "{0}.{1}".format(type(self.source).__module__, type(self.source).__name__),
            "controller_config_path": self.config.get("resolved_path"), "candidate_097_used": False,
            "publisher_created": self.publisher_created_at is not None,
            "publisher_created_at": self.publisher_created_at,
            "publisher_created_after_auto_hover": self.publisher_created_at is not None and self.auto_hover_confirmed_at is not None and self.publisher_created_at >= self.auto_hover_confirmed_at,
            "publisher_created_after_offboard": self.publisher_created_at is not None and self.offboard_confirmed_at is not None and self.publisher_created_at >= self.offboard_confirmed_at,
            "published_before_offboard_count": self.published_before_offboard_count,
            "prestream_message_count": self.prestream_message_count,
            "wait_ch5_entered_semantic": self.wait_ch5_entered_semantic,
            "wait_ch5_return_entered_semantic": self.wait_ch5_return_entered_semantic,
            "wait_ch6_entered_semantic": self.wait_ch6_entered_semantic,
            "wait_ch6_return_entered_semantic": self.wait_ch6_return_entered_semantic,
            "ch6_edge_seen": self.ch6_edge_seen,
            "active_started_at": self.active_started_at, "active_ended_at": self.active_ended_at,
            "active_duration_seconds": self.active_duration_seconds,
            "active_window_hard_limit_seconds": float(self.timing["active_command_window_seconds"]),
            "max_publish_gap_seconds": self.max_publish_gap_seconds,
            "cmd_stream_stopped_at": self.command_stream_stopped_at,
            "cmd_stale_confirmed": self.cmd_stale_confirmed,
            "final_px4_mode": final.get("px4_mode"), "final_px4ctrl_state": final.get("px4ctrl_state"),
            "publisher_unregistered": self.publisher_unregistered,
            "control_authority_final": self.control_authority(final),
            "actual_publish_count": self.published_count,
            "exit_reason": "complete" if self.state == "COMPLETE" else self.abort_reason,
        }
