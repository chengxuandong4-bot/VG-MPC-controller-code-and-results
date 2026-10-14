"""Offline and ROS-live entry point for the Alpha V1 audit-only migration."""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path
from typing import List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rpi_pixhawk_stage4.alpha_px4_adapter import (
    AlphaPx4AuditAdapter,
    load_alpha_config,
)
from rpi_pixhawk_stage4.flight_handover.v1_legacy import (
    LEGACY_ACTUAL_PUBLISH_ERROR,
    run_legacy_audit,
)
from rpi_pixhawk_stage4.flight_handover.v2_safe import load_v2_config, run_v2


REAL_PUBLISH_CONFIRMATION = "I_ACCEPT_ALPHA_REAL_PUBLISH_RISK"
BENCH_PUBLISH_CONFIRMATION = "ALPHA_BENCH_NO_CONSUMER"
BENCH_PX4CTRL_CONFIRMATION = "ALPHA_BENCH_PX4CTRL_CONSUMER"
FLIGHT_HANDOVER_HOLD_CONFIRMATION = "ALPHA_FLIGHT_HANDOVER_HOLD"


def _load_ros_noetic():
    """Load the receive-only ROS boundary only for explicit live-audit runs."""
    try:
        import rospy
        from mavros_msgs.msg import RCIn, State  # noqa: F401
        from nav_msgs.msg import Odometry  # noqa: F401
        from sensor_msgs.msg import BatteryState  # noqa: F401
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("live_audit_requires_ros_noetic_environment") from exc
    return rospy


def _load_ros_bench():
    """Load the narrow ROS publication boundary only for explicit bench mode."""
    try:
        rospy = _load_ros_noetic()
        import rosgraph
        import rosnode
        from quadrotor_msgs.msg import PositionCommand
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("bench_publish_requires_ros_noetic_environment") from exc
    return rospy, rosgraph, rosnode, PositionCommand


_REAL_LOAD_ROS_BENCH = _load_ros_bench


def _create_position_command_publisher(rospy, topic: str, message_type):
    """The sole production ROS Publisher construction boundary."""
    return rospy.Publisher(topic, message_type, queue_size=1)


def _publish_position_command(publisher, message) -> None:
    """The sole production PositionCommand publication boundary."""
    publisher.publish(message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Alpha V1 PX4 audit-only Stage 4 sidecar")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--results-dir",
        default="./rpi_pixhawk_stage4/results/alpha_audit",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--simulate", action="store_true", default=False)
    mode.add_argument("--live-audit", action="store_true", default=False)
    mode.add_argument(
        "--bench-publish-no-consumer", action="store_true", default=False
    )
    mode.add_argument(
        "--bench-px4ctrl-consumer", action="store_true", default=False
    )
    mode.add_argument("--flight-handover-hold", action="store_true", default=False)
    parser.add_argument("--flight-handover-version", choices=("v1", "v2"), default=None)
    parser.add_argument("--maximum-runtime-seconds", type=float, default=None)
    parser.add_argument("--enable-actual-publish", action="store_true", default=False)
    parser.add_argument("--actual-publish-confirmation", default="")
    return parser


def _simulation_event(adapter: AlphaPx4AuditAdapter) -> dict:
    now = time.monotonic()
    adapter.update_state(
        {"connected": True, "armed": False, "guided": False, "mode": "POSCTL"},
        now,
    )
    adapter.update_rc(
        {"channels": [1500, 1500, 1500, 1500, 999, 999, 1999, 1499]},
        now,
    )
    adapter.update_battery(
        {"voltage": 15.2, "current": 2.1, "percentage": 0.73},
        now,
    )
    adapter.update_odom(
        {
            "position": {"x": 0.0, "y": 0.0, "z": 0.8},
            "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "yaw": 0.0,
        },
        now,
    )
    return adapter.evaluate(
        {
            "vx_north_mps": 0.0,
            "vy_east_mps": 0.0,
            "vz_down_mps": 0.0,
            "yaw_rate_radps": 0.0,
        },
        now=now,
    )


def _message_status(adapter: AlphaPx4AuditAdapter, now: float) -> dict:
    freshness = adapter.telemetry_freshness(float(now))
    return {
        name: {
            "received": item["received"],
            "fresh": item["fresh"],
            "age_seconds": item["age_seconds"],
        }
        for name, item in freshness.items()
    }

def _run_live_audit(
    config: dict,
    results_dir: Path,
    maximum_runtime: float,
) -> int:
    rospy = _load_ros_noetic()
    adapter = AlphaPx4AuditAdapter(config=config, clock=time.monotonic)
    rospy.init_node("stage4_alpha_live_audit", anonymous=False)
    adapter.attach_ros_subscribers()

    audit_path = results_dir / "alpha_audit.jsonl"
    summary_path = results_dir / "run_summary.json"
    rate = rospy.Rate(10.0)
    started = time.monotonic()
    last_event = None
    event_count = 0
    would_publish_count = 0
    exit_reason = "maximum_runtime"
    error = None
    exit_code = 0

    try:
        with audit_path.open("w", encoding="utf-8") as handle:
            while True:
                now = time.monotonic()
                if now - started >= maximum_runtime:
                    exit_reason = "maximum_runtime"
                    break
                if rospy.is_shutdown():
                    exit_reason = "rospy_shutdown"
                    break

                last_event = adapter.evaluate({}, now=now)
                handle.write(json.dumps(last_event, sort_keys=True) + "\n")
                handle.flush()
                event_count += 1
                if last_event.get("event") == "WOULD_PUBLISH":
                    would_publish_count += 1
                rate.sleep()
    except KeyboardInterrupt:
        exit_reason = "keyboard_interrupt"
        exit_code = 130
    except Exception as exc:
        exit_reason = "exception:{0}".format(type(exc).__name__)
        error = str(exc)
        exit_code = 1

    finished = time.monotonic()
    last_rc_values = (
        last_event.get("rc_values", {})
        if last_event is not None
        else {"CH5": None, "CH6": None, "CH7": None, "CH8": None}
    )
    last_reject_reasons = (
        list(last_event.get("rejection_reasons", []))
        if last_event is not None
        else []
    )
    summary = {
        "platform": config["platform"],
        "backend": config["backend"],
        "mode": "live_audit",
        "exit_reason": exit_reason,
        "runtime_seconds": finished - started,
        "maximum_runtime_seconds": maximum_runtime,
        "event_count": event_count,
        "would_publish_count": would_publish_count,
        "actual_publish_count": adapter.actual_publish_count,
        "actual_publish_enabled": False,
        "audit_only": True,
        "last_gate_allowed": bool(
            last_event is not None and last_event.get("gate_authorized", False)
        ),
        "last_reject_reasons": last_reject_reasons,
        "last_rc_values": last_rc_values,
        "last_px4_mode": (adapter.state_snapshot or {}).get("mode", "UNKNOWN"),
        "message_status": _message_status(adapter, finished),
    }
    if error is not None:
        summary["error"] = error
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return exit_code


def _query_control_consumers(
    rospy, rosgraph, rosnode, command_topic: str, px4ctrl_node_name: str = "px4ctrl"
) -> dict:
    node_names = list(rosnode.get_node_names())
    expected_name = str(px4ctrl_node_name).lower()
    px4ctrl_nodes = [
        name for name in node_names if expected_name in str(name).lower()
    ]
    _, subscriptions, _ = rosgraph.Master(rospy.get_name()).getSystemState()
    subscriber_nodes = []
    for topic, nodes in subscriptions:
        if topic == command_topic:
            subscriber_nodes.extend(str(node) for node in nodes)
    subscriber_nodes = sorted(set(subscriber_nodes))
    return {
        "px4ctrl_detected": bool(px4ctrl_nodes),
        "px4ctrl_nodes": px4ctrl_nodes,
        "subscriber_count": len(subscriber_nodes),
        "subscriber_nodes": subscriber_nodes,
    }


def _bench_reason(reasons: List[str]) -> Optional[str]:
    mapping = {
        "ARMED_DURING_BENCH": "armed_detected",
        "MAVROS_DISCONNECTED": "mavros_disconnected",
        "CH7_NOT_LOCKED": "ch7_not_locked",
        "NONFINITE_TELEMETRY": "nonfinite_telemetry",
        "STATE_STALE": "state_stale",
        "RC_STALE": "rc_stale",
        "BATTERY_STALE": "battery_stale",
        "ODOM_STALE": "odom_stale",
        "RC_CHANNELS_INSUFFICIENT": "rc_channels_insufficient",
        "PX4CTRL_DETECTED": "px4ctrl_detected",
        "COMMAND_TOPIC_HAS_SUBSCRIBERS": "command_topic_has_subscribers",
        "PX4CTRL_MISSING": "px4ctrl_missing",
        "COMMAND_TOPIC_SUBSCRIBER_COUNT_NOT_ONE": (
            "command_topic_subscriber_count_not_one"
        ),
        "COMMAND_TOPIC_SUBSCRIBER_NOT_PX4CTRL": (
            "command_topic_subscriber_not_px4ctrl"
        ),
    }
    for reason in reasons:
        if reason in mapping:
            return mapping[reason]
    return None


def _bench_combined_safety_status(
    adapter: AlphaPx4AuditAdapter,
    consumers: dict,
    now_mono: float,
    require_px4ctrl_consumer: bool = False,
) -> dict:
    """Combine telemetry and ROS consumer gates at one monotonic instant."""
    status = adapter.bench_safety_status(float(now_mono))
    reasons = list(status["rejection_reasons"])
    subscriber_nodes = list(consumers["subscriber_nodes"])
    subscriber_is_px4ctrl = bool(
        len(subscriber_nodes) == 1
        and "px4ctrl" in subscriber_nodes[0].lower()
    )
    if require_px4ctrl_consumer:
        if not consumers["px4ctrl_detected"]:
            reasons.append("PX4CTRL_MISSING")
        if consumers["subscriber_count"] != 1:
            reasons.append("COMMAND_TOPIC_SUBSCRIBER_COUNT_NOT_ONE")
        elif not subscriber_is_px4ctrl:
            reasons.append("COMMAND_TOPIC_SUBSCRIBER_NOT_PX4CTRL")
    else:
        if consumers["px4ctrl_detected"]:
            reasons.append("PX4CTRL_DETECTED")
        if consumers["subscriber_count"] > 0:
            reasons.append("COMMAND_TOPIC_HAS_SUBSCRIBERS")
    combined = dict(status)
    combined["subscriber_is_px4ctrl"] = subscriber_is_px4ctrl
    combined["rejection_reasons"] = list(dict.fromkeys(reasons))
    return combined


def _stale_evidence(safety_status: dict) -> dict:
    reason_to_channel = {
        "STATE_STALE": "state",
        "RC_STALE": "rc",
        "BATTERY_STALE": "battery",
        "ODOM_STALE": "odom",
    }
    reasons = safety_status.get("rejection_reasons", [])
    freshness = safety_status.get("telemetry_freshness", {})
    for reason in reasons:
        channel = reason_to_channel.get(reason)
        if channel is None:
            continue
        item = freshness[channel]
        return {
            "stale_channel": channel,
            "stale_age_seconds": item["age_seconds"],
            "stale_timeout_seconds": item["timeout_seconds"],
            "stale_now_monotonic": item["now_monotonic"],
            "stale_last_receive_monotonic": item["last_receive_monotonic"],
        }
    return {
        "stale_channel": None,
        "stale_age_seconds": None,
        "stale_timeout_seconds": None,
        "stale_now_monotonic": None,
        "stale_last_receive_monotonic": None,
    }

def _run_bench_publish(
    config: dict,
    results_dir: Path,
    maximum_runtime: float,
    require_px4ctrl_consumer: bool = False,
) -> int:
    rospy, rosgraph, rosnode, position_command_type = _load_ros_bench()
    adapter = AlphaPx4AuditAdapter(
        config=config,
        clock=time.monotonic,
        bench_publish_mode=True,
    )
    node_name = (
        "stage4_alpha_bench_px4ctrl_consumer"
        if require_px4ctrl_consumer
        else "stage4_alpha_bench_publish_no_consumer"
    )
    rospy.init_node(node_name, anonymous=False)
    adapter.attach_ros_subscribers()

    audit_path = results_dir / "alpha_bench_publish.jsonl"
    summary_path = results_dir / "run_summary.json"
    warmup_rate = rospy.Rate(50.0)
    publish_rate = None
    started = time.monotonic()
    startup_finished_at = None
    startup_ready_at = None
    publish_started_at = None
    publisher_created_at = None
    publisher = None
    last_event = None
    last_message = None
    last_consumers = {
        "px4ctrl_detected": False,
        "px4ctrl_nodes": [],
        "subscriber_count": 0,
        "subscriber_nodes": [],
    }
    last_safety_status = None
    stale_evidence = _stale_evidence({})
    armed_seen = False
    safety_abort_reason = None
    exit_reason = "maximum_runtime"
    exit_code = 0
    error = None
    startup_ready = False
    startup_unmet_conditions: List[str] = []
    startup_warmup_limit_seconds = 5.0

    try:
        with audit_path.open("w", encoding="utf-8") as handle:
            # Publisher creation is forbidden until the 50 Hz warmup and an
            # immediate second check both pass through the same safety path.
            while publisher is None:
                if rospy.is_shutdown():
                    now_mono = time.monotonic()
                    exit_reason = "rospy_shutdown"
                    safety_abort_reason = "rospy_shutdown"
                    startup_finished_at = now_mono
                    exit_code = 1
                    break

                try:
                    last_consumers = _query_control_consumers(
                        rospy, rosgraph, rosnode, config["command_topic"]
                    )
                except Exception as exc:
                    now_mono = time.monotonic()
                    exit_reason = "safety_abort"
                    safety_abort_reason = "ros_master_query_failed"
                    startup_finished_at = now_mono
                    error = str(exc)
                    exit_code = 1
                    break

                # Capture once after the potentially blocking ROS Master query.
                now_mono = time.monotonic()
                last_safety_status = _bench_combined_safety_status(
                    adapter,
                    last_consumers,
                    now_mono,
                    require_px4ctrl_consumer=require_px4ctrl_consumer,
                )
                startup_unmet_conditions = list(
                    last_safety_status["rejection_reasons"]
                )
                armed_seen = armed_seen or bool(
                    (adapter.state_snapshot or {}).get("armed", False)
                )

                if not startup_unmet_conditions:
                    # Revalidate through the exact runtime safety path before
                    # constructing the sole Publisher. No grace or sleep is used.
                    try:
                        last_consumers = _query_control_consumers(
                            rospy, rosgraph, rosnode, config["command_topic"]
                        )
                    except Exception as exc:
                        now_mono = time.monotonic()
                        exit_reason = "safety_abort"
                        safety_abort_reason = "ros_master_query_failed"
                        startup_finished_at = now_mono
                        error = str(exc)
                        exit_code = 1
                        break
                    now_mono = time.monotonic()
                    last_safety_status = _bench_combined_safety_status(
                        adapter,
                        last_consumers,
                        now_mono,
                        require_px4ctrl_consumer=require_px4ctrl_consumer,
                    )
                    startup_unmet_conditions = list(
                        last_safety_status["rejection_reasons"]
                    )
                    if not startup_unmet_conditions:
                        startup_ready = True
                        startup_ready_at = now_mono
                        startup_finished_at = now_mono
                        publisher = _create_position_command_publisher(rospy, config["command_topic"], position_command_type)

                        publisher_created_at = float(now_mono)
                        publish_started_at = publisher_created_at
                        publish_rate = rospy.Rate(10.0)
                        break

                if now_mono - started >= startup_warmup_limit_seconds:
                    startup_finished_at = now_mono
                    exit_reason = "startup_timeout"
                    safety_abort_reason = "startup_not_ready"
                    exit_code = 1
                    break
                warmup_rate.sleep()

            while publisher is not None and safety_abort_reason is None:
                if rospy.is_shutdown():
                    exit_reason = "rospy_shutdown"
                    break

                try:
                    last_consumers = _query_control_consumers(
                        rospy, rosgraph, rosnode, config["command_topic"]
                    )
                except Exception as exc:
                    exit_reason = "safety_abort"
                    safety_abort_reason = "ros_master_query_failed"
                    error = str(exc)
                    exit_code = 1
                    break

                # One monotonic snapshot governs every freshness decision in
                # this loop iteration, after ROS Master queries have completed.
                now_mono = time.monotonic()
                last_safety_status = _bench_combined_safety_status(
                    adapter,
                    last_consumers,
                    now_mono,
                    require_px4ctrl_consumer=require_px4ctrl_consumer,
                )
                last_event = adapter.evaluate_bench_hold(
                    now_mono, safety_status=last_safety_status
                )
                armed_seen = armed_seen or bool(
                    (adapter.state_snapshot or {}).get("armed", False)
                )
                reason = _bench_reason(last_event["rejection_reasons"])
                if reason is None and last_event["rejection_reasons"]:
                    reason = last_event["rejection_reasons"][0].lower()
                if reason is not None:
                    stale_evidence = _stale_evidence(last_safety_status)
                    exit_reason = "safety_abort"
                    safety_abort_reason = reason
                    exit_code = 1
                    break

                if now_mono - float(publish_started_at) >= maximum_runtime:
                    exit_reason = "maximum_runtime"
                    break

                candidate = last_event["candidate_command"]
                message = adapter.to_position_command_message(
                    candidate,
                    position_command_type,
                    rospy.Time.now(),
                )
                _publish_position_command(publisher, message)
                adapter.record_bench_publish()
                last_message = candidate
                last_event["event"] = "BENCH_PUBLISH"
                last_event["actual_publish_count"] = adapter.actual_publish_count
                handle.write(json.dumps(last_event, sort_keys=True) + "\n")
                handle.flush()
                publish_rate.sleep()
    except KeyboardInterrupt:
        exit_reason = "keyboard_interrupt"
        safety_abort_reason = "keyboard_interrupt"
        exit_code = 130
    except Exception as exc:
        exit_reason = "exception:{0}".format(type(exc).__name__)
        safety_abort_reason = "exception"
        error = str(exc)
        exit_code = 1
    finally:
        if publisher is not None and hasattr(publisher, "unregister"):
            publisher.unregister()

    finished = time.monotonic()
    if startup_finished_at is None:
        startup_finished_at = finished
    diagnostics = adapter.receive_diagnostics(now=finished)
    state = adapter.state_snapshot or {}
    rc_values = adapter._rc_values()
    subscriber_is_px4ctrl = bool(
        last_consumers["subscriber_count"] == 1
        and "px4ctrl" in last_consumers["subscriber_nodes"][0].lower()
    )
    preflight_checks = {
        "authorized_cli_and_config": True,
        "telemetry_gate_passed": startup_ready,
        "mavros_connected": bool(state.get("connected", False)),
        "disarmed": not bool(state.get("armed", False)),
        "ch7_locked": (
            rc_values["CH7"] is not None
            and float(config["rc"]["ch7"]["locked_range"][0])
            <= rc_values["CH7"]
            <= float(config["rc"]["ch7"]["locked_range"][1])
        ),
        "px4ctrl_absent": not last_consumers["px4ctrl_detected"],
        "command_topic_has_no_subscribers": (
            last_consumers["subscriber_count"] == 0
        ),
        "px4ctrl_present_required": (
            require_px4ctrl_consumer and last_consumers["px4ctrl_detected"]
        ),
        "exact_single_px4ctrl_subscriber": (
            require_px4ctrl_consumer and subscriber_is_px4ctrl
        ),
    }
    publisher_created = publisher_created_at is not None
    publisher_created_after_startup_ready = bool(
        startup_ready
        and startup_ready_at is not None
        and publisher_created_at is not None
        and publisher_created_at >= startup_ready_at
    )
    summary = {
        "mode": (
            "bench_px4ctrl_consumer"
            if require_px4ctrl_consumer
            else "bench_publish_no_consumer"
        ),
        "preflight_checks": preflight_checks,
        "publish_topic": config["command_topic"],
        "runtime_seconds": finished - started,
        "maximum_runtime_seconds": maximum_runtime,
        "bench_publish_count": adapter.actual_publish_count,
        "actual_publish_count": adapter.actual_publish_count,
        "px4ctrl_detected": last_consumers["px4ctrl_detected"],
        "subscriber_count": last_consumers["subscriber_count"],
        "subscriber_nodes": last_consumers["subscriber_nodes"],
        "subscriber_is_px4ctrl": subscriber_is_px4ctrl,
        "armed_seen": armed_seen,
        "exit_reason": exit_reason,
        "last_message": last_message,
        "safety_abort_reason": safety_abort_reason,
        "startup_wait_seconds": startup_finished_at - started,
        "startup_ready": startup_ready,
        "startup_unmet_conditions": startup_unmet_conditions,
        "first_message_received": diagnostics["first_message_received"],
        "last_receive_age_seconds": diagnostics["last_receive_age_seconds"],
        "header_stamp": diagnostics["header_stamp"],
        "publisher_created": publisher_created,
        "publisher_created_after_startup_ready": (
            publisher_created_after_startup_ready
        ),
    }
    summary.update(stale_evidence)
    if error is not None:
        summary["error"] = error
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return exit_code
FLIGHT_HANDOVER_STATES = (
    "WAIT_SYSTEM",
    "WAIT_ARM",
    "PRESTREAM_HOLD",
    "READY_FOR_CH5",
    "READY_FOR_CH6",
    "ACTIVE_HOLD",
    "RETURN_TO_MANUAL",
    "EXIT",
)


def _flight_handover_combined_safety_status(
    adapter: AlphaPx4AuditAdapter,
    consumers: dict,
    now_mono: float,
    require_armed: bool = False,
    require_ch7_unlocked: bool = False,
    require_ch5_manual: bool = False,
    require_ch6_manual: bool = False,
) -> dict:
    status = adapter.flight_handover_safety_status(
        float(now_mono),
        require_armed=require_armed,
        require_ch7_unlocked=require_ch7_unlocked,
        require_ch5_manual=require_ch5_manual,
        require_ch6_manual=require_ch6_manual,
    )
    reasons = list(status["rejection_reasons"])
    subscriber_nodes = list(consumers["subscriber_nodes"])
    expected_subscriber = "/" + str(adapter.config["px4ctrl_node_name"]).strip("/")
    subscriber_is_px4ctrl = bool(
        len(subscriber_nodes) == 1 and subscriber_nodes[0] == expected_subscriber
    )
    if not consumers["px4ctrl_detected"]:
        reasons.append("PX4CTRL_MISSING")
    if consumers["subscriber_count"] != 1:
        reasons.append("COMMAND_TOPIC_SUBSCRIBER_COUNT_NOT_ONE")
    elif not subscriber_is_px4ctrl:
        reasons.append("COMMAND_TOPIC_SUBSCRIBER_NOT_PX4CTRL")
    combined = dict(status)
    combined["subscriber_is_px4ctrl"] = subscriber_is_px4ctrl
    combined["rejection_reasons"] = list(dict.fromkeys(reasons))
    return combined


def _flight_abort_reason(reasons: List[str], state_name: str) -> Optional[str]:
    if "ARMED_REQUIRED" in reasons:
        return (
            "armed_lost_during_hold"
            if state_name in ("ACTIVE_HOLD", "RETURN_TO_MANUAL")
            else "arm_lost_before_handover"
        )
    mapping = {
        "MAVROS_DISCONNECTED": "mavros_disconnected",
        "CH7_NOT_UNLOCKED": "ch7_locked",
        "NONFINITE_TELEMETRY": "odom_invalid",
        "STATE_MISSING": "telemetry_stale",
        "RC_MISSING": "telemetry_stale",
        "BATTERY_MISSING": "telemetry_stale",
        "ODOM_MISSING": "telemetry_stale",
        "STATE_STALE": "telemetry_stale",
        "RC_STALE": "telemetry_stale",
        "BATTERY_STALE": "telemetry_stale",
        "ODOM_STALE": "telemetry_stale",
        "RC_CHANNELS_INSUFFICIENT": "subscriber_invalid",
        "PX4CTRL_MISSING": "px4ctrl_missing",
        "COMMAND_TOPIC_SUBSCRIBER_COUNT_NOT_ONE": "subscriber_invalid",
        "COMMAND_TOPIC_SUBSCRIBER_NOT_PX4CTRL": "subscriber_invalid",
    }
    for reason in reasons:
        if reason in mapping:
            return mapping[reason]
    return reasons[0].lower() if reasons else None


def _run_flight_handover_hold(
    config: dict, results_dir: Path, maximum_runtime: float
) -> int:
    rospy, rosgraph, rosnode, position_command_type = _load_ros_bench()
    # V1 remains executable only under the existing fake-ROS characterization
    # harness. Reject a real rospy module before init_node or Publisher creation.
    if (
        isinstance(rospy, types.ModuleType)
        and getattr(rospy, "__name__", "") == "rospy"
        and getattr(rospy, "__file__", None)
    ):
        raise RuntimeError(LEGACY_ACTUAL_PUBLISH_ERROR)
    adapter = AlphaPx4AuditAdapter(
        config=config, clock=time.monotonic, flight_handover_hold_mode=True
    )
    rospy.init_node("stage4_alpha_flight_handover_hold", anonymous=False)
    adapter.attach_ros_subscribers()

    audit_path = results_dir / "alpha_flight_handover_hold.jsonl"
    summary_path = results_dir / "run_summary.json"
    handover_config = config["flight_handover"]
    rate_hz = float(handover_config["prestream_rate_hz"])
    rate = rospy.Rate(rate_hz)
    minimum_messages = int(handover_config["prestream_min_messages"])
    minimum_duration = float(handover_config["prestream_min_duration_s"])
    status_interval = float(handover_config["status_print_interval_s"])
    return_timeout = float(handover_config["return_to_manual_timeout_s"])

    started = time.monotonic()
    state_name = "WAIT_SYSTEM"
    state_trace = [{"state": state_name, "timestamp_monotonic": started}]
    publisher = None
    publisher_created_at = None
    publisher_unregistered = False
    hold_target = None
    latched_target_record = None
    hold_latch_monotonic = None
    hold_target_latched = False
    prestream_started_at = None
    prestream_completed_at = None
    prestream_publish_count = 0
    ready_for_ch5 = False
    ready_for_ch6 = False
    ch5_onboard_detected = False
    ch5_onboard_detected_monotonic = None
    ch6_tracking_detected = False
    ch6_tracking_detected_monotonic = None
    early_ch5_switch = False
    early_ch6_switch = False
    handover_detected = False
    active_hold_started = False
    active_hold_started_at = None
    return_to_manual_started_at = None
    hold_validation_passed = False
    maximum_position_error_m = 0.0
    maximum_velocity_mps = 0.0
    armed_seen = False
    armed_during_hold = False
    arm_lost = False
    startup_ready = False
    last_message = None
    last_publish_monotonic = None
    maximum_publish_gap_seconds = 0.0
    last_status_print = None
    last_status_text = None
    last_safety_status = {"telemetry_freshness": {}, "rejection_reasons": []}
    last_consumers = {
        "px4ctrl_detected": False,
        "px4ctrl_nodes": [],
        "subscriber_count": 0,
        "subscriber_nodes": [],
    }
    exit_reason = "maximum_runtime"
    safety_abort_reason = None
    error = None
    exit_code = 0

    def transition(next_state: str, now_mono: float) -> None:
        nonlocal state_name, last_status_print, last_status_text
        if next_state not in FLIGHT_HANDOVER_STATES:
            raise RuntimeError("unknown_flight_handover_state:{0}".format(next_state))
        if state_name != next_state:
            state_name = next_state
            state_trace.append(
                {"state": next_state, "timestamp_monotonic": float(now_mono)}
            )
            last_status_print = None
            last_status_text = None
            print(next_state, flush=True)

    def status(message: str, now_mono: float, force: bool = False) -> None:
        nonlocal last_status_print, last_status_text
        due = last_status_print is None or now_mono - last_status_print >= status_interval
        if force or due:
            print(message, flush=True)
            last_status_print = float(now_mono)
            last_status_text = message

    def unregister_publisher() -> None:
        nonlocal publisher, publisher_unregistered
        if publisher is not None:
            if hasattr(publisher, "unregister"):
                publisher.unregister()
            publisher = None
            publisher_unregistered = True

    def clear_active_target() -> None:
        nonlocal hold_target
        hold_target = None

    def fail(reason: str, now_mono: float, code: int = 1) -> None:
        nonlocal exit_reason, safety_abort_reason, exit_code
        exit_reason = reason
        safety_abort_reason = reason
        exit_code = code
        unregister_publisher()
        transition("EXIT", now_mono)

    def finish_at_maximum_runtime(now_mono: float) -> bool:
        nonlocal exit_reason, safety_abort_reason, exit_code
        if now_mono - started < maximum_runtime:
            return False
        exit_reason = "maximum_runtime"
        safety_abort_reason = None
        exit_code = 0
        unregister_publisher()
        transition("EXIT", now_mono)
        return True

    def query_safety(now_mono: float, **requirements) -> dict:
        nonlocal last_consumers, last_safety_status
        last_consumers = _query_control_consumers(
            rospy,
            rosgraph,
            rosnode,
            config["command_topic"],
            config["px4ctrl_node_name"],
        )
        # All receive ages in this iteration share this post-query monotonic time.
        safety_now = time.monotonic()
        last_safety_status = _flight_handover_combined_safety_status(
            adapter, last_consumers, safety_now, **requirements
        )
        return last_safety_status

    def publish_latched(handle, now_mono: float, event_state: str) -> None:
        nonlocal last_message, prestream_started_at, prestream_publish_count
        nonlocal last_publish_monotonic, maximum_publish_gap_seconds
        if hold_target is None or publisher is None:
            raise RuntimeError("flight_hold_publish_without_latched_target")
        if prestream_started_at is None:
            prestream_started_at = float(now_mono)
        if last_publish_monotonic is not None:
            publish_gap = max(0.0, float(now_mono) - last_publish_monotonic)
            maximum_publish_gap_seconds = max(maximum_publish_gap_seconds, publish_gap)
            if publish_gap > max(0.5, 3.0 / rate_hz):
                raise RuntimeError("flight_hold_publish_gap_exceeded")
        candidate = adapter.build_flight_hold_command(hold_target, now_mono)
        locked = hold_target["position"]
        command_position = candidate["position"]
        continuity_values = (
            command_position["x"] - locked["x"],
            command_position["y"] - locked["y"],
            command_position["z"] - locked["z"],
            candidate["yaw"] - hold_target["yaw"],
        )
        if any(value != 0.0 for value in continuity_values):
            raise RuntimeError("flight_hold_target_continuity_violation")
        message = adapter.to_position_command_message(
            candidate, position_command_type, rospy.Time.now()
        )
        _publish_position_command(publisher, message)
        adapter.record_flight_hold_publish()
        last_publish_monotonic = float(now_mono)
        if event_state == "PRESTREAM_HOLD":
            prestream_publish_count += 1
        last_message = candidate
        event = {
            "timestamp_monotonic": float(now_mono),
            "event": "FLIGHT_HOLD_PUBLISH",
            "state": event_state,
            "candidate_command": candidate,
            "actual_publish_count": adapter.actual_publish_count,
        }
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()

    def update_motion_metrics() -> None:
        nonlocal maximum_position_error_m, maximum_velocity_mps
        if hold_target is None or adapter.odom_snapshot is None:
            return
        position = adapter.odom_snapshot["position"]
        velocity = adapter.odom_snapshot["velocity"]
        error_norm = sum(
            (float(position[axis]) - float(hold_target["position"][axis])) ** 2
            for axis in ("x", "y", "z")
        ) ** 0.5
        speed_norm = sum(float(velocity[axis]) ** 2 for axis in ("x", "y", "z")) ** 0.5
        maximum_position_error_m = max(maximum_position_error_m, error_norm)
        maximum_velocity_mps = max(maximum_velocity_mps, speed_norm)

    try:
        with audit_path.open("w", encoding="utf-8") as handle:
            while state_name != "EXIT":
                now_mono = time.monotonic()
                if rospy.is_shutdown():
                    fail("ros_shutdown", now_mono)
                    break
                if finish_at_maximum_runtime(now_mono):
                    break
                armed_now = bool((adapter.state_snapshot or {}).get("armed", False))
                armed_seen = armed_seen or armed_now

                try:
                    if state_name == "WAIT_SYSTEM":
                        last_safety_status = query_safety(
                            now_mono,
                            require_ch5_manual=True,
                            require_ch6_manual=True,
                        )
                        now_mono = float(last_safety_status["now_monotonic"])
                        if finish_at_maximum_runtime(now_mono):
                            break
                        armed_now = bool((adapter.state_snapshot or {}).get("armed", False))
                        armed_seen = armed_seen or armed_now
                        reasons = last_safety_status["rejection_reasons"]
                        status(
                            "WAIT_SYSTEM: state={0} rc={1} battery={2} odom={3} px4ctrl={4}".format(
                                "ok" if "STATE_MISSING" not in reasons else "wait",
                                "ok" if "RC_MISSING" not in reasons else "wait",
                                "ok" if "BATTERY_MISSING" not in reasons else "wait",
                                "ok" if "ODOM_MISSING" not in reasons else "wait",
                                "ok" if last_consumers["px4ctrl_detected"] else "wait",
                            ),
                            now_mono,
                        )
                        if not reasons:
                            startup_ready = True
                            transition("WAIT_ARM", now_mono)
                        rate.sleep()
                        continue

                    if state_name == "WAIT_ARM":
                        last_safety_status = query_safety(
                            now_mono,
                            require_armed=True,
                            require_ch7_unlocked=True,
                            require_ch5_manual=True,
                            require_ch6_manual=True,
                        )
                        now_mono = float(last_safety_status["now_monotonic"])
                        if finish_at_maximum_runtime(now_mono):
                            break
                        armed_now = bool((adapter.state_snapshot or {}).get("armed", False))
                        armed_seen = armed_seen or armed_now
                        reasons = last_safety_status["rejection_reasons"]
                        waiting_only = {"ARMED_REQUIRED", "CH7_NOT_UNLOCKED"}
                        non_waiting = [reason for reason in reasons if reason not in waiting_only]
                        if non_waiting:
                            transition("WAIT_SYSTEM", now_mono)
                            rate.sleep()
                            continue
                        switches = last_safety_status["switches"]
                        if reasons:
                            status(
                                "WAIT_ARM: armed={0} CH7={1}".format(
                                    armed_now,
                                    "unlocked" if switches["ch7_unlocked"] else "locked",
                                ),
                                now_mono,
                            )
                            rate.sleep()
                            continue
                        status("WAIT_ARM: armed=True, latching hold target", now_mono, True)
                        odom_at_latch = json.loads(json.dumps(adapter.odom_snapshot))
                        hold_target = adapter.latch_flight_hold_target(odom_at_latch)
                        latched_target_record = json.loads(json.dumps(hold_target))
                        hold_latch_monotonic = float(now_mono)
                        hold_target_latched = True
                        publisher = _create_position_command_publisher(
                            rospy, config["command_topic"], position_command_type
                        )
                        publisher_created_at = float(now_mono)
                        transition("PRESTREAM_HOLD", now_mono)
                        rate.sleep()
                        continue

                    last_safety_status = query_safety(
                        now_mono,
                        require_armed=True,
                        require_ch7_unlocked=True,
                    )
                    now_mono = float(last_safety_status["now_monotonic"])
                    if finish_at_maximum_runtime(now_mono):
                        break
                    armed_now = bool((adapter.state_snapshot or {}).get("armed", False))
                    armed_seen = armed_seen or armed_now
                    reasons = last_safety_status["rejection_reasons"]
                    if reasons:
                        failure = _flight_abort_reason(reasons, state_name)
                        if failure and failure.startswith("arm_lost"):
                            arm_lost = True
                            print(
                                "ARM_LOST_BEFORE_HANDOVER"
                                if state_name not in ("ACTIVE_HOLD", "RETURN_TO_MANUAL")
                                else "ARMED_LOST_DURING_HOLD",
                                flush=True,
                            )
                        fail(failure or "safety_abort", now_mono)
                        if state_name != "ACTIVE_HOLD":
                            clear_active_target()
                        break

                    switches = last_safety_status["switches"]
                    armed_during_hold = armed_during_hold or armed_now

                    if state_name == "PRESTREAM_HOLD":
                        publish_latched(handle, now_mono, state_name)
                        if switches["ch5_onboard"] and not early_ch5_switch:
                            early_ch5_switch = True
                            print("EARLY_CH5_SWITCH: continuing hold stream", flush=True)
                        if switches["ch6_tracking"] and not early_ch6_switch:
                            early_ch6_switch = True
                            print("EARLY_CH6_SWITCH: continuing hold stream", flush=True)
                        prestream_duration = max(
                            0.0, now_mono - float(prestream_started_at)
                        )
                        status(
                            "PRESTREAM_HOLD: {0}/{1}".format(
                                prestream_publish_count, minimum_messages
                            ),
                            now_mono,
                        )
                        if (
                            prestream_publish_count >= minimum_messages
                            and prestream_duration >= minimum_duration
                        ):
                            prestream_completed_at = float(now_mono)
                            ready_for_ch5 = True
                            transition("READY_FOR_CH5", now_mono)
                            status(
                                "READY_FOR_CH5\nHold target locked: x={x:.3f}, y={y:.3f}, z={z:.3f}, yaw={yaw:.3f}\nPrestream count: {count}\nSwitch CH5 to onboard now. Keep CH6 unchanged.".format(
                                    x=hold_target["position"]["x"],
                                    y=hold_target["position"]["y"],
                                    z=hold_target["position"]["z"],
                                    yaw=hold_target["yaw"],
                                    count=prestream_publish_count,
                                ),
                                now_mono,
                                True,
                            )

                    elif state_name == "READY_FOR_CH5":
                        publish_latched(handle, now_mono, state_name)
                        if switches["ch6_tracking"]:
                            early_ch6_switch = True
                        if switches["ch5_onboard"]:
                            ch5_onboard_detected = True
                            ch5_onboard_detected_monotonic = float(now_mono)
                            ready_for_ch6 = True
                            transition("READY_FOR_CH6", now_mono)
                            status(
                                "READY_FOR_CH6\nCH5 onboard detected.\nSwitch CH6 to tracking now.",
                                now_mono,
                                True,
                            )

                    elif state_name == "READY_FOR_CH6":
                        publish_latched(handle, now_mono, state_name)
                        if not switches["ch5_onboard"]:
                            transition("READY_FOR_CH5", now_mono)
                        elif switches["ch6_tracking"]:
                            ch6_tracking_detected = True
                            ch6_tracking_detected_monotonic = float(now_mono)
                            handover_detected = True
                            active_hold_started = True
                            active_hold_started_at = float(now_mono)
                            transition("ACTIVE_HOLD", now_mono)

                    elif state_name == "ACTIVE_HOLD":
                        if not switches["ch5_onboard"]:
                            exit_reason = "operator_handover_complete"
                            safety_abort_reason = None
                            exit_code = 0
                            unregister_publisher()
                            transition("EXIT", now_mono)
                            break
                        publish_latched(handle, now_mono, state_name)
                        update_motion_metrics()
                        active_elapsed = max(
                            0.0, now_mono - float(active_hold_started_at)
                        )
                        if active_elapsed >= float(config["validation_duration_seconds"]):
                            hold_validation_passed = True
                        status(
                            "ACTIVE_HOLD t={0:.1f}s publish={1} error_xyz={2:.3f}m armed={3}".format(
                                active_elapsed,
                                adapter.actual_publish_count,
                                maximum_position_error_m,
                                armed_now,
                            ),
                            now_mono,
                        )
                        if not switches["ch6_tracking"]:
                            return_to_manual_started_at = float(now_mono)
                            transition("RETURN_TO_MANUAL", now_mono)
                            status(
                                "RETURN_TO_MANUAL: switch CH5 out of onboard",
                                now_mono,
                                True,
                            )

                    elif state_name == "RETURN_TO_MANUAL":
                        if not switches["ch5_onboard"]:
                            exit_reason = "operator_handover_complete"
                            safety_abort_reason = None
                            exit_code = 0
                            unregister_publisher()
                            transition("EXIT", now_mono)
                            break
                        if (
                            now_mono - float(return_to_manual_started_at)
                            >= return_timeout
                        ):
                            fail("return_to_manual_timeout", now_mono)
                            break
                        publish_latched(handle, now_mono, state_name)
                        update_motion_metrics()
                        status(
                            "RETURN_TO_MANUAL: switch CH5 out of onboard",
                            now_mono,
                        )
                    rate.sleep()
                except RuntimeError as exc:
                    message = str(exc)
                    failure = (
                        "publish_gap"
                        if message == "flight_hold_publish_gap_exceeded"
                        else "odom_invalid"
                        if message.startswith("flight_hold_")
                        else "runtime_exception"
                    )
                    error = message
                    fail(failure, now_mono)
                    break
                except Exception as exc:
                    error = str(exc)
                    fail("ros_master_query_failed", now_mono)
                    break
    except KeyboardInterrupt:
        now_mono = time.monotonic()
        fail("keyboard_interrupt", now_mono, 130)
    finally:
        unregister_publisher()

    finished = time.monotonic()
    transition("EXIT", finished)
    diagnostics = adapter.receive_diagnostics(now=finished)
    prestream_duration_seconds = 0.0
    if prestream_started_at is not None:
        end = prestream_completed_at if prestream_completed_at is not None else finished
        prestream_duration_seconds = max(0.0, float(end) - prestream_started_at)
    active_hold_duration_seconds = 0.0
    if active_hold_started_at is not None:
        active_hold_duration_seconds = max(0.0, finished - active_hold_started_at)
    hold_position = (
        None if latched_target_record is None else latched_target_record["position"]
    )
    hold_yaw = None if latched_target_record is None else latched_target_record["yaw"]
    switches = adapter.flight_switch_state()
    summary = {
        "mode": "flight_handover_hold",
        "state": "EXIT",
        "state_trace": state_trace,
        "startup_ready": startup_ready,
        "hold_target_latched": hold_target_latched,
        "hold_position": hold_position,
        "hold_yaw": hold_yaw,
        "hold_latch_monotonic": hold_latch_monotonic,
        "prestream_started": prestream_started_at is not None,
        "prestream_publish_count": prestream_publish_count,
        "prestream_duration_seconds": prestream_duration_seconds,
        "prestream_rate_hz": rate_hz,
        "ready_for_ch5": ready_for_ch5,
        "ch5_onboard_detected": ch5_onboard_detected,
        "ch5_onboard_detected_monotonic": ch5_onboard_detected_monotonic,
        "ready_for_ch6": ready_for_ch6,
        "ch6_tracking_detected": ch6_tracking_detected,
        "ch6_tracking_detected_monotonic": ch6_tracking_detected_monotonic,
        "early_ch5_switch": early_ch5_switch,
        "early_ch6_switch": early_ch6_switch,
        "handover_detected": handover_detected,
        "active_hold_started": active_hold_started,
        "active_hold_duration_seconds": active_hold_duration_seconds,
        "hold_validation_passed": hold_validation_passed,
        "publish_count": adapter.actual_publish_count,
        "actual_publish_count": adapter.actual_publish_count,
        "publisher_created": publisher_created_at is not None,
        "publisher_created_after_target_latch": bool(
            publisher_created_at is not None
            and hold_latch_monotonic is not None
            and publisher_created_at >= hold_latch_monotonic
        ),
        "publisher_unregistered": publisher_unregistered,
        "maximum_position_error_m": maximum_position_error_m,
        "maximum_velocity_mps": maximum_velocity_mps,
        "maximum_publish_gap_seconds": maximum_publish_gap_seconds,
        "armed_seen": armed_seen,
        "armed_during_hold": armed_during_hold,
        "arm_lost": arm_lost,
        "px4ctrl_detected": last_consumers["px4ctrl_detected"],
        "subscriber_count": last_consumers["subscriber_count"],
        "subscriber_nodes": last_consumers["subscriber_nodes"],
        "subscriber_is_px4ctrl": bool(last_safety_status.get("subscriber_is_px4ctrl", False)),
        "last_rc_values": switches["values"],
        "last_px4_mode": (adapter.state_snapshot or {}).get("mode", "UNKNOWN"),
        "last_rejection_reasons": list(last_safety_status["rejection_reasons"]),
        "exit_reason": exit_reason,
        "safety_abort_reason": safety_abort_reason,
        "runtime_seconds": finished - started,
        "maximum_runtime_seconds": maximum_runtime,
        "last_receive_age_seconds": diagnostics["last_receive_age_seconds"],
        "first_message_received": diagnostics["first_message_received"],
        "header_stamp": diagnostics["header_stamp"],
        "last_message": last_message,
    }
    summary.update(_stale_evidence(last_safety_status))
    if error is not None:
        summary["error"] = error
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return exit_code

def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.flight_handover_version == "v1":
        if args.enable_actual_publish or args.actual_publish_confirmation:
            raise RuntimeError(LEGACY_ACTUAL_PUBLISH_ERROR)
        if not args.flight_handover_hold:
            raise RuntimeError("flight_handover_v1_requires_flight_mode")
        return run_legacy_audit(Path(args.results_dir).expanduser().resolve())
    no_consumer_intent = bool(args.bench_publish_no_consumer or args.actual_publish_confirmation == BENCH_PUBLISH_CONFIRMATION)
    px4ctrl_consumer_intent = bool(args.bench_px4ctrl_consumer or args.actual_publish_confirmation == BENCH_PX4CTRL_CONFIRMATION)
    flight_handover_intent = bool(args.flight_handover_hold or args.actual_publish_confirmation == FLIGHT_HANDOVER_HOLD_CONFIRMATION)
    if flight_handover_intent:
        if not (args.flight_handover_hold and args.enable_actual_publish and args.actual_publish_confirmation == FLIGHT_HANDOVER_HOLD_CONFIRMATION):
            raise RuntimeError("flight_handover_hold_authorization_required")
    elif px4ctrl_consumer_intent:
        if not (args.bench_px4ctrl_consumer and args.enable_actual_publish and args.actual_publish_confirmation == BENCH_PX4CTRL_CONFIRMATION):
            raise RuntimeError("px4ctrl_consumer_authorization_required")
    elif no_consumer_intent:
        if not (args.bench_publish_no_consumer and args.enable_actual_publish and args.actual_publish_confirmation == BENCH_PUBLISH_CONFIRMATION):
            raise RuntimeError("bench_publish_authorization_required")
    elif args.enable_actual_publish or args.actual_publish_confirmation:
        raise RuntimeError("actual_publish_not_implemented_or_accepted_for_alpha_v1")
    if args.flight_handover_hold:
        configured_schema = None
        if args.config:
            try:
                configured_schema = json.loads(Path(args.config).expanduser().resolve().read_text(encoding="utf-8-sig")).get("schema_version")
            except Exception:
                configured_schema = None
        use_v2 = args.flight_handover_version == "v2" or args.config is None or configured_schema == "alpha_stage4_flight_handover_v2"
        if use_v2:
            config = load_v2_config(args.config)
            session_limit = float(config["timing"]["session_hard_limit_seconds"])
            requested = float(session_limit if args.maximum_runtime_seconds is None else args.maximum_runtime_seconds)
            if requested < 1.0 or requested > session_limit:
                raise ValueError("flight_handover_runtime_exceeds_hard_limit")
            return run_v2(config, Path(args.results_dir).expanduser().resolve(), requested)
    if not (args.simulate or args.live_audit or args.bench_publish_no_consumer or args.bench_px4ctrl_consumer or args.flight_handover_hold):
        build_parser().print_help()
        return 2
    bench_mode = bool(args.bench_publish_no_consumer or args.bench_px4ctrl_consumer)
    flight_mode = bool(args.flight_handover_hold)
    config = load_alpha_config(args.config, allow_bench_publish=bench_mode, allow_flight_handover_hold=flight_mode)
    configured_duration = float(config["max_session_duration_s"])
    requested_duration = configured_duration if args.maximum_runtime_seconds is None else float(args.maximum_runtime_seconds)
    if requested_duration <= 0.0:
        raise ValueError("maximum_runtime_seconds_must_be_positive")
    if bench_mode and requested_duration > 5.0:
        raise ValueError("bench_runtime_exceeds_hard_limit")
    if flight_mode and requested_duration > 60.0:
        raise ValueError("flight_handover_runtime_exceeds_hard_limit")
    hard_limit = 60.0 if flight_mode else 5.0
    maximum_runtime = min(hard_limit, configured_duration, requested_duration)
    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.flight_handover_hold:
        return _run_flight_handover_hold(config, results_dir, maximum_runtime)
    if args.live_audit:
        return _run_live_audit(config, results_dir, maximum_runtime)
    if args.bench_publish_no_consumer:
        return _run_bench_publish(config, results_dir, maximum_runtime)
    if args.bench_px4ctrl_consumer:
        return _run_bench_publish(
            config,
            results_dir,
            maximum_runtime,
            require_px4ctrl_consumer=True,
        )

    audit_path = results_dir / "alpha_audit.jsonl"
    summary_path = results_dir / "run_summary.json"
    adapter = AlphaPx4AuditAdapter(config=config)
    started = time.monotonic()
    events = []
    exit_reason = "audit_ready_no_input"
    try:
        if args.simulate:
            events.append(_simulation_event(adapter))
            exit_reason = "offline_simulation_complete"
    except Exception as exc:
        exit_reason = "exception:{0}".format(type(exc).__name__)
        summary = {
            "platform": config["platform"],
            "backend": config["backend"],
            "audit_only": True,
            "actual_publish_enabled": False,
            "actual_publish_count": adapter.actual_publish_count,
            "maximum_runtime_seconds": maximum_runtime,
            "elapsed_seconds": time.monotonic() - started,
            "exit_reason": exit_reason,
            "error": str(exc),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        raise
    finally:
        with audit_path.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
                handle.flush()

    summary = {
        "platform": config["platform"],
        "backend": config["backend"],
        "audit_only": True,
        "actual_publish_enabled": False,
        "actual_publish_count": adapter.actual_publish_count,
        "maximum_runtime_seconds": maximum_runtime,
        "elapsed_seconds": time.monotonic() - started,
        "event_count": len(events),
        "would_publish_count": sum(
            1 for event in events if event.get("event") == "WOULD_PUBLISH"
        ),
        "exit_reason": exit_reason,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
