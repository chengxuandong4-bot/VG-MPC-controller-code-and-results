from __future__ import annotations

import contextlib
import importlib
import io
import json
import math
import sys
import tempfile
import types
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock


STAGE_DIR = Path(__file__).resolve().parents[1]
BENCH_CONFIG = STAGE_DIR / "config" / "alpha_candidate_001_bench.yaml"
FLIGHT_HOLD_CONFIG = STAGE_DIR / "config" / "alpha_candidate_001_flight_hold.yaml"
PX4CTRL_BENCH_CONFIG = STAGE_DIR / "config" / "ctrl_param_fpv_stage4_bench.yaml"
PX4CTRL_BENCH_LAUNCH = STAGE_DIR / "launch" / "alpha_px4ctrl_consumer_bench.launch"


def healthy_messages(now=100.0):
    state = {
        "connected": True,
        "armed": False,
        "guided": False,
        "mode": "POSCTL",
    }
    channels = [1500, 1500, 1500, 1500, 999, 999, 1999, 1499]
    rc = {"channels": channels}
    battery = {"voltage": 15.2, "current": 2.1, "percentage": 0.73}
    odom = {
        "position": {"x": 1.0, "y": 2.0, "z": 0.8},
        "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        "yaw": 0.25,
    }
    return state, rc, battery, odom, now


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += float(seconds)


class FakeRosEnvironment:
    def __init__(self, clock, armed=False):
        self.clock = clock
        self.armed = armed
        self.node_calls = []
        self.subscribers = []
        self.rate_hz = None
        self.rate_history = []
        self.fed = False
        self.State = type("State", (), {})
        self.RCIn = type("RCIn", (), {})
        self.BatteryState = type("BatteryState", (), {})
        self.Odometry = type("Odometry", (), {})

        self.rospy = types.ModuleType("rospy")
        self.rospy.init_node = self.init_node
        self.rospy.Subscriber = self.subscriber
        self.rospy.Rate = self.rate
        self.rospy.is_shutdown = lambda: False

        mavros = types.ModuleType("mavros_msgs")
        mavros_msg = types.ModuleType("mavros_msgs.msg")
        mavros_msg.State = self.State
        mavros_msg.RCIn = self.RCIn
        mavros.msg = mavros_msg

        sensor = types.ModuleType("sensor_msgs")
        sensor_msg = types.ModuleType("sensor_msgs.msg")
        sensor_msg.BatteryState = self.BatteryState
        sensor.msg = sensor_msg

        nav = types.ModuleType("nav_msgs")
        nav_msg = types.ModuleType("nav_msgs.msg")
        nav_msg.Odometry = self.Odometry
        nav.msg = nav_msg
        self.modules = {
            "rospy": self.rospy,
            "mavros_msgs": mavros,
            "mavros_msgs.msg": mavros_msg,
            "sensor_msgs": sensor,
            "sensor_msgs.msg": sensor_msg,
            "nav_msgs": nav,
            "nav_msgs.msg": nav_msg,
        }

    def init_node(self, name, anonymous=False):
        self.node_calls.append((name, anonymous))

    def subscriber(self, topic, message_type, callback, queue_size=10):
        record = {
            "topic": topic,
            "message_type": message_type,
            "callback": callback,
            "queue_size": queue_size,
        }
        self.subscribers.append(record)
        return record

    def rate(self, hz):
        self.rate_hz = hz
        self.rate_history.append(float(hz))
        environment = self

        class FakeRate:
            def sleep(self):
                environment.feed_callbacks_once()
                environment.clock.advance(1.0 / float(hz))

        return FakeRate()

    def feed_callbacks_once(self):
        if self.fed:
            return
        self.fed = True
        messages = {
            "/mavros/state": {
                "connected": True,
                "armed": self.armed,
                "guided": True,
                "mode": "POSCTL",
            },
            "/mavros/rc/in": {
                "channels": [1500, 1500, 1500, 1500, 999, 999, 1999, 1499]
            },
            "/mavros/battery": {
                "voltage": 15.2,
                "current": 2.1,
                "percentage": 0.73,
            },
            "/ekf/ekf_odom": {
                "position": {"x": 1.0, "y": 2.0, "z": 0.8},
                "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
                "yaw": 0.25,
            },
        }
        for subscriber in self.subscribers:
            subscriber["callback"](messages[subscriber["topic"]])

class FakePositionCommand:
    def __init__(self):
        self.header = types.SimpleNamespace(stamp=None, frame_id="")
        self.position = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.velocity = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.acceleration = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.jerk = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.yaw = 0.0
        self.yaw_dot = 0.0
        self.kx = [0.0, 0.0, 0.0]
        self.kv = [0.0, 0.0, 0.0]
        self.trajectory_id = 0
        self.trajectory_flag = 0


class FakeBenchRosEnvironment(FakeRosEnvironment):
    def __init__(
        self,
        clock,
        armed=False,
        ch7=999,
        node_names=None,
        command_subscribers=None,
        omit_topic=None,
        nonfinite_odom=False,
        after_publish=None,
        topic_arrival_times=None,
        ch7_locked_at=None,
        header_stamp=0.0,
        query_delay_after_publisher=0.0,
    ):
        super().__init__(clock, armed=armed)
        self.ch7 = ch7
        self.node_names = list(node_names or [])
        self.command_subscribers = list(command_subscribers or [])
        self.omit_topic = omit_topic
        self.nonfinite_odom = nonfinite_odom
        self.after_publish = after_publish
        self.topic_arrival_times = dict(topic_arrival_times or {})
        self.ch7_locked_at = ch7_locked_at
        self.header_stamp = header_stamp
        self.query_delay_after_publisher = float(query_delay_after_publisher)
        self.publishers = []
        self.published_messages = []
        self.publisher_unregister_count = 0
        self.rospy.Publisher = self.publisher
        self.rospy.get_name = lambda: "/stage4_alpha_bench_publish_no_consumer"
        self.rospy.Time = types.SimpleNamespace(now=lambda: self.clock())

        quadrotor = types.ModuleType("quadrotor_msgs")
        quadrotor_msg = types.ModuleType("quadrotor_msgs.msg")
        quadrotor_msg.PositionCommand = FakePositionCommand
        quadrotor.msg = quadrotor_msg

        rosnode = types.ModuleType("rosnode")
        rosnode.get_node_names = lambda: list(self.node_names)

        environment = self
        rosgraph = types.ModuleType("rosgraph")

        class FakeMaster:
            def __init__(self, caller_id):
                self.caller_id = caller_id

            def getSystemState(self):
                if environment.publishers and environment.query_delay_after_publisher:
                    environment.clock.advance(environment.query_delay_after_publisher)
                    environment.feed_callbacks_once()
                subscribers = []
                if environment.command_subscribers:
                    subscribers.append(
                        ("/setpoints_cmd", list(environment.command_subscribers))
                    )
                return ([], subscribers, [])

        rosgraph.Master = FakeMaster
        self.modules.update(
            {
                "quadrotor_msgs": quadrotor,
                "quadrotor_msgs.msg": quadrotor_msg,
                "rosnode": rosnode,
                "rosgraph": rosgraph,
            }
        )

    def publisher(self, topic, message_type, queue_size=1):
        environment = self

        class FakePublisher:
            def publish(self, message):
                environment.published_messages.append(message)
                if environment.after_publish is not None:
                    environment.after_publish(environment)

            def unregister(self):
                environment.publisher_unregister_count += 1

        record = {
            "topic": topic,
            "message_type": message_type,
            "queue_size": queue_size,
            "created_at": self.clock(),
            "instance": FakePublisher(),
        }
        self.publishers.append(record)
        return record["instance"]

    def feed_callbacks_once(self):
        ch7 = self.ch7
        if self.ch7_locked_at is not None:
            ch7 = 999 if self.clock() >= float(self.ch7_locked_at) else 1999
        header = {"stamp": self.header_stamp}
        messages = {
            "/mavros/state": {
                "header": header,
                "connected": True,
                "armed": self.armed,
                "guided": False,
                "mode": "POSCTL",
            },
            "/mavros/rc/in": {
                "header": header,
                "channels": [1500, 1500, 1500, 1500, 999, 999, ch7, 1499],
            },
            "/mavros/battery": {
                "header": header,
                "voltage": 15.2,
                "current": 2.1,
                "percentage": 0.73,
            },
            "/ekf/ekf_odom": {
                "header": header,
                "position": {
                    "x": math.nan if self.nonfinite_odom else 1.0,
                    "y": 2.0,
                    "z": 0.8,
                },
                "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
                "yaw": 0.25,
            },
        }
        for subscriber in self.subscribers:
            topic = subscriber["topic"]
            arrival = float(self.topic_arrival_times.get(topic, 0.0))
            if topic != self.omit_topic and self.clock() >= arrival:
                subscriber["callback"](messages[topic])



class FakeFlightHoldRosEnvironment(FakeBenchRosEnvironment):
    """Offline fake ROS input for flight-handover-hold state-machine tests."""

    def __init__(
        self,
        clock,
        armed=False,
        ch5=1500,
        ch6=1999,
        ch7=1999,
        node_names=None,
        command_subscribers=None,
        after_publish=None,
        on_tick=None,
    ):
        super().__init__(
            clock,
            armed=armed,
            ch7=ch7,
            node_names=["/px4ctrl"] if node_names is None else node_names,
            command_subscribers=["/px4ctrl"] if command_subscribers is None else command_subscribers,
            after_publish=after_publish,
        )
        self.ch5 = ch5
        self.ch6 = ch6
        self.ch8 = 1499
        self.odom_position = [1.0, 2.0, 0.8]
        self.odom_yaw = 0.25
        self.connected = True
        self.omit_topic = None
        self.on_tick = on_tick
        self.rospy.get_name = lambda: "/stage4_alpha_flight_handover_hold"

    def feed_callbacks_once(self):
        if self.on_tick is not None:
            self.on_tick(self)
        header = {"stamp": self.header_stamp}
        messages = {
            "/mavros/state": {"header": header, "connected": self.connected, "armed": self.armed, "guided": False, "mode": "POSCTL"},
            "/mavros/rc/in": {"header": header, "channels": [1500, 1500, 1500, 1500, self.ch5, self.ch6, self.ch7, self.ch8]},
            "/mavros/battery": {"header": header, "voltage": 15.2, "current": 2.1, "percentage": 0.73},
            "/ekf/ekf_odom": {"header": header, "position": {"x": math.nan if self.nonfinite_odom else self.odom_position[0], "y": self.odom_position[1], "z": self.odom_position[2]}, "velocity": {"x": 0.0, "y": 0.0, "z": 0.0}, "yaw": self.odom_yaw},
        }
        for subscriber in self.subscribers:
            if subscriber["topic"] != self.omit_topic:
                            subscriber["callback"](messages[subscriber["topic"]])

class AlphaAuditTests(unittest.TestCase):
    def setUp(self):
        try:
            self.alpha = importlib.import_module("rpi_pixhawk_stage4.alpha_px4_adapter")
        except ModuleNotFoundError as exc:
            self.fail("Alpha audit adapter is not implemented: {0}".format(exc))

    def make_adapter(self, now=100.0):
        adapter = self.alpha.AlphaPx4AuditAdapter(clock=lambda: now)
        state, rc, battery, odom, stamp = healthy_messages(now)
        adapter.update_state(state, stamp)
        adapter.update_rc(rc, stamp)
        adapter.update_battery(battery, stamp)
        adapter.update_odom(odom, stamp)
        return adapter

    def run_bench(self, ros, runtime=0.3, consumer=False):
        runner = importlib.import_module("rpi_pixhawk_stage4.run_stage4_alpha")
        mode_flag = (
            "--bench-px4ctrl-consumer"
            if consumer
            else "--bench-publish-no-consumer"
        )
        confirmation = (
            "ALPHA_BENCH_PX4CTRL_CONSUMER"
            if consumer
            else "ALPHA_BENCH_NO_CONSUMER"
        )
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(sys.modules, ros.modules, clear=False):
                with mock.patch.object(runner.time, "monotonic", ros.clock):
                    with contextlib.redirect_stdout(io.StringIO()):
                        code = runner.main(
                            [

                                mode_flag,
                                "--enable-actual-publish",
                                "--actual-publish-confirmation",
                                confirmation,
                                "--config",
                                str(BENCH_CONFIG),
                                "--maximum-runtime-seconds",
                                str(runtime),
                                "--results-dir",
                                directory,
                            ]
                        )
            summary = json.loads(
                (Path(directory) / "run_summary.json").read_text(encoding="utf-8")
            )
            events = [
                json.loads(line)
                for line in (Path(directory) / "alpha_bench_publish.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
        return code, summary, events

    def run_flight_hold(self, ros, runtime=0.3):
        runner = importlib.import_module("rpi_pixhawk_stage4.run_stage4_alpha")
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(sys.modules, ros.modules, clear=False):
                with mock.patch.object(runner.time, "monotonic", ros.clock):
                    with contextlib.redirect_stdout(io.StringIO()):
                        code = runner.main(
                            [
                                "--flight-handover-hold",
                                "--enable-actual-publish",
                                "--actual-publish-confirmation",
                                "ALPHA_FLIGHT_HANDOVER_HOLD",
                                "--config", str(FLIGHT_HOLD_CONFIG),
                                "--maximum-runtime-seconds", str(runtime),
                                "--results-dir", directory,
                            ]
                        )
            summary = json.loads(
                (Path(directory) / "run_summary.json").read_text(encoding="utf-8")
            )
            events = [
                json.loads(line)
                for line in (Path(directory) / "alpha_flight_handover_hold.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
        return code, summary, events

    def test_flight_handover_hold_requires_exact_authorization(self):
        runner = importlib.import_module("rpi_pixhawk_stage4.run_stage4_alpha")
        for arguments in (
            ["--flight-handover-hold", "--config", str(FLIGHT_HOLD_CONFIG)],
            ["--flight-handover-hold", "--enable-actual-publish", "--config", str(FLIGHT_HOLD_CONFIG)],
            ["--enable-actual-publish", "--actual-publish-confirmation", "ALPHA_FLIGHT_HANDOVER_HOLD", "--config", str(FLIGHT_HOLD_CONFIG)],
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(RuntimeError, "flight_handover_hold_authorization_required"):
                    runner.main(arguments)

    def _schedule_handover(self, ros, validation=False):
        def advance(environment):
            count = len(environment.published_messages)
            if count >= 20:
                environment.ch5 = 999
            if count >= 22:
                environment.ch6 = 999
            if validation and count >= 30:
                environment.odom_position[0] = 1.03
            exit_count = 78 if validation else 30
            if count >= exit_count:
                environment.ch6 = 1999
            if count >= exit_count + 2:
                environment.ch5 = 1500
        ros.after_publish = advance

    def test_flight_handover_prestreams_before_ch5_or_ch6_handover(self):
        clock = FakeClock()
        ros = FakeFlightHoldRosEnvironment(
            clock, armed=True, ch5=1500, ch6=1999, ch7=1999
        )
        code, summary, events = self.run_flight_hold(ros, runtime=2.5)
        self.assertEqual(code, 0)
        self.assertTrue(summary["hold_target_latched"])
        self.assertTrue(summary["prestream_started"])
        self.assertGreaterEqual(summary["prestream_publish_count"], 20)
        self.assertGreaterEqual(summary["prestream_duration_seconds"], 2.0)
        self.assertEqual(summary["exit_reason"], "maximum_runtime")
        self.assertIsNone(summary["safety_abort_reason"])
        self.assertGreaterEqual(len(ros.published_messages), 20)
        self.assertTrue(all(event["state"] == "PRESTREAM_HOLD" for event in events[:20]))

    def test_flight_handover_wait_system_and_wait_arm_do_not_publish(self):
        cases = (
            (FakeFlightHoldRosEnvironment(FakeClock(), node_names=[]), "WAIT_SYSTEM"),
            (FakeFlightHoldRosEnvironment(FakeClock(), command_subscribers=[]), "WAIT_SYSTEM"),
            (FakeFlightHoldRosEnvironment(FakeClock(), ch5=999), "WAIT_SYSTEM"),
            (FakeFlightHoldRosEnvironment(FakeClock(), ch6=999), "WAIT_SYSTEM"),
            (FakeFlightHoldRosEnvironment(FakeClock(), armed=False), "WAIT_ARM"),
            (FakeFlightHoldRosEnvironment(FakeClock(), armed=True, ch7=999), "WAIT_ARM"),
        )
        for ros, expected_state in cases:
            with self.subTest(expected_state=expected_state):
                code, summary, events = self.run_flight_hold(ros, runtime=0.5)
                self.assertEqual(code, 0)
                self.assertEqual(summary["actual_publish_count"], 0)
                self.assertEqual(ros.publishers, [])
                self.assertEqual(events, [])
                self.assertIn(expected_state, [entry["state"] for entry in summary["state_trace"]])

    def test_flight_handover_latches_before_publisher_and_target_is_immutable(self):
        clock = FakeClock()
        ros = FakeFlightHoldRosEnvironment(clock, armed=True)

        def advance(environment):
            count = len(environment.published_messages)
            if count == 1:
                environment.odom_position = [9.0, 8.0, 7.0]
                environment.odom_yaw = -0.5
            if count >= 20:
                environment.ch5 = 999
            if count >= 22:
                environment.ch6 = 999
            if count >= 30:
                environment.ch6 = 1999
            if count >= 32:
                environment.ch5 = 1500
        ros.after_publish = advance
        code, summary, events = self.run_flight_hold(ros, runtime=6.0)
        self.assertEqual(code, 0)
        self.assertTrue(summary["publisher_created_after_target_latch"])
        self.assertEqual(summary["hold_position"], {"x": 1.0, "y": 2.0, "z": 0.8})
        self.assertEqual(summary["hold_yaw"], 0.25)
        self.assertGreater(len(events), 25)
        for message in ros.published_messages:
            self.assertEqual((message.position.x, message.position.y, message.position.z), (1.0, 2.0, 0.8))
            self.assertEqual(message.yaw, 0.25)
            self.assertEqual(message.yaw_dot, 0.0)
            for vector in (message.velocity, message.acceleration, message.jerk):
                self.assertEqual((vector.x, vector.y, vector.z), (0.0, 0.0, 0.0))
        identifiers = [message.trajectory_id for message in ros.published_messages]
        self.assertEqual(identifiers, list(range(identifiers[0], identifiers[0] + len(identifiers))))

    def test_flight_handover_early_switches_keep_stream_and_do_not_bypass_prestream(self):
        clock = FakeClock()
        ros = FakeFlightHoldRosEnvironment(clock, armed=True)

        def switch_early(environment):
            if len(environment.published_messages) == 1:
                environment.ch5 = 999
                environment.ch6 = 999
        ros.after_publish = switch_early
        code, summary, events = self.run_flight_hold(ros, runtime=3.2)
        self.assertEqual(code, 0)
        self.assertTrue(summary["early_ch5_switch"])
        self.assertTrue(summary["early_ch6_switch"])
        self.assertGreaterEqual(summary["prestream_publish_count"], 20)
        self.assertGreaterEqual(summary["prestream_duration_seconds"], 2.0)
        first_active = next(index for index, event in enumerate(events) if event["state"] == "ACTIVE_HOLD")
        self.assertGreaterEqual(first_active, 20)
        self.assertTrue(summary["handover_detected"])

    def test_flight_handover_full_state_sequence_and_normal_return(self):
        clock = FakeClock()
        ros = FakeFlightHoldRosEnvironment(clock, armed=True)
        self._schedule_handover(ros, validation=True)
        code, summary, events = self.run_flight_hold(ros, runtime=12.0)
        self.assertEqual(code, 0)
        states = [entry["state"] for entry in summary["state_trace"]]
        for required in (
            "WAIT_SYSTEM", "WAIT_ARM", "PRESTREAM_HOLD", "READY_FOR_CH5",
            "READY_FOR_CH6", "ACTIVE_HOLD", "RETURN_TO_MANUAL", "EXIT",
        ):
            self.assertIn(required, states)
        self.assertEqual(summary["exit_reason"], "operator_handover_complete")
        self.assertIsNone(summary["safety_abort_reason"])
        self.assertTrue(summary["hold_validation_passed"])
        self.assertTrue(summary["publisher_unregistered"])
        self.assertEqual(ros.publisher_unregister_count, 1)
        self.assertTrue(any(event["state"] == "RETURN_TO_MANUAL" for event in events))
        self.assertGreater(summary["maximum_position_error_m"], 0.0)

    def test_flight_handover_arm_loss_and_runtime_faults_fail_closed(self):
        cases = (
            (lambda env: setattr(env, "armed", False), "arm_lost_before_handover"),
            (lambda env: setattr(env, "ch7", 999), "ch7_locked"),
            (lambda env: setattr(env, "connected", False), "mavros_disconnected"),
            (lambda env: setattr(env, "node_names", []), "px4ctrl_missing"),
            (lambda env: setattr(env, "command_subscribers", ["/other"]), "subscriber_invalid"),
            (lambda env: setattr(env, "nonfinite_odom", True), "odom_invalid"),
        )
        for update, expected in cases:
            clock = FakeClock()
            ros = FakeFlightHoldRosEnvironment(clock, armed=True)
            def fault(environment, change=update):
                if len(environment.published_messages) == 1:
                    change(environment)
                    environment.after_publish = None
            ros.after_publish = fault
            with self.subTest(expected=expected):
                code, summary, _ = self.run_flight_hold(ros, runtime=2.0)
                self.assertNotEqual(code, 0)
                self.assertEqual(summary["exit_reason"], expected)
                self.assertEqual(summary["safety_abort_reason"], expected)
                self.assertTrue(summary["publisher_unregistered"])
                self.assertEqual(ros.publisher_unregister_count, 1)

    def test_flight_handover_active_arm_loss_and_ch7_lock_stop_immediately(self):
        for field, value, expected in (
            ("armed", False, "armed_lost_during_hold"),
            ("ch7", 999, "ch7_locked"),
        ):
            clock = FakeClock()
            ros = FakeFlightHoldRosEnvironment(clock, armed=True)
            def advance(environment, name=field, updated=value):
                count = len(environment.published_messages)
                if count >= 1:
                    environment.ch5 = 999
                    environment.ch6 = 999
                if count >= 25:
                    setattr(environment, name, updated)
                    environment.after_publish = None
            ros.after_publish = advance
            with self.subTest(expected=expected):
                code, summary, _ = self.run_flight_hold(ros, runtime=5.0)
                self.assertNotEqual(code, 0)
                self.assertTrue(summary["active_hold_started"])
                self.assertEqual(summary["exit_reason"], expected)
                self.assertEqual(ros.publisher_unregister_count, 1)

    def test_flight_handover_all_telemetry_stale_paths_stop(self):
        for topic, expected in (
            ("/mavros/state", "telemetry_stale"),
            ("/mavros/rc/in", "telemetry_stale"),
            ("/mavros/battery", "telemetry_stale"),
            ("/ekf/ekf_odom", "telemetry_stale"),
        ):
            clock = FakeClock()
            ros = FakeFlightHoldRosEnvironment(clock, armed=True)
            def omit(environment, selected=topic):
                if len(environment.published_messages) == 1:
                    environment.omit_topic = selected
                    environment.after_publish = None
            ros.after_publish = omit
            with self.subTest(topic=topic):
                code, summary, _ = self.run_flight_hold(ros, runtime=5.0)
                self.assertNotEqual(code, 0)
                self.assertEqual(summary["exit_reason"], expected)
                self.assertIsNotNone(summary["stale_channel"])

    def test_flight_handover_return_to_manual_keeps_stream_and_times_out_safely(self):
        clock = FakeClock()
        ros = FakeFlightHoldRosEnvironment(clock, armed=True)
        def advance(environment):
            count = len(environment.published_messages)
            if count >= 1:
                environment.ch5 = 999
                environment.ch6 = 999
            if count >= 25:
                environment.ch6 = 1999
        ros.after_publish = advance
        code, summary, events = self.run_flight_hold(ros, runtime=10.0)
        self.assertNotEqual(code, 0)
        return_events = [event for event in events if event["state"] == "RETURN_TO_MANUAL"]
        self.assertGreater(len(return_events), 1)
        self.assertEqual(summary["exit_reason"], "return_to_manual_timeout")
        self.assertEqual(summary["safety_abort_reason"], "return_to_manual_timeout")
        self.assertTrue(summary["publisher_unregistered"])

    def test_flight_handover_maximum_runtime_and_ch8_are_not_safety_aborts(self):
        clock = FakeClock()
        ros = FakeFlightHoldRosEnvironment(clock, armed=True)
        def change_ch8(environment):
            environment.ch8 = 999 if len(environment.published_messages) % 2 else 1999
        ros.after_publish = change_ch8
        code, summary, _ = self.run_flight_hold(ros, runtime=2.6)
        self.assertEqual(code, 0)
        self.assertEqual(summary["exit_reason"], "maximum_runtime")
        self.assertIsNone(summary["safety_abort_reason"])
        self.assertGreater(summary["actual_publish_count"], 0)

    def test_flight_handover_waits_for_arm_and_uses_monotonic_header_independent_time(self):
        clock = FakeClock()
        def arm_later(environment):
            if environment.clock() >= 0.3:
                environment.armed = True
        ros = FakeFlightHoldRosEnvironment(
            clock, armed=False, on_tick=arm_later
        )
        ros.header_stamp = 32503680000.0
        code, summary, _ = self.run_flight_hold(ros, runtime=2.8)
        self.assertEqual(code, 0)
        self.assertTrue(summary["hold_target_latched"])
        self.assertGreater(summary["hold_latch_monotonic"], 0.0)
        self.assertGreater(summary["actual_publish_count"], 0)

    def test_flight_handover_config_rejects_invalid_values_and_has_safe_defaults(self):
        base = json.loads(FLIGHT_HOLD_CONFIG.read_text(encoding="utf-8"))
        without_section = dict(base)
        without_section.pop("flight_handover")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "defaults.json"
            path.write_text(json.dumps(without_section), encoding="utf-8")
            loaded = self.alpha.load_alpha_config(
                str(path), allow_flight_handover_hold=True
            )
            self.assertEqual(loaded["flight_handover"]["prestream_min_messages"], 20)
            self.assertEqual(loaded["flight_handover"]["prestream_min_duration_s"], 2.0)

            invalid_cases = (
                ("prestream_min_messages", 0),
                ("prestream_min_messages", True),
                ("prestream_rate_hz", float("nan")),
                ("prestream_min_duration_s", -1.0),
                ("status_print_interval_s", float("inf")),
                ("return_to_manual_timeout_s", 0.0),
            )
            for name, value in invalid_cases:
                bad = json.loads(json.dumps(base))
                bad["flight_handover"][name] = value
                path.write_text(json.dumps(bad), encoding="utf-8")
                with self.subTest(name=name, value=value):
                    with self.assertRaises(self.alpha.AlphaConfigError):
                        self.alpha.load_alpha_config(
                            str(path), allow_flight_handover_hold=True
                        )
    def test_flight_handover_config_validation_and_static_boundaries(self):
        config = self.alpha.load_alpha_config(
            str(FLIGHT_HOLD_CONFIG), allow_flight_handover_hold=True
        )
        handover = config["flight_handover"]
        self.assertEqual(handover["prestream_rate_hz"], 10.0)
        self.assertEqual(handover["prestream_min_messages"], 20)
        self.assertEqual(handover["prestream_min_duration_s"], 2.0)
        self.assertEqual(handover["status_print_interval_s"], 0.5)
        self.assertEqual(handover["return_to_manual_timeout_s"], 5.0)
        source = (STAGE_DIR / "run_stage4_alpha.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("rospy.Publisher("), 1)
        self.assertEqual(source.count("publisher.publish("), 1)
        for token in (
            "ServiceProxy(", "CommandBool(", "SetMode(", "param/set",
            "COMMAND_LONG", "DO_SET_MODE", "NAV_TAKEOFF", "NAV_LAND",
        ):
            self.assertNotIn(token, source)
    def test_defaults_are_audit_only_and_ros_import_is_lazy(self):
        sys.modules.pop("rospy", None)
        module = importlib.reload(self.alpha)
        config = module.load_alpha_config()
        self.assertTrue(config["audit_only"])
        self.assertFalse(config["actual_publish_enabled"])
        self.assertEqual(config["max_session_duration_s"], 5.0)
        self.assertNotIn("rospy", sys.modules)

    def test_mavros_disconnect_is_rejected(self):
        adapter = self.make_adapter()
        adapter.update_state(
            {"connected": False, "armed": False, "guided": True, "mode": "OFFBOARD"},
            100.0,
        )
        result = adapter.evaluate({}, now=100.1)
        self.assertIn("MAVROS_DISCONNECTED", result["rejection_reasons"])

    def test_rc_missing_stale_and_short_are_rejected(self):
        for case, expected in (
            ("missing", "RC_MISSING"),
            ("stale", "RC_STALE"),
            ("short", "RC_CHANNELS_INSUFFICIENT"),
        ):
            with self.subTest(case=case):
                adapter = self.make_adapter()
                if case == "missing":
                    adapter.rc_snapshot = None
                elif case == "stale":
                    adapter.rc_received_at = 99.0
                else:
                    adapter.update_rc({"channels": [999] * 7}, 100.0)
                result = adapter.evaluate({}, now=100.1)
                self.assertIn(expected, result["rejection_reasons"])

    def test_odom_missing_and_stale_are_rejected(self):
        for case, expected in (("missing", "ODOM_MISSING"), ("stale", "ODOM_STALE")):
            with self.subTest(case=case):
                adapter = self.make_adapter()
                if case == "missing":
                    adapter.odom_snapshot = None
                else:
                    adapter.odom_received_at = 99.0
                result = adapter.evaluate({}, now=100.1)
                self.assertIn(expected, result["rejection_reasons"])

    def test_battery_missing_and_stale_are_rejected(self):
        for case, expected in (
            ("missing", "BATTERY_MISSING"),
            ("stale", "BATTERY_STALE"),
        ):
            with self.subTest(case=case):
                adapter = self.make_adapter()
                if case == "missing":
                    adapter.battery_snapshot = None
                else:
                    adapter.battery_received_at = 98.0
                result = adapter.evaluate({}, now=100.1)
                self.assertIn(expected, result["rejection_reasons"])

    def test_ch5_must_select_onboard(self):
        adapter = self.make_adapter()
        channels = list(adapter.rc_snapshot["channels"])
        channels[4] = 1499
        adapter.update_rc({"channels": channels}, 100.0)
        self.assertIn("CH5_NOT_ONBOARD", adapter.evaluate({}, now=100.1)["rejection_reasons"])

    def test_ch6_must_allow_tracking(self):
        adapter = self.make_adapter()
        channels = list(adapter.rc_snapshot["channels"])
        channels[5] = 1999
        adapter.update_rc({"channels": channels}, 100.0)
        self.assertIn(
            "CH6_TRACKING_NOT_ALLOWED",
            adapter.evaluate({}, now=100.1)["rejection_reasons"],
        )

    def test_ch7_locked_position_is_rejected(self):
        adapter = self.make_adapter()
        channels = list(adapter.rc_snapshot["channels"])
        channels[6] = 999
        adapter.update_rc({"channels": channels}, 100.0)
        self.assertIn("CH7_LOCKED", adapter.evaluate({}, now=100.1)["rejection_reasons"])

    def test_armed_vehicle_is_rejected_during_audit(self):
        adapter = self.make_adapter()
        adapter.update_state(
            {"connected": True, "armed": True, "guided": False, "mode": "POSCTL"},
            100.0,
        )
        self.assertIn(
            "ARMED_DURING_AUDIT",
            adapter.evaluate({}, now=100.1)["rejection_reasons"],
        )

    def test_nonfinite_telemetry_and_command_are_rejected(self):
        adapter = self.make_adapter()
        adapter.update_odom(
            {
                "position": {"x": math.nan, "y": 0.0, "z": 0.8},
                "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
                "yaw": 0.0,
            },
            100.0,
        )
        self.assertIn(
            "NONFINITE_TELEMETRY",
            adapter.evaluate({}, now=100.1)["rejection_reasons"],
        )
        adapter = self.make_adapter()
        result = adapter.evaluate(
            {
                "vx_north_mps": math.inf,
                "vy_east_mps": 0.0,
                "vz_down_mps": 0.0,
                "yaw_rate_radps": 0.0,
            },
            now=100.1,
        )
        self.assertIn("NONFINITE_COMMAND", result["rejection_reasons"])

    def test_limits_and_world_z_up_mapping_are_applied(self):
        adapter = self.make_adapter()
        result = adapter.evaluate(
            {
                "vx_north_mps": 1.0,
                "vy_east_mps": 1.0,
                "vz_down_mps": 1.0,
                "yaw_rate_radps": 1.0,
            },
            now=100.1,
        )
        command = result["candidate_command"]
        self.assertLessEqual(
            math.hypot(command["velocity"]["x"], command["velocity"]["y"]),
            0.10,
        )
        self.assertEqual(command["velocity"]["z"], -0.10)
        self.assertEqual(command["yaw_dot"], 0.08)
        self.assertTrue(result["limit_result"]["saturated"])

    def test_healthy_gate_generates_complete_would_publish_without_publish(self):
        adapter = self.make_adapter()
        result = adapter.evaluate(
            {
                "vx_north_mps": 0.02,
                "vy_east_mps": -0.01,
                "vz_down_mps": 0.03,
                "yaw_rate_radps": 0.02,
            },
            now=100.1,
        )
        self.assertTrue(result["gate_authorized"])
        self.assertEqual(result["event"], "WOULD_PUBLISH")
        self.assertEqual(result["rejection_reasons"], [])
        self.assertEqual(adapter.actual_publish_count, 0)
        self.assertEqual(result["actual_publish_count"], 0)
        command = result["candidate_command"]
        for field in (
            "header", "position", "velocity", "acceleration", "jerk",
            "yaw", "yaw_dot", "kx", "kv", "trajectory_id", "trajectory_flag",
        ):
            self.assertIn(field, command)
        self.assertEqual(command["kx"], [0.0, 0.0, 0.0])
        self.assertEqual(command["kv"], [0.0, 0.0, 0.0])
        self.assertEqual(command["trajectory_flag"], 1)
        self.assertEqual(result["mode"], "POSCTL")
        self.assertFalse(result["guided_recorded_only"])

    def test_trajectory_id_is_positive_and_monotonic(self):
        adapter = self.make_adapter()
        first = adapter.evaluate({}, now=100.1)["candidate_command"]["trajectory_id"]
        second = adapter.evaluate({}, now=100.2)["candidate_command"]["trajectory_id"]
        self.assertGreater(first, 0)
        self.assertEqual(second, first + 1)

    def test_simulate_and_live_audit_are_mutually_exclusive(self):
        runner = importlib.import_module("rpi_pixhawk_stage4.run_stage4_alpha")
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            with self.assertRaises(SystemExit) as caught:
                runner.main(["--simulate", "--live-audit"])
        self.assertNotEqual(caught.exception.code, 0)
        self.assertIn("not allowed with argument", stream.getvalue())

    def test_no_mode_prints_help_and_returns_nonzero(self):
        runner = importlib.import_module("rpi_pixhawk_stage4.run_stage4_alpha")
        stream = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stdout(stream):
                code = runner.main(["--results-dir", directory])
        self.assertNotEqual(code, 0)
        self.assertIn("--live-audit", stream.getvalue())
        self.assertIn("--simulate", stream.getvalue())

    def test_live_audit_requires_ros_noetic_environment(self):
        runner = importlib.import_module("rpi_pixhawk_stage4.run_stage4_alpha")
        original_import = __import__

        def reject_ros_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "rospy":
                raise ModuleNotFoundError("No module named 'rospy'")
            return original_import(name, globals, locals, fromlist, level)

        sys.modules.pop("rospy", None)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("builtins.__import__", side_effect=reject_ros_import):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "live_audit_requires_ros_noetic_environment",
                ):
                    runner.main(
                        [
                            "--live-audit",
                            "--maximum-runtime-seconds",
                            "0.1",
                            "--results-dir",
                            directory,
                        ]
                    )

    def test_live_audit_registers_four_subscribers_and_exits_at_runtime(self):
        runner = importlib.import_module("rpi_pixhawk_stage4.run_stage4_alpha")
        clock = FakeClock()
        ros = FakeRosEnvironment(clock, armed=False)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(sys.modules, ros.modules, clear=False):
                with mock.patch.object(runner.time, "monotonic", clock):
                    code = runner.main(
                        [
                            "--live-audit",
                            "--maximum-runtime-seconds",
                            "0.3",
                            "--results-dir",
                            directory,
                        ]
                    )
            summary = json.loads(
                (Path(directory) / "run_summary.json").read_text(encoding="utf-8")
            )
            events = [
                json.loads(line)
                for line in (Path(directory) / "alpha_audit.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]

        self.assertEqual(code, 0)
        self.assertEqual(ros.node_calls, [("stage4_alpha_live_audit", False)])
        self.assertEqual(ros.rate_hz, 10.0)
        self.assertEqual(len(ros.subscribers), 4)
        self.assertEqual(
            [item["topic"] for item in ros.subscribers],
            ["/mavros/state", "/mavros/rc/in", "/mavros/battery", "/ekf/ekf_odom"],
        )
        self.assertEqual(
            [item["message_type"] for item in ros.subscribers],
            [ros.State, ros.RCIn, ros.BatteryState, ros.Odometry],
        )
        self.assertTrue(any(item["event"] == "WOULD_PUBLISH" for item in events))
        self.assertEqual(summary["mode"], "live_audit")
        self.assertEqual(summary["exit_reason"], "maximum_runtime")
        self.assertGreater(summary["would_publish_count"], 0)
        self.assertEqual(summary["actual_publish_count"], 0)
        self.assertTrue(summary["last_gate_allowed"])
        self.assertEqual(summary["last_reject_reasons"], [])
        self.assertEqual(summary["last_rc_values"]["CH5"], 999.0)
        self.assertEqual(summary["last_px4_mode"], "POSCTL")
        self.assertTrue(
            all(item["received"] for item in summary["message_status"].values())
        )
        self.assertTrue(
            all(item["fresh"] for item in summary["message_status"].values())
        )

    def test_live_audit_armed_state_is_rejected_with_zero_publish(self):
        runner = importlib.import_module("rpi_pixhawk_stage4.run_stage4_alpha")
        clock = FakeClock()
        ros = FakeRosEnvironment(clock, armed=True)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(sys.modules, ros.modules, clear=False):
                with mock.patch.object(runner.time, "monotonic", clock):
                    code = runner.main(
                        [
                            "--live-audit",
                            "--maximum-runtime-seconds",
                            "0.3",
                            "--results-dir",
                            directory,
                        ]
                    )
            summary = json.loads(
                (Path(directory) / "run_summary.json").read_text(encoding="utf-8")
            )
        self.assertEqual(code, 0)
        self.assertEqual(summary["would_publish_count"], 0)
        self.assertEqual(summary["actual_publish_count"], 0)
        self.assertFalse(summary["last_gate_allowed"])
        self.assertIn("ARMED_DURING_AUDIT", summary["last_reject_reasons"])
    def test_default_modes_never_create_publisher(self):
        runner = importlib.import_module("rpi_pixhawk_stage4.run_stage4_alpha")
        clock = FakeClock()
        ros = FakeBenchRosEnvironment(clock)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(sys.modules, ros.modules, clear=False):
                with mock.patch.object(runner.time, "monotonic", clock):
                    code = runner.main(
                        [
                            "--live-audit",
                            "--maximum-runtime-seconds",
                            "0.2",
                            "--results-dir",
                            directory,
                        ]
                    )
            summary = json.loads(
                (Path(directory) / "run_summary.json").read_text(encoding="utf-8")
            )
        self.assertEqual(code, 0)
        self.assertEqual(ros.publishers, [])
        self.assertEqual(summary["actual_publish_count"], 0)

    def test_px4ctrl_consumer_cli_requires_all_three_danger_switches(self):
        runner = importlib.import_module("rpi_pixhawk_stage4.run_stage4_alpha")
        cases = (
            ["--bench-px4ctrl-consumer", "--config", str(BENCH_CONFIG)],
            [
                "--bench-px4ctrl-consumer",
                "--enable-actual-publish",
                "--config",
                str(BENCH_CONFIG),
            ],
            [
                "--enable-actual-publish",
                "--actual-publish-confirmation",
                "ALPHA_BENCH_PX4CTRL_CONSUMER",
                "--config",
                str(BENCH_CONFIG),
            ],
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(
                    RuntimeError, "px4ctrl_consumer_authorization_required"
                ):
                    runner.main(arguments)

    def test_px4ctrl_consumer_requires_exact_single_px4ctrl_subscriber(self):
        cases = (
            (
                FakeBenchRosEnvironment(
                    FakeClock(), command_subscribers=["/px4ctrl"]
                ),
                "PX4CTRL_MISSING",
            ),
            (
                FakeBenchRosEnvironment(FakeClock(), node_names=["/px4ctrl"]),
                "COMMAND_TOPIC_SUBSCRIBER_COUNT_NOT_ONE",
            ),
            (
                FakeBenchRosEnvironment(
                    FakeClock(),
                    node_names=["/px4ctrl"],
                    command_subscribers=["/px4ctrl", "/other"],
                ),
                "COMMAND_TOPIC_SUBSCRIBER_COUNT_NOT_ONE",
            ),
            (
                FakeBenchRosEnvironment(
                    FakeClock(),
                    node_names=["/px4ctrl"],
                    command_subscribers=["/other"],
                ),
                "COMMAND_TOPIC_SUBSCRIBER_NOT_PX4CTRL",
            ),
        )
        for ros, expected in cases:
            with self.subTest(expected=expected):
                code, summary, events = self.run_bench(
                    ros, runtime=0.2, consumer=True
                )
                self.assertNotEqual(code, 0)
                self.assertEqual(summary["exit_reason"], "startup_timeout")
                self.assertIn(expected, summary["startup_unmet_conditions"])
                self.assertEqual(summary["actual_publish_count"], 0)
                self.assertEqual(events, [])

    def test_px4ctrl_consumer_publishes_hold_only_with_exact_consumer(self):
        ros = FakeBenchRosEnvironment(
            FakeClock(),
            node_names=["/alpha/px4ctrl"],
            command_subscribers=["/alpha/px4ctrl"],
        )
        code, summary, events = self.run_bench(ros, runtime=0.3, consumer=True)
        self.assertEqual(code, 0)
        self.assertEqual(summary["mode"], "bench_px4ctrl_consumer")
        self.assertTrue(summary["px4ctrl_detected"])
        self.assertEqual(summary["subscriber_count"], 1)
        self.assertTrue(summary["subscriber_is_px4ctrl"])
        self.assertGreater(summary["actual_publish_count"], 0)
        self.assertGreater(len(events), 0)
        message = ros.published_messages[0]
        self.assertEqual(
            (message.position.x, message.position.y, message.position.z),
            (1.0, 2.0, 0.8),
        )
        for vector in (message.velocity, message.acceleration, message.jerk):
            self.assertEqual((vector.x, vector.y, vector.z), (0.0, 0.0, 0.0))
        self.assertEqual(message.yaw, 0.25)
        self.assertEqual(message.yaw_dot, 0.0)
        self.assertEqual(message.trajectory_flag, 1)

    def test_px4ctrl_consumer_stops_if_node_or_subscriber_disappears(self):
        def remove_px4ctrl(environment):
            environment.node_names = []
            environment.after_publish = None

        def remove_subscriber(environment):
            environment.command_subscribers = []
            environment.after_publish = None

        for callback, expected in (
            (remove_px4ctrl, "px4ctrl_missing"),
            (remove_subscriber, "command_topic_subscriber_count_not_one"),
        ):
            ros = FakeBenchRosEnvironment(
                FakeClock(),
                node_names=["/px4ctrl"],
                command_subscribers=["/px4ctrl"],
                after_publish=callback,
            )
            with self.subTest(expected=expected):
                code, summary, events = self.run_bench(
                    ros, runtime=0.5, consumer=True
                )
                self.assertNotEqual(code, 0)
                self.assertEqual(summary["safety_abort_reason"], expected)
                self.assertEqual(summary["actual_publish_count"], 1)
                self.assertEqual(len(events), 1)

    def test_px4ctrl_consumer_safety_overlay_and_single_node_launch(self):
        overlay = json.loads(PX4CTRL_BENCH_CONFIG.read_text(encoding="utf-8"))
        self.assertFalse(overlay["auto_takeoff_land"]["enable"])
        self.assertFalse(overlay["auto_takeoff_land"]["enable_auto_arm"])

        root = ET.fromstring(PX4CTRL_BENCH_LAUNCH.read_text(encoding="utf-8"))
        nodes = root.findall("node")
        self.assertEqual(len(nodes), 1)
        node = nodes[0]
        self.assertEqual(node.attrib["pkg"], "px4ctrl")
        self.assertEqual(node.attrib["type"], "px4ctrl_node")
        self.assertIn("px4ctrl", node.attrib["name"].lower())
        remaps = {
            item.attrib["from"]: item.attrib["to"]
            for item in node.findall("remap")
        }
        self.assertEqual(remaps["~odom"], "/ekf/ekf_odom")
        self.assertEqual(remaps["~cmd"], "/setpoints_cmd")
        self.assertEqual(remaps["~state"], "/mavros/state")
        self.assertEqual(remaps["~bat"], "/mavros/battery")
        self.assertEqual(remaps["~imu"], "/mavros/imu/data")
        self.assertEqual(remaps["~ctrl"], "/mavros/setpoint_raw/attitude")
        parameter_files = [item.attrib["file"] for item in node.findall("rosparam")]
        self.assertGreaterEqual(len(parameter_files), 2)
        self.assertIn("ctrl_param_fpv.yaml", parameter_files[0])
        self.assertIn("ctrl_param_fpv_stage4_bench.yaml", parameter_files[-1])
    def test_bench_requires_mode_enable_and_exact_confirmation(self):
        runner = importlib.import_module("rpi_pixhawk_stage4.run_stage4_alpha")
        cases = (
            ["--bench-publish-no-consumer", "--config", str(BENCH_CONFIG)],
            [
                "--bench-publish-no-consumer",
                "--enable-actual-publish",
                "--config",
                str(BENCH_CONFIG),
            ],
            [
                "--enable-actual-publish",
                "--actual-publish-confirmation",
                "ALPHA_BENCH_NO_CONSUMER",
                "--config",
                str(BENCH_CONFIG),
            ],
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(
                    RuntimeError, "bench_publish_authorization_required"
                ):
                    runner.main(arguments)

    def test_bench_rejects_runtime_above_hard_limit(self):
        runner = importlib.import_module("rpi_pixhawk_stage4.run_stage4_alpha")
        with self.assertRaisesRegex(ValueError, "bench_runtime_exceeds_hard_limit"):
            runner.main(
                [
                    "--bench-publish-no-consumer",
                    "--enable-actual-publish",
                    "--actual-publish-confirmation",
                    "ALPHA_BENCH_NO_CONSUMER",
                    "--config",
                    str(BENCH_CONFIG),
                    "--maximum-runtime-seconds",
                    "5.1",
                ]
            )

    def test_bench_startup_waits_for_unsafe_or_not_ready_conditions(self):
        cases = (
            (
                FakeBenchRosEnvironment(FakeClock(), armed=True),
                "ARMED_DURING_BENCH",
            ),
            (
                FakeBenchRosEnvironment(FakeClock(), ch7=1999),
                "CH7_NOT_LOCKED",
            ),
            (
                FakeBenchRosEnvironment(FakeClock(), node_names=["/px4ctrl"]),
                "PX4CTRL_DETECTED",
            ),
            (
                FakeBenchRosEnvironment(
                    FakeClock(), command_subscribers=["/hidden_controller"]
                ),
                "COMMAND_TOPIC_HAS_SUBSCRIBERS",
            ),
        )
        for ros, expected in cases:
            with self.subTest(expected=expected):
                code, summary, events = self.run_bench(ros)
                self.assertNotEqual(code, 0)
                self.assertEqual(summary["exit_reason"], "startup_timeout")
                self.assertEqual(summary["safety_abort_reason"], "startup_not_ready")
                self.assertFalse(summary["startup_ready"])
                self.assertIn(expected, summary["startup_unmet_conditions"])
                self.assertEqual(summary["actual_publish_count"], 0)
                self.assertEqual(ros.publishers, [])
                self.assertEqual(events, [])

    def test_bench_rejects_missing_stale_and_nonfinite_telemetry(self):
        config = self.alpha.load_alpha_config(
            str(BENCH_CONFIG), allow_bench_publish=True
        )
        adapter = self.alpha.AlphaPx4AuditAdapter(
            config=config, clock=lambda: 100.0, bench_publish_mode=True
        )
        state, rc, battery, odom, stamp = healthy_messages(100.0)
        rc["channels"][6] = 999
        adapter.update_state(state, stamp)
        adapter.update_rc(rc, stamp)
        adapter.update_battery(battery, stamp)
        adapter.update_odom(odom, stamp)
        self.assertEqual(adapter.bench_rejection_reasons(now=100.1), [])

        adapter.battery_snapshot = None
        self.assertIn(
            "BATTERY_MISSING", adapter.bench_rejection_reasons(now=100.1)
        )
        adapter.update_battery(battery, stamp)
        adapter.odom_received_at = 99.0
        self.assertIn("ODOM_STALE", adapter.bench_rejection_reasons(now=100.1))
        adapter.update_odom(
            {
                "position": {"x": math.nan, "y": 0.0, "z": 0.8},
                "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
                "yaw": 0.0,
            },
            100.0,
        )
        self.assertIn(
            "NONFINITE_TELEMETRY",
            adapter.bench_rejection_reasons(now=100.1),
        )

    def test_bench_config_keeps_required_runtime_freshness_limits(self):
        config = self.alpha.load_alpha_config(
            str(BENCH_CONFIG), allow_bench_publish=True
        )
        self.assertEqual(config["state_timeout_s"], 2.0)
        self.assertEqual(config["rc_timeout_s"], 0.5)
        self.assertEqual(config["odom_timeout_s"], 0.5)
        self.assertEqual(config["battery_timeout_s"], 3.0)

    def test_bench_warmup_accepts_delayed_out_of_order_first_frames(self):
        clock = FakeClock()
        ros = FakeBenchRosEnvironment(
            clock,
            topic_arrival_times={
                "/mavros/state": 0.4,
                "/mavros/rc/in": 0.8,
                "/mavros/battery": 1.2,
                "/ekf/ekf_odom": 2.0,
            },
        )
        code, summary, events = self.run_bench(ros, runtime=0.3)
        self.assertEqual(code, 0)
        self.assertTrue(summary["startup_ready"])
        self.assertGreaterEqual(summary["startup_wait_seconds"], 2.0)
        self.assertTrue(summary["publisher_created"])
        self.assertTrue(summary["publisher_created_after_startup_ready"])
        self.assertEqual(len(ros.publishers), 1)
        self.assertGreaterEqual(ros.publishers[0]["created_at"], 2.0)
        self.assertEqual(ros.rate_history[:2], [50.0, 10.0])
        self.assertTrue(all(summary["first_message_received"].values()))
        self.assertGreater(len(events), 0)

    def test_bench_warmup_waits_for_ch7_to_enter_locked_band(self):
        clock = FakeClock()
        ros = FakeBenchRosEnvironment(clock, ch7_locked_at=1.0)
        code, summary, events = self.run_bench(ros, runtime=0.2)
        self.assertEqual(code, 0)
        self.assertTrue(summary["startup_ready"])
        self.assertGreaterEqual(summary["startup_wait_seconds"], 1.0)
        self.assertGreater(len(events), 0)

    def test_bench_freshness_uses_monotonic_receive_time_not_header_stamp(self):
        clock = FakeClock()
        clock.value = 100.0
        ros = FakeBenchRosEnvironment(clock, header_stamp=0.0)
        config = self.alpha.load_alpha_config(
            str(BENCH_CONFIG), allow_bench_publish=True
        )
        adapter = self.alpha.AlphaPx4AuditAdapter(
            config=config, clock=clock, bench_publish_mode=True
        )
        state, rc, battery, odom, _ = healthy_messages()
        rc["channels"][6] = 999
        for message in (state, rc, battery, odom):
            message["header"] = {"stamp": 0.0}
        adapter.update_state(state)
        clock.advance(0.01)
        adapter.update_battery(battery)
        clock.advance(0.01)
        adapter.update_rc(rc)
        clock.advance(0.01)
        adapter.update_odom(odom)
        diagnostics = adapter.receive_diagnostics(now=100.04)
        self.assertEqual(adapter.bench_rejection_reasons(now=100.04), [])
        self.assertEqual(diagnostics["header_stamp"]["state"], 0.0)
        self.assertAlmostEqual(diagnostics["last_receive_age_seconds"]["state"], 0.04)
        self.assertTrue(all(diagnostics["first_message_received"].values()))

    def test_runtime_freshness_uses_one_monotonic_age_implementation(self):
        config = self.alpha.load_alpha_config(
            str(BENCH_CONFIG), allow_bench_publish=True
        )
        adapter = self.alpha.AlphaPx4AuditAdapter(
            config=config, clock=lambda: 100.0, bench_publish_mode=True
        )
        state, rc, battery, odom, _ = healthy_messages(100.0)
        rc["channels"][6] = 999
        adapter.update_state(state, 100.0)
        adapter.update_rc(rc, 100.0)
        adapter.update_battery(battery, 100.0)
        adapter.update_odom(odom, 100.0)

        freshness = adapter.telemetry_freshness(100.028)
        self.assertTrue(freshness["rc"]["fresh"])
        self.assertAlmostEqual(freshness["rc"]["age_seconds"], 0.028)
        self.assertEqual(freshness["rc"]["timeout_seconds"], 0.5)

        # A callback may run after the loop captured its single now snapshot.
        # That is age zero, not stale.
        adapter.rc_received_at = 100.050
        raced = adapter.telemetry_freshness(100.028)
        self.assertTrue(raced["rc"]["fresh"])
        self.assertEqual(raced["rc"]["age_seconds"], 0.0)

        adapter.rc_received_at = 100.0
        stale = adapter.telemetry_freshness(100.501)
        self.assertFalse(stale["rc"]["fresh"])
        self.assertAlmostEqual(stale["rc"]["age_seconds"], 0.501)
        self.assertEqual(stale["rc"]["timeout_seconds"], 0.5)
        self.assertEqual(stale["rc"]["reason"], "RC_STALE")

    def test_ros_header_epoch_future_and_jump_do_not_affect_freshness(self):
        clock = FakeClock()
        clock.value = 200.0
        config = self.alpha.load_alpha_config(
            str(BENCH_CONFIG), allow_bench_publish=True
        )
        adapter = self.alpha.AlphaPx4AuditAdapter(
            config=config, clock=clock, bench_publish_mode=True
        )
        _, rc, _, _, _ = healthy_messages(200.0)
        rc["channels"][6] = 999
        for header_stamp in (0.0, 4102444800.0, 1.0):
            rc["header"] = {"stamp": header_stamp}
            adapter.update_rc(rc)
            freshness = adapter.telemetry_freshness(clock() + 0.028)
            self.assertTrue(freshness["rc"]["fresh"])
            self.assertAlmostEqual(freshness["rc"]["age_seconds"], 0.028)
            self.assertEqual(
                adapter.receive_diagnostics(clock())["header_stamp"]["rc"],
                header_stamp,
            )
            clock.advance(0.1)

    def test_runtime_query_callback_race_does_not_false_trigger_rc_stale(self):
        ros = FakeBenchRosEnvironment(
            FakeClock(), query_delay_after_publisher=0.028
        )
        code, summary, events = self.run_bench(ros, runtime=0.3)
        self.assertEqual(code, 0)
        self.assertTrue(summary["startup_ready"])
        self.assertGreater(summary["actual_publish_count"], 0)
        self.assertGreater(len(events), 0)
        self.assertNotEqual(summary["safety_abort_reason"], "rc_stale")

    def test_runtime_rc_stale_summary_preserves_check_time_evidence(self):
        def stop_rc_updates(environment):
            environment.omit_topic = "/mavros/rc/in"
            environment.after_publish = None

        ros = FakeBenchRosEnvironment(FakeClock(), after_publish=stop_rc_updates)
        code, summary, events = self.run_bench(ros, runtime=1.0)
        self.assertNotEqual(code, 0)
        self.assertTrue(summary["startup_ready"])
        self.assertGreater(len(events), 0)
        self.assertEqual(summary["safety_abort_reason"], "rc_stale")
        self.assertEqual(summary["stale_channel"], "rc")
        self.assertGreater(summary["stale_age_seconds"], 0.5)
        self.assertEqual(summary["stale_timeout_seconds"], 0.5)
        self.assertEqual(
            summary["stale_age_seconds"],
            summary["stale_now_monotonic"]
            - summary["stale_last_receive_monotonic"],
        )

    def test_startup_and_runtime_share_freshness_implementation(self):
        runner = importlib.import_module("rpi_pixhawk_stage4.run_stage4_alpha")
        adapter_type = runner.AlphaPx4AuditAdapter
        original = adapter_type.telemetry_freshness
        calls = []

        def recording_freshness(adapter, now_mono):
            calls.append(float(now_mono))
            return original(adapter, now_mono)

        ros = FakeBenchRosEnvironment(FakeClock())
        with mock.patch.object(
            adapter_type,
            "telemetry_freshness",
            recording_freshness,
        ):
            code, summary, events = self.run_bench(ros, runtime=0.3)
        self.assertEqual(code, 0)
        self.assertGreater(len(events), 0)
        self.assertGreaterEqual(len(calls), 3)
        self.assertTrue(summary["publisher_created_after_startup_ready"])
    def test_bench_startup_timeout_does_not_create_publisher(self):
        clock = FakeClock()
        ros = FakeBenchRosEnvironment(clock, omit_topic="/ekf/ekf_odom")
        code, summary, events = self.run_bench(ros, runtime=0.2)
        self.assertNotEqual(code, 0)
        self.assertEqual(summary["exit_reason"], "startup_timeout")
        self.assertEqual(summary["safety_abort_reason"], "startup_not_ready")
        self.assertFalse(summary["startup_ready"])
        self.assertIn("ODOM_MISSING", summary["startup_unmet_conditions"])
        self.assertGreaterEqual(summary["startup_wait_seconds"], 5.0)
        self.assertEqual(summary["actual_publish_count"], 0)
        self.assertFalse(summary["publisher_created"])
        self.assertFalse(summary["publisher_created_after_startup_ready"])
        self.assertEqual(ros.publishers, [])
        self.assertEqual(events, [])
    def test_bench_publishes_mapped_hold_position_command(self):
        ros = FakeBenchRosEnvironment(FakeClock())
        code, summary, events = self.run_bench(ros)
        self.assertEqual(code, 0)
        self.assertEqual(len(ros.publishers), 1)
        self.assertEqual(ros.publishers[0]["topic"], "/setpoints_cmd")
        self.assertIs(ros.publishers[0]["message_type"], FakePositionCommand)
        self.assertEqual(ros.publishers[0]["queue_size"], 1)
        self.assertGreater(len(ros.published_messages), 0)
        message = ros.published_messages[0]
        self.assertEqual(
            (message.position.x, message.position.y, message.position.z),
            (1.0, 2.0, 0.8),
        )
        for vector in (message.velocity, message.acceleration, message.jerk):
            self.assertEqual((vector.x, vector.y, vector.z), (0.0, 0.0, 0.0))
        self.assertEqual(message.yaw, 0.25)
        self.assertEqual(message.yaw_dot, 0.0)
        self.assertEqual(message.kx, [0.0, 0.0, 0.0])
        self.assertEqual(message.kv, [0.0, 0.0, 0.0])
        self.assertGreater(message.trajectory_id, 0)
        self.assertEqual(message.trajectory_flag, 1)
        self.assertEqual(
            [item.trajectory_id for item in ros.published_messages],
            sorted(item.trajectory_id for item in ros.published_messages),
        )
        self.assertEqual(summary["mode"], "bench_publish_no_consumer")
        self.assertEqual(summary["bench_publish_count"], len(ros.published_messages))
        self.assertEqual(summary["actual_publish_count"], len(ros.published_messages))
        self.assertTrue(all(item["event"] == "BENCH_PUBLISH" for item in events))

    def test_bench_runtime_consumer_or_armed_change_stops_immediately(self):
        def add_subscriber(environment):
            environment.command_subscribers = ["/late_controller"]
            environment.after_publish = None

        def arm_vehicle(environment):
            environment.armed = True
            environment.feed_callbacks_once()
            environment.after_publish = None

        def unlock_ch7(environment):
            environment.ch7 = 1999
            environment.feed_callbacks_once()
            environment.after_publish = None

        def start_px4ctrl(environment):
            environment.node_names = ["/px4ctrl"]
            environment.after_publish = None

        for callback, expected in (
            (add_subscriber, "command_topic_has_subscribers"),
            (arm_vehicle, "armed_detected"),
            (unlock_ch7, "ch7_not_locked"),
            (start_px4ctrl, "px4ctrl_detected"),
        ):
            clock = FakeClock()
            ros = FakeBenchRosEnvironment(clock, after_publish=callback)
            with self.subTest(expected=expected):
                code, summary, events = self.run_bench(ros)
                self.assertNotEqual(code, 0)
                self.assertEqual(summary["safety_abort_reason"], expected)
                self.assertEqual(summary["actual_publish_count"], 1)
                self.assertEqual(len(events), 1)

    def test_nonbench_publish_switches_are_rejected_and_no_control_services_exist(self):
        runner = importlib.import_module("rpi_pixhawk_stage4.run_stage4_alpha")
        with self.assertRaisesRegex(RuntimeError, "actual_publish_not_implemented"):
            runner.main(["--enable-actual-publish"])
        with self.assertRaisesRegex(RuntimeError, "actual_publish_not_implemented"):
            runner.main(
                [
                    "--enable-actual-publish",
                    "--actual-publish-confirmation",
                    "I_ACCEPT_ALPHA_REAL_PUBLISH_RISK",
                ]
            )
        sources = (
            (STAGE_DIR / "alpha_px4_adapter.py").read_text(encoding="utf-8")
            + (STAGE_DIR / "run_stage4_alpha.py").read_text(encoding="utf-8")
        )
        self.assertNotIn("cmd/arming", sources)
        self.assertNotIn("set_mode", sources)
        self.assertNotIn("ServiceProxy", sources)

    def test_offline_simulation_writes_only_would_publish(self):
        runner = importlib.import_module("rpi_pixhawk_stage4.run_stage4_alpha")
        with tempfile.TemporaryDirectory() as directory:
            code = runner.main(["--simulate", "--results-dir", directory])
            summary = json.loads(
                (Path(directory) / "run_summary.json").read_text(encoding="utf-8")
            )
            events = [
                json.loads(line)
                for line in (Path(directory) / "alpha_audit.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
        self.assertEqual(code, 0)
        self.assertGreaterEqual(len(events), 1)
        self.assertTrue(all(item["event"] == "WOULD_PUBLISH" for item in events))
        self.assertEqual(summary["actual_publish_count"], 0)
        self.assertLessEqual(summary["maximum_runtime_seconds"], 5.0)


if __name__ == "__main__":
    unittest.main()