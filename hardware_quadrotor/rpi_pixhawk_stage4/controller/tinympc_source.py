"""Fail-closed Stage 5 setpoint source backed by the condensed TinyMPC solver."""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from .tinympc_solver import CondensedTinyMPCSolver, TinyMPCSolverError


X_SYSTEM_ID_KEYFRAMES = (
    (0.0, 0.0),
    (2.5, 1.0),
    (7.5, -1.0),
    (12.5, 1.0),
    (17.5, -1.0),
    (22.5, 1.0),
    (27.5, -1.0),
    (30.0, 0.0),
)

XY_SQUARE12CM_KEYFRAMES = (
    (0.0, (0.0, 0.0)),
    (6.0, (1.0, 0.0)),
    (7.0, (1.0, 0.0)),
    (13.0, (1.0, 1.0)),
    (14.0, (1.0, 1.0)),
    (20.0, (0.0, 1.0)),
    (21.0, (0.0, 1.0)),
    (27.0, (0.0, 0.0)),
    (30.0, (0.0, 0.0)),
)


class TinyMPCSafetyError(RuntimeError):
    """Raised before publication whenever the current MPC command is unsafe."""


def _vector(mapping: Any, name: str) -> np.ndarray:
    if not isinstance(mapping, dict):
        raise TinyMPCSafetyError("tinympc_{0}_missing".format(name))
    values = []
    for axis in ("x", "y", "z"):
        value = mapping.get(axis)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TinyMPCSafetyError("tinympc_{0}_{1}_invalid".format(name, axis))
        values.append(float(value))
    result = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(result)):
        raise TinyMPCSafetyError("tinympc_{0}_nonfinite".format(name))
    return result


class TinyMPCZeroMotionSource:
    name = "TinyMPCZeroMotionSource"

    def __init__(self, config: Dict[str, Any]) -> None:
        integral = config.get("integral_compensation") or {}
        if not isinstance(integral, dict):
            raise TinyMPCSafetyError("tinympc_integral_config_invalid")
        enabled = integral.get("enabled", False)
        if not isinstance(enabled, bool):
            raise TinyMPCSafetyError("tinympc_integral_enabled_invalid")

        augmented = config.get("augmented_integral")
        if augmented is None:
            augmented = {}
        if not isinstance(augmented, dict):
            raise TinyMPCSafetyError("tinympc_augmented_integral_config_invalid")
        augmented_enabled = augmented.get("enabled", False)
        if not isinstance(augmented_enabled, bool):
            raise TinyMPCSafetyError("tinympc_augmented_integral_enabled_invalid")
        if enabled and augmented_enabled:
            raise TinyMPCSafetyError("tinympc_integral_modes_conflict")

        self.config = config
        self.solver = CondensedTinyMPCSolver(config)
        self.solve_time_limit_seconds = float(config["solve_time_limit_seconds"])
        safety = config["safety"]
        self.position_limits = _vector(
            safety["max_position_error_m"], "position_error_limits"
        )
        self.horizontal_velocity_limit = float(safety["max_horizontal_velocity_mps"])
        self.vertical_velocity_limit = float(safety["max_vertical_velocity_mps"])
        self.target_offset = _vector(
            config.get(
                "target_offset_m",
                {"x": 0.0, "y": 0.0, "z": 0.0},
            ),
            "target_offset_m",
        )
        if np.any(np.abs(self.target_offset) > self.position_limits):
            raise TinyMPCSafetyError("tinympc_target_offset_exceeds_position_limits")

        zero3 = {"x": 0.0, "y": 0.0, "z": 0.0}

        self.integral_enabled = enabled
        self.integral_gain = _vector(
            integral.get("gain", zero3),
            "integral_gain",
        )
        self.integral_state_limit = _vector(
            integral.get("state_limit_m_s", zero3),
            "integral_state_limit",
        )
        self.integral_accel_limit = _vector(
            integral.get("max_correction_mps2", zero3),
            "integral_accel_limit",
        )

        if (
            np.any(self.integral_gain < 0.0)
            or np.any(self.integral_state_limit < 0.0)
            or np.any(self.integral_accel_limit < 0.0)
        ):
            raise TinyMPCSafetyError("tinympc_integral_config_negative")

        self._integral_error = np.zeros(3, dtype=float)
        self._integrator_active = False
        self.augmented_integral_enabled = augmented_enabled
        self.augmented_integral_state_limit = _vector(
            augmented.get("state_limit_m_s", zero3),
            "augmented_integral_state_limit",
        )
        if np.any(self.augmented_integral_state_limit < 0.0):
            raise TinyMPCSafetyError("tinympc_augmented_integral_config_negative")
        if self.augmented_integral_enabled and np.any(
            self.augmented_integral_state_limit <= 0.0
        ):
            raise TinyMPCSafetyError("tinympc_augmented_integral_limit_not_positive")
        self._aug_integral_state = np.zeros(3, dtype=float)
        self._aug_integrator_active = False

        reference_profile = config.get("reference_profile")
        self._reference_profile = None
        self._reference_active = False
        self._reference_started_at = None

        if reference_profile is not None:
            if not isinstance(reference_profile, dict):
                raise TinyMPCSafetyError(
                    "tinympc_reference_profile_invalid"
                )
            profile_type = reference_profile.get("type")
            if profile_type == "linear_ramp_hold":
                ramp_duration = reference_profile.get(
                    "ramp_duration_seconds"
                )
                if (
                    isinstance(ramp_duration, bool)
                    or not isinstance(ramp_duration, (int, float))
                    or not math.isfinite(float(ramp_duration))
                    or float(ramp_duration) <= 0.0
                ):
                    raise TinyMPCSafetyError(
                        "tinympc_reference_ramp_duration_invalid"
                    )

                self._reference_profile = {
                    "type": "linear_ramp_hold",
                    "ramp_duration_seconds": float(ramp_duration),
                }
            elif profile_type == "x_bidirectional_system_id":
                required_offset = np.asarray([0.05, 0.0, 0.0], dtype=float)
                if not np.allclose(
                    self.target_offset,
                    required_offset,
                    rtol=0.0,
                    atol=1.0e-12,
                ):
                    raise TinyMPCSafetyError(
                        "tinympc_x_system_id_target_offset_invalid"
                    )
                self._reference_profile = {
                    "type": "x_bidirectional_system_id",
                    "keyframes": X_SYSTEM_ID_KEYFRAMES,
                }
            elif profile_type == "xy_square12cm":
                required_offset = np.asarray([0.12, 0.12, 0.0], dtype=float)
                if not np.allclose(
                    self.target_offset,
                    required_offset,
                    rtol=0.0,
                    atol=1.0e-12,
                ):
                    raise TinyMPCSafetyError(
                        "tinympc_xy_square12cm_target_offset_invalid"
                    )
                self._reference_profile = {
                    "type": "xy_square12cm",
                    "keyframes": XY_SQUARE12CM_KEYFRAMES,
                }
            elif profile_type == "xy_square30cm":
                required_offset = np.asarray([0.30, 0.30, 0.0], dtype=float)
                if not np.allclose(
                    self.target_offset,
                    required_offset,
                    rtol=0.0,
                    atol=1.0e-12,
                ):
                    raise TinyMPCSafetyError(
                        "tinympc_xy_square30cm_target_offset_invalid"
                    )
                self._reference_profile = {
                    "type": "xy_square30cm",
                    "keyframes": XY_SQUARE12CM_KEYFRAMES,
                }
            elif profile_type == "relative_waypoints_linear":
                active_window = reference_profile.get("active_window_seconds")
                if (
                    isinstance(active_window, bool)
                    or not isinstance(active_window, (int, float))
                    or not math.isfinite(float(active_window))
                    or float(active_window) <= 0.0
                ):
                    raise TinyMPCSafetyError(
                        "tinympc_relative_waypoints_active_window_invalid"
                    )
                reference_offset_limits = _vector(
                    reference_profile.get("max_reference_offset_m"),
                    "reference_offset_limits",
                )
                if np.any(reference_offset_limits < 0.0):
                    raise TinyMPCSafetyError(
                        "tinympc_reference_offset_limits_negative"
                    )
                waypoints = reference_profile.get("waypoints")
                if not isinstance(waypoints, list) or len(waypoints) < 2:
                    raise TinyMPCSafetyError(
                        "tinympc_relative_waypoints_count_invalid"
                    )
                parsed_waypoints = []
                previous_time = None
                for index, waypoint in enumerate(waypoints):
                    if not isinstance(waypoint, dict):
                        raise TinyMPCSafetyError(
                            "tinympc_relative_waypoint_invalid"
                        )
                    time_s = waypoint.get("time_s")
                    if (
                        isinstance(time_s, bool)
                        or not isinstance(time_s, (int, float))
                        or not math.isfinite(float(time_s))
                    ):
                        raise TinyMPCSafetyError(
                            "tinympc_relative_waypoint_time_invalid"
                        )
                    time_s = float(time_s)
                    raw_offset = waypoint.get("offset_m")
                    if not isinstance(raw_offset, (list, tuple)) or len(raw_offset) != 3:
                        raise TinyMPCSafetyError(
                            "tinympc_relative_waypoint_offset_dimension_invalid"
                        )
                    offset_values = []
                    for value in raw_offset:
                        if (
                            isinstance(value, bool)
                            or not isinstance(value, (int, float))
                            or not math.isfinite(float(value))
                        ):
                            raise TinyMPCSafetyError(
                                "tinympc_relative_waypoint_offset_nonfinite"
                            )
                        offset_values.append(float(value))
                    offset = np.asarray(offset_values, dtype=float)
                    if index == 0:
                        if time_s != 0.0:
                            raise TinyMPCSafetyError(
                                "tinympc_relative_waypoints_first_time_not_zero"
                            )
                        if not np.array_equal(offset, np.zeros(3, dtype=float)):
                            raise TinyMPCSafetyError(
                                "tinympc_relative_waypoints_first_offset_not_zero"
                            )
                    elif time_s <= previous_time:
                        raise TinyMPCSafetyError(
                            "tinympc_relative_waypoint_time_not_increasing"
                        )
                    if time_s > float(active_window):
                        raise TinyMPCSafetyError(
                            "tinympc_relative_waypoint_exceeds_active_window"
                        )
                    if np.any(np.abs(offset) > reference_offset_limits):
                        raise TinyMPCSafetyError(
                            "tinympc_relative_waypoint_position_limit_exceeded"
                        )
                    parsed_waypoints.append((time_s, offset))
                    previous_time = time_s
                for index in range(len(parsed_waypoints) - 1):
                    start_time, start_offset = parsed_waypoints[index]
                    end_time, end_offset = parsed_waypoints[index + 1]
                    velocity = (end_offset - start_offset) / (end_time - start_time)
                    if (
                        abs(float(velocity[0])) > self.horizontal_velocity_limit
                        or abs(float(velocity[1])) > self.horizontal_velocity_limit
                        or abs(float(velocity[2])) > self.vertical_velocity_limit
                    ):
                        raise TinyMPCSafetyError(
                            "tinympc_relative_waypoint_velocity_limit_exceeded"
                        )
                self._reference_profile = {
                    "type": "relative_waypoints_linear",
                    "active_window_seconds": float(active_window),
                    "max_reference_offset_m": reference_offset_limits,
                    "waypoints": tuple(parsed_waypoints),
                }
            else:
                raise TinyMPCSafetyError(
                    "tinympc_reference_profile_type_invalid"
                )
        values = np.array(
            [
                self.solve_time_limit_seconds,
                self.horizontal_velocity_limit,
                self.vertical_velocity_limit,
            ]
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise TinyMPCSafetyError("tinympc_source_limits_invalid")

    def reset_integrator(self) -> None:
        self._integral_error.fill(0.0)
        self._integrator_active = True
        self._aug_integral_state.fill(0.0)
        self._aug_integrator_active = True

        # Ramp reference starts only after ACTIVE authorization.
        self._reference_active = True
        self._reference_started_at = None

    def _reference(
        self, timestamp: float, base_position: np.ndarray
    ):
        if not math.isfinite(float(timestamp)):
            raise TinyMPCSafetyError(
                "tinympc_reference_timestamp_nonfinite"
            )

        # Legacy static/step behavior remains exactly unchanged.
        if self._reference_profile is None:
            return (
                base_position + self.target_offset,
                np.zeros(3, dtype=float),
            )

        # PRESTREAM: hold the latched position, zero reference velocity.
        if not self._reference_active:
            return (
                base_position.copy(),
                np.zeros(3, dtype=float),
            )

        # First actual ACTIVE compute defines t=0.
        if self._reference_started_at is None:
            self._reference_started_at = float(timestamp)

        elapsed = max(
            0.0,
            float(timestamp) - self._reference_started_at,
        )

        if self._reference_profile["type"] == "linear_ramp_hold":
            duration = self._reference_profile[
                "ramp_duration_seconds"
            ]

            if elapsed < duration:
                alpha = elapsed / duration
                position_ref = (
                    base_position + alpha * self.target_offset
                )
                velocity_ref = (
                    self.target_offset / duration
                )
            else:
                position_ref = (
                    base_position + self.target_offset
                )
                velocity_ref = np.zeros(3, dtype=float)
        elif self._reference_profile["type"] == "x_bidirectional_system_id":
            keyframes = self._reference_profile["keyframes"]
            position_ref = base_position.copy()
            velocity_ref = np.zeros(3, dtype=float)
            if elapsed < keyframes[-1][0]:
                for index in range(len(keyframes) - 1):
                    start_time, start_scale = keyframes[index]
                    end_time, end_scale = keyframes[index + 1]
                    if start_time <= elapsed < end_time:
                        duration = end_time - start_time
                        fraction = (elapsed - start_time) / duration
                        scale = start_scale + fraction * (end_scale - start_scale)
                        position_ref[0] += scale * self.target_offset[0]
                        velocity_ref[0] = (
                            (end_scale - start_scale)
                            * self.target_offset[0]
                            / duration
                        )
                        break
        elif self._reference_profile["type"] in (
            "xy_square12cm",
            "xy_square30cm",
        ):
            keyframes = self._reference_profile["keyframes"]
            position_ref = base_position.copy()
            velocity_ref = np.zeros(3, dtype=float)
            if elapsed < keyframes[-1][0]:
                for index in range(len(keyframes) - 1):
                    start_time, start_scale_xy = keyframes[index]
                    end_time, end_scale_xy = keyframes[index + 1]
                    if start_time <= elapsed < end_time:
                        duration = end_time - start_time
                        fraction = (elapsed - start_time) / duration
                        start_scale = np.asarray(
                            [start_scale_xy[0], start_scale_xy[1], 0.0],
                            dtype=float,
                        )
                        end_scale = np.asarray(
                            [end_scale_xy[0], end_scale_xy[1], 0.0],
                            dtype=float,
                        )
                        scale = start_scale + fraction * (end_scale - start_scale)
                        position_ref = base_position + scale * self.target_offset
                        velocity_ref = (
                            (end_scale - start_scale)
                            * self.target_offset
                            / duration
                        )
                        break
        else:
            waypoints = self._reference_profile["waypoints"]
            position_ref = base_position + waypoints[-1][1]
            velocity_ref = np.zeros(3, dtype=float)
            if elapsed < waypoints[-1][0]:
                for index in range(len(waypoints) - 1):
                    start_time, start_offset = waypoints[index]
                    end_time, end_offset = waypoints[index + 1]
                    if start_time <= elapsed < end_time:
                        duration = end_time - start_time
                        fraction = (elapsed - start_time) / duration
                        position_ref = base_position + (
                            start_offset
                            + fraction * (end_offset - start_offset)
                        )
                        velocity_ref = (end_offset - start_offset) / duration
                        break

        if (
            not np.all(np.isfinite(position_ref))
            or not np.all(np.isfinite(velocity_ref))
        ):
            raise TinyMPCSafetyError(
                "tinympc_reference_nonfinite"
            )

        return position_ref, velocity_ref

    def compute(
        self, timestamp: float, odom: Dict[str, Any], target: Dict[str, Any]
    ) -> Dict[str, Any]:
        position = _vector(odom.get("position"), "odom_position")
        velocity = _vector(odom.get("velocity"), "odom_velocity")
        base_target_position = _vector(
            target.get("position"), "target_position"
        )
        target_position, target_velocity = self._reference(
            timestamp,
            base_target_position,
        )
        yaw = target.get("yaw")
        if isinstance(yaw, bool) or not isinstance(yaw, (int, float)) or not math.isfinite(float(yaw)):
            raise TinyMPCSafetyError("tinympc_target_yaw_nonfinite")

        error = position - target_position
        axis_names = ("x", "y", "z")
        for index, axis in enumerate(axis_names):
            if abs(error[index]) > self.position_limits[index]:
                raise TinyMPCSafetyError("tinympc_position_error_{0}_exceeded".format(axis))
        horizontal_speed = math.hypot(float(velocity[0]), float(velocity[1]))
        if horizontal_speed > self.horizontal_velocity_limit:
            raise TinyMPCSafetyError("tinympc_horizontal_velocity_exceeded")
        if abs(float(velocity[2])) > self.vertical_velocity_limit:
            raise TinyMPCSafetyError("tinympc_vertical_velocity_exceeded")

        if self.integral_enabled and self._integrator_active:
            self._integral_error = np.clip(
                self._integral_error
                + error * self.solver.dt_seconds,
                -self.integral_state_limit,
                self.integral_state_limit,
            )

        velocity_error = velocity - target_velocity

        if self.augmented_integral_enabled:
            state = np.concatenate(
                (
                    error,
                    velocity_error,
                    self._aug_integral_state,
                )
            )
        else:
            state = np.concatenate(
                (
                    error,
                    velocity_error,
                )
            )
        try:
            solution = self.solver.solve(state)
        except TinyMPCSolverError as exc:
            raise TinyMPCSafetyError("tinympc_solver_error:{0}".format(exc)) from exc
        if not solution["success"]:
            raise TinyMPCSafetyError(
                "tinympc_solver_not_converged:iterations={0}:residual={1}".format(
                    solution["iterations"], solution["residual"]
                )
            )
        solve_time = float(solution["solve_time_seconds"])
        if not math.isfinite(solve_time) or solve_time > self.solve_time_limit_seconds:
            raise TinyMPCSafetyError(
                "tinympc_solve_time_exceeded:{0}".format(solution["solve_time_seconds"])
            )

        control = np.asarray(solution["control"], dtype=float)
        if control.shape != (3,) or not np.all(np.isfinite(control)):
            raise TinyMPCSafetyError("tinympc_control_nonfinite")

        if self.integral_enabled and self._integrator_active:
            integral_correction = np.clip(
                -self.integral_gain * self._integral_error,
                -self.integral_accel_limit,
                self.integral_accel_limit,
            )
            control = control + integral_correction
        control = np.clip(
            control, -self.solver.axis_limits, self.solver.axis_limits
        )
        if not np.all(np.isfinite(control)):
            raise TinyMPCSafetyError("tinympc_control_nonfinite")
        if self.augmented_integral_enabled and self._aug_integrator_active:
            next_integral = np.clip(
                self._aug_integral_state + error * self.solver.dt_seconds,
                -self.augmented_integral_state_limit,
                self.augmented_integral_state_limit,
            )
            if not np.all(np.isfinite(next_integral)):
                raise TinyMPCSafetyError("tinympc_augmented_integral_nonfinite")
            self._aug_integral_state = next_integral
        return {
            "timestamp": float(timestamp),
            "position": dict(odom["position"]),
            "velocity": dict(odom["velocity"]),
            "acceleration": {
                "x": float(control[0]),
                "y": float(control[1]),
                "z": float(control[2]),
            },
            "jerk": {"x": 0.0, "y": 0.0, "z": 0.0},
            "yaw": float(yaw),
            "yaw_dot": 0.0,
        }
