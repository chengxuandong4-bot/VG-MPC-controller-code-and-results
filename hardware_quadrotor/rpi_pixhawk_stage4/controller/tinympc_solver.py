"""Small, deterministic condensed MPC for Stage 5 translational hold.

The implementation is deliberately ROS-free and depends only on NumPy.  It
solves the full horizon problem with projected gradient descent and never
returns a cached command when the current solve is invalid.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterable

import numpy as np


class TinyMPCSolverError(RuntimeError):
    """Raised when the numerical problem cannot be evaluated safely."""


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TinyMPCSolverError("tinympc_config_invalid:{0}".format(name))
    result = float(value)
    if not math.isfinite(result):
        raise TinyMPCSolverError("tinympc_config_nonfinite:{0}".format(name))
    return result


def _finite_vector(values: Iterable[Any], length: int, name: str) -> np.ndarray:
    try:
        result = np.asarray(list(values), dtype=float)
    except (TypeError, ValueError) as exc:
        raise TinyMPCSolverError("tinympc_{0}_invalid".format(name)) from exc
    if result.shape != (length,):
        raise TinyMPCSolverError("tinympc_{0}_shape_invalid".format(name))
    if not np.all(np.isfinite(result)):
        raise TinyMPCSolverError("tinympc_{0}_nonfinite".format(name))
    return result


class CondensedTinyMPCSolver:
    """Condensed MPC for a 3-D double integrator, optionally with integral state."""

    input_size = 3

    def __init__(self, config: Dict[str, Any], clock=time.perf_counter) -> None:
        if not isinstance(config, dict):
            raise TinyMPCSolverError("tinympc_config_not_object")
        self.clock = clock
        self.dt_seconds = _finite_float(config.get("dt_seconds"), "dt_seconds")
        self.horizon_steps = int(config.get("horizon_steps", 0))
        self.max_iterations = int(config.get("max_iterations", 0))
        self.residual_tolerance = _finite_float(
            config.get("residual_tolerance"), "residual_tolerance"
        )
        if self.dt_seconds <= 0.0 or self.horizon_steps <= 0:
            raise TinyMPCSolverError("tinympc_horizon_invalid")
        if self.max_iterations <= 0 or self.residual_tolerance <= 0.0:
            raise TinyMPCSolverError("tinympc_iteration_config_invalid")

        augmented_integral = config.get("augmented_integral")
        if augmented_integral is None:
            augmented_integral = {}
        if not isinstance(augmented_integral, dict):
            raise TinyMPCSolverError("tinympc_augmented_integral_not_object")
        augmented_enabled = augmented_integral.get("enabled", False)
        if not isinstance(augmented_enabled, bool):
            raise TinyMPCSolverError("tinympc_augmented_integral_enabled_invalid")
        self.augmented_integral_enabled = augmented_enabled
        self.state_size = 9 if augmented_enabled else 6

        q = _finite_vector(
            config.get("state_weights", []), self.state_size, "state_weights"
        )
        r = _finite_vector(config.get("input_weights", []), 3, "input_weights")
        if np.any(q <= 0.0) or np.any(r <= 0.0):
            raise TinyMPCSolverError("tinympc_weights_not_positive")

        limits = config.get("acceleration_limits_mps2")
        if not isinstance(limits, dict):
            raise TinyMPCSolverError("tinympc_acceleration_limits_missing")
        axis_limits = np.array(
            [
                _finite_float(limits.get("x"), "acceleration_limit_x"),
                _finite_float(limits.get("y"), "acceleration_limit_y"),
                _finite_float(limits.get("z"), "acceleration_limit_z"),
            ],
            dtype=float,
        )
        if np.any(axis_limits <= 0.0):
            raise TinyMPCSolverError("tinympc_acceleration_limits_not_positive")
        self.axis_limits = axis_limits

        dt = self.dt_seconds
        if self.augmented_integral_enabled:
            self.A = np.block(
                [
                    [np.eye(3), dt * np.eye(3), np.zeros((3, 3))],
                    [np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))],
                    [dt * np.eye(3), np.zeros((3, 3)), np.eye(3)],
                ]
            )
            self.B = np.vstack(
                (
                    0.5 * dt * dt * np.eye(3),
                    dt * np.eye(3),
                    np.zeros((3, 3)),
                )
            )
        else:
            self.A = np.block(
                [
                    [np.eye(3), dt * np.eye(3)],
                    [np.zeros((3, 3)), np.eye(3)],
                ]
            )
            self.B = np.vstack((0.5 * dt * dt * np.eye(3), dt * np.eye(3)))
        self.Sx, self.Su = self._prediction_matrices()
        q_bar = np.kron(np.eye(self.horizon_steps), np.diag(q))
        r_bar = np.kron(np.eye(self.horizon_steps), np.diag(r))
        hessian = 2.0 * (self.Su.T.dot(q_bar).dot(self.Su) + r_bar)
        self.H = 0.5 * (hessian + hessian.T) + 1.0e-9 * np.eye(
            self.horizon_steps * self.input_size
        )
        self._gradient_map = 2.0 * self.Su.T.dot(q_bar).dot(self.Sx)
        maximum_eigenvalue = float(np.linalg.eigvalsh(self.H)[-1])
        if not math.isfinite(maximum_eigenvalue) or maximum_eigenvalue <= 0.0:
            raise TinyMPCSolverError("tinympc_hessian_invalid")
        self.step_size = 1.0 / maximum_eigenvalue
        self.lower_bounds = np.tile(-axis_limits, self.horizon_steps)
        self.upper_bounds = np.tile(axis_limits, self.horizon_steps)
        self._warm_start = np.zeros(self.horizon_steps * self.input_size)

    def _prediction_matrices(self):
        state_rows = self.horizon_steps * self.state_size
        control_columns = self.horizon_steps * self.input_size
        sx = np.zeros((state_rows, self.state_size))
        su = np.zeros((state_rows, control_columns))
        powers = [np.eye(self.state_size)]
        for _ in range(self.horizon_steps):
            powers.append(powers[-1].dot(self.A))
        for row in range(self.horizon_steps):
            row_slice = slice(row * self.state_size, (row + 1) * self.state_size)
            sx[row_slice, :] = powers[row + 1]
            for column in range(row + 1):
                column_slice = slice(
                    column * self.input_size, (column + 1) * self.input_size
                )
                su[row_slice, column_slice] = powers[row - column].dot(self.B)
        return sx, su

    def reset_warm_start(self) -> None:
        self._warm_start.fill(0.0)

    def predict_states(self, state: Iterable[Any], controls: Iterable[Any]) -> np.ndarray:
        x0 = _finite_vector(state, self.state_size, "state")
        u = _finite_vector(
            controls, self.horizon_steps * self.input_size, "controls"
        )
        result = self.Sx.dot(x0) + self.Su.dot(u)
        if not np.all(np.isfinite(result)):
            raise TinyMPCSolverError("tinympc_prediction_nonfinite")
        return result

    def _projected_residual(self, controls: np.ndarray, gradient: np.ndarray) -> float:
        projected = np.clip(
            controls - gradient, self.lower_bounds, self.upper_bounds
        )
        return float(np.max(np.abs(controls - projected)))

    def solve(self, state: Iterable[Any]) -> Dict[str, Any]:
        started = self.clock()
        x0 = _finite_vector(state, self.state_size, "state")
        linear = self._gradient_map.dot(x0)
        controls = np.clip(
            self._warm_start.copy(), self.lower_bounds, self.upper_bounds
        )
        residual = math.inf
        iterations = 0
        success = False
        for iterations in range(1, self.max_iterations + 1):
            gradient = self.H.dot(controls) + linear
            if not np.all(np.isfinite(gradient)):
                self.reset_warm_start()
                raise TinyMPCSolverError("tinympc_gradient_nonfinite")
            controls = np.clip(
                controls - self.step_size * gradient,
                self.lower_bounds,
                self.upper_bounds,
            )
            residual = self._projected_residual(
                controls, self.H.dot(controls) + linear
            )
            if not math.isfinite(residual):
                self.reset_warm_start()
                raise TinyMPCSolverError("tinympc_residual_nonfinite")
            if residual <= self.residual_tolerance:
                success = True
                break

        solve_time = max(0.0, float(self.clock()) - float(started))
        if success:
            shifted = np.empty_like(controls)
            shifted[:-self.input_size] = controls[self.input_size :]
            shifted[-self.input_size :] = controls[-self.input_size :]
            self._warm_start = shifted
        else:
            self.reset_warm_start()
        return {
            "success": success,
            "control": controls[: self.input_size].copy(),
            "controls": controls.copy(),
            "iterations": iterations,
            "residual": residual,
            "solve_time_seconds": solve_time,
        }
