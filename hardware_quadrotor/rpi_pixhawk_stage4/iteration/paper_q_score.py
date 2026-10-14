"""Paper-compatible drone q-score adapter for the 9-state TinyMPC model.

The paper's drone benchmark used a 12-state hover-linearized plant and scored
eight metrics with fixed weights and normalizers.  The flight controller is a
9-state outer-loop model (position, velocity, integral error), so this module
lifts its achieved acceleration into the paper's hover-linearized attitude and
input coordinates without changing the legacy aggregation formula.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np


PAPER_BENCHMARK_SECONDS = 10.0
PAPER_VIOLATION_LAMBDA = 0.5
PAPER_WEIGHTS = {
    "IAE_pos": 0.35,
    "RMS_pos_error": 0.20,
    "max_pos_error": 0.10,
    "final_pos_error": 0.05,
    "IAE_att": 0.10,
    "control_energy": 0.05,
    "avg_solve_time_ms": 0.10,
    "saturation_rate": 0.05,
}
PAPER_NORMALIZERS = {
    "IAE_pos": 5.0,
    "RMS_pos_error": 0.5,
    "max_pos_error": 1.5,
    "final_pos_error": 0.5,
    "IAE_att": 2.0,
    "control_energy": 50.0,
    "avg_solve_time_ms": 50.0,
    "saturation_rate": 0.1,
}

# These are the physical constants used by the paper's drone_model.m.
PAPER_MASS_KG = 0.030
PAPER_GRAVITY_MPS2 = 9.81
PAPER_INERTIA_KGM2 = np.asarray((1.4e-5, 1.4e-5, 2.2e-5), dtype=float)


def aggregate_paper_q(
    metrics: Mapping[str, float],
    violation_residual: float = 0.0,
) -> Dict[str, Any]:
    """Apply the legacy paper aggregation exactly; larger q is better."""

    missing = sorted(set(PAPER_WEIGHTS) - set(metrics))
    if missing:
        raise ValueError("paper_q_missing_metrics:{0}".format(",".join(missing)))
    if not np.isclose(sum(PAPER_WEIGHTS.values()), 1.0):
        raise RuntimeError("paper_q_weights_do_not_sum_to_one")
    if not np.isfinite(violation_residual) or violation_residual < 0.0:
        raise ValueError("paper_q_invalid_violation_residual")

    normalized = {
        name: float(metrics[name]) / PAPER_NORMALIZERS[name]
        for name in PAPER_WEIGHTS
    }
    if not all(np.isfinite(value) and value >= 0.0 for value in normalized.values()):
        raise ValueError("paper_q_metric_not_finite_nonnegative")
    weighted = {
        name: PAPER_WEIGHTS[name] * normalized[name]
        for name in PAPER_WEIGHTS
    }
    normalized_cost = float(sum(weighted.values()))
    s_score = float(1.0 / (1.0 + normalized_cost))
    v_penalty = float(PAPER_VIOLATION_LAMBDA * violation_residual)
    return {
        "metrics": {name: float(metrics[name]) for name in PAPER_WEIGHTS},
        "normalized_metrics": normalized,
        "weighted_components": weighted,
        "normalized_cost": normalized_cost,
        "s_score": s_score,
        "violation_residual": float(violation_residual),
        "v_penalty": v_penalty,
        "q_score": float(s_score - v_penalty),
    }


def _trajectory(name: str, values: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
        raise ValueError("paper_q_{0}_must_be_nonempty_n_by_3".format(name))
    if not np.all(np.isfinite(array)):
        raise ValueError("paper_q_{0}_not_finite".format(name))
    return array


def _first_difference(values: np.ndarray, dt_seconds: float) -> np.ndarray:
    return np.diff(values, axis=0, prepend=values[[0]]) / dt_seconds


def trajectory_paper_q(
    position_errors_m: Sequence[Sequence[float]],
    achieved_acceleration_mps2: Sequence[Sequence[float]],
    commanded_acceleration_mps2: Sequence[Sequence[float]],
    solve_times_seconds: Sequence[float],
    dt_seconds: float,
    acceleration_limits_mps2: Sequence[float],
    violation_residual: float = 0.0,
) -> Dict[str, Any]:
    """Map a 9-state trajectory to the paper's eight drone metrics and q.

    Integral metrics are converted to a 10-second equivalent so the original
    paper normalizers remain meaningful for the 146-second Figure-8.  Attitude
    and body moments are virtual quantities from the same small-angle hover
    model used by the paper; no unobserved flight attitude is claimed.
    """

    errors = _trajectory("position_errors", position_errors_m)
    achieved_acceleration = _trajectory(
        "achieved_acceleration", achieved_acceleration_mps2
    )
    commanded_acceleration = _trajectory(
        "commanded_acceleration", commanded_acceleration_mps2
    )
    if not (len(errors) == len(achieved_acceleration) == len(commanded_acceleration)):
        raise ValueError("paper_q_trajectory_length_mismatch")
    dt = float(dt_seconds)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("paper_q_invalid_dt")
    duration_seconds = float(len(errors) * dt)
    duration_scale = PAPER_BENCHMARK_SECONDS / duration_seconds

    solve_times = np.asarray(solve_times_seconds, dtype=float)
    if solve_times.ndim != 1 or len(solve_times) != len(errors):
        raise ValueError("paper_q_solve_time_length_mismatch")
    if not np.all(np.isfinite(solve_times)) or np.any(solve_times < 0.0):
        raise ValueError("paper_q_invalid_solve_time")

    acceleration_limits = np.asarray(acceleration_limits_mps2, dtype=float)
    if acceleration_limits.shape != (3,) or not np.all(np.isfinite(acceleration_limits)):
        raise ValueError("paper_q_invalid_acceleration_limits")
    if np.any(acceleration_limits <= 0.0):
        raise ValueError("paper_q_nonpositive_acceleration_limit")

    position_error_norm = np.linalg.norm(errors, axis=1)
    iae_pos_raw = float(np.sum(position_error_norm) * dt)

    # Small-angle hover lift: ax = g*pitch, ay = -g*roll, yaw = 0.
    attitude = np.column_stack(
        (
            -achieved_acceleration[:, 1] / PAPER_GRAVITY_MPS2,
            achieved_acceleration[:, 0] / PAPER_GRAVITY_MPS2,
            np.zeros(len(achieved_acceleration), dtype=float),
        )
    )
    attitude_norm = np.linalg.norm(attitude, axis=1)
    iae_att_raw = float(np.sum(attitude_norm) * dt)

    # Lift outer-loop acceleration into the legacy [thrust, tau_x, tau_y,
    # tau_z] coordinates.  Angular acceleration is the second finite
    # difference of the virtual attitude.
    angular_rate = _first_difference(attitude, dt)
    angular_acceleration = _first_difference(angular_rate, dt)
    thrust = PAPER_MASS_KG * (
        PAPER_GRAVITY_MPS2 + achieved_acceleration[:, 2]
    )
    moments = angular_acceleration * PAPER_INERTIA_KGM2
    virtual_input = np.column_stack((thrust, moments))
    control_energy_raw = float(
        np.sum(np.linalg.norm(virtual_input, axis=1) ** 2) * dt
    )

    # The new outer-loop acceleration boundary is the actuator boundary visible
    # to this controller.  Match the old metric's per-step `any input saturated`
    # semantics rather than averaging individual saturated axes.
    saturated = np.any(
        np.abs(commanded_acceleration) >= 0.995 * acceleration_limits,
        axis=1,
    )
    saturation_rate = float(np.mean(saturated))
    warm_solve_times = solve_times[1:] if len(solve_times) > 1 else solve_times

    metrics = {
        "IAE_pos": iae_pos_raw * duration_scale,
        "RMS_pos_error": float(np.sqrt(np.mean(position_error_norm**2))),
        "max_pos_error": float(np.max(position_error_norm)),
        "final_pos_error": float(position_error_norm[-1]),
        "IAE_att": iae_att_raw * duration_scale,
        "control_energy": control_energy_raw * duration_scale,
        "avg_solve_time_ms": float(np.mean(warm_solve_times) * 1000.0),
        "saturation_rate": saturation_rate,
    }
    result = aggregate_paper_q(metrics, violation_residual=violation_residual)
    result.update(
        {
            "adapter_version": "paper_drone_q_v1_9state_hover_lift",
            "duration_seconds": duration_seconds,
            "duration_equivalent_seconds": PAPER_BENCHMARK_SECONDS,
            "raw_integrals": {
                "IAE_pos": iae_pos_raw,
                "IAE_att_proxy": iae_att_raw,
                "control_energy_proxy": control_energy_raw,
            },
            "mapping": {
                "attitude": "roll=-ay/g,pitch=ax/g,yaw=0",
                "input": "virtual_hover_thrust_and_body_moments",
                "saturation": "any_outer_acceleration_axis_at_limit_per_step",
            },
        }
    )
    return result
