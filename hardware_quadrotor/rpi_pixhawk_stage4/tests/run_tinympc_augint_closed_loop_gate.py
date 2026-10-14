"""Deterministic 30-second closed-loop comparison for the Stage 5 release gate."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from rpi_pixhawk_stage4.controller.tinympc_solver import CondensedTinyMPCSolver


STAGE4_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = STAGE4_ROOT / "config"
AUGMENTED_CONFIG = CONFIG_DIR / "alpha_stage5_tinympc_x5cm_n20_augint.json"
BASE_CONFIG = CONFIG_DIR / "alpha_stage5_tinympc_x5cm_n20.json"
STEPS = 200


def _controller_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)["controller"]


def _simulate(
    config: Dict[str, Any], disturbance: np.ndarray
) -> Tuple[Dict[str, Any], np.ndarray]:
    solver = CondensedTinyMPCSolver(config)
    dt = solver.dt_seconds
    augmented = solver.state_size == 9
    limits = None
    integral = np.zeros(3, dtype=float)
    if augmented:
        block = config["augmented_integral"]
        limits = np.asarray(
            [
                block["state_limit_m_s"]["x"],
                block["state_limit_m_s"]["y"],
                block["state_limit_m_s"]["z"],
            ],
            dtype=float,
        )

    error = np.array([-0.05, 0.0, 0.0], dtype=float)
    velocity = np.zeros(3, dtype=float)
    errors = []
    controls = []
    solve_times = []
    for _ in range(STEPS):
        state = np.concatenate((error, velocity, integral)) if augmented else np.concatenate((error, velocity))
        result = solver.solve(state)
        if not result["success"]:
            raise RuntimeError("closed_loop_solver_not_converged")
        control = np.asarray(result["control"], dtype=float)
        if not np.all(np.isfinite(control)):
            raise RuntimeError("closed_loop_control_nonfinite")
        errors.append(error.copy())
        controls.append(control.copy())
        solve_times.append(float(result["solve_time_seconds"]))
        if augmented:
            integral = np.clip(integral + error * dt, -limits, limits)
        acceleration = control + disturbance
        error = error + dt * velocity + 0.5 * dt * dt * acceleration
        velocity = velocity + dt * acceleration

    error_array = np.asarray(errors)
    control_array = np.asarray(controls)
    metrics = {
        "final_error_xyz_m": error.tolist(),
        "rmse_error_xyz_m": np.sqrt(np.mean(error_array * error_array, axis=0)).tolist(),
        "first_control_xyz_mps2": control_array[0].tolist(),
        "max_abs_control_xyz_mps2": np.max(np.abs(control_array), axis=0).tolist(),
    }
    return metrics, error_array


def main() -> int:
    base = _controller_config(BASE_CONFIG)
    augmented = _controller_config(AUGMENTED_CONFIG)
    disturbance = np.array([0.02, 0.03, -0.03], dtype=float)
    base_metrics, _ = _simulate(base, disturbance)
    augmented_metrics, _ = _simulate(augmented, disturbance)
    zero_disturbance_metrics, zero_errors = _simulate(augmented, np.zeros(3))

    base_final = np.abs(np.asarray(base_metrics["final_error_xyz_m"]))
    aug_final = np.abs(np.asarray(augmented_metrics["final_error_xyz_m"]))
    x_overshoot = max(0.0, float(np.max(zero_errors[:, 0])))
    gates = {
        "augmented_final_y_below_0_01": bool(aug_final[1] < 0.01),
        "augmented_final_z_below_0_01": bool(aug_final[2] < 0.01),
        "y_reduction_at_least_50_percent": bool(aug_final[1] <= 0.5 * base_final[1]),
        "z_reduction_at_least_50_percent": bool(aug_final[2] <= 0.5 * base_final[2]),
        "x_not_worse_by_more_than_0_003": bool(aug_final[0] <= base_final[0] + 0.003),
        "zero_disturbance_x_final_below_0_005": bool(abs(zero_disturbance_metrics["final_error_xyz_m"][0]) < 0.005),
        "zero_disturbance_x_overshoot_below_0_015": bool(x_overshoot < 0.015),
    }
    report = {
        "six_state": base_metrics,
        "nine_state_augmented": augmented_metrics,
        "nine_state_zero_disturbance_x_step": {
            **zero_disturbance_metrics,
            "max_x_overshoot_m": x_overshoot,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
