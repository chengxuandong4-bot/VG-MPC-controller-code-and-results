"""Deterministic 200-candidate robust tuning for the 146 s Figure-8 controller.

The flight configuration is never edited in place.  Candidates are evaluated
against the same 9-state condensed solver used by the flight source, while all
publication, safety-envelope, and flight-handover settings remain frozen.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from rpi_pixhawk_stage4.controller.tinympc_solver import (  # noqa: E402
    CondensedTinyMPCSolver,
    TinyMPCSolverError,
)
from rpi_pixhawk_stage4.iteration.paper_q_score import (  # noqa: E402
    trajectory_paper_q,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "alpha_stage5_tinympc_xy_figure8_r1m_146s_augint.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "iteration_results" / "tinympc_146s_iter200"
DEFAULT_BUDGET = 200
SEED = 20260811


@dataclass(frozen=True)
class Scenario:
    name: str
    gain: Tuple[float, float, float]
    lag_seconds: float
    delay_steps: int
    bias_mps2: Tuple[float, float, float]
    drag_per_second: Tuple[float, float, float]
    initial_error_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_velocity_mps: Tuple[float, float, float] = (0.0, 0.0, 0.0)


SCENARIOS = (
    Scenario(
        "nominal",
        gain=(1.0, 1.0, 1.0),
        lag_seconds=0.0,
        delay_steps=0,
        bias_mps2=(0.0, 0.0, 0.0),
        drag_per_second=(0.0, 0.0, 0.0),
    ),
    Scenario(
        "persistent_positive_disturbance",
        gain=(1.0, 1.0, 1.0),
        lag_seconds=0.10,
        delay_steps=1,
        bias_mps2=(0.020, 0.030, -0.030),
        drag_per_second=(0.03, 0.03, 0.02),
    ),
    Scenario(
        "persistent_negative_disturbance",
        gain=(1.0, 1.0, 1.0),
        lag_seconds=0.10,
        delay_steps=1,
        bias_mps2=(-0.020, -0.030, 0.030),
        drag_per_second=(0.03, 0.03, 0.02),
    ),
    Scenario(
        "weak_y_axis",
        gain=(0.88, 0.68, 0.85),
        lag_seconds=0.18,
        delay_steps=1,
        bias_mps2=(0.004, 0.015, -0.010),
        drag_per_second=(0.06, 0.08, 0.05),
        initial_error_m=(0.015, -0.025, 0.010),
    ),
    Scenario(
        "delayed_underactuated",
        gain=(0.78, 0.74, 0.82),
        lag_seconds=0.25,
        delay_steps=2,
        bias_mps2=(0.010, -0.012, 0.008),
        drag_per_second=(0.08, 0.08, 0.06),
        initial_error_m=(0.025, -0.030, 0.020),
        initial_velocity_mps=(0.010, -0.008, 0.005),
    ),
)


# All positive continuous parameters are searched in log space.  Bounds are
# deliberately conservative around the flown controller; safety/output limits
# are not tunable.
CONTINUOUS_BOUNDS = (
    ("q_pos_x", 8.0, 35.0),
    ("q_pos_y", 8.0, 45.0),
    ("q_pos_z", 15.0, 45.0),
    ("q_vel_x", 2.0, 10.0),
    ("q_vel_y", 2.0, 12.0),
    ("q_vel_z", 3.0, 14.0),
    ("q_int_x", 0.20, 6.0),
    ("q_int_y", 3.0, 22.0),
    ("q_int_z", 5.0, 24.0),
    ("r_x", 1.0, 6.0),
    ("r_y", 1.0, 6.0),
    ("r_z", 1.5, 8.0),
    ("i_limit_x", 0.060, 0.140),
    ("i_limit_y", 0.100, 0.180),
    ("i_limit_z", 0.080, 0.140),
)
HORIZON_BOUNDS = (14, 20)
PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47)


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("config_root_not_object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _reference_arrays(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    controller = config["controller"]
    dt = float(controller["dt_seconds"])
    profile = controller["reference_profile"]
    active_window = float(profile["active_window_seconds"])
    steps = int(round(active_window / dt))
    times = np.arange(steps, dtype=float) * dt
    waypoints = profile["waypoints"]
    waypoint_times = np.asarray([float(item["time_s"]) for item in waypoints])
    waypoint_positions = np.asarray([item["offset_m"] for item in waypoints], dtype=float)
    positions = np.column_stack(
        [np.interp(times, waypoint_times, waypoint_positions[:, axis]) for axis in range(3)]
    )
    velocities = np.zeros_like(positions)
    segment = 0
    for index, timestamp in enumerate(times):
        while segment + 1 < len(waypoint_times) - 1 and timestamp >= waypoint_times[segment + 1]:
            segment += 1
        if timestamp < waypoint_times[-1]:
            duration = waypoint_times[segment + 1] - waypoint_times[segment]
            velocities[index] = (
                waypoint_positions[segment + 1] - waypoint_positions[segment]
            ) / duration
    return times, positions, velocities


def _controller_parameters(controller: Dict[str, Any]) -> Dict[str, float]:
    q = [float(value) for value in controller["state_weights"]]
    r = [float(value) for value in controller["input_weights"]]
    limits = controller["augmented_integral"]["state_limit_m_s"]
    return {
        "q_pos_x": q[0],
        "q_pos_y": q[1],
        "q_pos_z": q[2],
        "q_vel_x": q[3],
        "q_vel_y": q[4],
        "q_vel_z": q[5],
        "q_int_x": q[6],
        "q_int_y": q[7],
        "q_int_z": q[8],
        "r_x": r[0],
        "r_y": r[1],
        "r_z": r[2],
        "i_limit_x": float(limits["x"]),
        "i_limit_y": float(limits["y"]),
        "i_limit_z": float(limits["z"]),
        "horizon_steps": int(controller["horizon_steps"]),
    }


def _unit_vector(parameters: Dict[str, float]) -> np.ndarray:
    values = []
    for name, lower, upper in CONTINUOUS_BOUNDS:
        value = min(upper, max(lower, float(parameters[name])))
        values.append((math.log(value) - math.log(lower)) / (math.log(upper) - math.log(lower)))
    return np.asarray(values, dtype=float)


def _parameters_from_unit(unit: Sequence[float], horizon_steps: int) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for coordinate, (name, lower, upper) in zip(unit, CONTINUOUS_BOUNDS):
        clipped = min(1.0, max(0.0, float(coordinate)))
        result[name] = math.exp(math.log(lower) + clipped * (math.log(upper) - math.log(lower)))
    result["horizon_steps"] = int(min(HORIZON_BOUNDS[1], max(HORIZON_BOUNDS[0], horizon_steps)))
    return result


def _candidate_config(base: Dict[str, Any], parameters: Dict[str, float]) -> Dict[str, Any]:
    candidate = copy.deepcopy(base)
    controller = candidate["controller"]
    controller["state_weights"] = [
        parameters["q_pos_x"],
        parameters["q_pos_y"],
        parameters["q_pos_z"],
        parameters["q_vel_x"],
        parameters["q_vel_y"],
        parameters["q_vel_z"],
        parameters["q_int_x"],
        parameters["q_int_y"],
        parameters["q_int_z"],
    ]
    controller["input_weights"] = [parameters["r_x"], parameters["r_y"], parameters["r_z"]]
    controller["horizon_steps"] = int(parameters["horizon_steps"])
    controller["augmented_integral"]["state_limit_m_s"] = {
        "x": parameters["i_limit_x"],
        "y": parameters["i_limit_y"],
        "z": parameters["i_limit_z"],
    }
    return candidate


def _halton(index: int, base: int) -> float:
    fraction = 1.0
    value = 0.0
    current = int(index)
    while current > 0:
        fraction /= float(base)
        value += fraction * float(current % base)
        current //= base
    return value


def _next_parameters(
    generation: int,
    baseline_unit: np.ndarray,
    incumbent_unit: np.ndarray,
    incumbent_horizon: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    dimension = len(CONTINUOUS_BOUNDS)
    if generation <= 48:
        global_point = np.asarray(
            [_halton(generation + 7, prime) for prime in PRIMES[:dimension]],
            dtype=float,
        )
        unit = np.clip(0.55 * baseline_unit + 0.45 * global_point, 0.0, 1.0)
        horizon = HORIZON_BOUNDS[0] + ((generation - 1) % (HORIZON_BOUNDS[1] - HORIZON_BOUNDS[0] + 1))
    elif generation <= 168:
        progress = float(generation - 49) / float(168 - 49)
        sigma = 0.18 * (1.0 - progress) + 0.045 * progress
        mask = rng.random(dimension) < (0.55 + 0.25 * progress)
        if not np.any(mask):
            mask[generation % dimension] = True
        perturbation = rng.normal(0.0, sigma, size=dimension) * mask
        unit = np.clip(incumbent_unit + perturbation, 0.0, 1.0)
        horizon = incumbent_horizon + int(rng.choice(np.asarray([-2, -1, 0, 0, 1])))
    else:
        offset = generation - 169
        coordinate = (offset // 2) % dimension
        direction = -1.0 if offset % 2 == 0 else 1.0
        step = 0.035 if offset < 2 * dimension else 0.018
        unit = incumbent_unit.copy()
        unit[coordinate] = np.clip(unit[coordinate] + direction * step, 0.0, 1.0)
        if generation in (199, 200):
            horizon = incumbent_horizon + (-1 if generation == 199 else 1)
        else:
            horizon = incumbent_horizon
    return _parameters_from_unit(unit, horizon)


def _scenario_metrics(
    controller: Dict[str, Any],
    references: Tuple[np.ndarray, np.ndarray, np.ndarray],
    scenario: Scenario,
) -> Dict[str, Any]:
    times, reference_position, reference_velocity = references
    dt = float(controller["dt_seconds"])
    position_limits = np.asarray(
        [
            controller["safety"]["max_position_error_m"][axis]
            for axis in ("x", "y", "z")
        ],
        dtype=float,
    )
    speed_limits = np.asarray(
        [
            controller["safety"]["max_horizontal_velocity_mps"],
            controller["safety"]["max_horizontal_velocity_mps"],
            controller["safety"]["max_vertical_velocity_mps"],
        ],
        dtype=float,
    )
    integral_limits = np.asarray(
        [controller["augmented_integral"]["state_limit_m_s"][axis] for axis in ("x", "y", "z")],
        dtype=float,
    )
    gain = np.asarray(scenario.gain, dtype=float)
    bias = np.asarray(scenario.bias_mps2, dtype=float)
    drag = np.asarray(scenario.drag_per_second, dtype=float)
    position = reference_position[0].copy() + np.asarray(scenario.initial_error_m, dtype=float)
    velocity = np.asarray(scenario.initial_velocity_mps, dtype=float).copy()
    acceleration = np.zeros(3, dtype=float)
    integral = np.zeros(3, dtype=float)
    delayed_controls: List[np.ndarray] = [
        np.zeros(3, dtype=float) for _ in range(scenario.delay_steps)
    ]
    errors: List[np.ndarray] = []
    velocity_errors: List[np.ndarray] = []
    controls: List[np.ndarray] = []
    achieved_accelerations: List[np.ndarray] = []
    iterations: List[int] = []
    residuals: List[float] = []
    solve_times: List[float] = []
    solver = CondensedTinyMPCSolver(controller)
    failure = None

    for index, _timestamp in enumerate(times):
        error = position - reference_position[index]
        velocity_error = velocity - reference_velocity[index]
        horizontal_speed = math.hypot(float(velocity[0]), float(velocity[1]))
        if np.any(np.abs(error) > position_limits):
            failure = "position_envelope"
            break
        if horizontal_speed > float(speed_limits[0]) or abs(float(velocity[2])) > float(speed_limits[2]):
            failure = "velocity_envelope"
            break
        state = np.concatenate((error, velocity_error, integral))
        try:
            solution = solver.solve(state)
        except TinyMPCSolverError as exc:
            failure = "solver_error:{0}".format(exc)
            break
        if not solution["success"]:
            failure = "solver_nonconvergence"
            break
        control = np.asarray(solution["control"], dtype=float)
        if not np.all(np.isfinite(control)):
            failure = "control_nonfinite"
            break
        errors.append(error.copy())
        velocity_errors.append(velocity_error.copy())
        controls.append(control.copy())
        iterations.append(int(solution["iterations"]))
        residuals.append(float(solution["residual"]))
        solve_times.append(float(solution["solve_time_seconds"]))
        integral = np.clip(integral + error * dt, -integral_limits, integral_limits)
        if scenario.delay_steps:
            delayed_controls.append(control.copy())
            applied_control = delayed_controls.pop(0)
        else:
            applied_control = control
        target_acceleration = gain * applied_control + bias - drag * velocity
        if scenario.lag_seconds <= 0.0:
            acceleration = target_acceleration
        else:
            alpha = 1.0 - math.exp(-dt / scenario.lag_seconds)
            acceleration = acceleration + alpha * (target_acceleration - acceleration)
        achieved_accelerations.append(acceleration.copy())
        position = position + dt * velocity + 0.5 * dt * dt * acceleration
        velocity = velocity + dt * acceleration

    completed = failure is None and len(errors) == len(times)
    if not errors:
        return {
            "name": scenario.name,
            "completed": False,
            "failure": failure or "no_steps",
            "loss": 1.0e6,
            "steps": 0,
        }

    error_array = np.asarray(errors)
    velocity_error_array = np.asarray(velocity_errors)
    control_array = np.asarray(controls)
    normalized_error = error_array / position_limits
    normalized_velocity = velocity_error_array / speed_limits
    normalized_control = control_array / solver.axis_limits
    delta_control = np.diff(normalized_control, axis=0) if len(control_array) > 1 else np.zeros((1, 3))
    peak_error_ratio_xyz = np.max(np.abs(normalized_error), axis=0)
    peak_horizontal_speed = float(np.max(np.linalg.norm(velocity_error_array[:, :2] + reference_velocity[: len(errors), :2], axis=1)))
    peak_vertical_speed = float(np.max(np.abs(velocity_error_array[:, 2] + reference_velocity[: len(errors), 2])))
    saturation_rate = float(np.mean(np.abs(normalized_control) >= 0.995))
    rmse_error_xyz = np.sqrt(np.mean(error_array * error_array, axis=0))
    terminal_error_xyz = np.abs(error_array[-1])
    position_rmse_normalized = float(np.sqrt(np.mean(normalized_error * normalized_error)))
    velocity_rmse_normalized = float(np.sqrt(np.mean(normalized_velocity * normalized_velocity)))
    control_rms_normalized = float(np.sqrt(np.mean(normalized_control * normalized_control)))
    control_delta_rms = float(np.sqrt(np.mean(delta_control * delta_control)))
    peak_error_normalized = float(np.max(peak_error_ratio_xyz))
    terminal_error_normalized = float(np.mean(terminal_error_xyz / position_limits))
    workload = float(np.mean(iterations)) / float(controller["max_iterations"])
    loss = (
        0.34 * position_rmse_normalized
        + 0.12 * velocity_rmse_normalized
        + 0.18 * peak_error_normalized
        + 0.08 * terminal_error_normalized
        + 0.08 * control_rms_normalized
        + 0.06 * control_delta_rms
        + 0.06 * saturation_rate
        + 0.08 * workload
    )
    if not completed:
        completion_ratio = float(len(errors)) / float(len(times))
        loss += 10.0 + 10.0 * (1.0 - completion_ratio)

    gate_passed = bool(
        completed
        and peak_error_ratio_xyz[0] <= 0.80
        and peak_error_ratio_xyz[1] <= 0.80
        and peak_error_ratio_xyz[2] <= 0.80
        and peak_horizontal_speed <= 0.90 * float(speed_limits[0])
        and peak_vertical_speed <= 0.90 * float(speed_limits[2])
        and saturation_rate <= 0.20
        and max(residuals) <= 1.05 * float(controller["residual_tolerance"])
    )
    paper_step_count = min(
        len(error_array),
        len(achieved_accelerations),
        len(control_array),
        len(solve_times),
    )
    paper_q_compat = None
    if paper_step_count:
        paper_q_compat = trajectory_paper_q(
            position_errors_m=error_array[:paper_step_count],
            achieved_acceleration_mps2=np.asarray(
                achieved_accelerations[:paper_step_count]
            ),
            commanded_acceleration_mps2=control_array[:paper_step_count],
            solve_times_seconds=solve_times[:paper_step_count],
            dt_seconds=dt,
            acceleration_limits_mps2=solver.axis_limits,
        )
    return {
        "name": scenario.name,
        "completed": completed,
        "gate_passed": gate_passed,
        "failure": failure,
        "steps": len(errors),
        "loss": loss,
        "rmse_error_xyz_m": rmse_error_xyz.tolist(),
        "peak_error_ratio_xyz": peak_error_ratio_xyz.tolist(),
        "terminal_abs_error_xyz_m": terminal_error_xyz.tolist(),
        "peak_horizontal_speed_mps": peak_horizontal_speed,
        "peak_vertical_speed_mps": peak_vertical_speed,
        "control_rms_normalized": control_rms_normalized,
        "control_delta_rms_normalized": control_delta_rms,
        "saturation_rate": saturation_rate,
        "mean_iterations": float(np.mean(iterations)),
        "max_iterations": int(max(iterations)),
        "max_residual": float(max(residuals)),
        "mean_solve_time_seconds": float(np.mean(solve_times)),
        "max_solve_time_seconds": float(max(solve_times)),
        "paper_q_compat": paper_q_compat,
    }


def evaluate_candidate(
    candidate: Dict[str, Any],
    references: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Dict[str, Any]:
    controller = candidate["controller"]
    scenario_results = [
        _scenario_metrics(controller, references, scenario) for scenario in SCENARIOS
    ]
    losses = [float(result["loss"]) for result in scenario_results]
    all_gates_passed = all(bool(result.get("gate_passed")) for result in scenario_results)
    aggregate_loss = float(np.mean(losses) + 0.35 * np.max(losses))
    workload_proxy = float(
        np.mean([result.get("mean_iterations", controller["max_iterations"]) for result in scenario_results])
        * int(controller["horizon_steps"])
    )
    return {
        "accepted_by_gates": all_gates_passed,
        "aggregate_loss": aggregate_loss,
        "workload_proxy": workload_proxy,
        "scenarios": scenario_results,
    }


def _frozen_fields(config: Dict[str, Any]) -> Dict[str, Any]:
    frozen = copy.deepcopy(config)
    controller = frozen["controller"]
    controller.pop("state_weights")
    controller.pop("input_weights")
    controller.pop("horizon_steps")
    controller["augmented_integral"] = dict(controller["augmented_integral"])
    controller["augmented_integral"].pop("state_limit_m_s")
    return frozen


def _assert_frozen(base: Dict[str, Any], candidate: Dict[str, Any]) -> None:
    if _frozen_fields(base) != _frozen_fields(candidate):
        raise RuntimeError("candidate_changed_frozen_flight_field")


def run(config_path: Path, output_dir: Path, budget: int = DEFAULT_BUDGET) -> Dict[str, Any]:
    if budget != DEFAULT_BUDGET:
        raise ValueError("iteration_budget_must_equal_200")
    base = _read_json(config_path)
    references = _reference_arrays(base)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "iterations.jsonl"
    if history_path.exists():
        raise FileExistsError("iteration_output_already_exists:{0}".format(output_dir))

    baseline_parameters = _controller_parameters(base["controller"])
    baseline_unit = _unit_vector(baseline_parameters)
    started = time.perf_counter()
    baseline_evaluation = evaluate_candidate(base, references)
    incumbent_config = copy.deepcopy(base)
    incumbent_parameters = dict(baseline_parameters)
    incumbent_unit = baseline_unit.copy()
    incumbent_horizon = int(baseline_parameters["horizon_steps"])
    incumbent_evaluation = baseline_evaluation
    baseline_record = {
        "generation": 0,
        "kind": "baseline",
        "parameters": baseline_parameters,
        "evaluation": baseline_evaluation,
        "updated_incumbent": True,
    }
    _append_jsonl(history_path, baseline_record)
    _write_json(output_dir / "baseline_evaluation.json", baseline_record)
    rng = np.random.default_rng(SEED)
    improvement_count = 0

    print(
        "[000/{0}] baseline loss={1:.8f} gates={2}".format(
            budget,
            baseline_evaluation["aggregate_loss"],
            baseline_evaluation["accepted_by_gates"],
        ),
        flush=True,
    )
    for generation in range(1, budget + 1):
        parameters = _next_parameters(
            generation,
            baseline_unit,
            incumbent_unit,
            incumbent_horizon,
            rng,
        )
        candidate = _candidate_config(base, parameters)
        _assert_frozen(base, candidate)
        evaluation = evaluate_candidate(candidate, references)
        better = bool(
            evaluation["accepted_by_gates"]
            and (
                not incumbent_evaluation["accepted_by_gates"]
                or evaluation["aggregate_loss"] < incumbent_evaluation["aggregate_loss"] - 1.0e-12
            )
        )
        if better:
            incumbent_config = candidate
            incumbent_parameters = parameters
            incumbent_unit = _unit_vector(parameters)
            incumbent_horizon = int(parameters["horizon_steps"])
            incumbent_evaluation = evaluation
            improvement_count += 1
        record = {
            "generation": generation,
            "kind": "candidate",
            "parameters": parameters,
            "evaluation": evaluation,
            "updated_incumbent": better,
            "incumbent_loss": incumbent_evaluation["aggregate_loss"],
        }
        _append_jsonl(history_path, record)
        if better or generation % 10 == 0 or generation == budget:
            marker = " NEW_BEST" if better else ""
            print(
                "[{0:03d}/{1}] loss={2:.8f} gates={3} best={4:.8f}{5}".format(
                    generation,
                    budget,
                    evaluation["aggregate_loss"],
                    evaluation["accepted_by_gates"],
                    incumbent_evaluation["aggregate_loss"],
                    marker,
                ),
                flush=True,
            )

    _assert_frozen(base, incumbent_config)
    best_config_path = output_dir / "alpha_stage5_tinympc_xy_figure8_r1m_146s_augint_iter200.json"
    delivery_config = copy.deepcopy(incumbent_config)
    delivery_config["audit_only"] = True
    delivery_config["actual_publish_enabled"] = False
    _write_json(best_config_path, delivery_config)
    summary = {
        "budget": budget,
        "seed": SEED,
        "scenario_count": len(SCENARIOS),
        "evaluated_candidates": budget,
        "improvement_count": improvement_count,
        "elapsed_seconds": time.perf_counter() - started,
        "baseline_parameters": baseline_parameters,
        "best_parameters": incumbent_parameters,
        "baseline_evaluation": baseline_evaluation,
        "best_evaluation": incumbent_evaluation,
        "absolute_loss_reduction": baseline_evaluation["aggregate_loss"]
        - incumbent_evaluation["aggregate_loss"],
        "relative_loss_reduction": (
            baseline_evaluation["aggregate_loss"] - incumbent_evaluation["aggregate_loss"]
        )
        / baseline_evaluation["aggregate_loss"],
        "source_config": str(config_path.resolve()),
        "best_config": str(best_config_path.resolve()),
        "frozen_fields_verified": True,
        "delivery_publish_disabled": True,
    }
    _write_json(output_dir / "summary.json", summary)
    print(
        "DONE candidates={0} improvements={1} relative_loss_reduction={2:.2%}".format(
            budget,
            improvement_count,
            summary["relative_loss_reduction"],
        ),
        flush=True,
    )
    return summary


def main(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    args = parser.parse_args(list(argv) if argv else None)
    run(Path(args.config), Path(args.output_dir), args.budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
