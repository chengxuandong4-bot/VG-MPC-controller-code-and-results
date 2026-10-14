"""Compare baseline and candidate on deterministic unseen robustness cases."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from .run_tinympc_200_iterations import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    Scenario,
    _assert_frozen,
    _read_json,
    _reference_arrays,
    _scenario_metrics,
    _write_json,
)


DEFAULT_CANDIDATE = (
    PROJECT_ROOT
    / "iteration_results"
    / "tinympc_146s_iter200"
    / "alpha_stage5_tinympc_xy_figure8_r1m_146s_augint_iter200.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "iteration_results" / "tinympc_146s_iter200" / "holdout_validation.json"
)
HOLDOUT_SEED = 20260812


def _assert_fail_closed_delivery(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> None:
    if candidate.get("audit_only") is not True:
        raise RuntimeError("candidate_delivery_not_audit_only")
    if candidate.get("actual_publish_enabled") is not False:
        raise RuntimeError("candidate_delivery_publish_enabled")
    normalized = dict(candidate)
    normalized["audit_only"] = baseline["audit_only"]
    normalized["actual_publish_enabled"] = baseline["actual_publish_enabled"]
    _assert_frozen(baseline, normalized)


def holdout_scenarios(count: int = 24) -> List[Scenario]:
    rng = np.random.default_rng(HOLDOUT_SEED)
    result = []
    for index in range(count):
        result.append(
            Scenario(
                name="holdout_{0:02d}".format(index + 1),
                gain=(
                    float(rng.uniform(0.72, 0.98)),
                    float(rng.uniform(0.62, 0.95)),
                    float(rng.uniform(0.75, 0.98)),
                ),
                lag_seconds=float(rng.uniform(0.05, 0.30)),
                delay_steps=int(rng.integers(0, 3)),
                bias_mps2=(
                    float(rng.uniform(-0.022, 0.022)),
                    float(rng.uniform(-0.032, 0.032)),
                    float(rng.uniform(-0.025, 0.025)),
                ),
                drag_per_second=(
                    float(rng.uniform(0.02, 0.10)),
                    float(rng.uniform(0.02, 0.10)),
                    float(rng.uniform(0.02, 0.08)),
                ),
                initial_error_m=(
                    float(rng.uniform(-0.04, 0.04)),
                    float(rng.uniform(-0.04, 0.04)),
                    float(rng.uniform(-0.025, 0.025)),
                ),
                initial_velocity_mps=(
                    float(rng.uniform(-0.015, 0.015)),
                    float(rng.uniform(-0.015, 0.015)),
                    float(rng.uniform(-0.008, 0.008)),
                ),
            )
        )
    return result


def _compact(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        name: result.get(name)
        for name in (
            "name",
            "completed",
            "gate_passed",
            "failure",
            "loss",
            "rmse_error_xyz_m",
            "peak_error_ratio_xyz",
            "peak_horizontal_speed_mps",
            "peak_vertical_speed_mps",
            "saturation_rate",
            "mean_iterations",
            "max_iterations",
        )
    }


def validate(baseline_path: Path, candidate_path: Path, count: int = 24) -> Dict[str, Any]:
    baseline = _read_json(baseline_path)
    candidate = _read_json(candidate_path)
    _assert_fail_closed_delivery(baseline, candidate)
    references = _reference_arrays(baseline)
    scenarios = holdout_scenarios(count)
    baseline_results = [
        _scenario_metrics(baseline["controller"], references, scenario)
        for scenario in scenarios
    ]
    candidate_results = [
        _scenario_metrics(candidate["controller"], references, scenario)
        for scenario in scenarios
    ]
    baseline_losses = np.asarray([item["loss"] for item in baseline_results], dtype=float)
    candidate_losses = np.asarray([item["loss"] for item in candidate_results], dtype=float)
    baseline_gates = sum(bool(item.get("gate_passed")) for item in baseline_results)
    candidate_gates = sum(bool(item.get("gate_passed")) for item in candidate_results)
    improved_cases = int(np.count_nonzero(candidate_losses < baseline_losses))
    return {
        "seed": HOLDOUT_SEED,
        "scenario_count": count,
        "frozen_fields_verified": True,
        "baseline_gate_pass_count": baseline_gates,
        "candidate_gate_pass_count": candidate_gates,
        "baseline_mean_loss": float(np.mean(baseline_losses)),
        "candidate_mean_loss": float(np.mean(candidate_losses)),
        "relative_mean_loss_reduction": float(
            (np.mean(baseline_losses) - np.mean(candidate_losses)) / np.mean(baseline_losses)
        ),
        "candidate_improved_case_count": improved_cases,
        "candidate_not_worse_case_count": int(np.count_nonzero(candidate_losses <= baseline_losses)),
        "passed": bool(
            candidate_gates >= baseline_gates
            and candidate_gates == count
            and improved_cases >= int(math.ceil(0.75 * count))
            and float(np.mean(candidate_losses)) < float(np.mean(baseline_losses))
        ),
        "cases": [
            {
                "scenario": scenarios[index].name,
                "baseline": _compact(baseline_results[index]),
                "candidate": _compact(candidate_results[index]),
                "relative_loss_reduction": float(
                    (baseline_losses[index] - candidate_losses[index]) / baseline_losses[index]
                ),
            }
            for index in range(count)
        ],
    }


def main(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=str(DEFAULT_CONFIG))
    parser.add_argument("--candidate", default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--count", type=int, default=24)
    args = parser.parse_args(list(argv) if argv else None)
    report = validate(Path(args.baseline), Path(args.candidate), args.count)
    _write_json(Path(args.output), report)
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
