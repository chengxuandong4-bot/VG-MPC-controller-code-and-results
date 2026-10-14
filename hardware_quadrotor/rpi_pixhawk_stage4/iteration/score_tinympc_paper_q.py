"""Score baseline and iter-200 TinyMPC configs with the paper drone q formula."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

from .paper_q_score import PAPER_NORMALIZERS, PAPER_WEIGHTS
from .run_tinympc_200_iterations import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    SCENARIOS,
    _read_json,
    _reference_arrays,
    _scenario_metrics,
    _write_json,
)
from .validate_tinympc_candidate import _assert_fail_closed_delivery


DEFAULT_CANDIDATE = (
    PROJECT_ROOT
    / "config"
    / "alpha_stage5_tinympc_xy_figure8_r1m_146s_augint_iter200_offline.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "iteration_results"
    / "tinympc_146s_iter200"
    / "paper_q_score_validation.json"
)


def _score_config(config: Dict[str, Any], references: Any) -> Dict[str, Any]:
    scenarios = [
        _scenario_metrics(config["controller"], references, scenario)
        for scenario in SCENARIOS
    ]
    q_values = np.asarray(
        [item["paper_q_compat"]["q_score"] for item in scenarios],
        dtype=float,
    )
    return {
        "accepted_scenario_count": int(
            sum(bool(item.get("gate_passed")) for item in scenarios)
        ),
        "primary_nominal_q": float(q_values[0]),
        "robust_mean_q": float(np.mean(q_values)),
        "robust_worst_q": float(np.min(q_values)),
        "scenarios": [
            {
                "name": item["name"],
                "gate_passed": bool(item.get("gate_passed")),
                "q_score": float(item["paper_q_compat"]["q_score"]),
                "s_score": float(item["paper_q_compat"]["s_score"]),
                "normalized_cost": float(
                    item["paper_q_compat"]["normalized_cost"]
                ),
                "metrics": item["paper_q_compat"]["metrics"],
            }
            for item in scenarios
        ],
    }


def _improvement(baseline: float, candidate: float) -> Dict[str, float]:
    return {
        "absolute": float(candidate - baseline),
        "relative": float((candidate - baseline) / baseline),
    }


def validate(baseline_path: Path, candidate_path: Path) -> Dict[str, Any]:
    baseline = _read_json(baseline_path)
    candidate = _read_json(candidate_path)
    _assert_fail_closed_delivery(baseline, candidate)
    references = _reference_arrays(baseline)
    baseline_score = _score_config(baseline, references)
    candidate_score = _score_config(candidate, references)
    scenario_count = len(SCENARIOS)
    return {
        "score_spec": {
            "adapter_version": "paper_drone_q_v1_9state_hover_lift",
            "formula": "q=s-0.5v; s=1/(1+sum(w_j*m_j/n_j))",
            "weights": PAPER_WEIGHTS,
            "normalizers": PAPER_NORMALIZERS,
            "primary_score_policy": "nominal_scenario_q",
            "integral_duration_policy": "10_second_equivalent",
            "legacy_formula_and_constants_unchanged": True,
            "absolute_q_directly_comparable_to_legacy_12state_runs": False,
            "baseline_relative_q_comparable_within_this_9state_benchmark": True,
        },
        "scenario_count": scenario_count,
        "baseline": baseline_score,
        "candidate": candidate_score,
        "improvement": {
            "primary_nominal_q": _improvement(
                baseline_score["primary_nominal_q"],
                candidate_score["primary_nominal_q"],
            ),
            "robust_mean_q": _improvement(
                baseline_score["robust_mean_q"],
                candidate_score["robust_mean_q"],
            ),
            "robust_worst_q": _improvement(
                baseline_score["robust_worst_q"],
                candidate_score["robust_worst_q"],
            ),
        },
        "passed": bool(
            baseline_score["accepted_scenario_count"] == scenario_count
            and candidate_score["accepted_scenario_count"] == scenario_count
            and candidate_score["primary_nominal_q"]
            > baseline_score["primary_nominal_q"]
            and candidate_score["robust_mean_q"] > baseline_score["robust_mean_q"]
            and candidate_score["robust_worst_q"] > baseline_score["robust_worst_q"]
        ),
    }


def main(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=str(DEFAULT_CONFIG))
    parser.add_argument("--candidate", default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(list(argv) if argv else None)
    report = validate(Path(args.baseline), Path(args.candidate))
    _write_json(Path(args.output), report)
    compact = {
        "baseline_primary_nominal_q": report["baseline"]["primary_nominal_q"],
        "candidate_primary_nominal_q": report["candidate"]["primary_nominal_q"],
        "primary_relative_improvement": report["improvement"]["primary_nominal_q"]["relative"],
        "baseline_robust_mean_q": report["baseline"]["robust_mean_q"],
        "candidate_robust_mean_q": report["candidate"]["robust_mean_q"],
        "robust_mean_relative_improvement": report["improvement"]["robust_mean_q"]["relative"],
        "passed": report["passed"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
