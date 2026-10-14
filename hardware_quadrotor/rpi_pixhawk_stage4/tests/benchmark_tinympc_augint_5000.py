"""Smooth 5000-solve numerical benchmark for the augmented Stage 5 solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rpi_pixhawk_stage4.controller.tinympc_solver import CondensedTinyMPCSolver


STAGE4_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = STAGE4_ROOT / "config" / "alpha_stage5_tinympc_x5cm_n20_augint.json"
DEFAULT_SAMPLES = 5000


def run_benchmark(config_path=DEFAULT_CONFIG, samples=DEFAULT_SAMPLES):
    with Path(config_path).open("r", encoding="utf-8-sig") as handle:
        config = json.load(handle)["controller"]
    solver = CondensedTinyMPCSolver(config)
    solve_times = []
    nonconverged = 0
    maximum_iterations = 0
    maximum_residual = 0.0
    for index in range(int(samples)):
        phase = 2.0 * np.pi * float(index) / 500.0
        state = np.array(
            [
                0.025 * np.sin(phase),
                0.020 * np.sin(phase + 0.7),
                0.015 * np.sin(phase + 1.4),
                0.030 * np.cos(phase),
                0.025 * np.cos(phase + 0.7),
                0.020 * np.cos(phase + 1.4),
                0.040 * np.sin(phase + 0.2),
                0.050 * np.sin(phase + 0.9),
                0.045 * np.sin(phase + 1.6),
            ],
            dtype=float,
        )
        result = solver.solve(state)
        solve_times.append(float(result["solve_time_seconds"]))
        if not result["success"]:
            nonconverged += 1
        maximum_iterations = max(maximum_iterations, int(result["iterations"]))
        maximum_residual = max(maximum_residual, float(result["residual"]))

    times = np.asarray(solve_times, dtype=float)
    over_limit = int(np.count_nonzero(times > 0.020))
    report = {
        "config": str(Path(config_path).resolve()),
        "samples": int(samples),
        "success_count": int(samples) - nonconverged,
        "nonconverged_count": nonconverged,
        "over_20ms_count": over_limit,
        "max_iterations": int(config["max_iterations"]),
        "observed_max_iterations": maximum_iterations,
        "max_residual": maximum_residual,
        "mean_solve_time_seconds": float(np.mean(times)),
        "p99_solve_time_seconds": float(np.percentile(times, 99.0)),
        "max_solve_time_seconds": float(np.max(times)),
        "passed": bool(
            nonconverged == 0
            and over_limit == 0
            and float(np.percentile(times, 99.0)) < 0.005
        ),
    }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    args = parser.parse_args(argv)
    report = run_benchmark(args.config, args.samples)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
