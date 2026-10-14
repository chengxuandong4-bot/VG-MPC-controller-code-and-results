"""Deterministic 1000-solve readiness benchmark for Stage 5 TinyMPC."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from rpi_pixhawk_stage4.controller.tinympc_solver import CondensedTinyMPCSolver


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "alpha_stage5_tinympc_zero_motion.json"


def run_benchmark(config_path=DEFAULT_CONFIG, solve_count=1000):
    config = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    solver = CondensedTinyMPCSolver(config["controller"])
    durations_ms = []
    successes = 0
    finite_count = 0
    maximum_iterations = 0
    maximum_residual = 0.0
    for index in range(int(solve_count)):
        phase = 2.0 * math.pi * float(index) / 100.0
        state = np.array(
            [
                0.08 * math.sin(phase),
                0.08 * math.cos(phase),
                0.05 * math.sin(0.5 * phase),
                0.03 * math.cos(phase),
                -0.03 * math.sin(phase),
                0.02 * math.cos(0.5 * phase),
            ],
            dtype=float,
        )
        started = time.perf_counter()
        result = solver.solve(state)
        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        durations_ms.append(elapsed_ms)
        successes += int(bool(result["success"]))
        finite_count += int(
            np.all(np.isfinite(result["control"]))
            and math.isfinite(float(result["residual"]))
            and math.isfinite(float(elapsed_ms))
        )
        maximum_iterations = max(maximum_iterations, int(result["iterations"]))
        maximum_residual = max(maximum_residual, float(result["residual"]))

    values = np.asarray(durations_ms, dtype=float)
    metrics = {
        "solve_count": int(solve_count),
        "success_rate": float(successes) / float(solve_count),
        "finite_rate": float(finite_count) / float(solve_count),
        "mean_ms": float(np.mean(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(np.max(values)),
        "max_iterations": maximum_iterations,
        "max_residual": maximum_residual,
    }
    ready = (
        metrics["success_rate"] == 1.0
        and metrics["finite_rate"] == 1.0
        and metrics["p99_ms"] < 10.0
        and metrics["max_ms"] < 20.0
    )
    metrics["status"] = (
        "STAGE5_OFFLINE_READY_HARDWARE_TEST_PENDING" if ready else "STAGE5_NOT_READY"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--solve-count", type=int, default=1000)
    args = parser.parse_args()
    metrics = run_benchmark(args.config, args.solve_count)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["status"] == "STAGE5_OFFLINE_READY_HARDWARE_TEST_PENDING" else 1


if __name__ == "__main__":
    raise SystemExit(main())
