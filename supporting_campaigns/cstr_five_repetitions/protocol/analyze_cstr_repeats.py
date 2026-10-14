#!/usr/bin/env python3
"""Validate and summarize the five frozen CSTR repetitions."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent.parent
MODES = ("single", "dual_objective", "dual_strategy")


def read_run(rep: int, mode: str):
    run_dir = ROOT / f"rep_{rep:02d}" / mode
    with (run_dir / "summary_metrics.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    with (run_dir / "trajectory.csv").open(newline="", encoding="utf-8") as handle:
        trajectory = list(csv.DictReader(handle))
    return run_dir, summary, trajectory


def direct_best(rows, source=None):
    selected = rows if source is None else [row for row in rows if row["source"] == source]
    if not selected:
        return None
    return max(selected, key=lambda row: float(row["q_score"]))


def source_of_formal_best(summary, rows):
    generation = int(summary["best_generation"])
    if generation == 0:
        return "seed", ""
    best_q = float(summary["best_q"])
    candidates = [row for row in rows if int(row["generation"]) == generation]
    row = min(candidates, key=lambda item: abs(float(item["q_score"]) - best_q))
    if not math.isclose(float(row["q_score"]), best_q, rel_tol=1e-9, abs_tol=1e-10):
        raise RuntimeError(f"Cannot identify formal best source at generation {generation}")
    return row["source"], row["role"]


def describe(values):
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sd_sample": statistics.stdev(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    runs = []
    trajectories = {}
    summaries = {}

    for rep in range(1, 6):
        for mode in MODES:
            run_dir, summary, rows = read_run(rep, mode)
            if summary.get("status") != "completed":
                raise RuntimeError(f"Incomplete summary: {run_dir}")
            if len(rows) != 200 or int(summary["total_candidates"]) != 200:
                raise RuntimeError(f"Expected 200 candidates: {run_dir}")

            sources = Counter(row["source"] for row in rows)
            if sum(sources.values()) != 200:
                raise RuntimeError(f"Source counts do not sum to 200: {run_dir}")

            formal_source, formal_role = source_of_formal_best(summary, rows)
            api_best = direct_best(rows, "llm_api")
            fallback_best = direct_best(rows, "llm_fallback")
            trajectories[(rep, mode)] = rows
            summaries[(rep, mode)] = summary

            runs.append(
                {
                    "replication": rep,
                    "mode": mode,
                    "baseline_q": float(summary["baseline_q"]),
                    "formal_best_q": float(summary["best_q"]),
                    "formal_best_generation": int(summary["best_generation"]),
                    "formal_best_role": formal_role,
                    "formal_best_direct_source": formal_source,
                    "gain_pct": float(summary["gain_pct"]),
                    "direct_api_candidate_best_q": float(api_best["q_score"]),
                    "direct_api_candidate_best_generation": int(api_best["generation"]),
                    "direct_api_candidate_best_role": api_best["role"],
                    "fallback_candidate_best_q": (
                        float(fallback_best["q_score"]) if fallback_best else ""
                    ),
                    "api_success_count": int(summary["api_success_count"]),
                    "fallback_count": int(summary["fallback_count"]),
                    "accepted_count": int(summary["accepted_count"]),
                    "updated_count": int(summary["updated_count"]),
                    "elapsed_s": float(summary["elapsed_s"]),
                    "seed_origin_q": summary.get("seed_origin_q", ""),
                    "seed_q_recomputed": summary.get("seed_q_recomputed", ""),
                    "seed_increment_pct": summary.get("seed_increment_pct", ""),
                    "seed_origin_source": "",
                    "fallback_in_lineage": False,
                }
            )

    by_key = {(row["replication"], row["mode"]): row for row in runs}
    for rep in range(1, 6):
        objective = by_key[(rep, "dual_objective")]
        strategy = by_key[(rep, "dual_strategy")]
        seed_source = objective["formal_best_direct_source"]
        strategy["seed_origin_source"] = seed_source
        strategy["fallback_in_lineage"] = seed_source == "llm_fallback"
        if strategy["formal_best_direct_source"] == "seed":
            strategy["formal_best_direct_source"] = f"seed_from_{seed_source}"
            strategy["fallback_in_lineage"] = seed_source == "llm_fallback"

    statistic_rows = []
    statistics_json = {}
    for mode in MODES:
        mode_runs = [row for row in runs if row["mode"] == mode]
        metrics = {
            "formal_best_q": [row["formal_best_q"] for row in mode_runs],
            "direct_api_candidate_best_q": [
                row["direct_api_candidate_best_q"] for row in mode_runs
            ],
            "gain_pct": [row["gain_pct"] for row in mode_runs],
            "api_success_rate": [row["api_success_count"] / 200 for row in mode_runs],
            "acceptance_rate": [row["accepted_count"] / 200 for row in mode_runs],
        }
        if mode == "dual_strategy":
            metrics["seed_increment_pct"] = [
                float(row["seed_increment_pct"]) for row in mode_runs
            ]
        statistics_json[mode] = {}
        for metric, values in metrics.items():
            stats = describe(values)
            statistics_json[mode][metric] = stats
            statistic_rows.append({"mode": mode, "metric": metric, **stats})

    run_fields = list(runs[0].keys())
    with (ROOT / "aggregate_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_fields)
        writer.writeheader()
        writer.writerows(runs)

    stat_fields = ["mode", "metric", "n", "mean", "sd_sample", "median", "min", "max"]
    with (ROOT / "aggregate_statistics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=stat_fields)
        writer.writeheader()
        writer.writerows(statistic_rows)

    paired_differences = {}
    for left, right, label in (
        ("single", "dual_objective", "dual_objective_minus_single"),
        ("dual_objective", "dual_strategy", "dual_strategy_minus_dual_objective"),
    ):
        values = [
            by_key[(rep, right)]["formal_best_q"]
            - by_key[(rep, left)]["formal_best_q"]
            for rep in range(1, 6)
        ]
        paired_differences[label] = {"values": values, **describe(values)}

    result = {
        "status": "completed_and_validated",
        "replications": 5,
        "modes": list(MODES),
        "formal_runs": 15,
        "candidates_per_run": 200,
        "total_formal_candidates": 3000,
        "runs": runs,
        "statistics": statistics_json,
        "paired_formal_best_differences": paired_differences,
        "interpretation_notes": [
            "Formal best follows the frozen legacy protocol, in which fallback candidates count and may update the incumbent.",
            "direct_api_candidate_best_q excludes fallback candidates and, for dual_strategy, excludes the warm-start seed.",
            "dual_strategy seed provenance is reported separately because an API-returned child can descend from a fallback-derived seed.",
            "Sample standard deviation uses n-1 in the denominator.",
        ],
    }
    with (ROOT / "aggregate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    lines = [
        "# CSTR × DeepSeek-v4-flash: five-repetition summary",
        "",
        "All 15 formal runs completed with 200 candidates each (3,000 candidates total).",
        "",
        "## Formal best score by repetition",
        "",
        "| Rep | Single | Dual-objective | Dual-strategy |",
        "|---:|---:|---:|---:|",
    ]
    for rep in range(1, 6):
        lines.append(
            f"| {rep} | {by_key[(rep, 'single')]['formal_best_q']:.6f} | "
            f"{by_key[(rep, 'dual_objective')]['formal_best_q']:.6f} | "
            f"{by_key[(rep, 'dual_strategy')]['formal_best_q']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Descriptive statistics for formal best score",
            "",
            "| Mode | Mean | SD | Median | Min | Max |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        stats = statistics_json[mode]["formal_best_q"]
        lines.append(
            f"| {mode} | {stats['mean']:.6f} | {stats['sd_sample']:.6f} | "
            f"{stats['median']:.6f} | {stats['min']:.6f} | {stats['max']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## API and acceptance accounting",
            "",
            "| Mode | API successes | Fallbacks | Accepted | Total |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        mode_runs = [row for row in runs if row["mode"] == mode]
        lines.append(
            f"| {mode} | {sum(row['api_success_count'] for row in mode_runs)} | "
            f"{sum(row['fallback_count'] for row in mode_runs)} | "
            f"{sum(row['accepted_count'] for row in mode_runs)} | 1000 |"
        )
    lines.extend(
        [
            "",
            "## Provenance caveat",
            "",
            "The frozen legacy protocol allows fallback candidates to update the incumbent. "
            "Therefore, the formal result and the best directly returned API candidate are both retained. "
            "See `aggregate_runs.csv` for per-run provenance and dual-strategy seed lineage.",
            "",
        ]
    )
    (ROOT / "experiment_results_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print("VALIDATED: 15 runs, 3000 candidates")
    for mode in MODES:
        stats = statistics_json[mode]["formal_best_q"]
        print(
            f"{mode}: mean={stats['mean']:.9f} sd={stats['sd_sample']:.9f} "
            f"median={stats['median']:.9f} min={stats['min']:.9f} max={stats['max']:.9f}"
        )


if __name__ == "__main__":
    main()
