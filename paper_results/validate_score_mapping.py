#!/usr/bin/env python3
"""Check that retained full-precision best scores round to the paper table."""

from __future__ import annotations

import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def round_six(value: str) -> Decimal:
    return Decimal(value).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)


def main() -> None:
    paper_rows = load_csv("paper_reported_scores.csv")
    retained_rows = load_csv("original_campaign_manifest.csv")
    if len(paper_rows) != 36 or len(retained_rows) != 36:
        raise SystemExit(f"Expected 36 rows in each table, got {len(paper_rows)} and {len(retained_rows)}")

    retained = {
        (row["benchmark"], row["mode"], row["model"]): row for row in retained_rows
    }
    baseline_differences: list[tuple[Decimal, tuple[str, str, str]]] = []
    missing_retained_baselines: list[tuple[str, str, str]] = []
    for paper in paper_rows:
        key = (paper["benchmark"], paper["mode"], paper["model"])
        source = retained[key]
        paper_best = Decimal(paper["reported_best_q"])
        retained_best = round_six(source["best_q"])
        if retained_best != paper_best:
            raise SystemExit(f"Best-score mismatch for {key}: {retained_best} != {paper_best}")
        retained_baseline = Decimal(source["baseline_q_in_retained_summary"])
        if not retained_baseline.is_finite():
            missing_retained_baselines.append(key)
            continue
        baseline_difference = abs(retained_baseline - Decimal(paper["reported_baseline_q"]))
        if baseline_difference > Decimal("0.0000005"):
            baseline_differences.append((baseline_difference, key))

    snapshot_counts: dict[str, int] = {}
    for row in retained_rows:
        status = row["snapshot_status"]
        snapshot_counts[status] = snapshot_counts.get(status, 0) + 1

    print("best_score_mapping: PASS (36/36)")
    print(f"conditions_missing_retained_baseline: {len(missing_retained_baselines)}")
    print(f"conditions_with_baseline_field_difference: {len(baseline_differences)}")
    if baseline_differences:
        largest, key = max(baseline_differences)
        print(f"largest_baseline_field_difference: {largest} at {key}")
    for status, count in sorted(snapshot_counts.items()):
        print(f"snapshot_status[{status}]: {count}")


if __name__ == "__main__":
    main()
