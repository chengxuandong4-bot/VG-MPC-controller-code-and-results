#!/usr/bin/env python3
"""Analyze and plot the retained 10-block paired UAV results."""

from __future__ import annotations

import csv
import itertools
import json
import math
import statistics
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "uav_ten_block_paired_results.csv"
PDF_PATH = ROOT / "uav_ten_block_paired_results.pdf"
PNG_PATH = ROOT / "uav_ten_block_paired_results.png"
STATS_PATH = ROOT / "uav_ten_block_statistics.json"

MODES = ("single", "dual_objective", "dual_strategy")
DISPLAY_MODES = ("Single", "Dual-objective", "Dual-strategy")
COLORS = ("#0077BB", "#EE7733", "#009988")
MARKERS = ("o", "s", "^")
PAIR_LINE = "#B8BDC2"
INK = "#202326"
GRID = "#D9DDE0"


def format_half_up(value: float, decimals: int) -> str:
    """Format reported decimals with conventional round-half-up behavior."""
    quantum = Decimal(1).scaleb(-decimals)
    rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
    return f"{rounded:.{decimals}f}"


def read_rows() -> list[list[float]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    return [[float(record[mode]) for mode in MODES] for record in records]


def average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        stop = start + 1
        while stop < len(indexed) and indexed[stop][1] == indexed[start][1]:
            stop += 1
        rank = (start + 1 + stop) / 2.0
        for position in range(start, stop):
            ranks[indexed[position][0]] = rank
        start = stop
    return ranks


def friedman_test(rows: list[list[float]]) -> tuple[float, float]:
    n = len(rows)
    k = len(rows[0])
    ranked_rows = [average_ranks(row) for row in rows]
    rank_sums = [sum(row[column] for row in ranked_rows) for column in range(k)]
    statistic = 12.0 / (n * k * (k + 1)) * sum(value**2 for value in rank_sums)
    statistic -= 3.0 * n * (k + 1)

    tie_sum = 0.0
    for row in rows:
        counts: dict[float, int] = {}
        for value in row:
            counts[value] = counts.get(value, 0) + 1
        tie_sum += sum(count**3 - count for count in counts.values())
    if tie_sum:
        statistic /= 1.0 - tie_sum / (n * (k**3 - k))

    # With three modes, the chi-square reference has two degrees of freedom.
    p_value = math.exp(-statistic / 2.0)
    return statistic, p_value


def exact_signed_rank(differences: list[float]) -> dict[str, float | int]:
    nonzero = [value for value in differences if value != 0.0]
    ranks = average_ranks([abs(value) for value in nonzero])
    w_positive = sum(rank for value, rank in zip(nonzero, ranks) if value > 0)
    w_negative = sum(rank for value, rank in zip(nonzero, ranks) if value < 0)
    rank_total = w_positive + w_negative
    observed_distance = abs(w_positive - rank_total / 2.0)

    extreme = 0
    for signs in itertools.product((False, True), repeat=len(nonzero)):
        permuted_w_positive = sum(rank for sign, rank in zip(signs, ranks) if sign)
        distance = abs(permuted_w_positive - rank_total / 2.0)
        if distance >= observed_distance - 1e-12:
            extreme += 1

    p_value = extreme / (2 ** len(nonzero))
    rank_biserial = (w_positive - w_negative) / rank_total
    return {
        "n_nonzero": len(nonzero),
        "w_positive": w_positive,
        "w_negative": w_negative,
        "p_raw": p_value,
        "rank_biserial": rank_biserial,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [0.0] * count
    running_maximum = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running_maximum = max(running_maximum, candidate)
        adjusted[index] = running_maximum
    return adjusted


def analyze(rows: list[list[float]]) -> dict[str, object]:
    columns = list(zip(*rows))
    summaries = {
        mode: {
            "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values),
        }
        for mode, values in zip(MODES, columns)
    }

    contrasts = {
        "dual_objective_minus_single": [row[1] - row[0] for row in rows],
        "dual_strategy_minus_single": [row[2] - row[0] for row in rows],
    }
    tests = [exact_signed_rank(values) for values in contrasts.values()]
    adjusted = holm_adjust([float(test["p_raw"]) for test in tests])

    contrast_summaries: dict[str, object] = {}
    for (name, values), test, p_holm in zip(contrasts.items(), tests, adjusted):
        contrast_summaries[name] = {
            "differences": values,
            "mean_difference": statistics.mean(values),
            "sample_sd_difference": statistics.stdev(values),
            "median_difference": statistics.median(values),
            "positive_blocks": sum(value > 0 for value in values),
            **test,
            "p_holm": p_holm,
        }

    friedman_statistic, friedman_p = friedman_test(rows)
    return {
        "n_paired_blocks": len(rows),
        "data_precision_decimals": 4,
        "mode_summaries": summaries,
        "friedman": {"df": 2, "statistic": friedman_statistic, "p": friedman_p},
        "prespecified_contrasts": contrast_summaries,
    }


def draw(rows: list[list[float]], statistics_record: dict[str, object]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    values = np.asarray(rows, dtype=float)
    x = np.arange(3, dtype=float)
    means = values.mean(axis=0)
    sample_sds = values.std(axis=0, ddof=1)

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 9.2,
            "axes.titlesize": 11.0,
            "axes.labelsize": 9.8,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.4,
            "axes.linewidth": 0.85,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
        }
    )

    fig, ax = plt.subplots(figsize=(6.85, 4.55), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for row in values:
        ax.plot(x, row, color=PAIR_LINE, linewidth=0.85, alpha=0.86, zorder=1)

    for mode_index, (color, marker) in enumerate(zip(COLORS, MARKERS)):
        ax.scatter(
            np.full(values.shape[0], mode_index),
            values[:, mode_index],
            s=30,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        ax.errorbar(
            mode_index,
            means[mode_index],
            yerr=sample_sds[mode_index],
            fmt="D",
            markersize=7.2,
            markerfacecolor=color,
            markeredgecolor=INK,
            markeredgewidth=0.75,
            ecolor=INK,
            elinewidth=1.25,
            capsize=4.5,
            capthick=1.15,
            zorder=5,
        )

    annotation_y = means + sample_sds + np.array([0.0040, 0.0030, 0.0030])
    for mode_index, (mean_value, y_value) in enumerate(zip(means, annotation_y)):
        ax.text(
            mode_index,
            y_value,
            f"mean = {format_half_up(mean_value, 4)}",
            ha="center",
            va="bottom",
            fontsize=8.4,
        )

    block_handle = Line2D(
        [0], [0], color=PAIR_LINE, marker="o", markerfacecolor="#777777",
        markeredgecolor="white", linewidth=0.85, markersize=4.5,
        label="Individual paired block",
    )
    mean_handle = Line2D(
        [0], [0], color=INK, marker="D", markerfacecolor="white",
        markeredgecolor=INK, linewidth=1.1, markersize=6.0,
        label="Mean $\\pm$ sample SD",
    )
    ax.legend(handles=[block_handle, mean_handle], frameon=False, loc="upper left")

    ax.set_title("UAV repeated-run comparison ($n=10$ paired blocks)", pad=9)
    ax.set_xlabel("Generation mode")
    ax.set_ylabel("Best accepted selection score, $q$")
    ax.set_xticks(x, DISPLAY_MODES)
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(0.592, 0.686)
    ax.set_yticks(np.arange(0.60, 0.681, 0.02))
    ax.grid(axis="y", color=GRID, linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3.2, width=0.8)

    break_style = {
        "transform": ax.transAxes,
        "color": INK,
        "clip_on": False,
        "linewidth": 1.05,
        "solid_capstyle": "round",
    }
    ax.plot((-0.012, 0.012), (-0.010, 0.010), **break_style)
    ax.plot((-0.012, 0.012), (0.012, 0.032), **break_style)
    ax.text(0.012, 0.018, "Truncated y-axis", transform=ax.transAxes, ha="left", va="bottom", fontsize=7.8)

    fig.tight_layout()
    metadata = {"Title": "UAV repeated-run comparison", "Creator": "Matplotlib"}
    fig.savefig(PDF_PATH, facecolor="white", metadata=metadata)
    fig.savefig(PNG_PATH, dpi=600, facecolor="white")
    plt.close(fig)

    STATS_PATH.write_text(
        json.dumps(statistics_record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    rows = read_rows()
    statistics_record = analyze(rows)
    draw(rows, statistics_record)
    print(json.dumps(statistics_record, indent=2, ensure_ascii=False))
    print(PDF_PATH)
    print(PNG_PATH)
    print(STATS_PATH)


if __name__ == "__main__":
    main()
