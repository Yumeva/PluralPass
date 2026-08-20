from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figures.style import PALETTE, save_publication_figure, set_nature_style


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        weight="bold",
        fontsize=9,
        va="bottom",
        ha="left",
    )


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def make_figure(
    output: str = "outputs/PluralPass_Figure5",
    source_dir: str = "outputs/coach_analysis",
) -> None:
    set_nature_style()
    source = Path(source_dir)
    report = _load_json(str(source / "PluralPass_coach_analysis.json"))
    summary = pd.read_csv(source / "condition_summary.csv").set_index("assistance_condition")
    diffs = pd.read_csv(source / "condition_differences.csv")
    h4 = report["h4"]

    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.05], height_ratios=[1, 1])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    conditions = ["pluralpass_set", "point_baseline"]
    labels = ["PluralPass set", "Point recommendation"]
    colors = [PALETTE["blue"], PALETTE["grey"]]

    # a: coverage endpoints
    coverage_metrics = [
        ("First-choice\ncoverage", h4["pluralpass_first_choice_coverage"], h4["point_first_choice_coverage"]),
        ("Any top-3\ncoverage", h4["pluralpass_any_top3_coverage"], h4["point_any_top3_coverage"]),
        (
            "Acceptable-set\ncoverage",
            h4["pluralpass_acceptable_set_proportion"],
            h4["point_acceptable_set_proportion"],
        ),
    ]
    x = np.arange(len(coverage_metrics))
    width = 0.34
    ax_a.bar(x - width / 2, [m[1] for m in coverage_metrics], width, color=colors[0], label=labels[0])
    ax_a.bar(x + width / 2, [m[2] for m in coverage_metrics], width, color=colors[1], label=labels[1])
    ax_a.set_xticks(x, [m[0] for m in coverage_metrics])
    ax_a.set_ylim(0, 1.08)
    ax_a.set_ylabel("Proportion of coach-endorsed choices")
    ax_a.set_title("Sets cover coach-endorsed options more often", loc="left", pad=7)
    ax_a.grid(axis="y", color="#E2E4E8", lw=0.6)
    ax_a.legend(loc="lower left", fontsize=6)
    ax_a.text(
        0.02,
        0.97,
        "First-choice difference = "
        f"{h4['first_choice_coverage_difference']:.3f}\n"
        "95% CI "
        f"{h4['first_choice_coverage_difference_ci_low']:.3f} to {h4['first_choice_coverage_difference_ci_high']:.3f}",
        transform=ax_a.transAxes,
        va="top",
        ha="left",
        fontsize=6,
        bbox={"facecolor": "white", "edgecolor": "#DADCE0", "boxstyle": "round,pad=0.25"},
    )
    panel_label(ax_a, "a")

    # b: rating differences on the same 1-7 scale
    rating_map = [
        ("recommendation_reasonableness_1_7", "Reasonableness"),
        ("information_gain_1_7", "Information\ngain"),
        ("intended_use_1_7", "Intended\nuse"),
        ("explanation_clarity_1_7", "Explanation\nclarity"),
    ]
    rating_diffs = diffs.loc[diffs["metric"].isin([m[0] for m in rating_map])].copy()
    order = [m[0] for m in rating_map]
    rating_diffs["order"] = rating_diffs["metric"].map({metric: i for i, metric in enumerate(order)})
    rating_diffs = rating_diffs.sort_values("order")
    y = np.arange(len(rating_diffs))[::-1]
    for yi, (_, row) in zip(y, rating_diffs.iterrows(), strict=False):
        value = row["pluralpass_minus_point"]
        color = PALETTE["blue"] if value >= 0 else PALETTE["orange"]
        ax_b.plot([row["bootstrap_ci_low"], row["bootstrap_ci_high"]], [yi, yi], color=color, lw=2.0, solid_capstyle="round")
        ax_b.scatter([value], [yi], s=42, color=color, edgecolor="white", lw=0.5, zorder=3)
        label_x = row["bootstrap_ci_high"] + 0.06 if value >= 0 else row["bootstrap_ci_low"] - 0.06
        ax_b.text(
            label_x,
            yi,
            f"{value:+.2f}",
            color=color,
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=7,
            weight="bold",
        )
    ax_b.axvline(0, color=PALETTE["dark"], lw=0.8)
    ax_b.set_yticks(y, [dict(rating_map)[m] for m in rating_diffs["metric"]])
    ax_b.set_xlim(-0.75, 1.72)
    ax_b.set_xlabel("PluralPass set − point recommendation")
    ax_b.set_title("Sets are rated more reasonable and informative", loc="left", pad=7)
    ax_b.grid(axis="x", color="#E2E4E8", lw=0.6)
    panel_label(ax_b, "b")

    # c: operational flags
    flag_metrics = [
        ("Missing important\noption", "missing_important_option_rate"),
        ("Bad option\nincluded", "bad_recommended_option_rate"),
        ("Changed first\nchoice", "changed_first_choice_after_assistance_rate"),
    ]
    x = np.arange(len(flag_metrics))
    ax_c.bar(x - width / 2, [summary.loc[conditions[0], metric] for _, metric in flag_metrics], width, color=colors[0], label=labels[0])
    ax_c.bar(x + width / 2, [summary.loc[conditions[1], metric] for _, metric in flag_metrics], width, color=colors[1], label=labels[1])
    ax_c.set_xticks(x, [label for label, _ in flag_metrics])
    ax_c.set_ylim(0, 0.9)
    ax_c.set_ylabel("Response rate")
    ax_c.set_title("Inclusiveness comes with larger-set costs", loc="left", pad=7)
    ax_c.grid(axis="y", color="#E2E4E8", lw=0.6)
    ax_c.legend(loc="upper left", fontsize=6)
    panel_label(ax_c, "c")

    # d: decision time
    means = [summary.loc[c, "stage2_decision_time_seconds_mean"] for c in conditions]
    sds = [summary.loc[c, "stage2_decision_time_seconds_sd"] for c in conditions]
    ax_d.bar([0, 1], means, yerr=sds, capsize=3, color=colors, width=0.56)
    ax_d.set_xticks([0, 1], labels)
    ax_d.set_ylabel("Stage-2 decision time (s)")
    ax_d.set_title("Set-valued advice takes longer to evaluate", loc="left", pad=7)
    ax_d.grid(axis="y", color="#E2E4E8", lw=0.6)
    time_row = diffs.loc[diffs["metric"] == "stage2_decision_time_seconds"].iloc[0]
    ax_d.text(
        0.04,
        0.96,
        "Difference = "
        f"{time_row['pluralpass_minus_point']:+.2f} s\n"
        "95% CI "
        f"{time_row['bootstrap_ci_low']:.2f} to {time_row['bootstrap_ci_high']:.2f}",
        transform=ax_d.transAxes,
        va="top",
        ha="left",
        fontsize=6,
        bbox={"facecolor": "white", "edgecolor": "#DADCE0", "boxstyle": "round,pad=0.25"},
    )
    ax_d.text(
        0.98,
        0.88,
        "Bars show means; whiskers show SD.",
        transform=ax_d.transAxes,
        ha="right",
        va="top",
        fontsize=6,
        color=PALETTE["grey"],
    )
    panel_label(ax_d, "d")

    fig.suptitle(
        "Coaches evaluate calibrated sets against point recommendations",
        x=0.01,
        ha="left",
        fontsize=9,
        weight="bold",
    )
    save_publication_figure(fig, output)
    save_publication_figure(fig, "artifacts/figures/figure5_coach_assistance")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
