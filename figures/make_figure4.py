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


def _distribution_to_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for part in str(text).split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        label, value = part.split(":", 1)
        counts[label.strip()] = int(value)
    return counts


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(float(x.min()), float(x.max()), 80)
    return xs, intercept + slope * xs


def make_figure(
    output: str = "outputs/PluralPass_Figure4",
    source_dir: str = "outputs/coach_analysis",
) -> None:
    set_nature_style()
    source = Path(source_dir)
    scenarios = pd.read_csv(source / "scenario_entropy_analysis.csv")
    report = _load_json(str(source / "PluralPass_coach_analysis.json"))
    h3 = report["h3"]

    fig = plt.figure(figsize=(7.2, 5.1), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.0], height_ratios=[1.0, 0.98])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    # a: primary H3 relationship
    x = scenarios["model_entropy"].to_numpy()
    y = scenarios["coach_choice_entropy_recomputed"].to_numpy()
    size = 22 + 7 * scenarios["candidate_count"].to_numpy()
    ax_a.scatter(x, y, s=size, color=PALETTE["blue"], alpha=0.78, edgecolor="white", linewidth=0.45)
    xs, ys = _fit_line(x, y)
    ax_a.plot(xs, ys, color=PALETTE["dark"], lw=1.0)
    ax_a.set_xlabel("Model receiver entropy")
    ax_a.set_ylabel("Coach first-choice entropy")
    ax_a.set_title("Model uncertainty tracks coach disagreement", loc="left", pad=7)
    ax_a.grid(color="#E2E4E8", lw=0.6)
    ax_a.text(
        0.03,
        0.97,
        "Spearman rho = "
        f"{h3['model_entropy_spearman_rho']:.2f}\n"
        "95% CI "
        f"{h3['model_entropy_bootstrap_ci_low']:.2f} to {h3['model_entropy_bootstrap_ci_high']:.2f}\n"
        f"p = {h3['model_entropy_spearman_p']:.4f}; n = {h3['n_scenarios']} scenes",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=6,
        bbox={"facecolor": "white", "edgecolor": "#DADCE0", "boxstyle": "round,pad=0.25"},
    )
    ax_a.text(
        0.98,
        0.03,
        "Point size: candidate count",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        color=PALETTE["grey"],
    )
    panel_label(ax_a, "a")

    # b: conformal set size relationship
    rng = np.random.default_rng(8)
    set_sizes = sorted(scenarios["conformal_set_size"].dropna().unique())
    positions = np.arange(len(set_sizes))
    for pos, set_size in zip(positions, set_sizes, strict=False):
        subset = scenarios.loc[scenarios["conformal_set_size"] == set_size, "coach_choice_entropy_recomputed"].to_numpy()
        jitter = rng.uniform(-0.08, 0.08, size=len(subset))
        ax_b.scatter(np.full(len(subset), pos) + jitter, subset, s=28, color=PALETTE["teal"], alpha=0.72, edgecolor="white", lw=0.4)
        if len(subset):
            ax_b.plot([pos - 0.18, pos + 0.18], [subset.mean(), subset.mean()], color=PALETTE["dark"], lw=1.0)
    ax_b.set_xticks(positions, [str(int(v)) for v in set_sizes])
    ax_b.set_xlabel("Conformal set size")
    ax_b.set_ylabel("Coach first-choice entropy")
    ax_b.set_title("Larger action sets tend to mark ambiguous scenes", loc="left", pad=7)
    ax_b.grid(axis="y", color="#E2E4E8", lw=0.6)
    ax_b.text(
        0.03,
        0.97,
        "rho = "
        f"{h3['conformal_set_size_spearman_rho']:.2f}\n"
        "95% CI "
        f"{h3['conformal_set_size_bootstrap_ci_low']:.2f} to {h3['conformal_set_size_bootstrap_ci_high']:.2f}",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=6,
        bbox={"facecolor": "white", "edgecolor": "#DADCE0", "boxstyle": "round,pad=0.25"},
    )
    panel_label(ax_b, "b")

    # c: representative choice distributions at low, medium and high model entropy
    low_pool = scenarios.loc[scenarios["model_entropy"] <= scenarios["model_entropy"].quantile(0.20)]
    high_pool = scenarios.loc[scenarios["model_entropy"] >= scenarios["model_entropy"].quantile(0.80)]
    low_idx = low_pool.sort_values(["coach_choice_entropy_recomputed", "model_entropy"]).index[0]
    mid_idx = (scenarios["model_entropy"] - scenarios["model_entropy"].median()).abs().sort_values().index[0]
    high_idx = high_pool.sort_values("coach_choice_entropy_recomputed", ascending=False).index[0]
    selected_indices = [low_idx, mid_idx, high_idx]
    case_labels = [
        "Low model\nuncertainty",
        "Median model\nuncertainty",
        "High model\nuncertainty",
    ]
    palette = [PALETTE["blue"], PALETTE["teal"], PALETTE["gold"], PALETTE["orange"], PALETTE["grey"]]
    y_positions = np.arange(len(selected_indices))[::-1]
    for row_i, (idx, ypos) in enumerate(zip(selected_indices, y_positions, strict=False)):
        row = scenarios.loc[idx]
        counts = _distribution_to_counts(row["rank1_distribution_recomputed"])
        total = sum(counts.values())
        left = 0
        for j, (choice, count) in enumerate(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]):
            width = count / total if total else 0
            ax_c.barh(ypos, width, left=left, color=palette[j % len(palette)], height=0.55)
            if width > 0.10:
                ax_c.text(left + width / 2, ypos, choice, ha="center", va="center", fontsize=6, color="white", weight="bold")
            left += width
        ax_c.text(
            1.02,
            ypos,
            f"H={0.0 if abs(row['coach_choice_entropy_recomputed']) < 1e-12 else row['coach_choice_entropy_recomputed']:.2f}",
            va="center",
            fontsize=6,
            color=PALETTE["dark"],
        )
    ax_c.set_yticks(y_positions, case_labels)
    ax_c.set_xlim(0, 1.18)
    ax_c.set_xlabel("Share of coaches selecting each first-choice receiver")
    ax_c.set_title("Example scenes range from consensus to plural choices", loc="left", pad=7)
    ax_c.spines["bottom"].set_visible(True)
    ax_c.grid(axis="x", color="#E2E4E8", lw=0.6)
    panel_label(ax_c, "c")

    # d: endpoint summary
    endpoints = [
        (
            "Model entropy\nvs coach entropy",
            h3["model_entropy_spearman_rho"],
            h3["model_entropy_bootstrap_ci_low"],
            h3["model_entropy_bootstrap_ci_high"],
            PALETTE["blue"],
        ),
        (
            "Set size\nvs coach entropy",
            h3["conformal_set_size_spearman_rho"],
            h3["conformal_set_size_bootstrap_ci_low"],
            h3["conformal_set_size_bootstrap_ci_high"],
            PALETTE["teal"],
        ),
    ]
    y = np.arange(len(endpoints))[::-1]
    for yi, (label, rho, low, high, color) in zip(y, endpoints, strict=False):
        ax_d.plot([low, high], [yi, yi], color=color, lw=2.0, solid_capstyle="round")
        ax_d.scatter([rho], [yi], s=42, color=color, edgecolor="white", lw=0.5, zorder=3)
        ax_d.text(high + 0.025, yi, f"{rho:.2f}", va="center", fontsize=7, color=color, weight="bold")
    ax_d.axvline(0, color=PALETTE["dark"], lw=0.8)
    ax_d.set_yticks(y, [label for label, *_ in endpoints])
    ax_d.set_xlim(-0.05, 0.82)
    ax_d.set_xlabel("Spearman rho with scenario-level coach disagreement")
    ax_d.set_title("Uncertainty corresponds to ambiguity, not causality", loc="left", pad=7)
    ax_d.grid(axis="x", color="#E2E4E8", lw=0.6)
    ax_d.text(
        0.02,
        0.96,
        "Bootstrap intervals cluster scenes,\nnot individual coach responses.",
        transform=ax_d.transAxes,
        ha="left",
        va="top",
        fontsize=6,
        color=PALETTE["grey"],
    )
    panel_label(ax_d, "d")

    fig.suptitle(
        "PluralPass uncertainty corresponds to tactical disagreement among coaches",
        x=0.01,
        ha="left",
        fontsize=9,
        weight="bold",
    )
    save_publication_figure(fig, output)
    save_publication_figure(fig, "artifacts/figures/figure4_coach_ambiguity")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
