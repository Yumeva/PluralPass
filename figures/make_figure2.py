from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figures.style import PALETTE, save_publication_figure, set_nature_style

DOMAIN_ORDER = [
    "FIFA World Cup 2022",
    "UEFA Euro 2020",
    "UEFA Euro 2024",
    "La Liga 2020/2021",
    "Ligue 1 2022/2023",
    "1. Bundesliga 2023/2024",
    "Ligue 1 2021/2022",
    "Major League Soccer 2023",
]

SHORT = {
    "FIFA World Cup 2022": "World Cup\n2022",
    "UEFA Euro 2020": "Euro\n2020",
    "UEFA Euro 2024": "Euro\n2024",
    "La Liga 2020/2021": "La Liga\n20/21",
    "Ligue 1 2022/2023": "Ligue 1\n22/23",
    "1. Bundesliga 2023/2024": "Bundesliga\n23/24",
    "Ligue 1 2021/2022": "Ligue 1\n21/22",
    "Major League Soccer 2023": "MLS\n2023",
}

TINY = {
    "FIFA World Cup 2022": "WC22",
    "UEFA Euro 2020": "EU20",
    "UEFA Euro 2024": "EU24",
    "La Liga 2020/2021": "LL20",
    "Ligue 1 2022/2023": "L1-22",
    "1. Bundesliga 2023/2024": "BUN23",
    "Ligue 1 2021/2022": "L1-21",
    "Major League Soccer 2023": "MLS23",
}


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


def _load_results() -> tuple[pd.DataFrame, dict[str, dict]]:
    table = pd.read_csv("outputs/tables/table1_benchmark_domains.csv")
    formal = json.loads(Path("outputs/PluralPass_formal_results_index.json").read_text())
    rows = []
    raw = {}
    for fold, payload in formal["folds"].items():
        domain = payload["domain"]
        raw[domain] = payload
        pp = payload["pluralpass"]
        diff = payload["pluralpass_minus_hgb"]
        rows.append(
            {
                "domain": domain,
                "fold": fold,
                "test_passes": payload["test_events"],
                "top1": pp["receiver_top1"],
                "top3": pp["receiver_top3"],
                "auroc": pp["completion_auroc"],
                "value_spearman": pp["value_spearman"],
                "coverage": pp["set_coverage"],
                "set_size": pp["mean_set_size"],
                "abstention": pp["abstention_rate"],
                "delta_top1": diff["receiver"]["top1"]["difference"],
                "delta_auroc": diff["completion"]["auroc"]["difference"],
                "delta_value": diff["value"]["spearman"]["difference"],
            }
        )
    result = pd.DataFrame(rows)
    order = {domain: i for i, domain in enumerate(DOMAIN_ORDER)}
    result["order"] = result["domain"].map(order)
    result = result.sort_values("order")
    table["order"] = table["domain"].map(order)
    return table.sort_values("order"), {"formal": formal, "result": result}


def _domain_colors(domains: list[str]) -> list[str]:
    colors = []
    for domain in domains:
        if "World Cup" in domain or "Euro" in domain:
            colors.append(PALETTE["blue"])
        elif "Major League Soccer" in domain:
            colors.append(PALETTE["orange"])
        else:
            colors.append(PALETTE["teal"])
    return colors


def make_figure(output: str = "outputs/PluralPass_Figure2") -> None:
    set_nature_style()
    domains, payload = _load_results()
    result = payload["result"]
    labels = [TINY[d] for d in result["domain"]]

    fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.02, 1], width_ratios=[1.05, 1.15])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    # a: data composition
    y = np.arange(len(domains))[::-1]
    ax_a.barh(
        y,
        domains["passes"].to_numpy() / 1000,
        color=_domain_colors(domains["domain"].tolist()),
        alpha=0.86,
        height=0.68,
    )
    for yi, (_, row) in zip(y, domains.iterrows(), strict=False):
        ax_a.text(
            row["passes"] / 1000 + 0.25,
            yi,
            f"{int(row['matches'])} matches",
            va="center",
            fontsize=6.5,
            color=PALETTE["dark"],
        )
    ax_a.set_yticks(y, [SHORT[d].replace("\n", " ") for d in domains["domain"]])
    ax_a.set_xlabel("Eligible open-play passes (×1,000)")
    ax_a.set_title("Version-pinned public 360 benchmark", loc="left", pad=7)
    ax_a.set_xlim(0, max(domains["passes"] / 1000) + 3.3)
    ax_a.grid(axis="x", color="#E2E4E8", lw=0.6)
    panel_label(ax_a, "a")

    # b: predictive decomposition
    metrics = [
        ("top1", "Receiver\nTop-1", PALETTE["blue"]),
        ("top3", "Receiver\nTop-3", PALETTE["navy"]),
        ("auroc", "Completion\nAUROC", PALETTE["teal"]),
        ("value_spearman", "Value\nSpearman", PALETTE["gold"]),
    ]
    x_offsets = np.linspace(-0.18, 0.18, len(metrics))
    x = np.arange(len(result))
    for offset, (metric, label, color) in zip(x_offsets, metrics, strict=False):
        ax_b.scatter(
            x + offset,
            result[metric],
            s=24,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            label=label,
            zorder=3,
        )
        ax_b.plot(x + offset, result[metric], color=color, lw=0.8, alpha=0.55)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, rotation=35, ha="right")
    ax_b.set_xlim(-0.55, len(result) - 0.15)
    ax_b.set_ylim(0.42, 0.95)
    ax_b.set_ylabel("External-test metric")
    ax_b.set_title("Choice, execution and value remain separable", loc="left", pad=7)
    ax_b.grid(axis="y", color="#E2E4E8", lw=0.6)
    for metric, label, color in metrics:
        ax_b.text(
            len(result) - 0.28,
            result[metric].iloc[-1],
            label.replace("\n", " "),
            color=color,
            fontsize=5.8,
            va="center",
            ha="left",
        )
    panel_label(ax_b, "b")

    # c: calibrated set behaviour
    x = np.arange(len(result))
    ax_c.axhline(0.90, color=PALETTE["orange"], lw=1.0, ls="--")
    ax_c.scatter(x, result["coverage"], s=34, color=PALETTE["blue"], zorder=4, label="APS coverage")
    ax_c.plot(x, result["coverage"], color=PALETTE["blue"], lw=0.9)
    ax_c.set_ylim(0.88, 1.00)
    ax_c.set_ylabel("Coverage")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(labels, rotation=35, ha="right")
    ax_c.set_title("Prediction sets are reliable but conservative", loc="left", pad=7)
    ax_c.text(
        len(result) - 0.15,
        0.902,
        "90% target",
        color=PALETTE["orange"],
        fontsize=6,
        ha="right",
        va="bottom",
    )
    ax_c.grid(axis="y", color="#E2E4E8", lw=0.6)
    ax_c2 = ax_c.twinx()
    ax_c2.bar(x, result["set_size"], color=PALETTE["grey"], alpha=0.22, width=0.62, label="Set size")
    ax_c2.scatter(
        x,
        result["abstention"] * 5,
        color=PALETTE["teal"],
        marker="v",
        s=24,
        zorder=5,
        label="Abstention ×5",
    )
    ax_c2.set_ylabel("Set size; abstention ×5")
    ax_c2.set_ylim(0, 5.3)
    handles1, labels1 = ax_c.get_legend_handles_labels()
    handles2, labels2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=3,
        fontsize=6,
        frameon=False,
    )
    panel_label(ax_c, "c")

    # d: paired gains over HGB
    gain_metrics = [
        ("delta_top1", "Top-1", PALETTE["blue"]),
        ("delta_auroc", "AUROC", PALETTE["teal"]),
        ("delta_value", "Value\nSpearman", PALETTE["gold"]),
    ]
    yy = np.arange(len(result))[::-1]
    for j, (metric, label, color) in enumerate(gain_metrics):
        vals = result[metric].to_numpy()
        ax_d.scatter(vals, yy + (j - 1) * 0.18, s=28, color=color, edgecolor="white", linewidth=0.4, label=label, zorder=3)
    ax_d.axvline(0, color=PALETTE["dark"], lw=0.7)
    ax_d.set_yticks(yy, [SHORT[d].replace("\n", " ") for d in result["domain"]])
    ax_d.set_xlabel("PluralPass − histogram-gradient boosting")
    ax_d.set_title("Paired gains are positive across held-out domains", loc="left", pad=7)
    ax_d.set_xlim(-0.005, 0.075)
    ax_d.grid(axis="x", color="#E2E4E8", lw=0.6)
    ax_d.legend(loc="lower right", fontsize=6)
    panel_label(ax_d, "d")

    fig.suptitle(
        "PluralPass external validation across a heterogeneous public 360 benchmark",
        x=0.01,
        ha="left",
        fontsize=9,
        weight="bold",
    )
    save_publication_figure(fig, output)
    save_publication_figure(fig, "artifacts/figures/figure2_external_validation")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
