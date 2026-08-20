from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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


def _valid_bins(payload: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, sizes = [], [], []
    for row in payload["bins"]:
        if row["count"] and row["mean_probability"] is not None:
            x.append(row["mean_probability"])
            y.append(row["empirical_frequency"])
            sizes.append(row["count"])
    sizes_arr = np.asarray(sizes, dtype=float)
    scaled = 18 + 80 * sizes_arr / sizes_arr.max()
    return np.asarray(x), np.asarray(y), scaled


def make_figure(output: str = "outputs/PluralPass_Figure3") -> None:
    set_nature_style()
    calibration = _load_json("outputs/PluralPass_calibration_diagnostics.json")
    observation = _load_json("outputs/PluralPass_observation_stress.json")
    counterfactual = _load_json("outputs/PluralPass_counterfactual_stress.json")
    neural_random = _load_json("outputs/PluralPass_neural_event_random_bias.json")
    hgb_random = _load_json("outputs/PluralPass_event_random_bias.json")

    fig = plt.figure(figsize=(7.2, 5.1), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.08], height_ratios=[1, 1])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    # a: pooled calibration curves
    ax_a.plot([0, 1], [0, 1], color="#B7BCC4", lw=0.9, ls="--", zorder=1)
    calibration_series = [
        ("receiver_top_label", "Receiver top label", PALETTE["blue"]),
        ("completion_pluralpass", "Completion, PluralPass", PALETTE["teal"]),
        ("completion_hgb", "Completion, HGB", PALETTE["gold"]),
    ]
    for key, label, color in calibration_series:
        x, y, sizes = _valid_bins(calibration["pooled"][key])
        ax_a.scatter(x, y, s=sizes, color=color, alpha=0.78, edgecolor="white", lw=0.4, label=label, zorder=3)
        ax_a.plot(x, y, color=color, lw=0.8, alpha=0.7)
    ax_a.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted probability", ylabel="Empirical frequency")
    ax_a.set_title("Pooled reliability is close to the diagonal", loc="left", pad=7)
    ax_a.grid(color="#E2E4E8", lw=0.6)
    ax_a.legend(loc="upper left", fontsize=5.8)
    ax_a.text(
        0.98,
        0.03,
        "Point size scales with bin count",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        color=PALETTE["grey"],
    )
    panel_label(ax_a, "a")

    # b: uncertainty strata
    strata = observation["pooled"]["epistemic"]
    order = ["epistemic_low", "epistemic_mid", "epistemic_high"]
    labels = ["Low", "Mid", "High"]
    x = np.arange(len(order))
    top1 = [strata[k]["receiver_top1"] for k in order]
    set_size = [strata[k]["mean_set_size"] for k in order]
    coverage = [strata[k]["set_coverage"] for k in order]
    ax_b.plot(x, top1, marker="o", color=PALETTE["blue"], lw=1.2, label="Receiver Top-1")
    ax_b.plot(x, coverage, marker="o", color=PALETTE["teal"], lw=1.2, label="Set coverage")
    ax_b.set_xticks(x, labels)
    ax_b.set_ylim(0.45, 1.01)
    ax_b.set_ylabel("Accuracy or coverage")
    ax_b.grid(axis="y", color="#E2E4E8", lw=0.6)
    ax_b2 = ax_b.twinx()
    ax_b2.bar(x, set_size, width=0.55, color=PALETTE["grey"], alpha=0.22, label="Mean set size")
    ax_b2.set_ylim(0, 5.2)
    ax_b2.set_ylabel("Mean set size")
    handles1, labels1 = ax_b.get_legend_handles_labels()
    handles2, labels2 = ax_b2.get_legend_handles_labels()
    ax_b.legend(handles1 + handles2, labels1 + labels2, loc="lower left", fontsize=6)
    ax_b.set_title("Epistemic uncertainty separates easy and hard scenes", loc="left", pad=7)
    panel_label(ax_b, "b")

    # c: counterfactual missingness
    scenarios = [
        ("context_dropout_20", "Context\n−20%"),
        ("opponent_dropout_30", "Opponents\n−30%"),
        ("nonreceiver_candidate_dropout_30", "Other options\n−30%"),
        ("receiver_hidden_10", "Receiver\nhidden 10%"),
    ]
    cf = counterfactual["pooled_macro_average"]
    top1_vals = [cf[k]["all_events.receiver.top1"] for k, _ in scenarios]
    coverage_vals = [cf[k]["all_events.recommendation_set.coverage"] for k, _ in scenarios]
    receiver_visible = [cf[k]["all_events.receiver_visible_rate"] for k, _ in scenarios]
    x = np.arange(len(scenarios))
    width = 0.34
    ax_c.bar(x - width / 2, top1_vals, width, color=PALETTE["blue"], label="Receiver Top-1")
    ax_c.bar(x + width / 2, coverage_vals, width, color=PALETTE["teal"], label="Set coverage")
    ax_c.plot(x, receiver_visible, color=PALETTE["orange"], marker="v", lw=1.0, label="Receiver visible")
    ax_c.axhline(0.90, color=PALETTE["orange"], lw=0.8, ls="--")
    ax_c.set_xticks(x, [label for _, label in scenarios])
    ax_c.set_xlim(-0.65, len(scenarios) + 0.20)
    ax_c.set_ylim(0.45, 1.02)
    ax_c.set_ylabel("Metric after recomputing predictions")
    ax_c.set_title("Coverage fails when the true receiver is unseen", loc="left", pad=7)
    ax_c.grid(axis="y", color="#E2E4E8", lw=0.6)
    label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.1}
    ax_c.text(
        3.36,
        coverage_vals[-1] - 0.018,
        "Set coverage",
        color=PALETTE["teal"],
        fontsize=6,
        va="center",
        ha="left",
        bbox=label_box,
    )
    ax_c.text(
        3.36,
        receiver_visible[-1] + 0.028,
        "Receiver visible",
        color=PALETTE["orange"],
        fontsize=6,
        va="center",
        ha="left",
        bbox=label_box,
    )
    ax_c.text(
        3.36,
        top1_vals[-1],
        "Top-1",
        color=PALETTE["blue"],
        fontsize=6,
        va="center",
        ha="left",
        bbox=label_box,
    )
    panel_label(ax_c, "c")

    # d: event-random leakage audit
    delta = neural_random["event_random_minus_leave_one_domain_macro"]
    metrics = [
        ("receiver.top1", "Top-1"),
        ("receiver.top3", "Top-3"),
        ("completion.auroc", "AUROC"),
        ("value.spearman", "Value\nSpearman"),
    ]
    vals = [delta[key] for key, _ in metrics]
    y = np.arange(len(metrics))[::-1]
    colors = [PALETTE["teal"] if v > 0 else PALETTE["orange"] for v in vals]
    ax_d.barh(y, vals, color=colors, alpha=0.86, height=0.55)
    ax_d.axvline(0, color=PALETTE["dark"], lw=0.8)
    ax_d.set_yticks(y, [label for _, label in metrics])
    ax_d.set_xlabel("Event-random − leave-one-domain macro")
    ax_d.set_title("Leakage is severe, optimism is not uniform", loc="left", pad=7)
    ax_d.grid(axis="x", color="#E2E4E8", lw=0.6)
    max_abs = max(abs(v) for v in vals)
    ax_d.set_xlim(-max_abs * 1.55, max_abs * 1.55)
    leakage = hgb_random["match_leakage"]
    ax_d.text(
        0.03,
        0.05,
        f"{leakage['test_matches_seen_in_train']}/{leakage['total_matches']} test matches also in train",
        transform=ax_d.transAxes,
        fontsize=6,
        color=PALETTE["dark"],
        bbox={"facecolor": "white", "edgecolor": "#DADCE0", "boxstyle": "round,pad=0.2"},
    )
    panel_label(ax_d, "d")

    fig.suptitle(
        "PluralPass calibration and boundary stress tests",
        x=0.01,
        ha="left",
        fontsize=9,
        weight="bold",
    )
    save_publication_figure(fig, output)
    save_publication_figure(fig, "artifacts/figures/figure3_calibration_stress")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
