from __future__ import annotations

import csv
import gzip
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )


def _draw_pitch(ax: plt.Axes) -> None:
    ax.add_patch(Rectangle((0, 0), 120, 80, facecolor="#F7F7F4", edgecolor="#4C566A", lw=1))
    ax.plot([60, 60], [0, 80], color="#9AA0A6", lw=0.7)
    ax.add_patch(plt.Circle((60, 40), 9.15, fill=False, edgecolor="#9AA0A6", lw=0.7))
    ax.add_patch(Rectangle((102, 18), 18, 44, fill=False, edgecolor="#9AA0A6", lw=0.7))
    ax.add_patch(Rectangle((0, 18), 18, 44, fill=False, edgecolor="#9AA0A6", lw=0.7))
    ax.set(xlim=(-1, 121), ylim=(81, -1), aspect="equal")
    ax.axis("off")


def _plot_scenario(
    row: dict[str, Any], blind_id: str, recommendation: list[int] | None = None
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.1, 4.8))
    _draw_pitch(ax)
    candidate_letter = 0
    for index, (node, is_candidate) in enumerate(zip(row["nodes"], row["candidate_mask"])):
        # Freeze-frame coordinates can sit fractionally outside the pitch.
        # Clamp only the rendered marker to a small safe inset so labels are
        # never clipped in the coach-facing PDF.
        x = max(1.5, min(118.5, (node[0] + 1) * 60))
        y = max(1.5, min(78.0, (node[1] + 1) * 40))
        if node[3] > 0.5:
            ax.scatter(x, y, s=85, marker="*", color="#202124", zorder=5)
            ax.text(x + 1.2, y - 1.4, "Passer", fontsize=7, weight="bold")
        elif is_candidate:
            label = chr(ord("A") + candidate_letter)
            candidate_letter += 1
            selected = recommendation is not None and index in recommendation
            color = "#D55E00" if selected else "#2F6B9A"
            ax.scatter(x, y, s=65, color=color, edgecolor="white", lw=0.8, zorder=4)
            ax.text(x + 1.0, y - 1.0, label, fontsize=8, weight="bold", color=color)
        else:
            ax.scatter(x, y, s=42, color="#8C8C8C", edgecolor="white", lw=0.5, zorder=3)
    title = f"Scenario {blind_id}: select every acceptable pass, then rank your top three"
    if recommendation is not None:
        title += "\nOrange markers: calibrated PluralPass recommendation set"
    ax.set_title(title, loc="left", fontsize=10, weight="bold")
    fig.tight_layout()
    return fig


def generate_stimuli(config: dict[str, Any], predictions_path: str | None = None) -> dict[str, Any]:
    _style()
    processed = Path(config["data"]["processed_dir"])
    eligible = []
    with gzip.open(processed / "passes.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["visible_players"] >= config["coach_study"]["minimum_visible_players"]:
                eligible.append(row)
    if len(eligible) < config["coach_study"]["scenarios"]:
        raise ValueError("Not enough high-visibility scenarios for the coach study")
    predictions = {}
    if predictions_path:
        with Path(predictions_path).open("r", encoding="utf-8") as handle:
            predictions = {row["event_id"]: row for row in map(json.loads, handle)}
    random.Random(config["coach_study"]["seed"]).shuffle(eligible)
    selected = []
    per_domain: dict[str, int] = {}
    domain_cap = max(
        2, config["coach_study"]["scenarios"] // max(1, len({x["domain"] for x in eligible}))
    )
    for row in eligible:
        if per_domain.get(row["domain"], 0) >= domain_cap + 1:
            continue
        selected.append(row)
        per_domain[row["domain"]] = per_domain.get(row["domain"], 0) + 1
        if len(selected) == config["coach_study"]["scenarios"]:
            break
    if len(selected) < config["coach_study"]["scenarios"]:
        for row in eligible:
            if row not in selected:
                selected.append(row)
            if len(selected) == config["coach_study"]["scenarios"]:
                break

    output = Path("artifacts/coach")
    output.mkdir(parents=True, exist_ok=True)
    private_rows, public_rows = [], []
    with PdfPages(output / "coach_scenarios.pdf") as pdf:
        for order, row in enumerate(selected, start=1):
            blind_id = hashlib.sha256(
                f"{config['coach_study']['seed']}:{row['event_id']}".encode()
            ).hexdigest()[:8]
            prediction = predictions.get(row["event_id"])
            recommendation = None if prediction is None else prediction.get("recommendation_set")
            fig = _plot_scenario(row, blind_id, recommendation)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            public_rows.append(
                {
                    "order": order,
                    "scenario_id": blind_id,
                    "candidate_count": sum(row["candidate_mask"]),
                }
            )
            private_rows.append(
                {
                    "order": order,
                    "scenario_id": blind_id,
                    "event_id": row["event_id"],
                    "domain": row["domain"],
                    "receiver_index": row["receiver_index"],
                    "pass_completed": row["pass_completed"],
                    "recommendation_set": recommendation,
                }
            )
    for path, rows in [
        (output / "scenario_index.csv", public_rows),
        (output / "private_answer_key.csv", private_rows),
    ]:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    survey_schema = {
        "stage_1": [
            "acceptable_choices",
            "top_1",
            "top_2",
            "top_3",
            "confidence_1_to_7",
            "ambiguity_1_to_7",
            "decision_seconds",
        ],
        "stage_2": [
            "condition",
            "reasonable_1_to_7",
            "information_gain_1_to_7",
            "use_intention_1_to_7",
            "revised_top_1",
        ],
    }
    (output / "survey_schema.json").write_text(
        json.dumps(survey_schema, indent=2), encoding="utf-8"
    )
    return {
        "scenarios": len(selected),
        "domains": per_domain,
        "output": str(output / "coach_scenarios.pdf"),
    }
