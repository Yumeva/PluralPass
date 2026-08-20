#!/usr/bin/env python3
"""Export 40-scene candidate scores from the correct leave-one-domain fold."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DOMAIN_TO_FOLD = {
    "1. Bundesliga|2023/2024": "1_bundesliga_2023_2024",
    "FIFA World Cup|2022": "fifa_world_cup_2022",
    "La Liga|2020/2021": "la_liga_2020_2021",
    "Ligue 1|2021/2022": "ligue_1_2021_2022",
    "Ligue 1|2022/2023": "ligue_1_2022_2023",
    "Major League Soccer|2023": "major_league_soccer_2023",
    "UEFA Euro|2020": "uefa_euro_2020",
    "UEFA Euro|2024": "uefa_euro_2024",
}


def load_prediction_file(path: Path) -> dict[str, dict[str, Any]]:
    with path.open() as handle:
        return {row["event_id"]: row for row in map(json.loads, handle)}


def load_hgb(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {
        str(event_id): np.asarray(score, dtype=float)
        for event_id, score in zip(data["event_id"], data["receiver_hist_gradient_boosting"])
    }


def node_xy(node: list[float]) -> tuple[float, float]:
    return (float(node[0]) + 1.0) * 60.0, (float(node[1]) + 1.0) * 40.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("scenario_key", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    scenarios = pd.read_csv(args.scenario_key)
    scenarios["scenario_id"] = scenarios["scenario_id"].astype(str)
    event_ids = set(scenarios["event_id"].astype(str))

    pass_rows: dict[str, dict[str, Any]] = {}
    with gzip.open(args.project / "data/processed/passes.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["event_id"] in event_ids:
                pass_rows[row["event_id"]] = row
                if len(pass_rows) == len(event_ids):
                    break

    fold_cache: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for _, scenario in scenarios.sort_values("order").iterrows():
        event_id = str(scenario["event_id"])
        scenario_id = str(scenario["scenario_id"])
        domain = str(scenario["domain"])
        fold = DOMAIN_TO_FOLD[domain]
        if fold not in fold_cache:
            fold_cache[fold] = {
                "pluralpass": load_prediction_file(
                    args.project / f"artifacts/results/pluralpass/{fold}/predictions.jsonl"
                ),
                "mlp": load_prediction_file(
                    args.project
                    / f"artifacts/results/pluralpass-neural-candidate-mlp/{fold}/predictions.jsonl"
                ),
                "hgb": load_hgb(
                    args.project / f"artifacts/baselines/pluralpass/{fold}/predictions.npz"
                ),
                "metrics": json.loads(
                    (args.project / f"artifacts/results/pluralpass/{fold}/metrics.json").read_text()
                ),
            }
        cache = fold_cache[fold]
        source = pass_rows[event_id]
        pp = cache["pluralpass"][event_id]
        mlp = cache["mlp"][event_id]
        hgb = cache["hgb"][event_id]
        candidate_nodes = [
            index for index, value in enumerate(source["candidate_mask"]) if bool(value)
        ]
        letter_by_node = {
            node_index: chr(ord("A") + rank)
            for rank, node_index in enumerate(candidate_nodes)
        }
        pp_set = [letter_by_node[index] for index in pp["recommendation_set"]]
        actual_letter = letter_by_node[int(source["receiver_index"])]
        probability = np.asarray(pp["receiver_probability"], dtype=float)
        valid_probability = probability[candidate_nodes]
        valid_probability = valid_probability / valid_probability.sum()
        model_entropy = float(
            -(valid_probability * np.log(np.clip(valid_probability, 1e-15, None))).sum()
        )
        for rank, node_index in enumerate(candidate_nodes, start=1):
            letter = letter_by_node[node_index]
            x, y = node_xy(source["nodes"][node_index])
            records.append(
                {
                    "scenario_order": int(scenario["order"]),
                    "scenario_id": scenario_id,
                    "event_id": event_id,
                    "domain": domain,
                    "fold": fold,
                    "match_id": source["match_id"],
                    "candidate_letter": letter,
                    "candidate_rank_in_pdf": rank,
                    "node_index": node_index,
                    "candidate_count": len(candidate_nodes),
                    "candidate_x": x,
                    "candidate_y": y,
                    "is_actual_receiver": int(letter == actual_letter),
                    "actual_receiver_letter": actual_letter,
                    "pluralpass_prob": pp["receiver_probability"][node_index],
                    "pluralpass_completion_prob": pp["completion_probability"][node_index],
                    "pluralpass_progression_mean": pp["value_mean"][node_index],
                    "pluralpass_epistemic_scene": pp["epistemic"],
                    "pluralpass_abstained": pp["abstained"],
                    "pluralpass_APS_set_letters": ",".join(pp_set),
                    "pluralpass_APS_set_size": len(pp_set),
                    "is_pluralpass_APS_member": int(node_index in pp["recommendation_set"]),
                    "fold_APS_qhat": cache["metrics"]["conformal_qhat"],
                    "fold_epistemic_abstention_threshold": cache["metrics"][
                        "epistemic_abstention_threshold"
                    ],
                    "hgb_receiver_score": hgb[node_index],
                    "mlp_prob": mlp["receiver_probability"][node_index],
                    "mlp_completion_prob": mlp["completion_probability"][node_index],
                    "mlp_progression_mean": mlp["value_mean"][node_index],
                    "model_entropy_from_external_fold": model_entropy,
                    "pass_completed": source["pass_completed"],
                    "visible_area_fraction": source["visible_area_fraction"],
                    "visible_players": source["visible_players"],
                    "receiver_match_distance_m": source["receiver_match_distance_m"],
                }
            )

    frame = pd.DataFrame(records).sort_values(
        ["scenario_order", "candidate_rank_in_pdf"]
    )
    csv_path = args.output / "PluralPass_40scenes_candidate_scores_external_fold.csv"
    xlsx_path = args.output / "PluralPass_40scenes_candidate_scores_external_fold.xlsx"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="candidate_scores", index=False)
        pd.DataFrame(
            [
                {
                    "rule": "fold selection",
                    "definition": "Each scene is read only from the leave-one-domain fold whose held domain equals the scene domain.",
                },
                {
                    "rule": "candidate ordering",
                    "definition": "Letters follow candidate_mask node order, matching the archived evaluation stimuli generator.",
                },
                {
                    "rule": "entropy",
                    "definition": "Shannon entropy of the archived external-fold PluralPass candidate probabilities.",
                },
            ]
        ).to_excel(writer, sheet_name="provenance", index=False)
    metadata = {
        "rows": len(frame),
        "scenarios": frame["scenario_id"].nunique(),
        "folds": sorted(frame["fold"].unique().tolist()),
        "candidate_probability_sum_max_error": float(
            frame.groupby("scenario_id")["pluralpass_prob"].sum().sub(1.0).abs().max()
        ),
    }
    (args.output / "external_fold_export_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
