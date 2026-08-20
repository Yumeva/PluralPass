from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def audit_dataset(config: dict[str, Any]) -> dict[str, Any]:
    processed = Path(config["data"]["processed_dir"])
    parquet = processed / "passes.parquet"
    csv = processed / "passes.csv.gz"
    if parquet.exists():
        data = pd.read_parquet(parquet)
    elif csv.exists():
        data = pd.read_csv(csv)
    else:
        raise FileNotFoundError(f"Run preprocess first: {csv}")
    by_domain = data.groupby("domain").agg(
        passes=("event_id", "size"),
        matches=("match_id", "nunique"),
        completion_rate=("pass_completed", "mean"),
        median_visible_players=("visible_players", "median"),
        median_receiver_match_m=("receiver_match_distance_m", "median"),
    )
    report = {
        "passes": len(data),
        "matches": int(data["match_id"].nunique()),
        "domains": int(data["domain"].nunique()),
        "publication_gate_30000_passes": bool(len(data) >= 30_000),
        "publication_gate_6_domains": bool(data["domain"].nunique() >= 6),
        "high_visibility_expert_candidates": int(
            (data["visible_players"] >= config["coach_study"]["minimum_visible_players"]).sum()
        ),
        "by_domain": by_domain.reset_index().to_dict(orient="records"),
    }
    output = Path(config["data"]["processed_dir"]) / "dataset_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
