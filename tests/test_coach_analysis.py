from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pluralpass.coach.analysis import analyse_coach_responses


def test_coach_analysis_accepts_deidentified_controlled_exports(tmp_path: Path) -> None:
    responses = pd.DataFrame(
        [
            {"coach_id": "A", "scenario_id": "1", "rank_1": "A"},
            {"coach_id": "B", "scenario_id": "1", "rank_1": "A"},
            {"coach_id": "A", "scenario_id": "2", "rank_1": "A"},
            {"coach_id": "B", "scenario_id": "2", "rank_1": "B"},
            {"coach_id": "A", "scenario_id": "3", "rank_1": "A"},
            {"coach_id": "B", "scenario_id": "3", "rank_1": "C"},
        ]
    )
    scenarios = pd.DataFrame(
        [
            {"scenario_id": "1", "model_entropy": 0.1},
            {"scenario_id": "2", "model_entropy": 0.5},
            {"scenario_id": "3", "model_entropy": 0.9},
        ]
    )
    response_path = tmp_path / "responses.csv"
    scenario_path = tmp_path / "scenarios.csv"
    output_path = tmp_path / "report.json"
    responses.to_csv(response_path, index=False)
    scenarios.to_csv(scenario_path, index=False)

    report = analyse_coach_responses(
        str(response_path), str(scenario_path), str(output_path)
    )
    assert report["coaches"] == 2
    assert report["scenarios"] == 3
    assert report["rho"] > 0
    stored = json.loads(output_path.read_text(encoding="utf-8"))
    assert "non-preregistered" in stored["status"].lower()
