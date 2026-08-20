from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy, spearmanr


def _parse_receivers(value: object) -> list[str]:
    """Parse a receiver-list cell exported by a survey platform.

    The questionnaire blueprint allows comma-, semicolon- or pipe-delimited
    receiver ids. Empty cells and explicit "none" values return an empty list.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "无", "无可接受传球"}:
        return []
    for delimiter in [";", "|", ","]:
        text = text.replace(delimiter, " ")
    return [item.strip() for item in text.split() if item.strip()]


def _first_existing(columns: Iterable[str], candidates: list[str]) -> str | None:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _stage1_rows(responses: pd.DataFrame) -> pd.DataFrame:
    if "stage" not in responses:
        return responses.copy()
    stage = responses["stage"].astype(str).str.lower()
    mask = stage.isin({"1", "stage1", "unaided", "unassisted", "independent"})
    if mask.any():
        return responses.loc[mask].copy()
    return responses.copy()


def _choice_entropy_by_scenario(responses: pd.DataFrame, choice_col: str) -> dict[str, float]:
    entropies: dict[str, float] = {}
    for scenario, group in responses.groupby("scenario_id"):
        counter = Counter(str(v) for v in group[choice_col].dropna() if str(v).strip())
        entropies[str(scenario)] = float(entropy(list(counter.values()), base=2))
    return entropies


def _coverage_rate(choices: pd.Series, recommendation_sets: pd.Series) -> float:
    hits = []
    for choice, recs in zip(choices, recommendation_sets, strict=False):
        parsed = set(_parse_receivers(recs))
        if not parsed or pd.isna(choice):
            continue
        hits.append(str(choice).strip() in parsed)
    return float(np.mean(hits)) if hits else float("nan")


def _any_top3_coverage(rows: pd.DataFrame, recommendation_col: str) -> float:
    hits = []
    for _, row in rows.iterrows():
        recommendation = set(_parse_receivers(row.get(recommendation_col)))
        if not recommendation:
            continue
        top3 = {
            str(row[col]).strip()
            for col in ["rank_1", "rank_2", "rank_3"]
            if col in row and pd.notna(row[col]) and str(row[col]).strip()
        }
        if top3:
            hits.append(bool(top3 & recommendation))
    return float(np.mean(hits)) if hits else float("nan")


def _acceptable_set_coverage(rows: pd.DataFrame, recommendation_col: str) -> float:
    proportions = []
    for _, row in rows.iterrows():
        acceptable = set(_parse_receivers(row.get("acceptable_receivers")))
        recommendation = set(_parse_receivers(row.get(recommendation_col)))
        if not acceptable or not recommendation:
            continue
        proportions.append(len(acceptable & recommendation) / len(acceptable))
    return float(np.mean(proportions)) if proportions else float("nan")


def _to_jsonable(value: object) -> object:
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def analyse_coach_responses(
    response_csv: str,
    scenario_csv: str,
    output_json: str = "outputs/PluralPass_coach_analysis.json",
) -> dict:
    responses = pd.read_csv(response_csv)
    scenarios = pd.read_csv(scenario_csv)

    choice_col = _first_existing(responses.columns, ["rank_1", "top_1", "first_choice"])
    if choice_col is None:
        raise ValueError("response_csv must include rank_1, top_1 or first_choice")
    if "scenario_id" not in responses or "coach_id" not in responses:
        raise ValueError("response_csv must include coach_id and scenario_id")
    if "scenario_id" not in scenarios:
        raise ValueError("scenario_csv must include scenario_id")

    stage1 = _stage1_rows(responses)
    coach_entropy = _choice_entropy_by_scenario(stage1, choice_col)
    scenarios["scenario_id"] = scenarios["scenario_id"].astype(str)
    responses["scenario_id"] = responses["scenario_id"].astype(str)
    stage1["scenario_id"] = stage1["scenario_id"].astype(str)
    scenarios["coach_choice_entropy"] = scenarios["scenario_id"].map(coach_entropy)

    model_entropy_col = _first_existing(
        scenarios.columns, ["model_entropy", "receiver_entropy", "predictive_entropy"]
    )
    if model_entropy_col is None:
        raise ValueError(
            "scenario_csv must include model_entropy, receiver_entropy or predictive_entropy "
            "for the primary scenario-level association analysis"
        )
    complete = scenarios.dropna(subset=["coach_choice_entropy", model_entropy_col])
    association = spearmanr(complete[model_entropy_col], complete["coach_choice_entropy"])

    merged_stage1 = stage1.merge(scenarios, on="scenario_id", how="left", suffixes=("", "_scenario"))
    pluralpass_col = _first_existing(
        merged_stage1.columns,
        ["pluralpass_set", "model_recommendation_set", "recommendation_set", "aps_set"],
    )
    point_col = _first_existing(
        merged_stage1.columns,
        ["point_receiver", "point_recommendation", "top_model_receiver", "model_top1"],
    )

    coverage: dict[str, dict[str, float]] = {}
    if pluralpass_col is not None:
        coverage["pluralpass_set"] = {
            "first_choice": _coverage_rate(merged_stage1[choice_col], merged_stage1[pluralpass_col]),
            "any_top3": _any_top3_coverage(merged_stage1, pluralpass_col),
            "acceptable_set_mean_proportion": _acceptable_set_coverage(
                merged_stage1, pluralpass_col
            ),
        }
    if point_col is not None:
        coverage["point_recommendation"] = {
            "first_choice": _coverage_rate(merged_stage1[choice_col], merged_stage1[point_col]),
            "any_top3": _any_top3_coverage(merged_stage1, point_col),
            "acceptable_set_mean_proportion": _acceptable_set_coverage(merged_stage1, point_col),
        }

    scenario_table = scenarios[
        ["scenario_id", model_entropy_col, "coach_choice_entropy"]
        + (["conformal_set_size"] if "conformal_set_size" in scenarios else [])
    ].to_dict(orient="records")
    report = {
        "primary_endpoint": (
            "Spearman association between prespecified model uncertainty and "
            "scenario-level coach first-choice entropy"
        ),
        "model_uncertainty_column": model_entropy_col,
        "rho": float(association.statistic),
        "p_value": float(association.pvalue),
        "scenarios": len(complete),
        "coaches": int(responses["coach_id"].nunique()),
        "valid_stage1_responses": len(stage1),
        "coach_choice_entropy_by_scenario": coach_entropy,
        "coverage": coverage,
        "scenario_table": scenario_table,
        "status": (
            "observed primary analysis; report the effect estimate, uncertainty interval and "
            "the study's non-preregistered status transparently"
        ),
    }
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_to_jsonable(report), indent=2), encoding="utf-8")
    return report
