#!/usr/bin/env python3
"""Rebuild 40-scene matched-size comparisons from archived candidate scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

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


def parse_letters(value: object) -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    return {
        item.strip().upper()
        for item in str(value).replace("，", ",").replace(";", ",").split(",")
        if item.strip()
    }


def joined(values: set[str] | list[str]) -> str:
    return ",".join(sorted(values))


def entropy_from_choices(values: pd.Series) -> float:
    counts = values.dropna().astype(str).value_counts().to_numpy(dtype=float)
    probability = counts / counts.sum()
    return float(-(probability * np.log(probability)).sum())


def probability_entropy(probability: np.ndarray) -> float:
    probability = probability / probability.sum()
    return float(-(probability * np.log(np.clip(probability, 1e-15, None))).sum())


def stable_order(frame: pd.DataFrame, score: str) -> list[str]:
    ordered = frame.sort_values([score, "node_index"], ascending=[False, True], kind="stable")
    return ordered["candidate_letter"].astype(str).tolist()


def conformal_set(frame: pd.DataFrame, qhat: float, lam: float = 0.0, k_reg: int = 3) -> list[str]:
    ordered = frame.sort_values(["pluralpass_prob", "node_index"], ascending=[False, True], kind="stable")
    cumulative = ordered["pluralpass_prob"].to_numpy(dtype=float).cumsum()
    size = np.arange(1, len(ordered) + 1)
    score = cumulative + lam * np.maximum(size - k_reg, 0)
    count = min(int(np.searchsorted(score, qhat, side="left") + 1), len(ordered))
    return ordered["candidate_letter"].astype(str).tolist()[:count]


def expected_random_any(n: int, k: int, marked: int) -> float:
    if marked <= 0:
        return 0.0
    if k > n - marked:
        return 1.0
    return float(1.0 - math.comb(n - marked, k) / math.comb(n, k))


def method_metrics(set_letters: set[str], group: pd.DataFrame, actual: str, poor: set[str]) -> dict[str, float]:
    rank1 = group["rank_1"].astype(str).str.upper()
    acceptable = group["acceptable_receivers"].apply(parse_letters)
    return {
        "expert_rank1_coverage": float(rank1.isin(set_letters).mean()),
        "expert_any_acceptable_coverage": float(acceptable.apply(lambda values: bool(values & set_letters)).mean()),
        "observed_receiver_coverage": float(actual in set_letters),
        "poor_option_inclusion": float(bool(poor & set_letters)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild matched-size set comparisons from external-fold candidate scores and a "
            "controlled, de-identified coach-evaluation workbook."
        )
    )
    parser.add_argument("candidate_table", type=Path)
    parser.add_argument("evaluation_workbook", type=Path)
    parser.add_argument("set_efficiency_json", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--responses-sheet",
        default="coach_responses",
        help="Worksheet containing de-identified coach responses (default: coach_responses).",
    )
    parser.add_argument(
        "--scenario-sheet",
        default="scenario_summary",
        help="Worksheet containing scenario metadata (default: scenario_summary).",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(args.candidate_table)
    responses = pd.read_excel(args.evaluation_workbook, sheet_name=args.responses_sheet)
    scenario_workbook = pd.read_excel(
        args.evaluation_workbook, sheet_name=args.scenario_sheet
    )
    scenario_workbook["scenario_id"] = scenario_workbook["scenario_id"].astype(str)
    scenario_workbook = scenario_workbook.set_index("scenario_id")
    efficiency = json.loads(args.set_efficiency_json.read_text())

    scenario_rows: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    random_expected_rows: list[dict[str, object]] = []
    aps_raw_mismatches = []

    grouped_responses = {str(key): group.copy() for key, group in responses.groupby("scenario_id")}
    for scenario_id, frame in candidates.groupby("scenario_id", sort=True):
        scenario_id = str(scenario_id)
        group = grouped_responses[scenario_id]
        domain = str(frame["domain"].iloc[0])
        fold = DOMAIN_TO_FOLD[domain]
        fold_rules = efficiency["folds"][fold]["variants"]
        aps = conformal_set(frame, float(fold_rules["APS"]["qhat"]))
        raps = conformal_set(
            frame,
            float(fold_rules["RAPS_lambda0.005_k3"]["qhat"]),
            lam=0.005,
            k_reg=3,
        )
        archived_raw = parse_letters(frame["pluralpass_APS_set_letters"].iloc[0])
        if set(aps) != archived_raw:
            aps_raw_mismatches.append(
                {
                    "scenario_id": scenario_id,
                    "recomputed_APS": joined(aps),
                    "archived_raw_APS": joined(archived_raw),
                }
            )
        k = len(aps)
        pp_order = stable_order(frame, "pluralpass_prob")
        hgb_order = stable_order(frame, "hgb_receiver_score")
        mlp_order = stable_order(frame, "mlp_prob")
        method_sets = {
            "PluralPass APS": aps,
            "PluralPass RAPS lambda=0.005 k=3": raps,
            "same-size PluralPass Top-k": pp_order[:k],
            "same-size HGB set": hgb_order[:k],
            "same-size candidate-wise MLP set": mlp_order[:k],
            "fixed Top-3": pp_order[: min(3, len(pp_order))],
            "fixed Top-4": pp_order[: min(4, len(pp_order))],
            "fixed Top-5": pp_order[: min(5, len(pp_order))],
            "point recommendation": pp_order[:1],
        }
        actual = str(frame["actual_receiver_letter"].iloc[0]).upper()
        poor = parse_letters(scenario_workbook.loc[scenario_id, "expert_disfavored_letters"])
        p = frame["pluralpass_prob"].to_numpy(dtype=float)
        model_entropy = probability_entropy(p)
        evaluator_entropy = entropy_from_choices(group["rank_1"])
        candidate_count = len(frame)

        for method, letters in method_sets.items():
            letters_set = set(letters)
            metrics = method_metrics(letters_set, group, actual, poor)
            scenario_rows.append(
                {
                    "scenario_id": scenario_id,
                    "domain": domain,
                    "scenario_order": int(scenario_workbook.loc[scenario_id, "scenario_order"]),
                    "candidate_count": candidate_count,
                    "model_entropy_from_archived_probabilities": model_entropy,
                    "evaluator_first_choice_entropy": evaluator_entropy,
                    "method": method,
                    "set_letters": joined(letters),
                    "set_size": len(letters),
                    **metrics,
                    "coverage_efficiency": metrics["expert_rank1_coverage"] / max(len(letters), 1),
                }
            )
            for _, response in group.iterrows():
                acceptable = parse_letters(response["acceptable_receivers"])
                rank1 = str(response["rank_1"]).upper()
                long_rows.append(
                    {
                        "evaluator_id": response["coach_id"],
                        "scenario_id": scenario_id,
                        "domain": domain,
                        "scenario_order": int(response["scenario_order"]),
                        "candidate_count": candidate_count,
                        "model_entropy_from_archived_probabilities": model_entropy,
                        "method": method,
                        "set_size": len(letters),
                        "rank1_covered": int(rank1 in letters_set),
                        "any_acceptable_covered": int(bool(acceptable & letters_set)),
                    }
                )

        union_acceptable = set().union(*(parse_letters(value) for value in group["acceptable_receivers"]))
        n = candidate_count
        random_expected_rows.append(
            {
                "scenario_id": scenario_id,
                "domain": domain,
                "candidate_count": n,
                "matched_set_size": k,
                "expected_rank1_coverage": k / n,
                "expected_any_union_acceptable_coverage": expected_random_any(n, k, len(union_acceptable)),
                "expected_observed_receiver_coverage": k / n,
                "expected_poor_option_inclusion": expected_random_any(n, k, len(poor)),
            }
        )

    scenario_frame = pd.DataFrame(scenario_rows)
    long_frame = pd.DataFrame(long_rows)
    random_frame = pd.DataFrame(random_expected_rows)
    scenario_frame.to_csv(args.output / "matched_size_scene_method_metrics.csv", index=False, encoding="utf-8-sig")
    long_frame.to_csv(args.output / "matched_size_evaluator_scene_method_long.csv", index=False, encoding="utf-8-sig")
    random_frame.to_csv(args.output / "same_size_random_exact_expectations.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(aps_raw_mismatches).to_csv(args.output / "APS_reconstruction_mismatches.csv", index=False, encoding="utf-8-sig")

    entropy_by_scene = scenario_frame.drop_duplicates("scenario_id")[[
        "scenario_id", "model_entropy_from_archived_probabilities"
    ]]
    lower_cut, upper_cut = entropy_by_scene[
        "model_entropy_from_archived_probabilities"
    ].quantile([1 / 3, 2 / 3]).to_numpy(dtype=float)
    scenario_frame["posthoc_entropy_stratum"] = pd.cut(
        scenario_frame["model_entropy_from_archived_probabilities"],
        [-np.inf, lower_cut, upper_cut, np.inf],
        labels=["low", "medium", "high"],
        include_lowest=True,
    )
    stratum_summary = (
        scenario_frame[
            scenario_frame["method"].isin(
                ["PluralPass APS", "PluralPass RAPS lambda=0.005 k=3"]
            )
        ]
        .groupby(["posthoc_entropy_stratum", "method"], observed=True, as_index=False)
        .agg(
            n_scenarios=("scenario_id", "nunique"),
            entropy_min=("model_entropy_from_archived_probabilities", "min"),
            entropy_max=("model_entropy_from_archived_probabilities", "max"),
            mean_set_size=("set_size", "mean"),
            expert_rank1_coverage=("expert_rank1_coverage", "mean"),
            observed_receiver_coverage=("observed_receiver_coverage", "mean"),
        )
    )
    stratum_summary.to_csv(
        args.output / "posthoc_model_entropy_strata_APS_RAPS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        scenario_frame.groupby("method", as_index=False)
        .agg(
            n_scenarios=("scenario_id", "nunique"),
            mean_set_size=("set_size", "mean"),
            expert_rank1_coverage=("expert_rank1_coverage", "mean"),
            expert_any_acceptable_coverage=("expert_any_acceptable_coverage", "mean"),
            observed_receiver_coverage=("observed_receiver_coverage", "mean"),
            poor_option_inclusion=("poor_option_inclusion", "mean"),
            coverage_efficiency=("coverage_efficiency", "mean"),
        )
    )
    random_summary = {
        "method": "same-size random set (exact expectation)",
        "n_scenarios": len(random_frame),
        "mean_set_size": random_frame["matched_set_size"].mean(),
        "expert_rank1_coverage": random_frame["expected_rank1_coverage"].mean(),
        "expert_any_acceptable_coverage": random_frame["expected_any_union_acceptable_coverage"].mean(),
        "observed_receiver_coverage": random_frame["expected_observed_receiver_coverage"].mean(),
        "poor_option_inclusion": random_frame["expected_poor_option_inclusion"].mean(),
        "coverage_efficiency": (
            random_frame["expected_rank1_coverage"] / random_frame["matched_set_size"]
        ).mean(),
    }
    summary = pd.concat([summary, pd.DataFrame([random_summary])], ignore_index=True)
    summary.to_csv(args.output / "matched_size_method_summary.csv", index=False, encoding="utf-8-sig")

    wide = scenario_frame.pivot(index="scenario_id", columns="method", values="expert_rank1_coverage")
    rng = np.random.default_rng(20260712)
    comparisons = [
        ("PluralPass APS", "same-size PluralPass Top-k"),
        ("PluralPass APS", "same-size HGB set"),
        ("PluralPass APS", "same-size candidate-wise MLP set"),
        ("PluralPass RAPS lambda=0.005 k=3", "same-size HGB set"),
        ("PluralPass RAPS lambda=0.005 k=3", "same-size candidate-wise MLP set"),
    ]
    bootstrap_rows = []
    for first, second in comparisons:
        differences = (wide[first] - wide[second]).to_numpy(dtype=float)
        boots = np.asarray(
            [rng.choice(differences, size=len(differences), replace=True).mean() for _ in range(50000)]
        )
        bootstrap_rows.append(
            {
                "comparison": f"{first} minus {second}",
                "mean_difference": differences.mean(),
                "scene_cluster_bootstrap_ci_low": np.quantile(boots, 0.025),
                "scene_cluster_bootstrap_ci_high": np.quantile(boots, 0.975),
                "bootstrap_repetitions": 50000,
                "bootstrap_seed": 20260712,
            }
        )
    pd.DataFrame(bootstrap_rows).to_csv(
        args.output / "matched_size_paired_scene_bootstrap.csv", index=False, encoding="utf-8-sig"
    )

    metadata = {
        "candidate_table_sha256": hashlib.sha256(args.candidate_table.read_bytes()).hexdigest(),
        "evaluation_workbook_sha256": hashlib.sha256(args.evaluation_workbook.read_bytes()).hexdigest(),
        "set_efficiency_sha256": hashlib.sha256(args.set_efficiency_json.read_bytes()).hexdigest(),
        "APS_reconstruction_mismatch_scenarios": len(aps_raw_mismatches),
        "posthoc_model_entropy_tertile_cutpoints": {
            "low_to_medium": lower_cut,
            "medium_to_high": upper_cut,
            "classification": "low <= first tertile; medium > first and <= second; high > second",
        },
        "random_set_method": "closed-form exact expectation under uniform sampling without replacement",
        "tie_breaking": "score descending, then node_index ascending",
        "human_evaluation_provenance": (
            "Set membership and model scores are reconstructed from archived external-fold "
            "predictions and linked to de-identified responses from the completed coach study. "
            "The study received Guangzhou Sport University ethics approval 2024LCLL-71 and "
            "participants provided informed consent. Individual-level source files remain "
            "restricted and are not redistributed by this script."
        ),
    }
    (args.output / "matched_size_analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"summary": summary.to_dict("records"), "metadata": metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
