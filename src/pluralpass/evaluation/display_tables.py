from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


def build_display_tables(config: dict[str, Any]) -> dict[str, Any]:
    """Build manuscript and supplementary tables from locked analysis outputs."""

    output_dir = Path("outputs") / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "table1_benchmark_domains": _benchmark_domains(config, output_dir),
        "table2_primary_external_validation": _primary_external_validation(output_dir),
        "supp_table_candidate_mlp": _candidate_mlp(output_dir),
        "supp_table_counterfactual_stress": _counterfactual_stress(output_dir),
        "supp_table_coach_experiment": _coach_experiment(output_dir),
    }
    report = {
        "tables": tables,
        "interpretation": (
            "These tables are generated from locked analysis artifacts. They should be "
            "treated as manuscript-display inputs rather than new analyses."
        ),
    }
    index_json = output_dir / "display_table_index.json"
    index_md = output_dir / "display_table_index.md"
    index_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    index_md.write_text(_index_markdown(report), encoding="utf-8")
    report["json_path"] = str(index_json)
    report["markdown_path"] = str(index_md)
    return report


def _benchmark_domains(config: dict[str, Any], output_dir: Path) -> dict[str, str]:
    passes_path = Path(config["data"]["processed_dir"]) / "passes.csv.gz"
    data = pd.read_csv(passes_path)
    grouped = (
        data.groupby(["competition", "season", "domain"], dropna=False)
        .agg(
            passes=("event_id", "count"),
            matches=("match_id", "nunique"),
            median_visible_players=("visible_players", "median"),
            median_visible_area=("visible_area_fraction", "median"),
            completion_rate=("pass_completed", "mean"),
        )
        .reset_index()
        .sort_values(["passes", "competition"], ascending=[False, True])
    )
    rows = [
        {
            "competition": row["competition"],
            "season": row["season"],
            "domain": row["domain"].replace("|", " "),
            "passes": int(row["passes"]),
            "matches": int(row["matches"]),
            "median_visible_players": round(float(row["median_visible_players"]), 1),
            "median_visible_area": round(float(row["median_visible_area"]), 3),
            "completion_rate": round(float(row["completion_rate"]), 3),
        }
        for _, row in grouped.iterrows()
    ]
    return _write_table(output_dir, "table1_benchmark_domains", rows)


def _primary_external_validation(output_dir: Path) -> dict[str, str]:
    report = _read_json(Path("outputs") / "PluralPass_formal_results_index.json")
    rows = []
    for fold, payload in report["folds"].items():
        pp = payload["pluralpass"]
        rows.append(
            {
                "fold": fold,
                "held_domain": payload["domain"],
                "test_passes": payload["test_events"],
                "top1": _r(pp["receiver_top1"]),
                "top3": _r(pp["receiver_top3"]),
                "completion_auroc": _r(pp["completion_auroc"]),
                "value_spearman": _r(pp["value_spearman"]),
                "aps_coverage": _r(pp["set_coverage"]),
                "mean_set_size": _r(pp["mean_set_size"], 2),
                "abstention_rate": _r(pp["abstention_rate"]),
            }
        )
    return _write_table(output_dir, "table2_primary_external_validation", rows)


def _candidate_mlp(output_dir: Path) -> dict[str, str]:
    report = _read_json(Path("outputs") / "PluralPass_candidate_mlp_comparison.json")
    rows = []
    for metric, payload in report["summary"].items():
        rows.append(
            {
                "metric": metric,
                "completed_folds": payload["completed_folds"],
                "pluralpass_better_folds": payload["pluralpass_better_folds"],
                "mean_delta_pluralpass_minus_candidate_mlp": _r(
                    payload["mean_difference_pluralpass_minus_candidate_mlp"], 4
                ),
            }
        )
    return _write_table(output_dir, "supp_table_candidate_mlp", rows)


def _counterfactual_stress(output_dir: Path) -> dict[str, str]:
    report = _read_json(Path("outputs") / "PluralPass_counterfactual_stress.json")
    rows = []
    for perturbation, payload in report["pooled_macro_average"].items():
        rows.append(
            {
                "perturbation": perturbation,
                "top1": _r(payload["all_events.receiver.top1"]),
                "top3": _r(payload["all_events.receiver.top3"]),
                "completion_auroc": _r(payload["all_events.completion.auroc"]),
                "value_spearman": _r(payload["all_events.value.spearman"]),
                "coverage": _r(payload["all_events.recommendation_set.coverage"]),
                "mean_set_size": _r(payload["all_events.recommendation_set.mean_set_size"], 2),
                "abstention_rate": _r(payload["all_events.recommendation_set.abstention_rate"]),
                "receiver_visible_rate": _r(payload["all_events.receiver_visible_rate"]),
                "mean_dropped_nodes": _r(payload["all_events.mean_dropped_nodes"], 2),
            }
        )
    return _write_table(output_dir, "supp_table_counterfactual_stress", rows)


def _coach_experiment(output_dir: Path) -> dict[str, str]:
    report = _read_json(Path("outputs") / "coach_analysis" / "PluralPass_coach_analysis.json")
    quality = report["data_quality"]
    h3 = report["h3"]
    h4 = report["h4"]
    rows = [
        {
            "endpoint": "complete_response_grid",
            "estimate": str(quality["complete_coach_scenario_grid"]),
            "ci_or_aux": f"{quality['n_coaches']} raters × {quality['n_scenarios']} scenarios",
        },
        {
            "endpoint": "attention_check_pass_rate",
            "estimate": _r(quality["attention_check_pass_rate"]),
            "ci_or_aux": "",
        },
        {
            "endpoint": "model_entropy_vs_coach_entropy_spearman",
            "estimate": _r(h3["model_entropy_spearman_rho"]),
            "ci_or_aux": (
                f"{_r(h3['model_entropy_bootstrap_ci_low'])}–"
                f"{_r(h3['model_entropy_bootstrap_ci_high'])}; "
                f"p={_r(h3['model_entropy_spearman_p'], 4)}"
            ),
        },
        {
            "endpoint": "conformal_set_size_vs_coach_entropy_spearman",
            "estimate": _r(h3["conformal_set_size_spearman_rho"]),
            "ci_or_aux": (
                f"{_r(h3['conformal_set_size_bootstrap_ci_low'])}–"
                f"{_r(h3['conformal_set_size_bootstrap_ci_high'])}; "
                f"p={_r(h3['conformal_set_size_spearman_p'], 4)}"
            ),
        },
        {
            "endpoint": "pluralpass_first_choice_coverage",
            "estimate": _r(h4["pluralpass_first_choice_coverage"]),
            "ci_or_aux": "",
        },
        {
            "endpoint": "point_first_choice_coverage",
            "estimate": _r(h4["point_first_choice_coverage"]),
            "ci_or_aux": "",
        },
        {
            "endpoint": "coverage_difference_pluralpass_minus_point",
            "estimate": _r(h4["first_choice_coverage_difference"]),
            "ci_or_aux": (
                f"{_r(h4['first_choice_coverage_difference_ci_low'])}–"
                f"{_r(h4['first_choice_coverage_difference_ci_high'])}"
            ),
        },
    ]
    return _write_table(output_dir, "supp_table_coach_experiment", rows)


def _write_table(output_dir: Path, stem: str, rows: list[dict[str, Any]]) -> dict[str, str]:
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    if not rows:
        raise ValueError(f"No rows generated for {stem}")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(_markdown_table(rows), encoding="utf-8")
    return {"csv": str(csv_path), "markdown": str(md_path), "rows": str(len(rows))}


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[key]) for key in headers) + " |")
    return "\n".join(lines) + "\n"


def _index_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PluralPass display table index",
        "",
        report["interpretation"],
        "",
        "| Table | Rows | CSV | Markdown |",
        "|---|---:|---|---|",
    ]
    for name, paths in report["tables"].items():
        lines.append(f"| {name} | {paths['rows']} | `{paths['csv']}` | `{paths['markdown']}` |")
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _r(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"
