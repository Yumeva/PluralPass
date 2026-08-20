from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pluralpass.config import artifact_run_name

METRICS = {
    "receiver.top1": ("receiver", "top1", True),
    "receiver.top3": ("receiver", "top3", True),
    "receiver.nll": ("receiver", "nll", False),
    "completion.auroc": ("completion", "auroc", True),
    "completion.brier": ("completion", "brier", False),
    "value.spearman": ("value", "spearman", True),
    "value.mae": ("value", "mae", False),
    "recommendation_set.coverage": ("recommendation_set", "coverage", True),
    "recommendation_set.mean_set_size": ("recommendation_set", "mean_set_size", False),
}


def compare_candidate_mlp_baseline(config: dict[str, Any]) -> dict[str, Any]:
    """Compare completed PluralPass folds against the candidate-wise MLP baseline."""

    primary_run = artifact_run_name(config)
    baseline_run = "pluralpass-neural-candidate-mlp"
    processed = Path(config["data"]["processed_dir"])
    manifest_path = processed / "splits" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result_root = Path("artifacts") / "results"

    folds: dict[str, Any] = {}
    for fold, metadata in sorted(manifest.items()):
        primary_path = result_root / primary_run / fold / "metrics.json"
        baseline_path = result_root / baseline_run / fold / "metrics.json"
        if not primary_path.exists() or not baseline_path.exists():
            continue
        primary = json.loads(primary_path.read_text(encoding="utf-8"))
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        metric_rows = {}
        for key, (section, metric, higher_is_better) in METRICS.items():
            primary_value = float(primary[section][metric])
            baseline_value = float(baseline[section][metric])
            difference = primary_value - baseline_value
            pluralpass_better = difference > 0 if higher_is_better else difference < 0
            metric_rows[key] = {
                "pluralpass": primary_value,
                "candidate_mlp": baseline_value,
                "difference_pluralpass_minus_candidate_mlp": difference,
                "higher_is_better": higher_is_better,
                "pluralpass_better": pluralpass_better,
            }
        folds[fold] = {
            "domain": str(metadata["held_domain"]).replace("|", " "),
            "test_events": int(primary["test_events"]),
            "metrics": metric_rows,
        }

    summary = _summarise(folds)
    report = {
        "primary_run": primary_run,
        "baseline_run": baseline_run,
        "completed_folds": sorted(folds),
        "missing_folds": sorted(set(manifest) - set(folds)),
        "folds": folds,
        "summary": summary,
        "interpretation": (
            "This report compares only completed leave-one-domain folds. It should "
            "be read as a neural-capacity comparator, not as the primary validation "
            "scheme. A grouped 10-fold analysis can be added as a supplementary "
            "stability check."
        ),
    }
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    json_path = output_dir / "PluralPass_candidate_mlp_comparison.json"
    markdown_path = output_dir / "PluralPass_candidate_mlp_comparison.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    report["json_path"] = str(json_path)
    report["markdown_path"] = str(markdown_path)
    return report


def compare_grouped10_stability(config: dict[str, Any]) -> dict[str, Any]:
    """Compare completed match-grouped 10-fold results against candidate-wise MLP.

    The function is intentionally conservative: it only reports folds for which both
    PluralPass and candidate-wise MLP metrics are present, and it records all missing
    folds rather than imputing them.
    """

    primary_run = artifact_run_name(config)
    baseline_run = "pluralpass-neural-candidate-mlp"
    processed = Path(config["data"]["processed_dir"])
    manifest_path = processed / "splits" / "grouped_10fold_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result_root = Path("artifacts") / "results"

    folds: dict[str, Any] = {}
    for fold, metadata in sorted(manifest.items()):
        primary_path = result_root / primary_run / fold / "metrics.json"
        baseline_path = result_root / baseline_run / fold / "metrics.json"
        if not primary_path.exists() or not baseline_path.exists():
            continue
        primary = json.loads(primary_path.read_text(encoding="utf-8"))
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        metric_rows = {}
        for key, (section, metric, higher_is_better) in METRICS.items():
            primary_value = float(primary[section][metric])
            baseline_value = float(baseline[section][metric])
            difference = primary_value - baseline_value
            pluralpass_better = difference > 0 if higher_is_better else difference < 0
            metric_rows[key] = {
                "pluralpass": primary_value,
                "candidate_mlp": baseline_value,
                "difference_pluralpass_minus_candidate_mlp": difference,
                "higher_is_better": higher_is_better,
                "pluralpass_better": pluralpass_better,
            }
        folds[fold] = {
            "scheme": metadata.get("scheme", "match_grouped_10fold"),
            "test_events": int(primary["test_events"]),
            "test_domains": list(metadata.get("test_domains", [])),
            "metrics": metric_rows,
        }

    summary = _summarise(folds)
    report = {
        "primary_run": primary_run,
        "baseline_run": baseline_run,
        "scheme": "match_grouped_10fold",
        "completed_folds": sorted(folds),
        "missing_folds": sorted(set(manifest) - set(folds)),
        "folds": folds,
        "summary": summary,
        "interpretation": (
            "This report is a supplementary match-grouped 10-fold stability check. "
            "Each fold holds out complete matches, so the same match cannot appear "
            "in both training and test partitions. It tests within-public-corpus "
            "stability and neural-baseline directionality; it does not replace the "
            "leave-one-competition-season analysis as the primary domain-shift "
            "validation."
        ),
    }
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    json_path = output_dir / "PluralPass_grouped10_stability.json"
    markdown_path = output_dir / "PluralPass_grouped10_stability.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(_grouped10_markdown(report), encoding="utf-8")
    _write_grouped10_table(report, output_dir / "tables" / "supp_table_grouped10_stability.csv")
    report["json_path"] = str(json_path)
    report["markdown_path"] = str(markdown_path)
    return report


def _summarise(folds: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for key in METRICS:
        rows = [fold["metrics"][key] for fold in folds.values()]
        if not rows:
            summary[key] = {
                "completed_folds": 0,
                "pluralpass_better_folds": 0,
                "mean_difference_pluralpass_minus_candidate_mlp": None,
            }
            continue
        summary[key] = {
            "completed_folds": len(rows),
            "pluralpass_better_folds": int(sum(row["pluralpass_better"] for row in rows)),
            "mean_difference_pluralpass_minus_candidate_mlp": float(
                np.mean([row["difference_pluralpass_minus_candidate_mlp"] for row in rows])
            ),
        }
    return summary


def _fmt(value: Any, signed: bool = False) -> str:
    if value is None:
        return "pending"
    number = float(value)
    if signed:
        return f"{number:+.3f}"
    return f"{number:.3f}"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PluralPass candidate-MLP neural baseline comparison",
        "",
        "This report compares completed leave-one-domain folds for PluralPass and a candidate-wise MLP neural baseline.",
        "",
        f"- Completed folds: {len(report['completed_folds'])}",
        f"- Missing folds: {', '.join(report['missing_folds']) if report['missing_folds'] else 'none'}",
        "",
        "## Stability summary",
        "",
        "| Metric | Completed folds | PluralPass better folds | Mean Δ PluralPass − candidate MLP |",
        "|---|---:|---:|---:|",
    ]
    for key, row in report["summary"].items():
        lines.append(
            f"| `{key}` | {row['completed_folds']} | {row['pluralpass_better_folds']} | "
            f"{_fmt(row['mean_difference_pluralpass_minus_candidate_mlp'], signed=True)} |"
        )
    lines.extend(
        [
            "",
            "## Fold details",
            "",
            "| Fold | Domain | Test events | ΔTop-1 | ΔTop-3 | ΔAUROC | ΔValue Spearman |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for fold, row in report["folds"].items():
        metrics = row["metrics"]
        lines.append(
            f"| `{fold}` | {row['domain']} | {row['test_events']} | "
            f"{_fmt(metrics['receiver.top1']['difference_pluralpass_minus_candidate_mlp'], signed=True)} | "
            f"{_fmt(metrics['receiver.top3']['difference_pluralpass_minus_candidate_mlp'], signed=True)} | "
            f"{_fmt(metrics['completion.auroc']['difference_pluralpass_minus_candidate_mlp'], signed=True)} | "
            f"{_fmt(metrics['value.spearman']['difference_pluralpass_minus_candidate_mlp'], signed=True)} |"
        )
    lines.extend(["", report["interpretation"], ""])
    return "\n".join(lines)


def _grouped10_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PluralPass match-grouped 10-fold stability check",
        "",
        "This report compares completed match-grouped 10-fold folds for PluralPass and a candidate-wise MLP neural baseline.",
        "",
        f"- Completed folds: {len(report['completed_folds'])}/10",
        f"- Missing folds: {', '.join(report['missing_folds']) if report['missing_folds'] else 'none'}",
        "",
        "## Stability summary",
        "",
        "| Metric | Completed folds | PluralPass better folds | Mean Δ PluralPass − candidate MLP |",
        "|---|---:|---:|---:|",
    ]
    for key, row in report["summary"].items():
        lines.append(
            f"| `{key}` | {row['completed_folds']} | {row['pluralpass_better_folds']} | "
            f"{_fmt(row['mean_difference_pluralpass_minus_candidate_mlp'], signed=True)} |"
        )
    lines.extend(
        [
            "",
            "## Fold details",
            "",
            "| Fold | Test events | Test domains | ΔTop-1 | ΔTop-3 | ΔAUROC | ΔValue Spearman | ΔSet size |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for fold, row in report["folds"].items():
        metrics = row["metrics"]
        domains = "; ".join(str(domain).replace("|", " ") for domain in row["test_domains"])
        lines.append(
            f"| `{fold}` | {row['test_events']} | {domains} | "
            f"{_fmt(metrics['receiver.top1']['difference_pluralpass_minus_candidate_mlp'], signed=True)} | "
            f"{_fmt(metrics['receiver.top3']['difference_pluralpass_minus_candidate_mlp'], signed=True)} | "
            f"{_fmt(metrics['completion.auroc']['difference_pluralpass_minus_candidate_mlp'], signed=True)} | "
            f"{_fmt(metrics['value.spearman']['difference_pluralpass_minus_candidate_mlp'], signed=True)} | "
            f"{_fmt(metrics['recommendation_set.mean_set_size']['difference_pluralpass_minus_candidate_mlp'], signed=True)} |"
        )
    lines.extend(["", report["interpretation"], ""])
    return "\n".join(lines)


def _write_grouped10_table(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "fold",
        "test_events",
        "test_domains",
        "metric",
        "pluralpass",
        "candidate_mlp",
        "difference_pluralpass_minus_candidate_mlp",
        "higher_is_better",
        "pluralpass_better",
    ]
    rows = [",".join(header)]
    for fold, fold_row in report["folds"].items():
        domains = "; ".join(str(domain).replace("|", " ") for domain in fold_row["test_domains"])
        for metric, metric_row in fold_row["metrics"].items():
            rows.append(
                ",".join(
                    [
                        fold,
                        str(fold_row["test_events"]),
                        json.dumps(domains, ensure_ascii=False),
                        metric,
                        f"{metric_row['pluralpass']:.12g}",
                        f"{metric_row['candidate_mlp']:.12g}",
                        f"{metric_row['difference_pluralpass_minus_candidate_mlp']:.12g}",
                        str(metric_row["higher_is_better"]),
                        str(metric_row["pluralpass_better"]),
                    ]
                )
            )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
