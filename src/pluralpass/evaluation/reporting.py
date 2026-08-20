from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pluralpass.config import artifact_run_name


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_number(value: float, digits: int, signed: bool = False) -> str:
    text = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return text.replace("-", "−")


def _format_interval(metric: dict[str, Any], section: str, name: str, digits: int = 3) -> str:
    value = float(metric[section][name])
    interval = metric.get("confidence_intervals_95", {}).get(section, {}).get(name)
    if not interval:
        return _format_number(value, digits)
    return (
        f"{_format_number(value, digits)} "
        f"({_format_number(float(interval['lower']), digits)}–"
        f"{_format_number(float(interval['upper']), digits)})"
    )


def _format_difference(baseline: dict[str, Any], section: str, name: str, digits: int = 3) -> str:
    differences = baseline.get("pluralpass_minus_hist_gradient_boosting")
    if not differences:
        return "pending"
    metric = differences[section][name]
    value = float(metric["difference"])
    return (
        f"{_format_number(value, digits, signed=True)} "
        f"({_format_number(float(metric['lower']), digits)}–"
        f"{_format_number(float(metric['upper']), digits)})"
    )


def _domain_label(fold: str, manifest: dict[str, Any] | None) -> str:
    if manifest and fold in manifest:
        return str(manifest[fold]["held_domain"]).replace("|", " ")
    return fold.replace("_", " ")


def collect_formal_results(config: dict[str, Any]) -> dict[str, Any]:
    """Collect completed leave-one-domain results and write manuscript-ready tables."""

    run_name = artifact_run_name(config)
    result_root = Path("artifacts/results") / run_name
    baseline_root = Path("artifacts/baselines") / run_name
    split_manifest_path = Path(config["data"]["processed_dir"]) / "splits" / "manifest.json"
    split_manifest = _read_json(split_manifest_path) if split_manifest_path.exists() else None

    expected_folds = sorted(split_manifest) if split_manifest else []
    completed: dict[str, dict[str, Any]] = {}
    pending = []
    for fold in expected_folds:
        metrics_path = result_root / fold / "metrics.json"
        baseline_path = baseline_root / fold / "metrics.json"
        if metrics_path.exists() and baseline_path.exists():
            completed[fold] = {
                "metrics": _read_json(metrics_path),
                "baseline": _read_json(baseline_path),
            }
        else:
            pending.append(
                {
                    "fold": fold,
                    "missing_metrics": not metrics_path.exists(),
                    "missing_baselines": not baseline_path.exists(),
                }
            )

    summary: dict[str, Any] = {
        "run_name": run_name,
        "completed_folds": list(completed),
        "pending_folds": pending,
        "folds": {},
    }
    for fold, payload in completed.items():
        metrics = payload["metrics"]
        baseline = payload["baseline"]
        summary["folds"][fold] = {
            "domain": _domain_label(fold, split_manifest),
            "test_events": metrics["test_events"],
            "pluralpass": {
                "receiver_top1": metrics["receiver"]["top1"],
                "receiver_top3": metrics["receiver"]["top3"],
                "completion_auroc": metrics["completion"]["auroc"],
                "completion_brier": metrics["completion"]["brier"],
                "value_spearman": metrics["value"]["spearman"],
                "set_coverage": metrics["recommendation_set"]["coverage"],
                "mean_set_size": metrics["recommendation_set"]["mean_set_size"],
                "abstention_rate": metrics["recommendation_set"]["abstention_rate"],
            },
            "pluralpass_minus_hgb": baseline.get("pluralpass_minus_hist_gradient_boosting"),
        }

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    json_path = output_dir / "PluralPass_formal_results_index.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown_path = output_dir / "PluralPass_formal_results_index.md"
    markdown_path.write_text(_markdown_report(summary, completed, split_manifest), encoding="utf-8")
    summary["json_path"] = str(json_path)
    summary["markdown_path"] = str(markdown_path)
    return summary


def _markdown_report(
    summary: dict[str, Any],
    completed: dict[str, dict[str, Any]],
    split_manifest: dict[str, Any] | None,
) -> str:
    lines = [
        "# PluralPass formal leave-one-domain results index",
        "",
        "This auto-generated report includes only folds with both PluralPass and",
        "histogram-gradient-boosting baseline metrics. Pending folds are listed but",
        "not imputed.",
        "",
        f"Completed folds: {len(summary['completed_folds'])}",
        f"Pending folds: {len(summary['pending_folds'])}",
        "",
        "## PluralPass external-test metrics",
        "",
        "| Held domain | Test passes | Top-1 | Top-3 | AUROC | Brier | Value Spearman | APS coverage | Set size | Abstention |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold, payload in completed.items():
        metrics = payload["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    _domain_label(fold, split_manifest),
                    str(int(metrics["test_events"])),
                    _format_interval(metrics, "receiver", "top1"),
                    _format_interval(metrics, "receiver", "top3"),
                    _format_interval(metrics, "completion", "auroc"),
                    _format_interval(metrics, "completion", "brier", digits=4),
                    _format_interval(metrics, "value", "spearman"),
                    _format_interval(metrics, "recommendation_set", "coverage"),
                    _format_interval(metrics, "recommendation_set", "mean_set_size", digits=2),
                    _format_interval(metrics, "recommendation_set", "abstention_rate"),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## PluralPass minus histogram-gradient-boosting baseline",
            "",
            "| Held domain | Top-1 | Top-3 | NLL | AUROC | Brier | Value MAE | Value Spearman |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for fold, payload in completed.items():
        baseline = payload["baseline"]
        lines.append(
            "| "
            + " | ".join(
                [
                    _domain_label(fold, split_manifest),
                    _format_difference(baseline, "receiver", "top1"),
                    _format_difference(baseline, "receiver", "top3"),
                    _format_difference(baseline, "receiver", "nll"),
                    _format_difference(baseline, "completion", "auroc"),
                    _format_difference(baseline, "completion", "brier", digits=4),
                    _format_difference(baseline, "value", "mae", digits=4),
                    _format_difference(baseline, "value", "spearman"),
                ]
            )
            + " |"
        )

    if summary["pending_folds"]:
        lines.extend(
            [
                "",
                "## Pending folds",
                "",
                "| Fold | Missing metrics | Missing baselines |",
                "|---|---:|---:|",
            ]
        )
        for row in summary["pending_folds"]:
            lines.append(
                f"| `{row['fold']}` | {row['missing_metrics']} | {row['missing_baselines']} |"
            )
    return "\n".join(lines) + "\n"
