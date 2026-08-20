from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from pluralpass.config import artifact_run_name


def collect_baseline_matrix(config: dict[str, Any]) -> dict[str, Any]:
    """Summarise all completed non-neural baselines across leave-one-domain folds."""
    run_name = artifact_run_name(config)
    baseline_root = Path("artifacts/baselines") / run_name
    manifest_path = Path(config["data"]["processed_dir"]) / "splits" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    folds = sorted(manifest)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    vaep_status: dict[str, int] = {}
    for fold in folds:
        path = baseline_root / fold / "metrics.json"
        if not path.exists():
            missing.append(fold)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        domain = _domain_label(fold, manifest)
        rows.extend(_metric_rows(domain, fold, payload))
        status = str(payload.get("vaep", {}).get("status", "not_reported"))
        vaep_status[status] = vaep_status.get(status, 0) + 1

    macro = _macro_average(rows)
    report = {
        "run_name": run_name,
        "completed_folds": sorted({row["fold"] for row in rows}),
        "missing_folds": missing,
        "macro_average": macro,
        "rows": rows,
        "vaep_status_counts": vaep_status,
        "interpretation": (
            "Grid-xT is a reproducible spatial-value baseline fitted within each training fold. "
            "The VAEP entry is reported as unavailable unless socceraction and full SPADL action "
            "sequences are present; no proxy is silently substituted for VAEP."
        ),
    }
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "PluralPass_baseline_matrix.json"
    markdown_path = output_dir / "PluralPass_baseline_matrix.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    report["json_path"] = str(json_path)
    report["markdown_path"] = str(markdown_path)
    return report


def _domain_label(fold: str, manifest: dict[str, Any]) -> str:
    entry = manifest.get(fold, {})
    domain = entry.get("test_domain", fold)
    return str(domain).replace("|", " ")


def _metric_rows(domain: str, fold: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, metrics in payload.get("receiver", {}).items():
        rows.append(
            {
                "fold": fold,
                "domain": domain,
                "task": "receiver",
                "model": model,
                "top1": metrics.get("top1"),
                "top3": metrics.get("top3"),
                "nll": metrics.get("nll"),
            }
        )
    for model, metrics in payload.get("completion", {}).items():
        rows.append(
            {
                "fold": fold,
                "domain": domain,
                "task": "completion",
                "model": model,
                "auroc": metrics.get("auroc"),
                "brier": metrics.get("brier"),
                "ece": metrics.get("ece"),
            }
        )
    for model, metrics in payload.get("value", {}).items():
        rows.append(
            {
                "fold": fold,
                "domain": domain,
                "task": "value",
                "model": model,
                "mae": metrics.get("mae"),
                "spearman": metrics.get("spearman"),
            }
        )
    return rows


def _macro_average(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["task"], row["model"]), []).append(row)
    output = []
    for (task, model), items in sorted(grouped.items()):
        metrics: dict[str, Any] = {
            "task": task,
            "model": model,
            "completed_folds": len(items),
        }
        for key in sorted(set().union(*(item.keys() for item in items))):
            if key in {"fold", "domain", "task", "model"}:
                continue
            values = [float(item[key]) for item in items if item.get(key) is not None]
            if values:
                metrics[key] = mean(values)
        output.append(metrics)
    return output


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PluralPass baseline matrix",
        "",
        (
            "This report summarises completed non-neural baselines across leave-one-domain "
            "folds. Missing folds are not imputed."
        ),
        "",
        f"- Completed folds: {len(report['completed_folds'])}",
        f"- Missing folds: {', '.join(report['missing_folds']) if report['missing_folds'] else 'none'}",
        f"- VAEP status counts: {report['vaep_status_counts']}",
        "",
        "## Macro-average performance",
        "",
        "| Task | Model | Folds | Top-1 | Top-3 | NLL | AUROC | Brier | ECE | MAE | Spearman |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["macro_average"]:
        lines.append(
            "| "
            f"{row['task']} | {row['model']} | {row['completed_folds']} | "
            f"{_fmt(row.get('top1'))} | {_fmt(row.get('top3'))} | {_fmt(row.get('nll'))} | "
            f"{_fmt(row.get('auroc'))} | {_fmt(row.get('brier'), 4)} | {_fmt(row.get('ece'), 4)} | "
            f"{_fmt(row.get('mae'), 4)} | {_fmt(row.get('spearman'))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            report["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)
