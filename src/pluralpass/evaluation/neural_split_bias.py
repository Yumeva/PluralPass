from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pluralpass.config import artifact_run_name

METRICS = {
    "receiver.top1": ("receiver", "top1"),
    "receiver.top3": ("receiver", "top3"),
    "receiver.nll": ("receiver", "nll"),
    "completion.auroc": ("completion", "auroc"),
    "completion.brier": ("completion", "brier"),
    "value.mae": ("value", "mae"),
    "value.spearman": ("value", "spearman"),
    "recommendation_set.coverage": ("recommendation_set", "coverage"),
    "recommendation_set.mean_set_size": ("recommendation_set", "mean_set_size"),
    "recommendation_set.abstention_rate": ("recommendation_set", "abstention_rate"),
}


def evaluate_neural_split_bias(config: dict[str, Any]) -> dict[str, Any]:
    """Compare the full neural model under event-random and leave-one-domain tests."""
    run_name = artifact_run_name(config)
    result_root = Path("artifacts/results") / run_name
    event_path = result_root / "event_random_bias" / "metrics.json"
    if not event_path.exists():
        raise FileNotFoundError(
            "Missing neural event-random metrics. Run "
            "`pluralpass train --fold event_random_bias` and "
            "`pluralpass evaluate --fold event_random_bias` first."
        )
    event_metrics = json.loads(event_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        (Path(config["data"]["processed_dir"]) / "splits" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    fold_metrics = {}
    for fold in sorted(manifest):
        path = result_root / fold / "metrics.json"
        if path.exists():
            fold_metrics[fold] = json.loads(path.read_text(encoding="utf-8"))
    if not fold_metrics:
        raise FileNotFoundError("No leave-one-domain neural metrics were found.")

    macro = {
        name: float(np.mean([_get(payload, path) for payload in fold_metrics.values()]))
        for name, path in METRICS.items()
    }
    weights = np.asarray([payload["test_events"] for payload in fold_metrics.values()], dtype=float)
    weighted = {
        name: float(
            np.average([_get(payload, path) for payload in fold_metrics.values()], weights=weights)
        )
        for name, path in METRICS.items()
    }
    event = {name: _get(event_metrics, path) for name, path in METRICS.items()}
    report = {
        "run_name": run_name,
        "event_random_fold": "event_random_bias",
        "event_random_test_events": event_metrics["test_events"],
        "leave_one_domain_folds": sorted(fold_metrics),
        "leave_one_domain_test_events": int(sum(payload["test_events"] for payload in fold_metrics.values())),
        "event_random": event,
        "leave_one_domain_macro": macro,
        "leave_one_domain_weighted": weighted,
        "event_random_minus_leave_one_domain_macro": {
            name: event[name] - macro[name] for name in METRICS
        },
        "event_random_minus_leave_one_domain_weighted": {
            name: event[name] - weighted[name] for name in METRICS
        },
        "interpretation": (
            "This diagnostic trains and evaluates the full neural ensemble on the intentionally "
            "leaky event-random partition. It is not used as the primary performance estimate. "
            "Positive accuracy or AUROC differences indicate event-random optimism relative to "
            "leave-one-domain testing; mixed differences should be reported as leakage evidence "
            "rather than as a uniform inflation claim."
        ),
    }
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "PluralPass_neural_event_random_bias.json"
    markdown_path = output_dir / "PluralPass_neural_event_random_bias.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    report["json_path"] = str(json_path)
    report["markdown_path"] = str(markdown_path)
    return report


def _get(payload: dict[str, Any], path: tuple[str, str]) -> float:
    section, metric = path
    return float(payload[section][metric])


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Neural event-random split diagnostic",
        "",
        (
            "This report compares the full PluralPass neural ensemble under the intentionally "
            "leaky event-random split against the locked leave-one-domain estimates."
        ),
        "",
        f"- Event-random test events: {report['event_random_test_events']}",
        f"- Leave-one-domain folds: {len(report['leave_one_domain_folds'])}",
        f"- Leave-one-domain test events: {report['leave_one_domain_test_events']}",
        "",
        "## Event-random versus leave-one-domain",
        "",
        "| Metric | Event-random | LODO macro | Δ vs macro | LODO weighted | Δ vs weighted |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in METRICS:
        lines.append(
            "| "
            f"{name} | {_fmt(report['event_random'][name])} | "
            f"{_fmt(report['leave_one_domain_macro'][name])} | "
            f"{_fmt(report['event_random_minus_leave_one_domain_macro'][name])} | "
            f"{_fmt(report['leave_one_domain_weighted'][name])} | "
            f"{_fmt(report['event_random_minus_leave_one_domain_weighted'][name])} |"
        )
    lines.extend(["", "## Interpretation boundary", "", report["interpretation"], ""])
    return "\n".join(lines)
