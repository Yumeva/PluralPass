from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pluralpass.config import artifact_run_name
from pluralpass.evaluation.metrics import receiver_metrics


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _domain_label(fold: str, manifest: dict[str, Any] | None) -> str:
    if manifest and fold in manifest:
        return str(manifest[fold]["held_domain"]).replace("|", " ")
    return fold.replace("_", " ")


def bin_visible_area(value: float) -> str:
    if value < 0.25:
        return "visible_area_lt_0.25"
    if value < 0.50:
        return "visible_area_0.25_to_0.50"
    return "visible_area_ge_0.50"


def bin_candidate_count(value: int) -> str:
    if value <= 7:
        return "candidates_le_7"
    if value <= 10:
        return "candidates_8_to_10"
    return "candidates_ge_11"


def evaluate_observation_stress(config: dict[str, Any]) -> dict[str, Any]:
    """Stratify saved predictions by observation completeness and uncertainty."""

    run_name = artifact_run_name(config)
    prediction_root = Path("artifacts/results") / run_name
    manifest_path = Path(config["data"]["processed_dir"]) / "splits" / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else None
    if not manifest:
        raise FileNotFoundError(f"Missing split manifest at {manifest_path}")

    folds = {}
    all_rows = []
    for fold in sorted(manifest):
        prediction_path = prediction_root / fold / "predictions.jsonl"
        rows = _read_predictions(prediction_path)
        fold_report = _summarize_rows(rows)
        fold_report["domain"] = _domain_label(fold, manifest)
        fold_report["test_events"] = len(rows)
        folds[fold] = fold_report
        all_rows.extend(rows)

    report = {
        "run_name": run_name,
        "folds": folds,
        "pooled": _summarize_rows(all_rows),
        "interpretation": (
            "This stress analysis stratifies saved external-test predictions by visible-area "
            "coverage, number of visible candidate receivers and epistemic uncertainty. It is "
            "not a counterfactual occlusion experiment because node locations are not perturbed "
            "and model predictions are not recomputed."
        ),
    }
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    json_path = output_dir / "PluralPass_observation_stress.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path = output_dir / "PluralPass_observation_stress.md"
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    report["json_path"] = str(json_path)
    report["markdown_path"] = str(markdown_path)
    return report


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions at {path}")
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    for row in rows:
        row["candidate_count"] = int(np.asarray(row["candidate_mask"], dtype=bool).sum())
        row["set_size"] = len(row["recommendation_set"])
        row["included"] = int(row["receiver_index"] in row["recommendation_set"])
        row["visible_area_bin"] = bin_visible_area(float(row["visible_area_fraction"]))
        row["candidate_count_bin"] = bin_candidate_count(row["candidate_count"])
    if rows:
        epistemic = np.asarray([float(row["epistemic"]) for row in rows])
        lower, upper = np.quantile(epistemic, [1 / 3, 2 / 3])
        for row in rows:
            value = float(row["epistemic"])
            if value <= lower:
                row["epistemic_bin"] = "epistemic_low"
            elif value <= upper:
                row["epistemic_bin"] = "epistemic_mid"
            else:
                row["epistemic_bin"] = "epistemic_high"
    return rows


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": _metrics(rows),
        "visible_area": _group_metrics(rows, "visible_area_bin"),
        "candidate_count": _group_metrics(rows, "candidate_count_bin"),
        "epistemic": _group_metrics(rows, "epistemic_bin"),
    }


def _group_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups = sorted({row[key] for row in rows})
    return {group: _metrics([row for row in rows if row[key] == group]) for group in groups}


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"events": 0}
    probability = np.asarray([row["receiver_probability"] for row in rows], dtype=float)
    probability = probability / np.clip(probability.sum(axis=1, keepdims=True), 1e-12, None)
    labels = np.asarray([row["receiver_index"] for row in rows], dtype=int)
    receiver = receiver_metrics(probability, labels)
    return {
        "events": len(rows),
        "receiver_top1": receiver["top1"],
        "receiver_top3": receiver["top3"],
        "receiver_nll": receiver["nll"],
        "set_coverage": float(np.mean([row["included"] for row in rows])),
        "mean_set_size": float(np.mean([row["set_size"] for row in rows])),
        "abstention_rate": float(np.mean([row["abstained"] for row in rows])),
        "mean_visible_area": float(np.mean([row["visible_area_fraction"] for row in rows])),
        "mean_candidate_count": float(np.mean([row["candidate_count"] for row in rows])),
        "mean_epistemic": float(np.mean([row["epistemic"] for row in rows])),
    }


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace("-", "−")


def _markdown(report: dict[str, Any]) -> str:
    pooled = report["pooled"]
    lines = [
        "# Observation-completeness stress analysis",
        "",
        "This analysis stratifies saved external-test predictions. It does not recompute",
        "predictions under controlled counterfactual occlusion, so it should be interpreted as an",
        "observational stress analysis rather than a counterfactual missingness experiment.",
        "",
        "## Pooled external-test strata",
        "",
    ]
    for title, section in [
        ("Visible-area strata", "visible_area"),
        ("Visible candidate-count strata", "candidate_count"),
        ("Epistemic-uncertainty strata", "epistemic"),
    ]:
        lines.extend(
            [
                f"### {title}",
                "",
                "| Stratum | Events | Top-1 | Top-3 | Coverage | Set size | Abstention | Mean visible area | Mean candidates |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for group, metrics in pooled[section].items():
            lines.append(
                f"| {group} | {metrics['events']} | {_fmt(metrics['receiver_top1'])} | "
                f"{_fmt(metrics['receiver_top3'])} | {_fmt(metrics['set_coverage'])} | "
                f"{_fmt(metrics['mean_set_size'], 2)} | {_fmt(metrics['abstention_rate'])} | "
                f"{_fmt(metrics['mean_visible_area'])} | {_fmt(metrics['mean_candidate_count'], 2)} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Per-domain overview",
            "",
            "| Domain | Events | Top-1 | Coverage | Set size | Abstention | Mean visible area |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for fold in report["folds"].values():
        metrics = fold["overall"]
        lines.append(
            f"| {fold['domain']} | {metrics['events']} | {_fmt(metrics['receiver_top1'])} | "
            f"{_fmt(metrics['set_coverage'])} | {_fmt(metrics['mean_set_size'], 2)} | "
            f"{_fmt(metrics['abstention_rate'])} | {_fmt(metrics['mean_visible_area'])} |"
        )
    lines.extend(["", f"Interpretation boundary: {report['interpretation']}"])
    return "\n".join(lines) + "\n"
