from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from pluralpass.config import artifact_run_name
from pluralpass.evaluation.metrics import expected_calibration_error


def probability_bins(
    probability: np.ndarray,
    outcome: np.ndarray,
    bins: int = 10,
) -> list[dict[str, float | int | None]]:
    """Return reliability bins for binary outcomes or top-label correctness."""

    probability = np.asarray(probability, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for lower, upper in pairwise(edges):
        mask = (probability >= lower) & (probability < upper if upper < 1 else probability <= upper)
        count = int(mask.sum())
        rows.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_probability": float(probability[mask].mean()) if count else None,
                "empirical_frequency": float(outcome[mask].mean()) if count else None,
                "absolute_gap": (
                    float(abs(probability[mask].mean() - outcome[mask].mean()))
                    if count
                    else None
                ),
            }
        )
    return rows


def evaluate_calibration_diagnostics(config: dict[str, Any]) -> dict[str, Any]:
    """Build calibration tables from saved external-test predictions."""

    run_name = artifact_run_name(config)
    result_root = Path("artifacts/results") / run_name
    baseline_root = Path("artifacts/baselines") / run_name
    manifest_path = Path(config["data"]["processed_dir"]) / "splits" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    folds = {}
    pooled: dict[str, list[float]] = {
        "receiver_confidence": [],
        "receiver_correct": [],
        "pluralpass_completion_probability": [],
        "hgb_completion_probability": [],
        "completion_outcome": [],
    }
    for fold, metadata in sorted(manifest.items()):
        neural_rows = _read_predictions(result_root / fold / "predictions.jsonl")
        baseline = np.load(baseline_root / fold / "predictions.npz", allow_pickle=False)
        event_ids = [str(row["event_id"]) for row in neural_rows]
        baseline_ids = [str(value) for value in baseline["event_id"]]
        if event_ids != baseline_ids:
            raise ValueError(f"Prediction order mismatch for {fold}")
        arrays = _arrays(neural_rows, baseline)
        folds[fold] = {
            "domain": str(metadata["held_domain"]).replace("|", " "),
            "events": len(neural_rows),
            "receiver_top_label": _calibration_summary(
                arrays["receiver_confidence"], arrays["receiver_correct"]
            ),
            "completion_pluralpass": _calibration_summary(
                arrays["pluralpass_completion_probability"], arrays["completion_outcome"]
            ),
            "completion_hgb": _calibration_summary(
                arrays["hgb_completion_probability"], arrays["completion_outcome"]
            ),
        }
        for key, values in pooled.items():
            values.extend(arrays[key].tolist())

    pooled_summary = {
        "receiver_top_label": _calibration_summary(
            np.asarray(pooled["receiver_confidence"]),
            np.asarray(pooled["receiver_correct"]),
        ),
        "completion_pluralpass": _calibration_summary(
            np.asarray(pooled["pluralpass_completion_probability"]),
            np.asarray(pooled["completion_outcome"]),
        ),
        "completion_hgb": _calibration_summary(
            np.asarray(pooled["hgb_completion_probability"]),
            np.asarray(pooled["completion_outcome"]),
        ),
    }
    report = {
        "run_name": run_name,
        "folds": folds,
        "pooled": pooled_summary,
        "interpretation": (
            "Receiver calibration is top-label calibration: the confidence of the "
            "highest-probability receiver is compared with whether that receiver was observed. "
            "Completion calibration uses the probability assigned to the observed receiver. "
            "Calibration diagnostics are descriptive and do not imply causal validity for "
            "unplayed passes."
        ),
    }
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    json_path = output_dir / "PluralPass_calibration_diagnostics.json"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    markdown_path = output_dir / "PluralPass_calibration_diagnostics.md"
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    report["json_path"] = str(json_path)
    report["markdown_path"] = str(markdown_path)
    return report


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _arrays(rows: list[dict[str, Any]], baseline: Any) -> dict[str, np.ndarray]:
    receiver_probability = np.asarray([row["receiver_probability"] for row in rows], dtype=float)
    receiver_probability = receiver_probability / np.clip(
        receiver_probability.sum(axis=1, keepdims=True), 1e-12, None
    )
    labels = np.asarray([row["receiver_index"] for row in rows], dtype=int)
    top_label = receiver_probability.argmax(axis=1)
    receiver_confidence = receiver_probability[np.arange(len(rows)), top_label]
    completion_probability = np.asarray(
        [row["completion_probability"][int(row["receiver_index"])] for row in rows],
        dtype=float,
    )
    return {
        "receiver_confidence": receiver_confidence,
        "receiver_correct": (top_label == labels).astype(float),
        "pluralpass_completion_probability": completion_probability,
        "hgb_completion_probability": np.asarray(
            baseline["completion_hist_gradient_boosting"], dtype=float
        ),
        "completion_outcome": np.asarray([row["pass_completed"] for row in rows], dtype=float),
    }


def _calibration_summary(probability: np.ndarray, outcome: np.ndarray) -> dict[str, Any]:
    bins = probability_bins(probability, outcome)
    return {
        "events": len(probability),
        "mean_probability": float(np.mean(probability)),
        "empirical_frequency": float(np.mean(outcome)),
        "ece": expected_calibration_error(outcome, probability),
        "bins": bins,
    }


def _fmt(value: float, digits: int = 3, signed: bool = False) -> str:
    text = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return text.replace("-", "−")


def _markdown(report: dict[str, Any]) -> str:
    pooled = report["pooled"]
    lines = [
        "# PluralPass calibration diagnostics",
        "",
        "## Pooled external-test calibration",
        "",
        "| Target | Events | Mean probability | Empirical frequency | ECE |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in [
        ("Receiver top-label", "receiver_top_label"),
        ("Completion PluralPass", "completion_pluralpass"),
        ("Completion HGB", "completion_hgb"),
    ]:
        row = pooled[key]
        lines.append(
            f"| {label} | {row['events']} | {_fmt(row['mean_probability'])} | "
            f"{_fmt(row['empirical_frequency'])} | {_fmt(row['ece'])} |"
        )
    lines.extend(
        [
            "",
            "## Per-domain calibration summary",
            "",
            "| Domain | Receiver ECE | Completion ECE PluralPass | Completion ECE HGB | Difference |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for fold in report["folds"].values():
        pluralpass_ece = fold["completion_pluralpass"]["ece"]
        hgb_ece = fold["completion_hgb"]["ece"]
        lines.append(
            f"| {fold['domain']} | {_fmt(fold['receiver_top_label']['ece'])} | "
            f"{_fmt(pluralpass_ece)} | {_fmt(hgb_ece)} | "
            f"{_fmt(pluralpass_ece - hgb_ece, signed=True)} |"
        )
    lines.extend(
        [
            "",
            "## Pooled completion reliability bins",
            "",
            "| Model | Bin | Count | Mean probability | Empirical frequency | Absolute gap |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, key in [("PluralPass", "completion_pluralpass"), ("HGB", "completion_hgb")]:
        for row in pooled[key]["bins"]:
            if row["count"] == 0:
                continue
            lines.append(
                f"| {label} | {row['lower']:.1f}–{row['upper']:.1f} | {row['count']} | "
                f"{_fmt(float(row['mean_probability']))} | "
                f"{_fmt(float(row['empirical_frequency']))} | "
                f"{_fmt(float(row['absolute_gap']))} |"
            )
    lines.extend(["", f"Interpretation boundary: {report['interpretation']}"])
    return "\n".join(lines) + "\n"
