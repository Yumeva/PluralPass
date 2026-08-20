from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from pluralpass.config import artifact_run_name
from pluralpass.training import _dataset, _device, _load_models, _predict
from pluralpass.uncertainty.conformal import (
    AdaptivePredictionSet,
    RegularizedAdaptivePredictionSet,
    should_abstain,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _domain_label(fold: str, manifest: dict[str, Any] | None) -> str:
    if manifest and fold in manifest:
        return str(manifest[fold]["held_domain"]).replace("|", " ")
    return fold.replace("_", " ")


def _coverage(prediction_sets: list[np.ndarray], labels: np.ndarray) -> float:
    included = [int(label in prediction) for label, prediction in zip(labels, prediction_sets)]
    return float(np.mean(included))


def _selective_coverage(
    prediction_sets: list[np.ndarray], labels: np.ndarray, abstained: np.ndarray
) -> float:
    keep = ~abstained.astype(bool)
    if not keep.any():
        return float("nan")
    included = np.asarray(
        [int(label in prediction) for label, prediction in zip(labels, prediction_sets)]
    )
    return float(included[keep].mean())


def _mean_size(prediction_sets: list[np.ndarray]) -> float:
    return float(np.mean([len(prediction) for prediction in prediction_sets]))


def _variant_specs() -> list[dict[str, Any]]:
    return [
        {"name": "APS", "method": "aps"},
        {"name": "RAPS_lambda0.005_k3", "method": "raps", "lambda": 0.005, "k": 3},
        {"name": "RAPS_lambda0.010_k3", "method": "raps", "lambda": 0.010, "k": 3},
        {"name": "RAPS_lambda0.020_k3", "method": "raps", "lambda": 0.020, "k": 3},
        {"name": "RAPS_lambda0.010_k2", "method": "raps", "lambda": 0.010, "k": 2},
    ]


def _build_predictor(spec: dict[str, Any], config: dict[str, Any]):
    kwargs = {
        "coverage": config["conformal"]["coverage"],
        "max_set_size": config["conformal"]["max_set_size"],
    }
    if spec["method"] == "aps":
        return AdaptivePredictionSet(**kwargs)
    return RegularizedAdaptivePredictionSet(
        lambda_regularization=float(spec["lambda"]),
        k_regularization=int(spec["k"]),
        **kwargs,
    )


def evaluate_set_efficiency(config: dict[str, Any]) -> dict[str, Any]:
    """Compare APS and RAPS recommendation-set efficiency across completed folds."""

    processed = Path(config["data"]["processed_dir"])
    manifest_path = processed / "splits" / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else None
    if not manifest:
        raise FileNotFoundError(f"Missing split manifest at {manifest_path}")

    device = _device()
    batch_size = config["training"]["batch_size"]
    variants = _variant_specs()
    fold_reports: dict[str, Any] = {}

    for fold in sorted(manifest):
        models = _load_models(config, fold, device)
        calibration = _predict(
            models, DataLoader(_dataset(config, fold, "calibration"), batch_size=batch_size), device
        )
        test = _predict(
            models, DataLoader(_dataset(config, fold, "test"), batch_size=batch_size), device
        )
        epistemic = test["receiver_epistemic"].sum(axis=1)
        threshold = float(
            np.quantile(
                calibration["receiver_epistemic"].sum(axis=1),
                config["conformal"]["epistemic_abstention_quantile"],
            )
        )
        minimum_area = float(config["conformal"]["minimum_visible_area_fraction"])
        abstained = np.asarray(
            [
                should_abstain(score, threshold, area, minimum_area)
                for score, area in zip(epistemic, test["visible_area_fraction"])
            ]
        )

        fold_variants = {}
        for spec in variants:
            predictor = _build_predictor(spec, config).fit(
                calibration["receiver_probability"], calibration["labels"]
            )
            prediction_sets = predictor.predict(
                test["receiver_probability"], test["candidate_mask"]
            )
            fold_variants[spec["name"]] = {
                "qhat": predictor.qhat,
                "coverage": _coverage(prediction_sets, test["labels"]),
                "selective_coverage": _selective_coverage(
                    prediction_sets, test["labels"], abstained
                ),
                "mean_set_size": _mean_size(prediction_sets),
                "abstention_rate": float(abstained.mean()),
            }

        fold_reports[fold] = {
            "domain": _domain_label(fold, manifest),
            "test_events": len(test["labels"]),
            "variants": fold_variants,
        }

    summary = {
        "run_name": artifact_run_name(config),
        "coverage_target": config["conformal"]["coverage"],
        "folds": fold_reports,
        "macro_average": _macro_average(fold_reports, min_test_events=0),
        "macro_average_excluding_small_domains": _macro_average(
            fold_reports, min_test_events=5000
        ),
    }
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    json_path = output_dir / "PluralPass_set_efficiency.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown_path = output_dir / "PluralPass_set_efficiency.md"
    markdown_path.write_text(_markdown(summary), encoding="utf-8")
    summary["json_path"] = str(json_path)
    summary["markdown_path"] = str(markdown_path)
    return summary


def _macro_average(fold_reports: dict[str, Any], min_test_events: int) -> dict[str, Any]:
    names = list(next(iter(fold_reports.values()))["variants"]) if fold_reports else []
    result = {}
    selected = [
        fold for fold in fold_reports.values() if int(fold["test_events"]) >= min_test_events
    ]
    for name in names:
        rows = [fold["variants"][name] for fold in selected]
        result[name] = {
            "folds": len(rows),
            "coverage": float(np.mean([row["coverage"] for row in rows])),
            "selective_coverage": float(np.mean([row["selective_coverage"] for row in rows])),
            "mean_set_size": float(np.mean([row["mean_set_size"] for row in rows])),
            "abstention_rate": float(np.mean([row["abstention_rate"] for row in rows])),
        }
    return result


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace("-", "−")


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# PluralPass set-efficiency sensitivity",
        "",
        f"Coverage target: {_fmt(float(summary['coverage_target']))}",
        "",
        "## Macro-average across all eight external domains",
        "",
        "| Variant | Folds | Coverage | Selective coverage | Mean set size | Abstention |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary["macro_average"].items():
        lines.append(
            f"| {name} | {row['folds']} | {_fmt(row['coverage'])} | "
            f"{_fmt(row['selective_coverage'])} | {_fmt(row['mean_set_size'], 2)} | "
            f"{_fmt(row['abstention_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Macro-average excluding domains with fewer than 5,000 test passes",
            "",
            "| Variant | Folds | Coverage | Selective coverage | Mean set size | Abstention |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in summary["macro_average_excluding_small_domains"].items():
        lines.append(
            f"| {name} | {row['folds']} | {_fmt(row['coverage'])} | "
            f"{_fmt(row['selective_coverage'])} | {_fmt(row['mean_set_size'], 2)} | "
            f"{_fmt(row['abstention_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Per-domain set size and coverage",
            "",
            "| Domain | Variant | Test passes | Coverage | Selective coverage | Mean set size | Abstention |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for fold in summary["folds"].values():
        for name, row in fold["variants"].items():
            lines.append(
                f"| {fold['domain']} | {name} | {fold['test_events']} | "
                f"{_fmt(row['coverage'])} | {_fmt(row['selective_coverage'])} | "
                f"{_fmt(row['mean_set_size'], 2)} | {_fmt(row['abstention_rate'])} |"
            )
    lines.extend(
        [
            "",
            (
                "Interpretation boundary: RAPS variants are sensitivity analyses for set "
                "efficiency. They do not replace the prespecified APS primary result unless "
                "selected before the coach experiment and reported as such."
            ),
        ]
    )
    return "\n".join(lines) + "\n"
