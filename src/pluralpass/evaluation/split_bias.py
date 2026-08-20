from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pluralpass.config import artifact_run_name
from pluralpass.evaluation.baselines import evaluate_baselines

FOLD_NAME = "event_random_bias"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def assign_event_random_splits(
    data: pd.DataFrame,
    seed: int,
    test_fraction: float,
    calibration_fraction: float,
    validation_fraction: float,
) -> pd.DataFrame:
    """Assign split labels by event, intentionally allowing match leakage."""

    total_fraction = test_fraction + calibration_fraction + validation_fraction
    if not 0 < total_fraction < 1:
        raise ValueError("test + calibration + validation fractions must sum to between 0 and 1")
    rng = np.random.default_rng(seed)
    shuffled = data[["event_id", "match_id", "domain"]].sample(
        frac=1.0, random_state=seed
    )
    n = len(shuffled)
    labels = np.full(n, "train", dtype=object)
    draws = rng.random(n)
    test_cut = test_fraction
    cal_cut = test_cut + calibration_fraction
    val_cut = cal_cut + validation_fraction
    labels[draws < test_cut] = "test"
    labels[(draws >= test_cut) & (draws < cal_cut)] = "calibration"
    labels[(draws >= cal_cut) & (draws < val_cut)] = "validation"
    split = shuffled.copy()
    split["split"] = labels
    return split.sort_values("event_id").reset_index(drop=True)


def evaluate_event_random_bias(config: dict[str, Any]) -> dict[str, Any]:
    """Run an event-random baseline diagnostic and compare it with domain holdouts."""

    processed = Path(config["data"]["processed_dir"])
    data_path = processed / "passes.parquet"
    data = pd.read_parquet(data_path) if data_path.exists() else pd.read_csv(processed / "passes.csv.gz")
    split_dir = processed / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    random_cfg = config.get("split_bias", {})
    test_fraction = float(random_cfg.get("event_random_test_fraction", 0.15))
    split = assign_event_random_splits(
        data,
        seed=int(config["project"]["seed"]) + 9000,
        test_fraction=test_fraction,
        calibration_fraction=float(config["splits"]["calibration_fraction"]),
        validation_fraction=float(config["splits"]["validation_fraction"]),
    )
    split_path = split_dir / f"{FOLD_NAME}.csv"
    split.to_csv(split_path, index=False)

    event_random = evaluate_baselines(config, FOLD_NAME)
    leave_one_domain = _collect_leave_one_domain_baselines(config)
    report = {
        "fold": FOLD_NAME,
        "split_path": str(split_path),
        "split_events": split["split"].value_counts().to_dict(),
        "split_matches_per_split": split.groupby("split")["match_id"].nunique().to_dict(),
        "match_leakage": _match_leakage(split),
        "event_random_hist_gradient_boosting": _extract_hgb(event_random),
        "leave_one_domain_hist_gradient_boosting": leave_one_domain,
        "event_random_minus_leave_one_domain_macro": _difference(
            _extract_hgb(event_random), leave_one_domain["macro_average"]
        ),
        "event_random_minus_leave_one_domain_weighted": _difference(
            _extract_hgb(event_random), leave_one_domain["weighted_average"]
        ),
        "interpretation": (
            "This is a baseline-level leakage diagnostic. It intentionally splits events at "
            "random, allowing the same matches and domains to appear in train and test. It "
            "does not show broad inflation for the histogram-gradient-boosting baseline and "
            "does not replace the primary leave-one-domain PluralPass evaluation."
        ),
    }
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    json_path = output_dir / "PluralPass_event_random_bias.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path = output_dir / "PluralPass_event_random_bias.md"
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    report["json_path"] = str(json_path)
    report["markdown_path"] = str(markdown_path)
    return report


def _match_leakage(split: pd.DataFrame) -> dict[str, Any]:
    split_counts = split.groupby("match_id")["split"].nunique()
    leaking_matches = int((split_counts > 1).sum())
    total_matches = int(split_counts.shape[0])
    test_matches = set(split.loc[split["split"] == "test", "match_id"].astype(str))
    train_matches = set(split.loc[split["split"] == "train", "match_id"].astype(str))
    return {
        "matches_with_multiple_splits": leaking_matches,
        "total_matches": total_matches,
        "share_matches_with_multiple_splits": leaking_matches / total_matches,
        "test_matches_seen_in_train": len(test_matches & train_matches),
        "test_match_share_seen_in_train": len(test_matches & train_matches) / max(len(test_matches), 1),
    }


def _collect_leave_one_domain_baselines(config: dict[str, Any]) -> dict[str, Any]:
    run_name = artifact_run_name(config)
    baseline_root = Path("artifacts/baselines") / run_name
    manifest_path = Path(config["data"]["processed_dir"]) / "splits" / "manifest.json"
    manifest = _read_json(manifest_path)
    folds = {}
    for fold, metadata in sorted(manifest.items()):
        metrics_path = baseline_root / fold / "metrics.json"
        metrics = _read_json(metrics_path)
        folds[fold] = {
            "domain": metadata["held_domain"],
            "test_events": metrics["test_events"],
            "hist_gradient_boosting": _extract_hgb(metrics),
        }
    return {
        "folds": folds,
        "macro_average": _aggregate(folds, weighted=False),
        "weighted_average": _aggregate(folds, weighted=True),
    }


def _extract_hgb(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "receiver": metrics["receiver"]["hist_gradient_boosting"],
        "completion": metrics["completion"]["hist_gradient_boosting"],
        "value": metrics["value"]["hist_gradient_boosting"],
        "test_events": metrics["test_events"],
    }


def _aggregate(folds: dict[str, Any], weighted: bool) -> dict[str, Any]:
    weights = np.asarray(
        [row["test_events"] for row in folds.values()] if weighted else [1 for _ in folds],
        dtype=float,
    )
    weights = weights / weights.sum()
    return {
        "receiver": {
            metric: _weighted_mean(
                [row["hist_gradient_boosting"]["receiver"][metric] for row in folds.values()],
                weights,
            )
            for metric in ["top1", "top3", "nll"]
        },
        "completion": {
            metric: _weighted_mean(
                [row["hist_gradient_boosting"]["completion"][metric] for row in folds.values()],
                weights,
            )
            for metric in ["auroc", "auprc", "brier", "ece"]
        },
        "value": {
            metric: _weighted_mean(
                [row["hist_gradient_boosting"]["value"][metric] for row in folds.values()],
                weights,
            )
            for metric in ["mae", "spearman"]
        },
        "test_events": int(sum(row["test_events"] for row in folds.values())),
    }


def _weighted_mean(values: list[float], weights: np.ndarray) -> float:
    return float(np.asarray(values, dtype=float) @ weights)


def _difference(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        "receiver": {
            metric: float(left["receiver"][metric] - right["receiver"][metric])
            for metric in ["top1", "top3", "nll"]
        },
        "completion": {
            metric: float(left["completion"][metric] - right["completion"][metric])
            for metric in ["auroc", "auprc", "brier", "ece"]
        },
        "value": {
            metric: float(left["value"][metric] - right["value"][metric])
            for metric in ["mae", "spearman"]
        },
    }


def _fmt(value: float, digits: int = 3, signed: bool = False) -> str:
    text = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return text.replace("-", "−")


def _markdown(report: dict[str, Any]) -> str:
    event = report["event_random_hist_gradient_boosting"]
    macro = report["leave_one_domain_hist_gradient_boosting"]["macro_average"]
    weighted = report["leave_one_domain_hist_gradient_boosting"]["weighted_average"]
    macro_diff = report["event_random_minus_leave_one_domain_macro"]
    weighted_diff = report["event_random_minus_leave_one_domain_weighted"]
    lines = [
        "# Event-random split bias diagnostic",
        "",
        "This diagnostic uses the histogram-gradient-boosting baseline, not the full",
        "PluralPass ensemble. The split is intentionally event-random and therefore",
        "allows the same match and domain to appear in both train and test.",
        "",
        "## Split leakage audit",
        "",
        f"- Split file: `{report['split_path']}`",
        f"- Events by split: {report['split_events']}",
        f"- Matches by split: {report['split_matches_per_split']}",
        (
            f"- Matches appearing in multiple splits: "
            f"{report['match_leakage']['matches_with_multiple_splits']} / "
            f"{report['match_leakage']['total_matches']} "
            f"({_fmt(report['match_leakage']['share_matches_with_multiple_splits'])})"
        ),
        (
            f"- Test matches also present in train: "
            f"{report['match_leakage']['test_matches_seen_in_train']} "
            f"({_fmt(report['match_leakage']['test_match_share_seen_in_train'])} of test matches)"
        ),
        "",
        "## HGB performance under event-random and leave-one-domain evaluation",
        "",
        "| Metric | Event-random | Leave-one-domain macro | Difference | Leave-one-domain weighted | Difference |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    rows = [
        ("Receiver Top-1", "receiver", "top1", 3, False),
        ("Receiver Top-3", "receiver", "top3", 3, False),
        ("Receiver NLL", "receiver", "nll", 3, False),
        ("Completion AUROC", "completion", "auroc", 3, False),
        ("Completion Brier", "completion", "brier", 4, False),
        ("Value Spearman", "value", "spearman", 3, False),
    ]
    for label, section, metric, digits, _higher in rows:
        lines.append(
            f"| {label} | {_fmt(event[section][metric], digits)} | "
            f"{_fmt(macro[section][metric], digits)} | "
            f"{_fmt(macro_diff[section][metric], digits, signed=True)} | "
            f"{_fmt(weighted[section][metric], digits)} | "
            f"{_fmt(weighted_diff[section][metric], digits, signed=True)} |"
        )
    lines.extend(
        [
            "",
            "Interpretation boundary: this is a leakage diagnostic for the strong non-neural",
            "baseline. Despite complete match leakage, it does not show broad event-random",
            "inflation across metrics. A full neural event-random experiment would be needed",
            "before reporting a PluralPass-specific inflation estimate.",
        ]
    )
    return "\n".join(lines) + "\n"
