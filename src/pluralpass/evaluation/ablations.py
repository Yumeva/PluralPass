from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from pluralpass.config import artifact_run_name

NODE_FEATURES = [
    "x",
    "y",
    "teammate",
    "actor",
    "keeper",
    "dx_from_passer",
    "dy_from_passer",
    "distance_from_passer",
    "angle_from_passer",
    "nearest_opponent_distance",
    "pass_corridor_blockage",
    "distance_to_goal",
    "visible_area_fraction",
    "actor_duplicate",
]


ABLATION_REGISTRY: list[dict[str, Any]] = [
    {
        "id": "full_primary",
        "label": "Full PluralPass",
        "type": "primary",
        "status": "completed_external_validation",
        "scientific_question": "Reference model combining partial-observation features, graph interaction, mirror averaging, ensemble uncertainty and APS.",
        "overrides": {},
        "outputs": "Already reported under the base pluralpass artifact namespace.",
    },
    {
        "id": "no_touchline_mirror",
        "label": "No touchline mirror averaging",
        "type": "trainable_ablation",
        "status": "ready_to_train",
        "scientific_question": "Tests whether the restricted symmetry assumption improves out-of-domain prediction.",
        "overrides": {"project.name": "pluralpass-ablate-no-mirror", "model.mirror_ensemble": False},
        "outputs": "Train and evaluate all eight leave-one-domain folds under the generated config.",
    },
    {
        "id": "no_graph_interaction",
        "label": "No graph interaction blocks",
        "type": "trainable_ablation",
        "status": "ready_to_train",
        "scientific_question": "Tests whether multi-agent interaction adds information beyond candidate-wise node features.",
        "overrides": {"project.name": "pluralpass-ablate-no-graph", "model.layers": 0},
        "outputs": "Train and evaluate all eight leave-one-domain folds under the generated config.",
    },
    {
        "id": "no_visible_area_feature",
        "label": "No visible-area node feature",
        "type": "trainable_ablation",
        "status": "ready_to_train",
        "scientific_question": "Tests whether explicit partial-observation information improves prediction and abstention behaviour.",
        "overrides": {
            "project.name": "pluralpass-ablate-no-visible-area",
            "model.zero_feature_indices": [12],
        },
        "outputs": "Train and evaluate all eight leave-one-domain folds under the generated config.",
    },
    {
        "id": "no_pressure_blockage_features",
        "label": "No local pressure or pass-corridor blockage",
        "type": "trainable_ablation",
        "status": "ready_to_train",
        "scientific_question": "Tests whether the hand-engineered defensive-pressure channels explain completion and receiver-choice gains.",
        "overrides": {
            "project.name": "pluralpass-ablate-no-pressure-blockage",
            "model.zero_feature_indices": [9, 10],
        },
        "outputs": "Train and evaluate all eight leave-one-domain folds under the generated config.",
    },
    {
        "id": "single_member",
        "label": "Single neural member",
        "type": "trainable_ablation",
        "status": "ready_to_train",
        "scientific_question": "Tests the contribution of deep ensembling to uncertainty-aware abstention and set construction.",
        "overrides": {
            "project.name": "pluralpass-ablate-single-member",
            "model.ensemble_members": 1,
        },
        "outputs": "Train and evaluate all eight leave-one-domain folds under the generated config; interpret abstention cautiously because epistemic variance collapses.",
    },
    {
        "id": "no_selective_abstention",
        "label": "No selective abstention",
        "type": "evaluation_ablation",
        "status": "ready_after_primary_predictions",
        "scientific_question": "Tests whether the abstention rule improves selective-risk reporting relative to always returning a set.",
        "overrides": {
            "project.name": "pluralpass-ablate-no-abstention",
            "conformal.epistemic_abstention_quantile": 1.0,
            "conformal.minimum_visible_area_fraction": 0.0,
        },
        "outputs": "Can be evaluated from primary checkpoints or retrained under the generated config if strict artifact separation is preferred.",
    },
    {
        "id": "point_recommendation",
        "label": "Point recommendation instead of APS",
        "type": "evaluation_ablation",
        "status": "ready_after_primary_predictions",
        "scientific_question": "Contrasts calibrated action sets with the top-ranked receiver alone.",
        "overrides": {},
        "outputs": "Report top-1 set coverage, mean set size of one and coach-choice coverage when the restricted de-identified coach-response artifact is supplied.",
    },
]


def build_ablation_plan(config: dict[str, Any]) -> dict[str, Any]:
    """Write the prespecified ablation matrix and runnable config variants."""

    config_dir = Path("configs") / "ablations"
    output_dir = Path("outputs")
    config_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    variants = []
    for spec in ABLATION_REGISTRY:
        variant = copy.deepcopy(spec)
        variant["artifact_namespace"] = _namespace_for_variant(config, spec)
        if spec["overrides"]:
            ablation_config = _apply_overrides(config, spec["overrides"])
            for transient_key in ["_config_path", "_config_sha256"]:
                ablation_config.pop(transient_key, None)
            config_path = config_dir / f"{spec['id']}.yaml"
            _write_yaml(config_path, ablation_config)
            variant["config_path"] = str(config_path)
        else:
            variant["config_path"] = config.get("_config_path")
        variant["primary_metrics"] = [
            "receiver.top1",
            "receiver.nll",
            "completion.auroc",
            "completion.brier",
            "value.spearman",
            "recommendation_set.coverage",
            "recommendation_set.mean_set_size",
            "recommendation_set.abstention_rate",
            "recommendation_set.selective_coverage",
        ]
        variants.append(variant)

    plan = {
        "base_run_name": artifact_run_name(config),
        "node_features": NODE_FEATURES,
        "variants": variants,
        "interpretation_rules": [
            "Treat ablations as mechanism evidence, not as independent benchmark claims.",
            "Compare each ablation against the full model within the same leave-one-domain folds.",
            "Do not use event-random splits for ablation claims.",
            "Report MLS separately as a small low-visibility stress domain.",
            "Do not claim the pluralistic decision hypothesis from ablations alone; coach-choice data remain required.",
        ],
    }
    json_path = output_dir / "PluralPass_ablation_plan.json"
    json_path.write_text(json.dumps(plan, indent=2, allow_nan=False), encoding="utf-8")
    markdown_path = output_dir / "PluralPass_ablation_plan.md"
    markdown_path.write_text(_markdown(plan), encoding="utf-8")
    plan["json_path"] = str(json_path)
    plan["markdown_path"] = str(markdown_path)
    return plan


def collect_ablation_status(config: dict[str, Any]) -> dict[str, Any]:
    """Collect completion status for the ablation matrix without imputing missing results."""

    manifest_path = Path(config["data"]["processed_dir"]) / "splits" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    expected_folds = sorted(manifest)
    recommendation_ablation_folds = _completed_recommendation_ablation_folds()
    variants = []
    for spec in ABLATION_REGISTRY:
        namespace = _namespace_for_variant(config, spec)
        result_root = Path("artifacts") / "results" / namespace
        recommendation_metrics_available = spec["id"] in {
            "no_selective_abstention",
            "point_recommendation",
        }
        standard_metrics_available = not recommendation_metrics_available
        folds = []
        for fold in expected_folds:
            metrics_path = result_root / fold / "metrics.json"
            completed = (
                fold in recommendation_ablation_folds.get(spec["id"], set())
                if recommendation_metrics_available
                else metrics_path.exists()
            )
            folds.append(
                {
                    "fold": fold,
                    "domain": str(manifest[fold]["held_domain"]).replace("|", " ")
                    if fold in manifest
                    else fold,
                    "metrics_path": str(metrics_path),
                    "completed": bool(completed),
                }
            )
        variants.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "type": spec["type"],
                "status": spec["status"],
                "artifact_namespace": namespace,
                "standard_metrics_available": standard_metrics_available,
                "recommendation_metrics_available": recommendation_metrics_available,
                "completed_folds": sum(1 for row in folds if row["completed"]),
                "expected_folds": len(expected_folds),
                "folds": folds,
            }
        )

    report = {
        "base_run_name": artifact_run_name(config),
        "expected_folds": expected_folds,
        "variants": variants,
        "interpretation": (
            "This status report records completed primary-result artifacts and "
            "recommendation-layer ablation reports. Missing ablation metrics are not imputed, "
            "and trainable-ablation claims should compare completed variants against the "
            "full model within identical leave-one-domain folds."
        ),
    }
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    json_path = output_dir / "PluralPass_ablation_status.json"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    markdown_path = output_dir / "PluralPass_ablation_status.md"
    markdown_path.write_text(_status_markdown(report), encoding="utf-8")
    report["json_path"] = str(json_path)
    report["markdown_path"] = str(markdown_path)
    return report


def evaluate_recommendation_ablations(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate recommendation-only ablations from saved primary predictions."""

    manifest_path = Path(config["data"]["processed_dir"]) / "splits" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if not manifest:
        raise FileNotFoundError(f"Missing split manifest at {manifest_path}")

    run_name = artifact_run_name(config)
    result_root = Path("artifacts") / "results" / run_name
    folds: dict[str, Any] = {}
    for fold in sorted(manifest):
        predictions_path = result_root / fold / "predictions.jsonl"
        if not predictions_path.exists():
            raise FileNotFoundError(f"Missing primary predictions for {fold}: {predictions_path}")
        rows = _read_prediction_rows(predictions_path)
        labels = np.asarray([int(row["receiver_index"]) for row in rows])
        recommendation_sets = [
            np.asarray(row["recommendation_set"], dtype=int) for row in rows
        ]
        receiver_probability = np.asarray(
            [row["receiver_probability"] for row in rows], dtype=float
        )
        point_sets = [np.asarray([int(np.argmax(probability))]) for probability in receiver_probability]
        primary_abstained = np.asarray([bool(row["abstained"]) for row in rows], dtype=bool)

        variants = {
            "primary_aps_with_abstention": _set_summary(
                recommendation_sets, labels, primary_abstained
            ),
            "no_selective_abstention": _set_summary(
                recommendation_sets, labels, np.zeros(len(rows), dtype=bool)
            ),
            "point_recommendation": _set_summary(
                point_sets, labels, np.zeros(len(rows), dtype=bool)
            ),
        }
        folds[fold] = {
            "domain": str(manifest[fold]["held_domain"]).replace("|", " "),
            "test_events": len(rows),
            "variants": variants,
        }

    report = {
        "run_name": run_name,
        "folds": folds,
        "macro_average": _recommendation_macro_average(folds, min_test_events=0),
        "macro_average_excluding_small_domains": _recommendation_macro_average(
            folds, min_test_events=5000
        ),
        "interpretation": (
            "These are recommendation-layer ablations computed from saved primary predictions. "
            "They do not test graph architecture, feature representation or retraining effects. "
            "Point recommendation is the highest-probability receiver returned without abstention."
        ),
    }
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    json_path = output_dir / "PluralPass_recommendation_ablations.json"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    markdown_path = output_dir / "PluralPass_recommendation_ablations.md"
    markdown_path.write_text(_recommendation_markdown(report), encoding="utf-8")
    report["json_path"] = str(json_path)
    report["markdown_path"] = str(markdown_path)
    return report


def summarize_trainable_ablations(config: dict[str, Any]) -> dict[str, Any]:
    """Summarize completed trainable ablation metrics against the full model."""

    manifest_path = Path(config["data"]["processed_dir"]) / "splits" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if not manifest:
        raise FileNotFoundError(f"Missing split manifest at {manifest_path}")

    base_namespace = artifact_run_name(config)
    base_root = Path("artifacts") / "results" / base_namespace
    variants: dict[str, Any] = {}
    for spec in ABLATION_REGISTRY:
        if spec["type"] != "trainable_ablation":
            continue
        namespace = _namespace_for_variant(config, spec)
        result_root = Path("artifacts") / "results" / namespace
        folds = {}
        for fold, metadata in sorted(manifest.items()):
            full_path = base_root / fold / "metrics.json"
            ablation_path = result_root / fold / "metrics.json"
            if not full_path.exists() or not ablation_path.exists():
                continue
            full = json.loads(full_path.read_text(encoding="utf-8"))
            ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
            folds[fold] = _ablation_comparison(
                domain=str(metadata["held_domain"]).replace("|", " "),
                full=full,
                ablation=ablation,
            )
        variants[spec["id"]] = {
            "label": spec["label"],
            "artifact_namespace": namespace,
            "completed_folds": len(folds),
            "folds": folds,
            "macro_average_difference": _macro_average_differences(folds),
        }

    report = {
        "base_run_name": base_namespace,
        "variants": variants,
        "interpretation": (
            "This report compares only trainable ablation folds with completed metrics. "
            "Missing folds are not imputed. Findings from a small number of folds are "
            "treated as partial mechanism evidence rather than full mechanism evidence."
        ),
    }
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    json_path = output_dir / "PluralPass_trainable_ablation_summary.json"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    markdown_path = output_dir / "PluralPass_trainable_ablation_summary.md"
    markdown_path.write_text(_trainable_ablation_markdown(report), encoding="utf-8")
    report["json_path"] = str(json_path)
    report["markdown_path"] = str(markdown_path)
    return report


def _namespace_for_variant(config: dict[str, Any], spec: dict[str, Any]) -> str:
    if not spec["overrides"]:
        return artifact_run_name(config)
    variant_config = _apply_overrides(config, spec["overrides"])
    return artifact_run_name(variant_config)


def _apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    for dotted_key, value in overrides.items():
        target = updated
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    return updated


def _read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _metric_value(metrics: dict[str, Any], dotted_key: str) -> float:
    section, name = dotted_key.split(".")
    return float(metrics[section][name])


def _ablation_comparison(
    domain: str, full: dict[str, Any], ablation: dict[str, Any]
) -> dict[str, Any]:
    keys = [
        "receiver.top1",
        "receiver.top3",
        "receiver.nll",
        "completion.auroc",
        "completion.brier",
        "completion.ece",
        "value.spearman",
        "recommendation_set.coverage",
        "recommendation_set.mean_set_size",
        "recommendation_set.abstention_rate",
        "recommendation_set.selective_coverage",
    ]
    metrics = {}
    for key in keys:
        full_value = _metric_value(full, key)
        ablation_value = _metric_value(ablation, key)
        metrics[key] = {
            "full": full_value,
            "ablation": ablation_value,
            "difference_ablation_minus_full": ablation_value - full_value,
        }
    return {
        "domain": domain,
        "test_events": int(ablation["test_events"]),
        "metrics": metrics,
    }


def _macro_average_differences(folds: dict[str, Any]) -> dict[str, float]:
    if not folds:
        return {}
    keys = list(next(iter(folds.values()))["metrics"])
    return {
        key: float(
            np.mean(
                [
                    fold["metrics"][key]["difference_ablation_minus_full"]
                    for fold in folds.values()
                ]
            )
        )
        for key in keys
    }


def _set_summary(
    prediction_sets: list[np.ndarray], labels: np.ndarray, abstained: np.ndarray
) -> dict[str, float]:
    included = np.asarray(
        [int(label in prediction) for label, prediction in zip(labels, prediction_sets)]
    )
    sizes = np.asarray([len(prediction) for prediction in prediction_sets], dtype=float)
    retained = ~abstained.astype(bool)
    return {
        "coverage": float(included.mean()),
        "selective_coverage": float(included[retained].mean()) if retained.any() else float("nan"),
        "mean_set_size": float(sizes.mean()),
        "abstention_rate": float(abstained.mean()),
    }


def _recommendation_macro_average(
    folds: dict[str, Any], min_test_events: int
) -> dict[str, dict[str, float]]:
    selected = [fold for fold in folds.values() if int(fold["test_events"]) >= min_test_events]
    if not selected:
        return {}
    variant_names = list(selected[0]["variants"])
    summary = {}
    for variant in variant_names:
        rows = [fold["variants"][variant] for fold in selected]
        summary[variant] = {
            "folds": len(rows),
            "coverage": float(np.mean([row["coverage"] for row in rows])),
            "selective_coverage": float(np.mean([row["selective_coverage"] for row in rows])),
            "mean_set_size": float(np.mean([row["mean_set_size"] for row in rows])),
            "abstention_rate": float(np.mean([row["abstention_rate"] for row in rows])),
        }
    return summary


def _completed_recommendation_ablation_folds() -> dict[str, set[str]]:
    path = Path("outputs") / "PluralPass_recommendation_ablations.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    completed = {"no_selective_abstention": set(), "point_recommendation": set()}
    for fold, report in payload.get("folds", {}).items():
        variants = report.get("variants", {})
        for name, completed_folds in completed.items():
            if name in variants:
                completed_folds.add(fold)
    return completed


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml

        path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except ImportError:
        path.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# PluralPass ablation plan",
        "",
        "This generated plan records the ablations required to support mechanism claims.",
        "Ablations are interpreted within the leave-one-domain design and are not a",
        "replacement for coach validation.",
        "",
        "## Node feature indices",
        "",
        "| Index | Feature |",
        "|---:|---|",
    ]
    for index, feature in enumerate(plan["node_features"]):
        lines.append(f"| {index} | `{feature}` |")

    lines.extend(
        [
            "",
            "## Variants",
            "",
            "| Variant | Type | Status | Artifact namespace | Config | Scientific question |",
            "|---|---|---|---|---|---|",
        ]
    )
    for variant in plan["variants"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{variant['id']}`",
                    variant["type"],
                    variant["status"],
                    f"`{variant['artifact_namespace']}`",
                    f"`{variant['config_path']}`",
                    variant["scientific_question"],
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Primary metrics",
            "",
            ", ".join(f"`{metric}`" for metric in plan["variants"][0]["primary_metrics"]),
            "",
            "## Interpretation rules",
            "",
        ]
    )
    for rule in plan["interpretation_rules"]:
        lines.append(f"- {rule}")
    return "\n".join(lines) + "\n"


def _status_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PluralPass ablation status",
        "",
        report["interpretation"],
        "",
        "| Variant | Type | Artifact namespace | Completed folds | Status |",
        "|---|---|---|---:|---|",
    ]
    for variant in report["variants"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{variant['id']}`",
                    variant["type"],
                    f"`{variant['artifact_namespace']}`",
                    f"{variant['completed_folds']}/{variant['expected_folds']}",
                    variant["status"],
                ]
            )
            + " |"
        )

    lines.extend(["", "## Missing folds", ""])
    for variant in report["variants"]:
        missing = [row["fold"] for row in variant["folds"] if not row["completed"]]
        if missing:
            lines.append(f"- `{variant['id']}`: " + ", ".join(f"`{fold}`" for fold in missing))
    return "\n".join(lines) + "\n"


def _fmt(value: float, digits: int = 3, signed: bool = False) -> str:
    text = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return text.replace("-", "−")


def _recommendation_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PluralPass recommendation-layer ablations",
        "",
        report["interpretation"],
        "",
        "## Macro-average across all eight external domains",
        "",
        "| Variant | Folds | Coverage | Selective coverage | Mean set size | Abstention |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in report["macro_average"].items():
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
    for name, row in report["macro_average_excluding_small_domains"].items():
        lines.append(
            f"| {name} | {row['folds']} | {_fmt(row['coverage'])} | "
            f"{_fmt(row['selective_coverage'])} | {_fmt(row['mean_set_size'], 2)} | "
            f"{_fmt(row['abstention_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Per-domain results",
            "",
            "| Domain | Variant | Test passes | Coverage | Selective coverage | Mean set size | Abstention |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for fold in report["folds"].values():
        for name, row in fold["variants"].items():
            lines.append(
                f"| {fold['domain']} | {name} | {fold['test_events']} | "
                f"{_fmt(row['coverage'])} | {_fmt(row['selective_coverage'])} | "
                f"{_fmt(row['mean_set_size'], 2)} | {_fmt(row['abstention_rate'])} |"
            )
    return "\n".join(lines) + "\n"


def _trainable_ablation_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PluralPass trainable ablation summary",
        "",
        report["interpretation"],
        "",
        "## Completed-fold summary",
        "",
        "| Variant | Completed folds | ΔTop-1 | ΔTop-3 | ΔAUROC | ΔBrier | ΔValue Spearman | ΔCoverage | ΔSet size |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant_id, variant in report["variants"].items():
        diff = variant["macro_average_difference"]
        if not diff:
            lines.append(
                f"| `{variant_id}` | 0 | pending | pending | pending | pending | pending | pending | pending |"
            )
            continue
        lines.append(
            f"| `{variant_id}` | {variant['completed_folds']} | "
            f"{_fmt(diff['receiver.top1'], signed=True)} | "
            f"{_fmt(diff['receiver.top3'], signed=True)} | "
            f"{_fmt(diff['completion.auroc'], signed=True)} | "
            f"{_fmt(diff['completion.brier'], signed=True)} | "
            f"{_fmt(diff['value.spearman'], signed=True)} | "
            f"{_fmt(diff['recommendation_set.coverage'], signed=True)} | "
            f"{_fmt(diff['recommendation_set.mean_set_size'], signed=True)} |"
        )
    lines.extend(
        [
            "",
            "## Per-completed-fold comparisons",
            "",
            "| Variant | Domain | Test passes | Full Top-1 | Ablation Top-1 | ΔTop-1 | Full AUROC | Ablation AUROC | ΔAUROC |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant_id, variant in report["variants"].items():
        for fold in variant["folds"].values():
            metrics = fold["metrics"]
            lines.append(
                f"| `{variant_id}` | {fold['domain']} | {fold['test_events']} | "
                f"{_fmt(metrics['receiver.top1']['full'])} | "
                f"{_fmt(metrics['receiver.top1']['ablation'])} | "
                f"{_fmt(metrics['receiver.top1']['difference_ablation_minus_full'], signed=True)} | "
                f"{_fmt(metrics['completion.auroc']['full'])} | "
                f"{_fmt(metrics['completion.auroc']['ablation'])} | "
                f"{_fmt(metrics['completion.auroc']['difference_ablation_minus_full'], signed=True)} |"
            )
    return "\n".join(lines) + "\n"
