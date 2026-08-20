from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pluralpass.config import artifact_run_name
from pluralpass.evaluation.metrics import (
    binary_metrics,
    receiver_metrics,
    set_metrics,
    value_metrics,
)
from pluralpass.training import _dataset, _device, _load_models, _predict
from pluralpass.uncertainty.conformal import AdaptivePredictionSet, should_abstain


@dataclass(frozen=True)
class Perturbation:
    id: str
    label: str
    drop_fraction: float
    target: str
    preserve_receiver: bool = True
    event_probability: float = 1.0


PERTURBATIONS = [
    Perturbation(
        id="context_dropout_20",
        label="Drop 20% of non-candidate visible context players",
        drop_fraction=0.20,
        target="context",
    ),
    Perturbation(
        id="opponent_dropout_30",
        label="Drop 30% of visible opponents",
        drop_fraction=0.30,
        target="opponents",
    ),
    Perturbation(
        id="nonreceiver_candidate_dropout_30",
        label="Drop 30% of non-receiver teammate candidates",
        drop_fraction=0.30,
        target="nonreceiver_candidates",
    ),
    Perturbation(
        id="receiver_hidden_10",
        label="Hide the observed receiver in 10% of events",
        drop_fraction=1.00,
        target="receiver",
        preserve_receiver=False,
        event_probability=0.10,
    ),
]


class PerturbedDataset(Dataset):
    """Apply deterministic counterfactual visibility perturbations to graph tensors."""

    def __init__(self, base: Dataset, perturbation: Perturbation, seed: int):
        self.base = base
        self.perturbation = perturbation
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.base[index]
        out = {
            key: value.clone() if torch.is_tensor(value) else value for key, value in row.items()
        }
        out["counterfactual_dropped_nodes"] = torch.tensor(0, dtype=torch.long)
        receiver = int(out["receiver_index"])
        node_mask = out["node_mask"]
        candidate_mask = out["candidate_mask"]
        nodes = out["nodes"]
        out["counterfactual_receiver_visible"] = torch.tensor(
            bool(candidate_mask[receiver]), dtype=torch.bool
        )
        existing = torch.nonzero(node_mask, as_tuple=False).flatten().cpu().numpy()
        rng = np.random.default_rng(self.seed + index * 9973)
        if rng.random() > self.perturbation.event_probability:
            return out

        removable = self._removable_indices(
            existing=existing,
            nodes=nodes,
            candidate_mask=candidate_mask,
            receiver=receiver,
        )
        if not removable:
            return out
        if self.perturbation.target == "receiver":
            to_drop = [receiver]
        else:
            count = max(1, int(np.ceil(len(removable) * self.perturbation.drop_fraction)))
            count = min(count, len(removable))
            to_drop = rng.choice(removable, size=count, replace=False).tolist()
        for dropped in to_drop:
            node_mask[dropped] = False
            candidate_mask[dropped] = False
            nodes[dropped] = 0.0
        out["counterfactual_dropped_nodes"] = torch.tensor(len(to_drop), dtype=torch.long)
        out["counterfactual_receiver_visible"] = torch.tensor(
            bool(candidate_mask[receiver]), dtype=torch.bool
        )
        return out

    def _removable_indices(
        self,
        existing: np.ndarray,
        nodes: torch.Tensor,
        candidate_mask: torch.Tensor,
        receiver: int,
    ) -> list[int]:
        if self.perturbation.target == "receiver":
            return [receiver] if receiver < len(candidate_mask) and bool(candidate_mask[receiver]) else []
        removable = []
        for index in existing.tolist():
            is_receiver = index == receiver
            is_actor = bool(nodes[index, 3] > 0.5)
            is_teammate = bool(nodes[index, 2] > 0.5)
            is_candidate = bool(candidate_mask[index])
            if is_actor or (self.perturbation.preserve_receiver and is_receiver):
                continue
            if self.perturbation.target == "context" and not is_candidate or self.perturbation.target == "opponents" and not is_teammate or self.perturbation.target == "nonreceiver_candidates" and is_candidate:
                removable.append(index)
        return removable


def evaluate_counterfactual_stress(config: dict[str, Any]) -> dict[str, Any]:
    """Recompute predictions under controlled counterfactual player missingness."""

    device = _device()
    run_name = artifact_run_name(config)
    manifest_path = Path(config["data"]["processed_dir"]) / "splits" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    batch_size = int(config["training"]["batch_size"])
    seed = int(config["project"]["seed"])

    folds: dict[str, Any] = {}
    pooled: dict[str, list[dict[str, Any]]] = {p.id: [] for p in PERTURBATIONS}
    for fold in sorted(manifest):
        models = _load_models(config, fold, device)
        calibration = _predict(
            models, DataLoader(_dataset(config, fold, "calibration"), batch_size=batch_size), device
        )
        aps = AdaptivePredictionSet(
            coverage=config["conformal"]["coverage"],
            max_set_size=config["conformal"]["max_set_size"],
        ).fit(calibration["receiver_probability"], calibration["labels"])
        epistemic_threshold = float(
            np.quantile(
                calibration["receiver_epistemic"].sum(axis=1),
                config["conformal"]["epistemic_abstention_quantile"],
            )
        )
        minimum_area = float(config["conformal"]["minimum_visible_area_fraction"])
        fold_rows = {}
        test_base = _dataset(config, fold, "test")
        for perturbation in PERTURBATIONS:
            test = _predict_perturbed(
                models=models,
                dataset=PerturbedDataset(test_base, perturbation, seed=seed),
                batch_size=batch_size,
                device=device,
            )
            summary = _summarize_prediction(
                test=test,
                aps=aps,
                epistemic_threshold=epistemic_threshold,
                minimum_area=minimum_area,
            )
            summary["label"] = perturbation.label
            fold_rows[perturbation.id] = summary
            pooled[perturbation.id].append(summary)
        folds[fold] = {
            "domain": str(manifest[fold]["held_domain"]).replace("|", " "),
            "test_events": len(test_base),
            "perturbations": fold_rows,
        }

    report = {
        "run_name": run_name,
        "folds": folds,
        "pooled_macro_average": {
            perturbation.id: _macro_average(rows) for perturbation in PERTURBATIONS for rows in [pooled[perturbation.id]]
        },
        "interpretation": (
            "Counterfactual stress tests recompute PluralPass predictions after deterministic "
            "controlled counterfactual player-missingness perturbations in the external test "
            "folds. APS qhat and "
            "epistemic abstention thresholds are kept from the unperturbed calibration split. "
            "Receiver-hidden results are a boundary/failure-mode analysis, not a fair accuracy "
            "comparison, because the observed receiver is deliberately removed from the visible "
            "candidate set in a subset of events."
        ),
    }
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    json_path = output_dir / "PluralPass_counterfactual_stress.json"
    markdown_path = output_dir / "PluralPass_counterfactual_stress.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    report["json_path"] = str(json_path)
    report["markdown_path"] = str(markdown_path)
    return report


def _predict_perturbed(
    models: list[Any], dataset: Dataset, batch_size: int, device: torch.device
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size)
    test = _predict(models, loader, device)
    dropped = []
    receiver_visible = []
    for row in loader:
        dropped.extend(row.get("counterfactual_dropped_nodes", torch.zeros(len(row["event_id"]))).numpy())
        receiver_visible.extend(
            row.get("counterfactual_receiver_visible", torch.ones(len(row["event_id"]), dtype=torch.bool))
            .numpy()
            .astype(bool)
        )
    test["counterfactual_dropped_nodes"] = np.asarray(dropped, dtype=float)
    test["counterfactual_receiver_visible"] = np.asarray(receiver_visible, dtype=bool)
    return test


def _summarize_prediction(
    test: dict[str, Any],
    aps: AdaptivePredictionSet,
    epistemic_threshold: float,
    minimum_area: float,
) -> dict[str, Any]:
    rows = np.arange(len(test["labels"]))
    prediction_sets = aps.predict(test["receiver_probability"], test["candidate_mask"])
    chosen_completion = test["completion_probability"][rows, test["labels"]]
    chosen_value = test["value_mean"][rows, test["labels"]]
    epistemic = test["receiver_epistemic"].sum(axis=1)
    abstained = np.asarray(
        [
            should_abstain(score, epistemic_threshold, area, minimum_area)
            for score, area in zip(epistemic, test["visible_area_fraction"])
        ]
    )
    receiver_visible = test["counterfactual_receiver_visible"].astype(bool)
    all_events = {
        "events": len(rows),
        "receiver_visible_rate": float(receiver_visible.mean()),
        "mean_dropped_nodes": float(test["counterfactual_dropped_nodes"].mean()),
        "receiver": receiver_metrics(test["receiver_probability"], test["labels"]),
        "completion": binary_metrics(test["completed"], chosen_completion),
        "value": value_metrics(test["value"], chosen_value),
        "recommendation_set": set_metrics(prediction_sets, test["labels"], abstained),
        "abstention_components": {
            "high_epistemic_rate": float((epistemic > epistemic_threshold).mean()),
            "low_visibility_rate": float((test["visible_area_fraction"] < minimum_area).mean()),
        },
    }
    visible_events = None
    if receiver_visible.any():
        visible_sets = [prediction_sets[i] for i in np.flatnonzero(receiver_visible)]
        visible_events = {
            "events": int(receiver_visible.sum()),
            "receiver": receiver_metrics(
                test["receiver_probability"][receiver_visible], test["labels"][receiver_visible]
            ),
            "recommendation_set": set_metrics(
                visible_sets, test["labels"][receiver_visible], abstained[receiver_visible]
            ),
        }
    return {
        "all_events": all_events,
        "receiver_visible_subset": visible_events,
    }


def _macro_average(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        ("all_events", "receiver", "top1"),
        ("all_events", "receiver", "top3"),
        ("all_events", "completion", "auroc"),
        ("all_events", "value", "spearman"),
        ("all_events", "recommendation_set", "coverage"),
        ("all_events", "recommendation_set", "mean_set_size"),
        ("all_events", "recommendation_set", "abstention_rate"),
        ("all_events", "receiver_visible_rate"),
        ("all_events", "mean_dropped_nodes"),
    ]
    out: dict[str, float] = {}
    for key in keys:
        values = []
        for row in rows:
            value: Any = row
            for part in key:
                value = value[part]
            values.append(float(value))
        out[".".join(key)] = float(np.mean(values))
    return out


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace("-", "−")


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PluralPass counterfactual missingness stress test",
        "",
        "Predictions were recomputed after controlled counterfactual player-missingness perturbations",
        "in the external test folds. Calibration quantiles were kept fixed from the original",
        "calibration split.",
        "",
        "## Macro-average across leave-one-domain folds",
        "",
        "| Perturbation | Top-1 | Top-3 | AUROC | Value Spearman | Coverage | Set size | Abstention | Receiver visible | Dropped nodes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for perturbation in PERTURBATIONS:
        row = report["pooled_macro_average"][perturbation.id]
        lines.append(
            f"| {perturbation.label} | {_fmt(row['all_events.receiver.top1'])} | "
            f"{_fmt(row['all_events.receiver.top3'])} | {_fmt(row['all_events.completion.auroc'])} | "
            f"{_fmt(row['all_events.value.spearman'])} | "
            f"{_fmt(row['all_events.recommendation_set.coverage'])} | "
            f"{_fmt(row['all_events.recommendation_set.mean_set_size'], 2)} | "
            f"{_fmt(row['all_events.recommendation_set.abstention_rate'])} | "
            f"{_fmt(row['all_events.receiver_visible_rate'])} | "
            f"{_fmt(row['all_events.mean_dropped_nodes'], 2)} |"
        )
    lines.extend(
        [
            "",
            "## Per-domain Top-1 and coverage",
            "",
            "| Domain | Perturbation | Top-1 | Coverage | Set size | Abstention | Receiver visible |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for fold in report["folds"].values():
        for perturbation in PERTURBATIONS:
            row = fold["perturbations"][perturbation.id]["all_events"]
            lines.append(
                f"| {fold['domain']} | {perturbation.id} | {_fmt(row['receiver']['top1'])} | "
                f"{_fmt(row['recommendation_set']['coverage'])} | "
                f"{_fmt(row['recommendation_set']['mean_set_size'], 2)} | "
                f"{_fmt(row['recommendation_set']['abstention_rate'])} | "
                f"{_fmt(row['receiver_visible_rate'])} |"
            )
    lines.extend(["", f"Interpretation boundary: {report['interpretation']}"])
    return "\n".join(lines) + "\n"
