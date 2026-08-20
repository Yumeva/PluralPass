from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from pluralpass.config import artifact_run_name, set_seed, write_run_metadata
from pluralpass.dataset import PassGraphDataset
from pluralpass.evaluation.bootstrap import cluster_bootstrap_intervals
from pluralpass.evaluation.metrics import (
    binary_metrics,
    receiver_metrics,
    set_metrics,
    value_metrics,
)
from pluralpass.models.ensemble import predict_ensemble
from pluralpass.models.graph import CandidateMLPModel, PluralPassModel, pluralpass_loss
from pluralpass.uncertainty.conformal import AdaptivePredictionSet, should_abstain


def _artifact_dir(config: dict[str, Any], kind: str, fold: str) -> Path:
    return Path("artifacts") / kind / artifact_run_name(config) / fold


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _dataset(config: dict[str, Any], fold: str, split: str) -> PassGraphDataset:
    processed = Path(config["data"]["processed_dir"])
    return PassGraphDataset(
        processed / "passes.jsonl.gz",
        processed / "splits" / f"{fold}.csv",
        split,
        max_nodes=config["model"]["max_nodes"],
        max_samples=config["training"].get("max_samples_per_split"),
    )


def build_model(config: dict[str, Any]) -> PluralPassModel | CandidateMLPModel:
    model_cfg = config["model"]
    model_type = model_cfg.get("type", "graph_transformer")
    common = {
        "node_features": model_cfg["node_features"],
        "hidden_dim": model_cfg["hidden_dim"],
        "dropout": model_cfg["dropout"],
        "mirror_ensemble": model_cfg["mirror_ensemble"],
        "zero_feature_indices": model_cfg.get("zero_feature_indices"),
    }
    if model_type == "graph_transformer":
        return PluralPassModel(
            **common,
            heads=model_cfg["heads"],
            layers=model_cfg["layers"],
        )
    if model_type == "candidate_mlp":
        return CandidateMLPModel(**common)
    raise ValueError(f"Unknown model.type: {model_type}")


def _validation_nll(
    model: PluralPassModel | CandidateMLPModel, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            tensor_batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            output = model(
                tensor_batch["nodes"], tensor_batch["node_mask"], tensor_batch["candidate_mask"]
            )
            loss = torch.nn.functional.cross_entropy(
                output.receiver_logits, tensor_batch["receiver_index"]
            )
            losses.append(float(loss))
    return float(np.mean(losses)) if losses else float("inf")


def train_member(config: dict[str, Any], fold: str, seed: int) -> Path:
    set_seed(seed)
    checkpoint_dir = _artifact_dir(config, "checkpoints", fold)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"seed_{seed}.pt"
    metadata_path = checkpoint.with_suffix(".metadata.json")
    log_path = _artifact_dir(config, "logs", fold) / f"seed_{seed}.json"
    if checkpoint.exists() and metadata_path.exists() and log_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("config_sha256") == config.get("_config_sha256"):
            return checkpoint

    device = _device()
    train = _dataset(config, fold, "train")
    validation = _dataset(config, fold, "validation")
    train_loader = DataLoader(train, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(validation, batch_size=config["training"]["batch_size"], shuffle=False)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    weights = (
        config["training"]["receiver_loss_weight"],
        config["training"]["completion_loss_weight"],
        config["training"]["value_loss_weight"],
    )
    best, stale = float("inf"), 0
    history = []
    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            tensor_batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            optimizer.zero_grad(set_to_none=True)
            output = model(
                tensor_batch["nodes"], tensor_batch["node_mask"], tensor_batch["candidate_mask"]
            )
            loss, _ = pluralpass_loss(
                output,
                tensor_batch["receiver_index"],
                tensor_batch["pass_completed"],
                tensor_batch["value_delta"],
                weights,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        val_nll = _validation_nll(model, val_loader, device)
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "val_receiver_nll": val_nll,
        }
        history.append(epoch_record)
        print(json.dumps({"fold": fold, "seed": seed, **epoch_record}), flush=True)
        if val_nll < best - 1e-4:
            best, stale = val_nll, 0
            torch.save({"state_dict": model.state_dict(), "seed": seed, "fold": fold}, checkpoint)
        else:
            stale += 1
            if stale >= config["training"]["early_stopping_patience"]:
                break
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(
            {"best_validation_nll": best, "epochs_completed": len(history), "history": history},
            indent=2,
        ),
        encoding="utf-8",
    )
    write_run_metadata(metadata_path, config, fold=fold, member_seed=seed, device=str(device))
    return checkpoint


def train_ensemble(config: dict[str, Any], fold: str, seed: int | None = None) -> list[Path]:
    base = int(seed if seed is not None else config["project"]["seed"])
    return [
        train_member(config, fold, base + member)
        for member in range(config["model"]["ensemble_members"])
    ]


def _load_models(
    config: dict[str, Any], fold: str, device: torch.device
) -> list[PluralPassModel | CandidateMLPModel]:
    base = int(config["project"]["seed"])
    expected = [
        _artifact_dir(config, "checkpoints", fold) / f"seed_{base + i}.pt"
        for i in range(config["model"]["ensemble_members"])
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected checkpoints for fold {fold}: {missing}")
    models = []
    for path in expected:
        model = build_model(config).to(device)
        state_dict = torch.load(path, map_location=device, weights_only=True)["state_dict"]
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            result = model.load_state_dict(state_dict, strict=False)
            allowed_missing = {"feature_mask"}
            if set(result.missing_keys) - allowed_missing or result.unexpected_keys:
                raise
        model.eval()
        models.append(model)
    return models


def _predict(
    models: list[PluralPassModel | CandidateMLPModel], loader: DataLoader, device: torch.device
) -> dict[str, Any]:
    collected: dict[str, list[Any]] = {
        key: []
        for key in [
            "receiver_probability",
            "completion_probability",
            "value_mean",
            "receiver_epistemic",
            "labels",
            "completed",
            "value",
            "candidate_mask",
            "visible_area_fraction",
            "event_id",
            "match_id",
            "domain",
        ]
    }
    for batch in loader:
        tensors = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        output = predict_ensemble(models, tensors)
        for key in [
            "receiver_probability",
            "completion_probability",
            "value_mean",
            "receiver_epistemic",
        ]:
            collected[key].append(getattr(output, key).cpu().numpy())
        collected["labels"].append(tensors["receiver_index"].cpu().numpy())
        collected["completed"].append(tensors["pass_completed"].cpu().numpy())
        collected["value"].append(tensors["value_delta"].cpu().numpy())
        collected["candidate_mask"].append(tensors["candidate_mask"].cpu().numpy())
        collected["visible_area_fraction"].append(tensors["visible_area_fraction"].cpu().numpy())
        collected["event_id"].extend(batch["event_id"])
        collected["match_id"].extend(batch["match_id"])
        collected["domain"].extend(batch["domain"])
    for key in [
        "receiver_probability",
        "completion_probability",
        "value_mean",
        "receiver_epistemic",
        "labels",
        "completed",
        "value",
        "candidate_mask",
        "visible_area_fraction",
    ]:
        collected[key] = np.concatenate(collected[key], axis=0)
    return collected


def evaluate_fold(config: dict[str, Any], fold: str) -> dict[str, Any]:
    device = _device()
    models = _load_models(config, fold, device)
    batch_size = config["training"]["batch_size"]
    calibration = _predict(
        models, DataLoader(_dataset(config, fold, "calibration"), batch_size=batch_size), device
    )
    test = _predict(
        models, DataLoader(_dataset(config, fold, "test"), batch_size=batch_size), device
    )
    aps = AdaptivePredictionSet(
        coverage=config["conformal"]["coverage"], max_set_size=config["conformal"]["max_set_size"]
    ).fit(calibration["receiver_probability"], calibration["labels"])
    prediction_sets = aps.predict(test["receiver_probability"], test["candidate_mask"])

    rows = np.arange(len(test["labels"]))
    chosen_completion = test["completion_probability"][rows, test["labels"]]
    chosen_value = test["value_mean"][rows, test["labels"]]
    epistemic = test["receiver_epistemic"].sum(axis=1)
    threshold = float(
        np.quantile(
            calibration["receiver_epistemic"].sum(axis=1),
            config["conformal"]["epistemic_abstention_quantile"],
        )
    )
    minimum_area = float(config["conformal"]["minimum_visible_area_fraction"])
    low_visibility = test["visible_area_fraction"] < minimum_area
    high_epistemic = epistemic > threshold
    abstained = np.asarray(
        [
            should_abstain(score, threshold, area, minimum_area)
            for score, area in zip(epistemic, test["visible_area_fraction"])
        ]
    )
    included = np.asarray(
        [int(label in prediction) for label, prediction in zip(test["labels"], prediction_sets)]
    )
    set_size = np.asarray([len(prediction) for prediction in prediction_sets])
    report = {
        "fold": fold,
        "receiver": receiver_metrics(test["receiver_probability"], test["labels"]),
        "completion": binary_metrics(test["completed"], chosen_completion),
        "value": value_metrics(test["value"], chosen_value),
        "recommendation_set": set_metrics(prediction_sets, test["labels"], abstained),
        "conformal_qhat": aps.qhat,
        "epistemic_abstention_threshold": threshold,
        "minimum_visible_area_fraction": minimum_area,
        "abstention_components": {
            "high_epistemic_rate": float(high_epistemic.mean()),
            "low_visibility_rate": float(low_visibility.mean()),
        },
        "test_events": len(test["labels"]),
    }
    report["confidence_intervals_95"] = cluster_bootstrap_intervals(
        groups=np.asarray(test["match_id"]),
        receiver_probability=test["receiver_probability"],
        labels=test["labels"],
        completed=test["completed"],
        chosen_completion=chosen_completion,
        value=test["value"],
        chosen_value=chosen_value,
        included=included,
        set_size=set_size,
        abstained=abstained,
        repetitions=int(config.get("statistics", {}).get("cluster_bootstrap_repetitions", 1000)),
        seed=int(config["project"]["seed"]),
    )
    result_dir = _artifact_dir(config, "results", fold)
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (result_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for i, event_id in enumerate(test["event_id"]):
            handle.write(
                json.dumps(
                    {
                        "event_id": event_id,
                        "match_id": test["match_id"][i],
                        "domain": test["domain"][i],
                        "receiver_index": int(test["labels"][i]),
                        "pass_completed": float(test["completed"][i]),
                        "value_delta": float(test["value"][i]),
                        "candidate_mask": test["candidate_mask"][i].tolist(),
                        "visible_area_fraction": float(test["visible_area_fraction"][i]),
                        "receiver_probability": test["receiver_probability"][i].tolist(),
                        "completion_probability": test["completion_probability"][i].tolist(),
                        "value_mean": test["value_mean"][i].tolist(),
                        "epistemic": float(epistemic[i]),
                        "recommendation_set": prediction_sets[i].tolist(),
                        "abstained": bool(abstained[i]),
                    }
                )
                + "\n"
            )
    return report
