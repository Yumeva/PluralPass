from __future__ import annotations

from dataclasses import dataclass

import torch

from pluralpass.models.graph import PluralPassModel


@dataclass
class EnsembleOutput:
    receiver_probability: torch.Tensor
    completion_probability: torch.Tensor
    value_mean: torch.Tensor
    receiver_epistemic: torch.Tensor
    completion_epistemic: torch.Tensor
    completion_aleatoric: torch.Tensor
    value_epistemic: torch.Tensor
    value_aleatoric: torch.Tensor
    embedding: torch.Tensor


@torch.no_grad()
def predict_ensemble(
    models: list[PluralPassModel], batch: dict[str, torch.Tensor]
) -> EnsembleOutput:
    outputs = [
        model(batch["nodes"], batch["node_mask"], batch["candidate_mask"]) for model in models
    ]
    receiver = torch.stack([torch.softmax(out.receiver_logits, dim=-1) for out in outputs])
    completion = torch.stack([torch.sigmoid(out.completion_logits) for out in outputs])
    value = torch.stack([out.value_mean for out in outputs])
    value_variance = torch.stack([torch.exp(out.value_log_variance) for out in outputs])
    embeddings = torch.stack([out.embedding for out in outputs])
    return EnsembleOutput(
        receiver_probability=receiver.mean(0),
        completion_probability=completion.mean(0),
        value_mean=value.mean(0),
        receiver_epistemic=receiver.var(0, unbiased=False),
        completion_epistemic=completion.var(0, unbiased=False),
        completion_aleatoric=(completion * (1 - completion)).mean(0),
        value_epistemic=value.var(0, unbiased=False),
        value_aleatoric=value_variance.mean(0),
        embedding=embeddings.mean(0),
    )
