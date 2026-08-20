from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PluralPassOutput:
    receiver_logits: torch.Tensor
    completion_logits: torch.Tensor
    value_mean: torch.Tensor
    value_log_variance: torch.Tensor
    embedding: torch.Tensor


class GraphBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(
            hidden, hidden, hidden, key_padding_mask=~mask, need_weights=False
        )
        hidden = self.norm1(hidden + self.dropout(attended))
        hidden = self.norm2(hidden + self.dropout(self.feed_forward(hidden)))
        return hidden * mask.unsqueeze(-1)


class PluralPassModel(nn.Module):
    """Variable-size partially observed multi-agent encoder with candidate heads."""

    def __init__(
        self,
        node_features: int = 14,
        hidden_dim: int = 96,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.15,
        mirror_ensemble: bool = True,
        zero_feature_indices: list[int] | None = None,
    ):
        super().__init__()
        if zero_feature_indices:
            feature_mask = torch.ones(node_features)
            for index in zero_feature_indices:
                if index < 0 or index >= node_features:
                    raise ValueError(
                        f"zero_feature_indices contains {index}, but node_features={node_features}"
                    )
                feature_mask[index] = 0.0
        else:
            feature_mask = torch.ones(node_features)
        self.register_buffer("feature_mask", feature_mask)
        self.input = nn.Sequential(nn.Linear(node_features, hidden_dim), nn.GELU())
        self.blocks = nn.ModuleList([GraphBlock(hidden_dim, heads, dropout) for _ in range(layers)])
        self.receiver = nn.Linear(hidden_dim, 1)
        self.completion = nn.Linear(hidden_dim, 1)
        self.value = nn.Linear(hidden_dim, 2)
        self.mirror_ensemble = mirror_ensemble

    def _encode(self, nodes: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nodes = nodes * self.feature_mask.to(device=nodes.device, dtype=nodes.dtype)
        hidden = self.input(nodes)
        for block in self.blocks:
            hidden = block(hidden, mask)
        pooled = hidden.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return hidden, pooled

    def _heads(
        self, hidden: torch.Tensor, pooled: torch.Tensor, candidate_mask: torch.Tensor
    ) -> PluralPassOutput:
        receiver = self.receiver(hidden).squeeze(-1).masked_fill(~candidate_mask, -1e9)
        completion = self.completion(hidden).squeeze(-1).masked_fill(~candidate_mask, -1e9)
        value = self.value(hidden)
        return PluralPassOutput(
            receiver, completion, value[..., 0], value[..., 1].clamp(-8, 5), pooled
        )

    def forward(
        self, nodes: torch.Tensor, node_mask: torch.Tensor, candidate_mask: torch.Tensor
    ) -> PluralPassOutput:
        hidden, pooled = self._encode(nodes, node_mask)
        output = self._heads(hidden, pooled, candidate_mask)
        if not self.mirror_ensemble:
            return output
        mirrored = nodes.clone()
        mirrored[..., 1] *= -1
        mirrored[..., 6] *= -1
        mirrored[..., 8] *= -1
        hidden_m, pooled_m = self._encode(mirrored, node_mask)
        output_m = self._heads(hidden_m, pooled_m, candidate_mask)
        return PluralPassOutput(
            receiver_logits=(output.receiver_logits + output_m.receiver_logits) / 2,
            completion_logits=(output.completion_logits + output_m.completion_logits) / 2,
            value_mean=(output.value_mean + output_m.value_mean) / 2,
            value_log_variance=torch.logsumexp(
                torch.stack([output.value_log_variance, output_m.value_log_variance]), dim=0
            )
            - torch.log(torch.tensor(2.0, device=nodes.device)),
            embedding=(pooled + pooled_m) / 2,
        )


class CandidateMLPModel(nn.Module):
    """Candidate-wise neural baseline without graph/self-attention interaction.

    Each node is encoded independently and combined only with a pooled scene
    summary before the three task heads are applied. This preserves the same
    inputs, losses, ensemble machinery and conformal evaluation as PluralPass,
    but removes candidate-candidate message passing.
    """

    def __init__(
        self,
        node_features: int = 14,
        hidden_dim: int = 96,
        dropout: float = 0.15,
        mirror_ensemble: bool = True,
        zero_feature_indices: list[int] | None = None,
    ):
        super().__init__()
        if zero_feature_indices:
            feature_mask = torch.ones(node_features)
            for index in zero_feature_indices:
                if index < 0 or index >= node_features:
                    raise ValueError(
                        f"zero_feature_indices contains {index}, but node_features={node_features}"
                    )
                feature_mask[index] = 0.0
        else:
            feature_mask = torch.ones(node_features)
        self.register_buffer("feature_mask", feature_mask)
        self.input = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.context = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.receiver = nn.Linear(hidden_dim, 1)
        self.completion = nn.Linear(hidden_dim, 1)
        self.value = nn.Linear(hidden_dim, 2)
        self.mirror_ensemble = mirror_ensemble

    def _encode(self, nodes: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nodes = nodes * self.feature_mask.to(device=nodes.device, dtype=nodes.dtype)
        independent = self.input(nodes) * mask.unsqueeze(-1)
        pooled = independent.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled_expanded = pooled.unsqueeze(1).expand_as(independent)
        hidden = self.context(torch.cat([independent, pooled_expanded], dim=-1))
        hidden = hidden * mask.unsqueeze(-1)
        return hidden, pooled

    def _heads(
        self, hidden: torch.Tensor, pooled: torch.Tensor, candidate_mask: torch.Tensor
    ) -> PluralPassOutput:
        receiver = self.receiver(hidden).squeeze(-1).masked_fill(~candidate_mask, -1e9)
        completion = self.completion(hidden).squeeze(-1).masked_fill(~candidate_mask, -1e9)
        value = self.value(hidden)
        return PluralPassOutput(
            receiver, completion, value[..., 0], value[..., 1].clamp(-8, 5), pooled
        )

    def forward(
        self, nodes: torch.Tensor, node_mask: torch.Tensor, candidate_mask: torch.Tensor
    ) -> PluralPassOutput:
        hidden, pooled = self._encode(nodes, node_mask)
        output = self._heads(hidden, pooled, candidate_mask)
        if not self.mirror_ensemble:
            return output
        mirrored = nodes.clone()
        mirrored[..., 1] *= -1
        mirrored[..., 6] *= -1
        mirrored[..., 8] *= -1
        hidden_m, pooled_m = self._encode(mirrored, node_mask)
        output_m = self._heads(hidden_m, pooled_m, candidate_mask)
        return PluralPassOutput(
            receiver_logits=(output.receiver_logits + output_m.receiver_logits) / 2,
            completion_logits=(output.completion_logits + output_m.completion_logits) / 2,
            value_mean=(output.value_mean + output_m.value_mean) / 2,
            value_log_variance=torch.logsumexp(
                torch.stack([output.value_log_variance, output_m.value_log_variance]), dim=0
            )
            - torch.log(torch.tensor(2.0, device=nodes.device)),
            embedding=(pooled + pooled_m) / 2,
        )


def pluralpass_loss(
    output: PluralPassOutput,
    receiver_index: torch.Tensor,
    completed: torch.Tensor,
    value_delta: torch.Tensor,
    weights: tuple[float, float, float] = (1.0, 0.5, 0.5),
) -> tuple[torch.Tensor, dict[str, float]]:
    receiver_loss = nn.functional.cross_entropy(output.receiver_logits, receiver_index)
    batch = torch.arange(receiver_index.shape[0], device=receiver_index.device)
    completion_logits = output.completion_logits[batch, receiver_index]
    completion_loss = nn.functional.binary_cross_entropy_with_logits(completion_logits, completed)
    mean = output.value_mean[batch, receiver_index]
    log_variance = output.value_log_variance[batch, receiver_index]
    value_loss = 0.5 * torch.mean(
        torch.exp(-log_variance) * (value_delta - mean) ** 2 + log_variance
    )
    total = weights[0] * receiver_loss + weights[1] * completion_loss + weights[2] * value_loss
    return total, {
        "receiver_loss": float(receiver_loss.detach()),
        "completion_loss": float(completion_loss.detach()),
        "value_loss": float(value_loss.detach()),
    }
