from __future__ import annotations

import torch

from pluralpass.models.graph import PluralPassModel, pluralpass_loss


def test_graph_model_shapes_masks_and_loss() -> None:
    torch.manual_seed(7)
    model = PluralPassModel(
        node_features=14,
        hidden_dim=16,
        heads=4,
        layers=1,
        dropout=0.0,
        mirror_ensemble=True,
    )
    nodes = torch.randn(2, 5, 14)
    node_mask = torch.tensor(
        [[True, True, True, True, False], [True, True, True, False, False]]
    )
    candidate_mask = torch.tensor(
        [[False, True, True, False, False], [False, True, True, False, False]]
    )
    output = model(nodes, node_mask, candidate_mask)

    assert output.receiver_logits.shape == (2, 5)
    assert output.completion_logits.shape == (2, 5)
    assert output.value_mean.shape == (2, 5)
    assert torch.all(output.receiver_logits[~candidate_mask] < -1e8)

    loss, parts = pluralpass_loss(
        output,
        receiver_index=torch.tensor([1, 2]),
        completed=torch.tensor([1.0, 0.0]),
        value_delta=torch.tensor([0.2, -0.1]),
    )
    assert torch.isfinite(loss)
    assert set(parts) == {"receiver_loss", "completion_loss", "value_loss"}

