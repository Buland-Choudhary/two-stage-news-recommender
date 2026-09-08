from __future__ import annotations

import torch

from newsrec.losses import in_batch_contrastive_loss
from newsrec.towers import TwoTowerRetrievalModel


def test_two_tower_shapes_and_normalization() -> None:
    model = TwoTowerRetrievalModel(input_dim=384, embed_dim=128, heads=4, max_history=5, dropout=0.0)
    history = torch.randn(3, 5, 384)
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, False, False, False],
            [True, True, True, True, True],
        ]
    )
    target = torch.randn(3, 384)

    user_vec, item_vec = model(history, mask, target)

    assert user_vec.shape == (3, 128)
    assert item_vec.shape == (3, 128)
    assert torch.allclose(user_vec.norm(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(item_vec.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_in_batch_loss_accepts_logq_correction() -> None:
    user_vec = torch.nn.functional.normalize(torch.randn(4, 128), dim=-1)
    item_vec = torch.nn.functional.normalize(torch.randn(4, 128), dim=-1)
    target_idx = torch.tensor([1, 5, 5, 9])
    logq = torch.zeros(10)

    loss = in_batch_contrastive_loss(user_vec, item_vec, target_item_idx=target_idx, logq=logq)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
