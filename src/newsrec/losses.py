"""Retrieval losses."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def in_batch_contrastive_loss(
    user_vec: torch.Tensor,
    item_vec: torch.Tensor,
    target_item_idx: torch.Tensor | None = None,
    logq: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """InfoNCE loss with optional logQ correction on in-batch item columns."""

    if user_vec.shape != item_vec.shape:
        raise ValueError("user_vec and item_vec must have the same [B, D] shape")
    logits = user_vec @ item_vec.T
    logits = logits / temperature
    if logq is not None:
        if target_item_idx is None:
            raise ValueError("target_item_idx is required when logq is provided")
        logits = logits - logq[target_item_idx][None, :].to(logits.device, dtype=logits.dtype)
    labels = torch.arange(user_vec.shape[0], device=user_vec.device)
    return F.cross_entropy(logits, labels)
