"""Two-tower retrieval modules for frozen article embeddings."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ItemProjection(nn.Module):
    """Projection head from frozen MiniLM space into retrieval space."""

    def __init__(self, input_dim: int = 384, embed_dim: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, item_vectors: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(item_vectors), dim=-1)


class AttentionUserTower(nn.Module):
    """History-only self-attention user tower.

    There is deliberately no user-id embedding table. The test split has mostly
    unseen users, so the tower must generalize from article history alone.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        heads: int = 4,
        max_history: int = 50,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.max_history = max_history
        self.position = nn.Embedding(max_history, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.attention = nn.Linear(embed_dim, 1)
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, history_vectors: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        if history_vectors.ndim != 3:
            raise ValueError("history_vectors must have shape [B, T, D]")
        if history_mask.ndim != 2:
            raise ValueError("history_mask must have shape [B, T]")
        positions = torch.arange(history_vectors.shape[1], device=history_vectors.device)
        x = history_vectors + self.position(positions)[None, :, :]
        encoded = self.encoder(x, src_key_padding_mask=~history_mask)
        logits = self.attention(encoded).squeeze(-1)
        logits = logits.masked_fill(~history_mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), encoded).squeeze(1)
        return F.normalize(self.out_norm(pooled), dim=-1)


class TwoTowerRetrievalModel(nn.Module):
    """Shared item projection plus history-only user tower."""

    def __init__(
        self,
        input_dim: int = 384,
        embed_dim: int = 128,
        heads: int = 4,
        max_history: int = 50,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.item_tower = ItemProjection(input_dim=input_dim, embed_dim=embed_dim, dropout=dropout)
        self.user_tower = AttentionUserTower(
            embed_dim=embed_dim,
            heads=heads,
            max_history=max_history,
            dropout=dropout,
        )

    def encode_items(self, item_vectors: torch.Tensor) -> torch.Tensor:
        return self.item_tower(item_vectors)

    def encode_user(self, history_vectors: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        projected_history = self.item_tower(history_vectors)
        return self.user_tower(projected_history, history_mask)

    def forward(
        self,
        history_vectors: torch.Tensor,
        history_mask: torch.Tensor,
        target_vectors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        user_vec = self.encode_user(history_vectors, history_mask)
        item_vec = self.encode_items(target_vectors)
        return user_vec, item_vec
