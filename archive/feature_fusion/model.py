"""Feature-level fusion (single / concat / mean / attention) + MLP classifier."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFusion(nn.Module):
    """Softmax-weighted view fusion with a small learned scoring network."""

    def __init__(self, dim: int, hidden: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, views: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack(views, dim=1)                 # [B, V, D]
        logits = self.score(stacked).squeeze(-1)            # [B, V]
        weights = torch.softmax(logits, dim=1)              # [B, V]
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        return fused, weights


class FusionClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        num_views: int = 1,
        fusion: str = "single",
        hidden_dims: Sequence[int] = (512, 256),
        dropout: float = 0.2,
        activation: str = "gelu",
        input_norm: bool = True,
    ) -> None:
        super().__init__()
        self.fusion = fusion.lower()
        self.num_views = int(num_views)
        act_cls = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation.lower()]

        if self.fusion in ("concat", "mean", "attention") and self.num_views < 2:
            raise ValueError(f"fusion={fusion} requires at least 2 views")

        if self.fusion == "concat":
            fused_dim = in_dim * self.num_views
        elif self.fusion in ("mean", "attention"):
            fused_dim = in_dim
        else:
            fused_dim = in_dim
        self.fused_dim = int(fused_dim)
        self.attention = AttentionFusion(in_dim, dropout=dropout) if self.fusion == "attention" else None

        layers: list[nn.Module] = []
        if input_norm:
            layers.append(nn.LayerNorm(fused_dim))
        prev = fused_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, int(h)), act_cls(), nn.Dropout(dropout)]
            prev = int(h)
        layers.append(nn.Linear(prev, int(num_classes)))
        self.mlp = nn.Sequential(*layers)

    def fuse(self, views: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        if len(views) != self.num_views:
            raise ValueError(f"expected {self.num_views} views, got {len(views)}")
        if self.fusion == "single":
            return views[0], None
        if self.fusion == "concat":
            return torch.cat(views, dim=1), None
        stacked = torch.stack(views, dim=1)
        if self.fusion == "mean":
            return stacked.mean(dim=1), None
        if self.fusion == "attention":
            return self.attention(views)
        raise ValueError(f"unknown fusion: {self.fusion}")

    def forward(self, views: list[torch.Tensor]) -> torch.Tensor:
        fused, _ = self.fuse(views)
        return self.mlp(fused)
