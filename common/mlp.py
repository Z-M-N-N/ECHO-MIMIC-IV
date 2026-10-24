"""Multi-task MLP regression head (>=2 hidden layers + activation + dropout)."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Sequence[int] = (512, 256),
        dropout: float = 0.2,
        activation: str = "gelu",
        input_norm: bool = True,
    ) -> None:
        super().__init__()
        act_cls = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation.lower()]
        layers: list[nn.Module] = []
        if input_norm:
            layers.append(nn.LayerNorm(in_dim))
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, int(h)), act_cls(), nn.Dropout(dropout)]
            prev = int(h)
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
