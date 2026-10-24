"""Multi-task MLP classifier: shared trunk + one linear head per task."""
from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: Mapping[str, int],
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
        self.trunk = nn.Sequential(*layers)
        self.tasks = list(num_classes.keys())
        self.num_classes = {t: int(c) for t, c in num_classes.items()}
        self.heads = nn.ModuleDict({t: nn.Linear(prev, int(c)) for t, c in self.num_classes.items()})
        self.in_dim = int(in_dim)
        self.hidden_dim = int(prev)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(x)
        return {t: self.heads[t](h) for t in self.tasks}

    def forward_stacked(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.forward(x)[t] for t in self.tasks], dim=1)

    @property
    def total_logits(self) -> int:
        return sum(self.num_classes.values())
