"""Minimal LoRA injection for the Qwen2.5-VL language model."""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = self.alpha / self.r
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(self.r, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        a = self.lora_A.to(dtype=x.dtype)
        b = self.lora_B.to(dtype=x.dtype)
        lora = (self.dropout(x) @ a.t()) @ b.t()
        return out + lora.to(out.dtype) * self.scaling


def inject_lora(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_suffixes: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"),
) -> int:
    """Replace matching LLM Linear layers; returns number of injected modules."""
    targets = set(target_suffixes)
    replaced = 0
    for name, module in list(model.named_modules()):
        if "language_model" not in name or not isinstance(module, nn.Linear):
            continue
        if name.split(".")[-1] not in targets:
            continue
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
        replaced += 1
    return replaced


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    missing = [k for k in state if k not in current]
    if missing:
        raise KeyError(f"LoRA keys not in model: {missing[:5]}")
    current.update(state)
    model.load_state_dict(current, strict=True)


def count_trainable(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
