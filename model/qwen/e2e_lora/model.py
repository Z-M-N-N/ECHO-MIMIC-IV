"""End-to-end Qwen2.5-VL + LLM LoRA + MLP classification head."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from lora_utils import count_trainable, inject_lora


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_dims: Sequence[int] = (512, 256),
                 dropout: float = 0.2, input_norm: bool = True):
        super().__init__()
        layers: list[nn.Module] = []
        if input_norm:
            layers.append(nn.LayerNorm(in_dim))
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, int(h)), nn.GELU(), nn.Dropout(dropout)]
            prev = int(h)
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)
        self.in_dim = int(in_dim)
        self.out_dim = int(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QwenE2EClassifier(nn.Module):
    def __init__(self, cfg: dict, device="cuda"):
        super().__init__()
        from transformers import Qwen2_5_VLForConditionalGeneration

        mcfg = cfg["model"]
        self.qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            mcfg["path"],
            torch_dtype=torch.bfloat16 if mcfg.get("dtype", "bfloat16") == "bfloat16" else torch.float32,
            attn_implementation=mcfg.get("attn_implementation", "flash_attention_2"),
            low_cpu_mem_usage=True,
        )
        self.qwen.config.use_cache = False
        for p in self.qwen.parameters():
            p.requires_grad_(False)

        lcfg = cfg["lora"]
        self.num_lora = inject_lora(
            self.qwen, r=int(lcfg["r"]), alpha=int(lcfg["alpha"]), dropout=float(lcfg["dropout"]),
            target_suffixes=tuple(lcfg["target_modules"]),
        )
        if self.num_lora == 0:
            raise RuntimeError("no LoRA modules injected")

        if bool(mcfg.get("gradient_checkpointing", True)):
            self.qwen.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(self.qwen, "enable_input_require_grads"):
                self.qwen.enable_input_require_grads()

        hcfg = cfg["head"]
        hidden = int(self.qwen.config.text_config.hidden_size)
        nc = cfg["num_classes"]
        if isinstance(nc, dict):
            nc = nc[cfg["task"]]
        self.head = MLPHead(
            in_dim=hidden, num_classes=int(nc),
            hidden_dims=tuple(hcfg.get("hidden_dims", [512, 256])),
            dropout=float(hcfg.get("dropout", 0.2)),
            input_norm=bool(hcfg.get("input_norm", True)),
        ).to(device)
        self.layer = int(cfg.get("representation", {}).get("layer", -1))
        self.pooling = str(cfg.get("representation", {}).get("pooling", "last"))
        self.device = device

    def forward(self, inputs: dict, return_features: bool = False):
        out = self.qwen(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        h = out.hidden_states[self.layer]                  # [B, L, H]
        mask = inputs["attention_mask"]
        if self.pooling in ("mean", "visual_mean"):
            if self.pooling == "visual_mean":
                vid = int(self.qwen.config.video_token_id)
                w = (inputs["input_ids"] == vid).unsqueeze(-1).to(h.dtype)
                feat = (h * w).sum(1) / w.sum(1).clamp(min=1)
            else:
                w = mask.unsqueeze(-1).to(h.dtype)
                feat = (h * w).sum(1) / w.sum(1).clamp(min=1)
        else:
            idx = mask.sum(dim=1).clamp(min=1) - 1
            feat = h[torch.arange(h.size(0), device=h.device), idx]
        logits = self.head(feat.float())
        return (logits, feat) if return_features else logits

    def trainable_summary(self) -> dict:
        tr, tot = count_trainable(self)
        return {"trainable": tr, "total": tot, "ratio": tr / max(tot, 1), "lora_modules": self.num_lora}


def build_optimizer(model: QwenE2EClassifier, cfg: dict):
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": float(cfg["train"]["lr_lora"])},
            {"params": head_params, "lr": float(cfg["train"]["lr_head"])},
        ],
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    return opt, lora_params, head_params
