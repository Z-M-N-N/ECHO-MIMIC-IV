"""Qwen2.5-VL vision tower with LoRA adapters + MLP regression head."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "__init__.py").exists())
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common.mlp import MLPHead  # noqa: E402
from common.qwen_vision import QwenVisionEncoder, count_trainable, inject_lora  # noqa: E402


class QwenLoraRegressor(nn.Module):
    """Frozen vision tower + LoRA adapters + MLP head (end-to-end training)."""

    def __init__(self, cfg: dict, device="cpu") -> None:
        super().__init__()
        enc_cfg = cfg["encoder"]
        lora_cfg = cfg.get("lora", {})
        head_cfg = cfg.get("head", {})
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
            str(enc_cfg.get("dtype", "bfloat16"))
        ]

        self.encoder = QwenVisionEncoder(
            enc_cfg["model_path"],
            device=device,
            dtype=dtype,
            pooling=enc_cfg.get("pooling", "mean"),
            attn_implementation=enc_cfg.get("attn_implementation", "sdpa"),
        ).to(device)
        self.encoder.visual = inject_lora(
            self.encoder.visual,
            r=int(lora_cfg.get("r", 8)),
            alpha=int(lora_cfg.get("alpha", 16)),
            dropout=float(lora_cfg.get("dropout", 0.05)),
            target_modules=tuple(lora_cfg.get("target_modules", ["attn.qkv", "attn.proj"])),
            layers=lora_cfg.get("layers"),
        )
        self.head = MLPHead(
            in_dim=self.encoder.out_dim,
            out_dim=int(cfg.get("num_tasks", len(cfg["tasks"]))),
            hidden_dims=tuple(head_cfg.get("hidden_dims", [512, 256])),
            dropout=float(head_cfg.get("dropout", 0.2)),
            activation=str(head_cfg.get("activation", "gelu")),
            input_norm=bool(head_cfg.get("input_norm", True)),
        ).to(device)

        if bool(cfg.get("train", {}).get("gradient_checkpointing", True)):
            try:
                self.encoder.visual.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except Exception:  # noqa: BLE001
                pass

    def forward(self, frames: torch.Tensor, return_features: bool = False):
        features = self.encoder(frames)
        preds = self.head(features.float())
        return (preds, features) if return_features else preds

    def trainable_summary(self) -> dict:
        lora_trainable, lora_total = count_trainable(self.encoder.visual)
        head_trainable, head_total = count_trainable(self.head)
        return {
            "lora_trainable": lora_trainable,
            "lora_total": lora_total,
            "head_trainable": head_trainable,
            "head_total": head_total,
            "trainable_params": lora_trainable + head_trainable,
            "total_params": lora_total + head_total,
        }
