"""Qwen2.5-VL **vision tower only** utilities (no language model is loaded).

The checkpoint layout stores the vision tower under the ``visual.*`` prefix, so
we build ``Qwen2_5_VisionTransformerPretrainedModel`` from
``config.json['vision_config']`` and load only those tensors from the
safetensors shards.  This keeps GPU memory ~1.3 GB instead of ~7.5 GB.

Frame preprocessing reproduces ``Qwen2VLVideoProcessor`` exactly (verified
numerically against transformers 4.57): resize to a multiple of 28, rescale to
[0,1], normalise with the Qwen mean/std, and pack patches in the
``(grid_t, grid_h, grid_w, merge_h, merge_w, channel, temporal, p, p)`` order.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

PATCH_SIZE = 14
TEMPORAL_PATCH = 2
MERGE_SIZE = 2
QWEN_MEAN = (0.48145466, 0.4578275, 0.40821073)
QWEN_STD = (0.26862954, 0.26130258, 0.27577711)


def load_vision_tower(
    model_path: str | Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
):
    """Load only ``visual.*`` weights and return ``(vision_model, info)``."""
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VisionTransformerPretrainedModel,
        Qwen2_5_VLVisionConfig,
    )
    from safetensors import safe_open

    model_path = Path(model_path).expanduser()
    cfg_path = model_path / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Qwen config not found: {cfg_path}")
    full_cfg = json.loads(cfg_path.read_text())
    vision_cfg = dict(full_cfg.get("vision_config", {}))
    if not vision_cfg:
        raise ValueError(f"no vision_config in {cfg_path}")
    cfg = Qwen2_5_VLVisionConfig(**vision_cfg)

    model = Qwen2_5_VisionTransformerPretrainedModel._from_config(
        cfg, attn_implementation=attn_implementation
    )

    shards = sorted(model_path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards in {model_path}")
    state, n_keys = {}, 0
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("visual."):
                    state[key[len("visual."):]] = f.get_tensor(key)
                    n_keys += 1
    if not state:
        raise RuntimeError(f"no 'visual.*' tensors found in {model_path}")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"vision tower weights incomplete: {len(missing)} missing, e.g. {missing[:5]}")
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, {
        "model_path": str(model_path),
        "vision_keys_loaded": n_keys,
        "missing": len(missing),
        "unexpected": len(unexpected),
        "dtype": str(dtype),
        "attn_implementation": attn_implementation,
        "out_hidden_size": int(vision_cfg.get("out_hidden_size", cfg.out_hidden_size)),
    }


def preprocess_frames(frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[B,T,3,H,W]`` uint8/float -> ``(pixel_values [B*N, 1176], grid_thw [B,3])``.

    ``H`` and ``W`` must be divisible by ``PATCH_SIZE * MERGE_SIZE`` (28).
    """
    if frames.ndim != 5:
        raise ValueError(f"expected [B,T,3,H,W], got {tuple(frames.shape)}")
    if frames.dtype == torch.uint8:
        x = frames.float().div_(255.0)
    else:
        x = frames.float()
    b, t, c, h, w = x.shape
    if t % TEMPORAL_PATCH:
        raise ValueError(f"num_frames must be a multiple of {TEMPORAL_PATCH}, got {t}")
    if h % (PATCH_SIZE * MERGE_SIZE) or w % (PATCH_SIZE * MERGE_SIZE):
        raise ValueError(f"frame size must be divisible by {PATCH_SIZE * MERGE_SIZE}, got {h}x{w}")

    mean = torch.tensor(QWEN_MEAN, device=x.device, dtype=x.dtype).view(1, 1, 3, 1, 1)
    std = torch.tensor(QWEN_STD, device=x.device, dtype=x.dtype).view(1, 1, 3, 1, 1)
    x = (x - mean) / std

    gt, gh, gw = t // TEMPORAL_PATCH, h // PATCH_SIZE, w // PATCH_SIZE
    x = x.reshape(b, gt, TEMPORAL_PATCH, c, gh // MERGE_SIZE, MERGE_SIZE, PATCH_SIZE,
                  gw // MERGE_SIZE, MERGE_SIZE, PATCH_SIZE)
    x = x.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9).reshape(b, -1, c * TEMPORAL_PATCH * PATCH_SIZE * PATCH_SIZE)
    grid_thw = torch.tensor([[gt, gh, gw]] * b, dtype=torch.long, device=x.device)
    return x, grid_thw


def tokens_per_sample(grid_thw: torch.Tensor) -> int:
    gt, gh, gw = (int(v) for v in grid_thw[0].tolist())
    return gt * (gh // MERGE_SIZE) * (gw // MERGE_SIZE)


def pool_tokens(hidden_states: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    if mode == "mean":
        return hidden_states.mean(dim=1)
    if mode == "max":
        return hidden_states.max(dim=1).values
    if mode == "mean_std":
        return torch.cat([hidden_states.mean(dim=1), hidden_states.std(dim=1, unbiased=False)], dim=1)
    raise ValueError(f"unknown pooling mode: {mode}")


class QwenVisionEncoder(nn.Module):
    """Frozen Qwen vision tower: ``[B,T,3,H,W]`` -> pooled embedding ``[B,D]``."""

    def __init__(
        self,
        model_path: str | Path,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        pooling: str = "mean",
        attn_implementation: str = "sdpa",
    ) -> None:
        super().__init__()
        self.visual, self.load_info = load_vision_tower(
            model_path, device=device, dtype=dtype, attn_implementation=attn_implementation
        )
        self.pooling = pooling
        self.out_dim = int(self.load_info["out_hidden_size"]) * (2 if pooling == "mean_std" else 1)
        for p in self.visual.parameters():
            p.requires_grad_(False)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        pixel_values, grid_thw = preprocess_frames(frames)
        hidden = self.visual(hidden_states=pixel_values, grid_thw=grid_thw)
        per_sample = tokens_per_sample(grid_thw)
        hidden = hidden.reshape(grid_thw.shape[0], per_sample, -1)
        return pool_tokens(hidden, self.pooling)


def inject_lora(
    vision: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Sequence[str] = ("attn.qkv", "attn.proj"),
    layers: Sequence[int] | None = None,
):
    """Wrap the vision tower with LoRA adapters (base weights stay frozen)."""
    from peft import LoraConfig, get_peft_model

    kwargs = {}
    if layers:
        kwargs = {"layers_pattern": "blocks", "layers_to_transform": list(layers)}
    lora_cfg = LoraConfig(
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        bias="none",
        target_modules=list(target_modules),
        **kwargs,
    )
    model = get_peft_model(vision, lora_cfg)
    return model


def count_trainable(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
