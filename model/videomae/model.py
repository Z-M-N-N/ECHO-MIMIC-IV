"""Frozen VideoMAE encoder + pooling for regression features."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def _strip_prefix(state_dict: dict, prefix: str = "module.") -> dict:
    if state_dict and all(k.startswith(prefix) for k in state_dict):
        return {k[len(prefix) :]: v for k, v in state_dict.items()}
    return state_dict


def load_videomae(weights_path: str | Path, num_frames: int = 16, image_size: int = 224, device="cpu"):
    """Load the VideoMAE checkpoint into ``VideoMAEForPreTraining`` and return the bare encoder."""
    from transformers import VideoMAEConfig, VideoMAEForPreTraining

    weights_path = Path(weights_path).expanduser()
    if not weights_path.exists():
        raise FileNotFoundError(f"VideoMAE checkpoint not found: {weights_path}")

    cfg = VideoMAEConfig(image_size=int(image_size), num_frames=int(num_frames))
    model = VideoMAEForPreTraining(cfg)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "state_dict"):
        if isinstance(state_dict, dict) and key in state_dict and isinstance(state_dict[key], dict):
            state_dict = state_dict[key]
            break
    state_dict = _strip_prefix(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    encoder_keys = [k for k in missing if k.startswith("videomae.")]
    if encoder_keys:
        raise RuntimeError(f"missing encoder weights ({len(encoder_keys)}), e.g. {encoder_keys[:5]}")

    encoder = model.videomae.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder, {"missing": len(missing), "unexpected": len(unexpected)}


def pool_tokens(hidden_states: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """``[B, L, D] -> [B, D]`` (``mean_std`` -> ``[B, 2D]``)."""
    if mode == "mean":
        return hidden_states.mean(dim=1)
    if mode == "max":
        return hidden_states.max(dim=1).values
    if mode == "mean_std":
        return torch.cat([hidden_states.mean(dim=1), hidden_states.std(dim=1, unbiased=False)], dim=1)
    raise ValueError(f"unknown pooling mode: {mode}")


class VideoMAEEncoder(nn.Module):
    """Wraps the frozen backbone: ``[B,T,3,H,W]`` in ``[0,1]`` -> pooled feature."""

    def __init__(self, weights_path: str | Path, num_frames: int = 16, image_size: int = 224,
                 pooling: str = "mean", device="cpu") -> None:
        super().__init__()
        self.backbone, self.load_info = load_videomae(weights_path, num_frames, image_size, device=device)
        self.pooling = pooling
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)

    @property
    def out_dim(self) -> int:
        dim = int(self.backbone.config.hidden_size)
        return dim * (2 if self.pooling == "mean_std" else 1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [B,T,3,H,W] uint8 or float in [0,1]
        if frames.dtype == torch.uint8:
            frames = frames.float().div_(255.0)
        out = self.backbone(pixel_values=frames, return_dict=True)
        return pool_tokens(out.last_hidden_state, self.pooling)
