"""Frozen Qwen2.5-VL vision tower as a feature extractor (no LM loaded)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "__init__.py").exists())
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common.qwen_vision import QwenVisionEncoder, load_vision_tower, pool_tokens, preprocess_frames  # noqa: E402,F401


class QwenFeatureExtractor(QwenVisionEncoder):
    """Alias kept explicit for the linear-probe subproject."""


def build_encoder(cfg: dict, device="cpu") -> QwenVisionEncoder:
    enc = cfg["encoder"]
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        str(enc.get("dtype", "bfloat16"))
    ]
    return QwenVisionEncoder(
        enc["model_path"],
        device=device,
        dtype=dtype,
        pooling=enc.get("pooling", "mean"),
        attn_implementation=enc.get("attn_implementation", "sdpa"),
    )
