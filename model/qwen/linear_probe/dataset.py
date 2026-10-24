"""Qwen linear-probe dataset: raw 224px frames, decoded with decord."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "__init__.py").exists())
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common.data import (  # noqa: E402,F401
    META_COLS,
    TASK_COLS,
    VideoRegressionDataset,
    collate_batch,
    load_records,
)


def build_dataset(df, tasks, num_frames: int = 8, image_size: int = 224) -> VideoRegressionDataset:
    return VideoRegressionDataset(df, tasks, num_frames=num_frames, image_size=image_size)
