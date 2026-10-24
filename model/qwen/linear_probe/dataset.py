"""Qwen linear-probe dataset: raw 224px frames, decoded with decord."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "__init__.py").exists())
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common.data import (  # noqa: E402,F401
    CLASS_NAMES,
    META_COLS,
    TASK_COLS,
    TASK_NUM_CLASSES,
    VideoClassificationDataset,
    collate_batch,
    load_records,
    make_labels,
)


def build_dataset(df, tasks, num_frames: int = 8, image_size: int = 224) -> VideoClassificationDataset:
    return VideoClassificationDataset(df, tasks, num_frames=num_frames, image_size=image_size)
