"""Dataset / CSV utilities for the ECHO-MIMIC-IV **classification** tasks.

``data/classification.csv`` has one row per echocardiography video, a fixed
``split`` column (train/val/test) and three binary targets.  Missing targets
are encoded as ``-1`` and masked out of the loss / metrics, so every labelled
row is usable for multi-task training.

Each item returned by :class:`VideoClassificationDataset` is::

    {
        "frames": uint8 tensor [T, 3, H, W],
        "labels": int64 tensor [K]   (-1 = missing),
        "meta":   {"row_id", "subject_id", "study_id", "measurement_id",
                   "video_path", "split"},
    }
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TASK_COLS: tuple[str, ...] = (
    "cls_rv_function",
    "cls_la_size",
    "cls_ra_size",
)
TASK_NUM_CLASSES: dict[str, int] = {t: 2 for t in TASK_COLS}
CLASS_NAMES: dict[str, list[str]] = {t: [str(i) for i in range(TASK_NUM_CLASSES[t])] for t in TASK_COLS}
META_COLS: tuple[str, ...] = ("subject_id", "study_id", "measurement_id")
SPLITS: tuple[str, ...] = ("train", "val", "test")


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------
def load_records(
    csv_path: str | Path,
    tasks: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
    require_any_label: bool = True,
) -> pd.DataFrame:
    """Load ``classification.csv`` and validate columns / split values."""
    csv_path = Path(csv_path).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"csv not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))
    df["split"] = df["split"].astype(str).str.lower()

    tasks = list(tasks) if tasks is not None else list(TASK_COLS)
    missing_cols = [c for c in ("video_path", "split", *tasks) if c not in df.columns]
    if missing_cols:
        raise ValueError(f"missing column(s) in {csv_path}: {missing_cols}")

    bad_splits = sorted(set(df["split"]) - set(SPLITS))
    if bad_splits:
        raise ValueError(f"unexpected split value(s): {bad_splits}")
    for t in tasks:
        vals = df[t].dropna()
        if (vals < 0).any() or (vals % 1 != 0).any():
            raise ValueError(f"{t}: labels must be non-negative integers")

    if splits is not None:
        df = df[df["split"].isin(list(splits))].copy()
    if require_any_label:
        df = df[df[tasks].notna().any(axis=1)].copy()
    return df.reset_index(drop=True)


def make_labels(df: pd.DataFrame, tasks: Sequence[str]) -> np.ndarray:
    """``[N, K]`` int64 labels with ``-1`` for missing values."""
    out = np.full((len(df), len(tasks)), -1, dtype=np.int64)
    for j, t in enumerate(tasks):
        col = df[t].to_numpy()
        valid = ~pd.isna(col)
        out[valid, j] = col[valid].astype(np.int64)
    return out


def class_stats(
    df: pd.DataFrame, tasks: Sequence[str], num_classes: Mapping[str, int] | None = None
) -> dict[str, dict]:
    """Per-task class counts and balanced class weights computed on train.

    ``num_classes`` may be supplied by the config; otherwise it is inferred
    from the observed label values.
    """
    stats: dict[str, dict] = {}
    for t in tasks:
        if num_classes and t in num_classes:
            n_classes = int(num_classes[t])
        else:
            observed = df[t].dropna().astype(int)
            n_classes = int(observed.max()) + 1 if len(observed) else 2
        col = df.loc[df["split"] == "train", t].dropna().astype(int)
        counts = [int((col == c).sum()) for c in range(n_classes)]
        n = int(sum(counts))
        weights = [float(n / (n_classes * c)) if c > 0 else 0.0 for c in counts]
        stats[t] = {
            "num_classes": n_classes,
            "class_names": [str(i) for i in range(n_classes)],
            "train_counts": counts,
            "train_freq": [c / n if n else 0.0 for c in counts],
            "class_weights": weights,
            "n_train": n,
        }
    return stats


def save_split_manifest(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ("row_id", "subject_id", "study_id", "measurement_id", "split") if c in df.columns]
    df[cols].to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Video decoding / sampling
# ---------------------------------------------------------------------------
def sample_indices(num_frames: int, count: int) -> np.ndarray:
    if num_frames <= 0:
        raise ValueError("video has no decodable frames")
    if count <= 0:
        raise ValueError("num_frames must be positive")
    if num_frames >= count:
        idx = np.linspace(0, num_frames - 1, count)
    else:
        idx = np.concatenate([np.arange(num_frames), np.full(count - num_frames, num_frames - 1)])
    return np.clip(np.round(idx).astype(np.int64), 0, num_frames - 1)


def decode_video(path: str | Path, num_frames: int) -> np.ndarray:
    """Return ``uint8`` frames ``[T, H, W, 3]`` sampled uniformly over the clip."""
    path = str(path)
    try:
        from decord import VideoReader, cpu

        vr = VideoReader(path, ctx=cpu(0), num_threads=1)
        idx = sample_indices(len(vr), num_frames)
        frames = vr.get_batch(idx).asnumpy()
    except Exception:  # pragma: no cover - fallback only
        from torchvision.io import read_video

        vframes, _, _ = read_video(path, pts_unit="sec", output_format="THWC")
        idx = torch.from_numpy(sample_indices(int(vframes.shape[0]), num_frames)).long()
        frames = vframes[idx].numpy()
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"unexpected decoded shape {frames.shape} for {path}")
    return np.ascontiguousarray(frames)


def resize_uint8_frames(frames: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(frames.shape[-2:]) == tuple(size):
        return frames
    out = torch.nn.functional.interpolate(
        frames.float(), size=tuple(size), mode="bilinear", align_corners=False, antialias=True
    )
    return out.round().clamp_(0, 255).to(torch.uint8)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class VideoClassificationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tasks: Sequence[str],
        num_frames: int = 16,
        image_size: int | tuple[int, int] | None = 224,
        check_paths: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tasks = list(tasks)
        self.num_frames = int(num_frames)
        if image_size is None:
            self.image_size = None
        elif isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = tuple(image_size)

        if check_paths:
            missing = [p for p in self.df["video_path"] if not Path(p).exists()]
            if missing:
                preview = "\n  ".join(missing[:5])
                raise FileNotFoundError(f"{len(missing)} video(s) missing, e.g.:\n  {preview}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        rec = self.df.iloc[i]
        frames = decode_video(rec["video_path"], self.num_frames)
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()  # [T,3,H,W]
        if self.image_size is not None:
            tensor = resize_uint8_frames(tensor, self.image_size)

        labels = torch.tensor(
            [-1 if pd.isna(rec[t]) else int(rec[t]) for t in self.tasks], dtype=torch.int64
        )
        meta = {
            "row_id": int(rec["row_id"]),
            "subject_id": int(rec["subject_id"]),
            "study_id": int(rec["study_id"]),
            "measurement_id": None if pd.isna(rec.get("measurement_id")) else int(rec["measurement_id"]),
            "video_path": str(rec["video_path"]),
            "split": str(rec["split"]),
        }
        return {"frames": tensor, "labels": labels, "meta": meta}


def collate_batch(items: list[dict]) -> dict:
    frames = torch.stack([it["frames"] for it in items], dim=0)  # [B,T,3,H,W]
    labels = torch.stack([it["labels"] for it in items], dim=0)  # [B,K]
    keys = items[0]["meta"].keys()
    meta = {k: [it["meta"][k] for it in items] for k in keys}
    return {"frames": frames, "labels": labels, "meta": meta}
