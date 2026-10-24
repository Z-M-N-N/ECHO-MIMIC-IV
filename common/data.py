"""Dataset / CSV utilities shared by all three regression subprojects.

The single source of truth is ``data/regression.csv``.  Each row is one
echocardiography video with an explicit ``split`` column (train/val/test) and up
to six regression targets.  Missing targets are kept as ``NaN`` and masked out
by the loss / metrics, so multi-task training uses every labelled row.

Every item returned by :class:`VideoRegressionDataset` is a dict::

    {
        "frames": uint8 tensor [T, 3, H, W],
        "labels": float32 tensor [K]   (NaN where the target is missing),
        "meta":   {"row_id", "subject_id", "study_id", "measurement_id",
                   "video_path", "split"},
    }

Frame decoding uses decord (fast, random access) with a torchvision fallback.
Sampling is deterministic uniform sampling over the decoded clip, so the same
csv row always yields the same frames for a given ``num_frames``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TASK_COLS: tuple[str, ...] = (
    "reg_lvef",
    "reg_la_4ch_length",
    "reg_ra_length",
    "reg_la_vol",
    "reg_biplane_lvedv",
    "reg_biplane_lvesv",
)
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
    """Load ``regression.csv``, validate columns and optionally subset rows.

    ``row_id`` is the original CSV row index and is preserved as the join key
    between cached features, predictions and labels.
    """
    csv_path = Path(csv_path).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"csv not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))
    df["split"] = df["split"].astype(str).str.lower()

    tasks = list(tasks) if tasks is not None else list(TASK_COLS)
    unknown = [t for t in tasks if t not in TASK_COLS]
    if unknown:
        raise ValueError(f"unknown task column(s): {unknown}; available={list(TASK_COLS)}")
    missing_cols = [c for c in ("video_path", "split", *tasks) if c not in df.columns]
    if missing_cols:
        raise ValueError(f"missing column(s) in {csv_path}: {missing_cols}")

    bad_splits = sorted(set(df["split"]) - set(SPLITS))
    if bad_splits:
        raise ValueError(f"unexpected split value(s): {bad_splits}")

    if splits is not None:
        df = df[df["split"].isin(list(splits))].copy()
    if require_any_label:
        df = df[df[tasks].notna().any(axis=1)].copy()
    return df.reset_index(drop=True)


def label_stats(df: pd.DataFrame, tasks: Sequence[str]) -> dict[str, dict[str, float]]:
    """Per-task mean/std computed on the *train* split (NaN-safe)."""
    stats: dict[str, dict[str, float]] = {}
    for t in tasks:
        col = df.loc[df["split"] == "train", t].dropna().astype(float)
        mean = float(col.mean()) if len(col) else 0.0
        std = float(col.std(ddof=0)) if len(col) else 1.0
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0
        stats[t] = {"mean": mean, "std": std, "n_train": int(len(col))}
    return stats


def save_split_manifest(df: pd.DataFrame, path: str | Path) -> None:
    """Persist the (fixed) split assignment for reproducibility."""
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
    else:  # very short clip: repeat the last frame
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
    """Bilinear resize of a ``[T, 3, H, W]`` uint8 tensor."""
    if tuple(frames.shape[-2:]) == tuple(size):
        return frames
    out = torch.nn.functional.interpolate(
        frames.float(), size=tuple(size), mode="bilinear", align_corners=False, antialias=True
    )
    return out.round().clamp_(0, 255).to(torch.uint8)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class VideoRegressionDataset(Dataset):
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
            [np.nan if pd.isna(rec[t]) else float(rec[t]) for t in self.tasks], dtype=torch.float32
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
    """Stack frames / labels and keep metadata as python lists."""
    frames = torch.stack([it["frames"] for it in items], dim=0)  # [B,T,3,H,W]
    labels = torch.stack([it["labels"] for it in items], dim=0)  # [B,K]
    keys = items[0]["meta"].keys()
    meta = {k: [it["meta"][k] for it in items] for k in keys}
    return {"frames": frames, "labels": labels, "meta": meta}


def dataframe_from_records(
    df: pd.DataFrame, tasks: Sequence[str], indices: Iterable[int] | None = None
) -> pd.DataFrame:
    if indices is not None:
        df = df.iloc[list(indices)]
    return df.reset_index(drop=True)
