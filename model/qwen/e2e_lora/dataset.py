"""EchoNet/MIMIC multi-view dataset for end-to-end Qwen2.5-VL LoRA.

A2C has no classification labels, so labels are carried from A4C by study_id.
Only studies with both views and a non-null label are kept, ensuring identical
samples/splits for the A4C / A2C / joint experiments.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common.video import decode_video, resize_uint8_frames


def read_view_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    need = {"subject_id", "study_id", "video_path", "split"}
    missing = need.difference(df.columns)
    if missing:
        raise ValueError(f"{p}: missing {sorted(missing)}")
    df["split"] = df["split"].astype(str).str.lower()
    return df.drop_duplicates("study_id").reset_index(drop=True)


def build_alignment(a4c_csv: str | Path, a2c_csv: str | Path, task: str) -> pd.DataFrame:
    a = read_view_csv(a4c_csv)
    b = read_view_csv(a2c_csv)
    if task not in a.columns:
        raise ValueError(f"{task!r} not in {a4c_csv}")
    a = a[["subject_id", "study_id", "split", "video_path", task]].rename(
        columns={"video_path": "video_path_A4C", "split": "split_A4C", task: "label"})
    b = b[["study_id", "split", "video_path"]].rename(
        columns={"video_path": "video_path_A2C", "split": "split_A2C"})
    df = a.merge(b, on="study_id", how="inner")
    df = df[df["label"].notna()].copy()
    if (df["split_A4C"] != df["split_A2C"]).any():
        raise ValueError("inconsistent A4C/A2C split")
    df["split"] = df["split_A4C"].astype(str)
    df["label"] = df["label"].astype(int)
    df = df.reset_index(drop=True)
    df.insert(0, "sample_idx", np.arange(len(df), dtype=np.int64))
    return df


class E2EDataset(Dataset):
    def __init__(self, df: pd.DataFrame, views: Sequence[str], num_frames: int = 16,
                 image_size: int = 224, view_csvs: Mapping[str, str] | None = None):
        self.df = df.reset_index(drop=True)
        self.views = list(views)
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.view_csvs = view_csvs or {}
        # The alignment table already contains video_path_<view> columns.
        self.path_cols = {v: f"video_path_{v}" for v in self.views}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        rec = self.df.iloc[i]
        frames = []
        for v in self.views:
            arr = decode_video(rec[self.path_cols[v]], self.num_frames)
            t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
            t = resize_uint8_frames(t, (self.image_size, self.image_size))
            frames.append(t)
        return {"frames": frames, "label": int(rec["label"]),
                "study_id": int(rec["study_id"]), "sample_idx": int(rec["sample_idx"])}


def collate_single(items: list[dict]) -> dict:
    return items[0]
