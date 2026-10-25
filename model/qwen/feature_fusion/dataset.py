"""Data alignment and cached-feature datasets for multi-view fusion.

A2C has no classification labels, so labels are carried from A4C by
``study_id``.  For a strict A4C vs A2C vs fusion comparison we keep only
studies that have **both** views and a non-null label for the selected task.
All three experiments therefore use exactly the same samples and split.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

VIEW_CSV_COLUMNS = ("subject_id", "study_id", "video_path", "split")


def read_view_csv(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"csv not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in VIEW_CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df["split"] = df["split"].astype(str).str.lower()
    if "study_id" in df.columns:
        df = df.drop_duplicates(subset=["study_id"], keep="first")
    return df.reset_index(drop=True)


def build_alignment(
    label_csv: str | Path,
    view_csvs: Mapping[str, str | Path],
    task: str,
    label_view: str = "A4C",
) -> pd.DataFrame:
    """Return the common multi-view sample table.

    Labels and the split come from ``label_view``; every other view is
    inner-joined by ``study_id``.  All views must agree on the split.
    """
    base = read_view_csv(label_csv)
    if task not in base.columns:
        raise ValueError(f"task column {task!r} not found in {label_csv}")

    df = base[["subject_id", "study_id", "split", "video_path", task]].rename(
        columns={"video_path": f"video_path_{label_view}", "split": "split_label", task: "label"}
    )
    for view, csv_path in view_csvs.items():
        if view == label_view:
            continue
        v = read_view_csv(csv_path)
        v = v[["study_id", "split", "video_path"]].rename(
            columns={"video_path": f"video_path_{view}", "split": f"split_{view}"}
        )
        df = df.merge(v, on="study_id", how="inner")
        mismatch = df["split_label"] != df[f"split_{view}"]
        if mismatch.any():
            raise ValueError(f"{int(mismatch.sum())} studies have inconsistent {label_view}/{view} splits")

    df = df[df["label"].notna()].copy()
    df["split"] = df["split_label"].astype(str)
    df["label"] = df["label"].astype(int)
    df = df.reset_index(drop=True)
    df.insert(0, "sample_idx", np.arange(len(df), dtype=np.int64))
    view_order = list(view_csvs.keys())
    cols = ["sample_idx", "subject_id", "study_id", "split", "label"] + \
           [f"video_path_{v}" for v in view_order]
    return df[cols]


def load_feature_cache(path: str | Path) -> dict:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"feature cache not found: {path}")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def align_features(cache: Mapping[str, np.ndarray], study_ids: Sequence[int], view: str) -> np.ndarray:
    """Reorder cached features to ``study_ids`` order."""
    feats = np.asarray(cache["features"], dtype=np.float32)
    ids = np.asarray(cache["study_id"], dtype=np.int64)
    pos = {int(s): i for i, s in enumerate(ids)}
    missing = [int(s) for s in study_ids if int(s) not in pos]
    if missing:
        raise ValueError(f"{view}: {len(missing)} studies missing from feature cache, e.g. {missing[:5]}")
    idx = np.asarray([pos[int(s)] for s in study_ids], dtype=np.int64)
    return feats[idx]


class FeatureDataset(Dataset):
    """Aligned multi-view cached features + a single classification label."""

    def __init__(self, features: Mapping[str, np.ndarray], labels: np.ndarray):
        self.views = list(features.keys())
        self.features = {v: torch.as_tensor(features[v], dtype=torch.float32) for v in self.views}
        self.labels = torch.as_tensor(np.asarray(labels), dtype=torch.int64)
        n = len(self.labels)
        for v, f in self.features.items():
            if len(f) != n:
                raise ValueError(f"{v}: {len(f)} rows != {n} labels")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        return {"features": [self.features[v][i] for v in self.views],
                "label": self.labels[i]}


def collate_features(items: list[dict]) -> dict:
    n_views = len(items[0]["features"])
    return {
        "features": [torch.stack([it["features"][j] for it in items], dim=0) for j in range(n_views)],
        "labels": torch.stack([it["label"] for it in items], dim=0),
    }
