"""Extract cached Qwen2.5-VL vision-tower features for A4C and A2C.

Only the vision tower is loaded (``visual.*``); the language model is never
instantiated.  Features are pooled embeddings (2048-D, mean over tokens).

    python extract_features.py --config config.yaml --view all --device cuda

Caches: ``out/features/qwen_feature_fusion/<VIEW>/<tag>.npz`` with
``features``, ``study_id``, ``subject_id``, ``split`` and ``video_path``.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "__init__.py").exists())
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common.qwen_vision import QwenVisionEncoder  # noqa: E402
from common.utils import (  # noqa: E402
    dump_json, environment_snapshot, git_info, load_config, resolve_device, seed_everything, setup_logger,
    worker_init_fn,
)
from common.video import RawViewDataset, collate_raw  # noqa: E402
from dataset import read_view_csv  # noqa: E402


def view_rows(cfg: dict, view: str) -> pd.DataFrame:
    """Rows to cache: studies present in all configured views with >=1 A4C cls label."""
    label_view = cfg.get("label_view", "A4C")
    base = read_view_csv(cfg["views"][label_view]["csv"])
    cls_cols = [c for c in base.columns if c.startswith("cls_")]
    if not cls_cols:
        raise ValueError(f"{label_view} csv has no cls_* columns")
    keep_studies = set(base.loc[base[cls_cols].notna().any(axis=1), "study_id"].astype(int))
    frames = {}
    for v, spec in cfg["views"].items():
        df = read_view_csv(spec["csv"])
        frames[v] = df
        if v != label_view:
            keep_studies &= set(df["study_id"].astype(int))
    df = frames[view]
    df = df[df["study_id"].astype(int).isin(keep_studies)].reset_index(drop=True)
    return df[["subject_id", "study_id", "video_path", "split"]]


def extract_view(cfg: dict, view: str, device, logger, overrides: dict) -> Path:
    enc = cfg["encoder"]
    out_dir = Path(cfg["feature"]["dir"]).expanduser() / view
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = cfg["feature"]["tag"]
    out_path = out_dir / f"{tag}.npz"
    if out_path.exists() and not overrides["overwrite"]:
        logger.info(f"[skip] {view} cache exists: {out_path} (use --overwrite to redo)")
        return out_path

    df = view_rows(cfg, view)
    if overrides["max_videos"]:
        df = df.iloc[: overrides["max_videos"]].reset_index(drop=True)
    logger.info(f"{view}: extracting {len(df)} videos")

    ds = RawViewDataset(df, num_frames=enc["num_frames"], image_size=enc["image_size"])
    loader = DataLoader(
        ds,
        batch_size=int(overrides["batch_size"] or enc.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(overrides["num_workers"] if overrides["num_workers"] is not None else enc.get("num_workers", 8)),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_raw,
        worker_init_fn=worker_init_fn,
    )
    encoder = QwenVisionEncoder(
        enc["model_path"], device=device,
        dtype={"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[enc.get("dtype", "bfloat16")],
        pooling=enc.get("pooling", "mean"),
        attn_implementation=enc.get("attn_implementation", "sdpa"),
    ).to(device)
    logger.info(f"{view}: vision tower loaded {encoder.load_info}")

    feats, meta = [], {k: [] for k in ("study_id", "subject_id", "split", "video_path")}
    t0 = time.time()
    for step, batch in enumerate(tqdm(loader, desc=f"{view}", dynamic_ncols=True), 1):
        frames = batch["frames"].to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            f = encoder(frames)
        feats.append(f.float().cpu().numpy().astype(np.float16))
        for k in meta:
            meta[k].extend(batch[k])
        if step % 20 == 0:
            done = min(step * loader.batch_size, len(df))
            logger.info(f"{view} step {step}/{len(loader)} | {done}/{len(df)} | {done / max(time.time() - t0, 1e-6):.1f} vid/s")

    features = np.concatenate(feats, axis=0)
    np.savez_compressed(
        out_path,
        features=features,
        study_id=np.asarray(meta["study_id"], dtype=np.int64),
        subject_id=np.asarray(meta["subject_id"], dtype=np.int64),
        split=np.asarray(meta["split"]),
        video_path=np.asarray(meta["video_path"]),
    )
    dump_json(
        {
            "view": view,
            "tag": tag,
            "features": int(features.shape[0]),
            "feature_dim": int(features.shape[1]),
            "dtype": "float16",
            "num_frames": enc["num_frames"],
            "image_size": enc["image_size"],
            "pooling": enc.get("pooling", "mean"),
            "vision_tower_only": True,
            "load_info": encoder.load_info,
            "git": git_info(ROOT),
            "environment": environment_snapshot(),
            "created": datetime.now().isoformat(timespec="seconds"),
        },
        out_path.with_suffix(".json"),
    )
    logger.info(f"{view}: saved {features.shape} -> {out_path} ({time.time() - t0:.1f}s)")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Extract frozen Qwen vision features for A4C/A2C")
    p.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.yaml"))
    p.add_argument("--view", default="all")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--max-videos", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    logger = setup_logger("qwen.feature_fusion.extract")
    device = resolve_device(args.device)
    views = list(cfg["views"].keys()) if args.view == "all" else [args.view]
    overrides = {"batch_size": args.batch_size, "num_workers": args.num_workers,
                 "max_videos": args.max_videos, "overwrite": args.overwrite}
    for view in views:
        extract_view(cfg, view, device, logger, overrides)


if __name__ == "__main__":
    main()
