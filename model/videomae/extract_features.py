"""Extract frozen VideoMAE features for every row of ``data/classification.csv``.

Example
-------
    python extract_features.py --config config.yaml --device cuda

Output: ``out/features/videomae/<tag>.npz`` (+ ``<tag>.json`` metadata) with
``features [N,D]``, ``labels [N,K]`` (NaN = missing), ``row_id`` and metadata.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "__init__.py").exists())
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common.data import (CLASS_NAMES, TASK_NUM_CLASSES, VideoClassificationDataset, collate_batch,
                         load_records, make_labels)  # noqa: E402
from common.utils import (  # noqa: E402
    dump_json, environment_snapshot, git_info, load_config, resolve_device, seed_everything, setup_logger,
    worker_init_fn,
)
from model import VideoMAEEncoder  # noqa: E402


def default_tag(cfg: dict) -> str:
    enc = cfg["encoder"]
    return f"videomae_{enc['num_frames']}f_{enc['image_size']}px_{enc['pooling']}"


def build_loader(cfg: dict, df, device, batch_size: int | None, num_workers: int | None) -> DataLoader:
    enc = cfg["encoder"]
    ds = VideoClassificationDataset(df, cfg["tasks"], num_frames=enc["num_frames"], image_size=enc["image_size"])
    return DataLoader(
        ds,
        batch_size=int(batch_size or enc.get("batch_size", 16)),
        shuffle=False,
        num_workers=int(num_workers if num_workers is not None else enc.get("num_workers", 8)),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_batch,
        persistent_workers=False,
        worker_init_fn=worker_init_fn,
    )


@torch.inference_mode()
def extract(
    cfg: dict,
    logger,
    device: torch.device,
    tag: str | None = None,
    max_videos: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    overwrite: bool = False,
    num_shards: int = 1,
    shard_id: int = 0,
) -> Path:
    enc_cfg = cfg["encoder"]
    feature_dir = Path(cfg["feature"]["dir"]).expanduser()
    tag = tag or default_tag(cfg)
    if max_videos and tag == default_tag(cfg):
        tag = f"{tag}_n{max_videos}"
    feature_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_shard{shard_id}of{num_shards}" if num_shards > 1 else ""
    out_path = feature_dir / f"{tag}{suffix}.npz"
    if out_path.exists() and not overwrite:
        logger.info(f"[skip] feature cache already exists: {out_path} (use --overwrite to redo)")
        return out_path

    df = load_records(cfg["csv"], cfg["tasks"])
    if max_videos:
        df = df.iloc[: int(max_videos)].reset_index(drop=True)
    if num_shards > 1:
        keep = list(range(shard_id, len(df), num_shards))
        df = df.iloc[keep].reset_index(drop=True)
        logger.info(f"[shard {shard_id}/{num_shards}] {len(df)} rows")

    encoder = VideoMAEEncoder(
        enc_cfg["weights"], num_frames=enc_cfg["num_frames"], image_size=enc_cfg["image_size"],
        pooling=enc_cfg["pooling"], device=device,
    ).to(device)
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(str(enc_cfg.get("amp_dtype", "bfloat16")))
    use_amp = device.type == "cuda" and amp_dtype is not None

    loader = build_loader(cfg, df, device, batch_size, num_workers)
    labels_all = make_labels(df, cfg["tasks"]).astype(np.int8)
    feats = []
    meta_all = {k: [] for k in ("row_id", "subject_id", "study_id", "measurement_id", "video_path", "split")}
    t0 = time.time()
    for step, batch in enumerate(tqdm(loader, desc=f"extract {tag}", dynamic_ncols=True), 1):
        pixel = batch["frames"].to(device, non_blocking=True).float().div_(255.0)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            f = encoder(pixel)
        feats.append(f.float().cpu().numpy().astype(np.float16))
        for k in meta_all:
            meta_all[k].extend(batch["meta"][k])
        if step % 20 == 0:
            done = min(step * loader.batch_size, len(df))
            logger.info(f"step {step}/{len(loader)} | {done}/{len(df)} videos | {done / max(time.time() - t0, 1e-6):.1f} vid/s")

    features = np.concatenate(feats, axis=0)
    np.savez_compressed(
        out_path,
        features=features,
        labels=labels_all,
        row_id=np.asarray(meta_all["row_id"], dtype=np.int64),
        subject_id=np.asarray(meta_all["subject_id"], dtype=np.int64),
        study_id=np.asarray(meta_all["study_id"], dtype=np.int64),
        measurement_id=np.asarray([np.nan if m is None else m for m in meta_all["measurement_id"]], dtype=np.float64),
        video_path=np.asarray(meta_all["video_path"]),
        split=np.asarray(meta_all["split"]),
    )
    dump_json(
        {
            "tag": tag,
            "kind": "videomae",
            "weights": str(enc_cfg["weights"]),
            "weights_mtime": datetime.fromtimestamp(Path(enc_cfg["weights"]).stat().st_mtime).isoformat(),
            "num_frames": enc_cfg["num_frames"],
            "image_size": enc_cfg["image_size"],
            "pooling": enc_cfg["pooling"],
            "feature_dim": int(features.shape[1]),
            "dtype": "float16",
            "n_samples": int(features.shape[0]),
            "task_type": "classification",
            "tasks": cfg["tasks"],
            "num_classes": {t: TASK_NUM_CLASSES[t] for t in cfg["tasks"]},
            "class_names": {t: CLASS_NAMES[t] for t in cfg["tasks"]},
            "load_info": encoder.load_info,
            "git": git_info(ROOT),
            "environment": environment_snapshot(),
            "created": datetime.now().isoformat(timespec="seconds"),
        },
        out_path.with_suffix(".json"),
    )
    logger.info(f"saved {features.shape} features -> {out_path} ({time.time() - t0:.1f}s)")
    return out_path


def merge_shards(cfg: dict, logger, tag: str | None = None, num_shards: int = 1) -> Path:
    feature_dir = Path(cfg["feature"]["dir"]).expanduser()
    tag = tag or default_tag(cfg)
    paths = [feature_dir / f"{tag}_shard{i}of{num_shards}.npz" for i in range(num_shards)]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing shard file(s): {missing}")
    parts = [np.load(p, allow_pickle=False) for p in paths]
    keys = parts[0].files
    merged = {}
    for k in keys:
        merged[k] = np.concatenate([p[k] for p in parts], axis=0)
    order = np.argsort(merged["row_id"])
    merged = {k: v[order] for k, v in merged.items()}
    out_path = feature_dir / f"{tag}.npz"
    np.savez_compressed(out_path, **merged)
    logger.info(f"merged {num_shards} shards -> {out_path} ({len(order)} rows)")
    return out_path


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.yaml"))
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--tag", default=None)
    p.add_argument("--max-videos", type=int, default=None, help="debug: only first N rows")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--merge", action="store_true")


def main() -> None:
    p = argparse.ArgumentParser(description="Extract frozen VideoMAE classification features")
    add_common_args(p)
    args = p.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    logger = setup_logger("videomae.extract")
    device = resolve_device(args.device)
    logger.info(f"device={device} cuda_visible={torch.cuda.device_count()} visible_devices={__import__('os').environ.get('CUDA_VISIBLE_DEVICES','')}")
    if args.merge:
        merge_shards(cfg, logger, args.tag, args.num_shards)
        return
    extract(
        cfg, logger, device, tag=args.tag, max_videos=args.max_videos, batch_size=args.batch_size,
        num_workers=args.num_workers, overwrite=args.overwrite, num_shards=args.num_shards, shard_id=args.shard_id,
    )


if __name__ == "__main__":
    main()
