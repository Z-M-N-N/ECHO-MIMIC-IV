"""Train the MLP regression head on frozen VideoMAE features.

    python train.py --config config.yaml --device cuda

The feature cache is created automatically when missing.  Results are written
to timestamped ``out/results/videomae/<run>/`` directories (plus ``latest``
symlinks) and figures to ``out/figures/videomae/<run>/``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "__init__.py").exists())
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common.data import TASK_COLS, label_stats, load_records, save_split_manifest  # noqa: E402
from common.metrics import format_metrics_table, masked_regression_metrics  # noqa: E402
from common.plots import plot_loss_curve, plot_pred_vs_true  # noqa: E402
from common.trainer import fit_mlp, inverse_standardize, predict_mlp, save_predictions  # noqa: E402
from common.utils import (  # noqa: E402
    append_experiment_log, dump_json, environment_snapshot, git_info, link_latest_files, load_config,
    make_run_dir, resolve_device, save_config, seed_everything, setup_logger,
)
from extract_features import default_tag, extract  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VideoMAE frozen features + MLP regression")
    p.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.yaml"))
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--tag", default=None, help="feature cache tag")
    p.add_argument("--reuse-features", action="store_true", help="never re-extract; fail if cache missing")
    p.add_argument("--overwrite-features", action="store_true")
    p.add_argument("--max-videos", type=int, default=None, help="debug subset (also tags the cache)")
    return p.parse_args()


def load_feature_cache(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    seed_everything(seed)
    device = resolve_device(args.device)
    logger = setup_logger("videomae.train")

    tasks = list(cfg.get("tasks") or TASK_COLS)
    cfg["tasks"] = tasks
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)

    run_name = args.run_name or cfg.get("output", {}).get("run_name", "videomae_mlp")
    run_dir = make_run_dir(cfg["output"]["root"], "results/videomae", run_name)
    fig_dir = Path(cfg["output"]["root"]) / "figures/videomae" / run_dir.name
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    logger.info(f"run dir: {run_dir}")

    # --- frozen features (extract once, reuse afterwards) -------------------
    tag = args.tag or default_tag(cfg)
    if args.max_videos and args.tag is None:
        tag = f"{tag}_n{args.max_videos}"
    feat_path = Path(cfg["feature"]["dir"]).expanduser() / f"{tag}.npz"
    if not feat_path.exists():
        if args.reuse_features:
            raise FileNotFoundError(f"feature cache missing: {feat_path}")
        logger.info(f"feature cache missing, extracting: {feat_path}")
        feat_path = extract(
            cfg, logger, device, tag=tag, max_videos=args.max_videos,
            overwrite=args.overwrite_features,
        )
    elif args.overwrite_features:
        feat_path = extract(cfg, logger, device, tag=tag, max_videos=args.max_videos, overwrite=True)

    cache = load_feature_cache(feat_path)
    features = cache["features"].astype(np.float32)
    labels = cache["labels"].astype(np.float32)
    split = cache["split"].astype(str)
    logger.info(f"features: {features.shape} from {feat_path}")

    # DataFrame aligned with the cache (same order as stored row_id).
    df_all = load_records(cfg["csv"], tasks)
    id_map = df_all.set_index("row_id")
    df = id_map.loc[cache["row_id"]].reset_index()
    assert (df["split"].to_numpy() == split).all(), "split mismatch between cache and csv"
    save_split_manifest(df, run_dir / "split_manifest.csv")

    stats = label_stats(df, tasks)
    dump_json(stats, run_dir / "label_stats.json")

    tr, va, te = (np.where(split == s)[0] for s in ("train", "val", "test"))
    logger.info(f"split sizes: train={len(tr)} val={len(va)} test={len(te)}")
    for j, t in enumerate(tasks):
        logger.info(f"  {t}: train={np.isfinite(labels[tr, j]).sum()} val={np.isfinite(labels[va, j]).sum()} test={np.isfinite(labels[te, j]).sum()}")

    train_out = fit_mlp(
        features[tr], labels[tr], features[va], labels[va], stats, tasks, cfg["train"], seed, device, logger
    )
    head, history, best_epoch = train_out["head"], train_out["history"], train_out["best_epoch"]

    # --- evaluation ---------------------------------------------------------
    results, all_preds = {}, {}
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        preds_std = predict_mlp(head, torch.as_tensor(features[idx]), device=device)
        preds = inverse_standardize(preds_std, stats, tasks)
        all_preds[name] = preds
        results[name] = masked_regression_metrics(labels[idx], preds, tasks)
        logger.info(f"[{name}]\n" + format_metrics_table(results[name], tasks))

    plot_loss_curve(history, fig_dir / "loss_curve.png", best_epoch=best_epoch)
    fit_info = {}
    for name in ("val", "test"):
        idx = {"val": va, "test": te}[name]
        for j, task in enumerate(tasks):
            fname = f"pred_vs_true_{task}.png" if name == "test" else f"pred_vs_true_{name}_{task}.png"
            fit_info[f"{name}/{task}"] = plot_pred_vs_true(
                labels[idx, j], all_preds[name][:, j], task, fig_dir / fname, results[name]["per_task"][task]
            )

    # --- save artifacts -----------------------------------------------------
    torch.save({"state_dict": head.state_dict(), "in_dim": head.in_dim, "out_dim": head.out_dim,
                "tasks": tasks, "stats": stats, "config": cfg["train"]}, run_dir / "mlp_head.pt")
    pred_df = pd.concat(
        [save_predictions(df.iloc[idx].reset_index(drop=True), all_preds[name], tasks, run_dir / f"predictions_{name}.csv", split_label=name)
         for name, idx in (("train", tr), ("val", va), ("test", te))],
        ignore_index=True,
    )
    pred_df.to_csv(run_dir / "predictions.csv", index=False)

    metrics_out = {
        "subproject": "videomae",
        "feature_cache": str(feat_path),
        "feature_dim": int(features.shape[1]),
        "num_frames": cfg["encoder"]["num_frames"],
        "image_size": cfg["encoder"]["image_size"],
        "pooling": cfg["encoder"]["pooling"],
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": train_out["best_val_loss"],
        "label_stats": stats,
        "metrics": results,
        "pred_fit": fit_info,
        "history": history,
        "git": git_info(ROOT),
        "environment": environment_snapshot(),
    }
    dump_json(metrics_out, run_dir / "metrics.json")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    link_latest_files(run_dir, ["metrics.json", "predictions.csv", "config.yaml", "mlp_head.pt", "label_stats.json"])

    test_macro = results["test"]["macro"]
    append_experiment_log(
        ROOT,
        f"VideoMAE 冻结特征 + MLP 回归（{run_name}）",
        {
            "purpose": "VideoMAE 冻结 backbone 提取特征，训练 MLP 多任务回归头，报告各任务 MAE/RMSE/R2。",
            "command": " ".join(sys.argv),
            "params": f"seed={seed}, frames={cfg['encoder']['num_frames']}, size={cfg['encoder']['image_size']}, "
                      f"pooling={cfg['encoder']['pooling']}, hidden={cfg['head']['hidden_dims']}, "
                      f"lr={cfg['train']['lr']}, bs={cfg['train']['batch_size']}, epochs={cfg['train']['epochs']}, "
                      f"best_epoch={best_epoch}",
            "input": cfg["csv"],
            "output": str(run_dir),
            "result": f"test macro MAE={test_macro['mae']:.4f}, RMSE={test_macro['rmse']:.4f}, R2={test_macro['r2']:.4f}\n"
                      + format_metrics_table(results["test"], tasks),
            "conclusion": f"特征缓存 {feat_path}；最佳 epoch={best_epoch}；详细指标见 metrics.json。",
        },
    )
    logger.info(f"done. results={run_dir}")
    logger.info(f"figures={fig_dir}")


if __name__ == "__main__":
    main()
