"""Train the MLP classification head on frozen VideoMAE features.

    python train.py --config config.yaml --device cuda

The feature cache is created automatically when missing.  Results go to
timestamped ``out/results/videomae/<run>/`` directories (plus ``latest``
symlinks); figures to ``out/figures/videomae/<run>/``.
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

from common.data import (  # noqa: E402
    CLASS_NAMES, TASK_COLS, TASK_NUM_CLASSES, class_stats, load_records, save_split_manifest,
)
from common.metrics import format_metrics_table, masked_classification_metrics  # noqa: E402
from common.plots import plot_confusion_matrix, plot_loss_curve, plot_pred_distribution, plot_roc_curve, plot_val_metric  # noqa: E402
from common.trainer import fit_classifier, predict_proba, save_predictions  # noqa: E402
from common.utils import (  # noqa: E402
    append_experiment_log, dump_json, environment_snapshot, git_info, link_latest_files, load_config,
    make_run_dir, resolve_device, save_config, seed_everything, setup_logger,
)
from extract_features import default_tag, extract  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VideoMAE frozen features + MLP classification")
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
    cfg["num_classes"] = {
        t: int(cfg.get("num_classes", {}).get(t, TASK_NUM_CLASSES.get(t, 2))) for t in tasks
    }
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)

    run_name = args.run_name or cfg.get("output", {}).get("run_name", "videomae_mlp")
    run_dir = make_run_dir(cfg["output"]["root"], "results/videomae", run_name)
    fig_dir = Path(cfg["output"]["root"]) / "figures/videomae" / run_dir.name
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    logger.info(f"run dir: {run_dir}")

    # --- frozen features ----------------------------------------------------
    tag = args.tag or cfg.get("feature", {}).get("tag") or default_tag(cfg)
    if args.max_videos and args.tag is None:
        tag = f"{tag}_n{args.max_videos}"
    feat_path = Path(cfg["feature"]["dir"]).expanduser() / f"{tag}.npz"
    if not feat_path.exists():
        if args.reuse_features:
            raise FileNotFoundError(f"feature cache missing: {feat_path}")
        logger.info(f"feature cache missing, extracting: {feat_path}")
        feat_path = extract(cfg, logger, device, tag=tag, max_videos=args.max_videos,
                            overwrite=args.overwrite_features)
    elif args.overwrite_features:
        feat_path = extract(cfg, logger, device, tag=tag, max_videos=args.max_videos, overwrite=True)

    cache = load_feature_cache(feat_path)
    features = cache["features"].astype(np.float32)
    labels = cache["labels"].astype(np.int64)
    split = cache["split"].astype(str)
    logger.info(f"features: {features.shape} from {feat_path}")

    df_all = load_records(cfg["csv"], tasks)
    df = df_all.set_index("row_id").loc[cache["row_id"]].reset_index()
    assert (df["split"].to_numpy() == split).all(), "split mismatch between cache and csv"
    save_split_manifest(df, run_dir / "split_manifest.csv")

    stats = class_stats(df, tasks, cfg.get("num_classes"))
    dump_json(stats, run_dir / "label_stats.json")
    tr, va, te = (np.where(split == s)[0] for s in ("train", "val", "test"))
    logger.info(f"split sizes: train={len(tr)} val={len(va)} test={len(te)}")
    for j, t in enumerate(tasks):
        logger.info(f"  {t}: train={(labels[tr, j] >= 0).sum()} val={(labels[va, j] >= 0).sum()} "
                    f"test={(labels[te, j] >= 0).sum()} classes={cfg['num_classes'][t]}")

    out = fit_classifier(
        features[tr], labels[tr], features[va], labels[va], stats, tasks, cfg["train"], seed, device, logger
    )
    model, history, best_epoch = out["model"], out["history"], out["best_epoch"]

    # --- evaluation ---------------------------------------------------------
    results, all_probs = {}, {}
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        probs = predict_proba(model, torch.as_tensor(features[idx]), tasks, device=device)
        all_probs[name] = probs
        results[name] = masked_classification_metrics(labels[idx], probs, tasks, cfg["num_classes"])
        logger.info(f"[{name}]\n" + format_metrics_table(results[name], tasks))

    plot_loss_curve(history, fig_dir / "loss_curve.png", best_epoch=best_epoch)
    plot_val_metric(history, "val_macro_f1", "macro-F1", "Validation macro-F1", fig_dir / "val_macro_f1.png")
    for name, idx in (("val", va), ("test", te)):
        suffix = "" if name == "test" else "_val"
        for j, task in enumerate(tasks):
            m = results[name]["per_task"][task]
            plot_confusion_matrix(np.asarray(m["confusion_matrix"]), [str(i) for i in range(cfg["num_classes"][task])], task,
                                  fig_dir / f"confusion_matrix{suffix}_{task}.png")
            plot_roc_curve(labels[idx, j], all_probs[name][task], task,
                           fig_dir / f"roc_curve{suffix}_{task}.png", [str(i) for i in range(cfg["num_classes"][task])])
            if name == "test":
                plot_pred_distribution(labels[idx, j], all_probs[name][task], task,
                                       fig_dir / f"pred_dist_{task}.png")

    # --- artifacts ----------------------------------------------------------
    torch.save({"state_dict": model.state_dict(), "in_dim": model.in_dim,
                "num_classes": cfg["num_classes"], "tasks": tasks, "stats": stats,
                "config": cfg["train"]}, run_dir / "mlp_head.pt")
    pred_df = pd.concat(
        [save_predictions(df.iloc[idx].reset_index(drop=True), all_probs[name], tasks,
                          run_dir / f"predictions_{name}.csv", split_label=name)
         for name, idx in (("train", tr), ("val", va), ("test", te))],
        ignore_index=True,
    )
    pred_df.to_csv(run_dir / "predictions.csv", index=False)

    metrics_out = {
        "subproject": "videomae",
        "task_type": "classification",
        "feature_cache": str(feat_path),
        "feature_dim": int(features.shape[1]),
        "num_frames": cfg["encoder"]["num_frames"],
        "image_size": cfg["encoder"]["image_size"],
        "pooling": cfg["encoder"]["pooling"],
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": out["best_val_macro_f1"],
        "label_stats": stats,
        "metrics": results,
        "history": history,
        "git": git_info(ROOT),
        "environment": environment_snapshot(),
    }
    dump_json(metrics_out, run_dir / "metrics.json")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    link_latest_files(run_dir, ["metrics.json", "predictions.csv", "config.yaml", "mlp_head.pt", "label_stats.json"])

    tm = results["test"]["macro"]
    append_experiment_log(
        ROOT,
        f"VideoMAE 冻结特征 + MLP 分类（{run_name}）",
        {
            "purpose": "VideoMAE 冻结 backbone 提取特征，训练 MLP 多任务分类头；报告 Accuracy/macro-F1/macro-AUC 与混淆矩阵。",
            "command": " ".join(sys.argv),
            "params": f"seed={seed}, frames={cfg['encoder']['num_frames']}, size={cfg['encoder']['image_size']}, "
                      f"pooling={cfg['encoder']['pooling']}, hidden={cfg['head']['hidden_dims']}, "
                      f"lr={cfg['train']['lr']}, bs={cfg['train']['batch_size']}, epochs={cfg['train']['epochs']}, "
                      f"class_weight={cfg['train'].get('class_weight', 'balanced')}, best_epoch={best_epoch}",
            "input": cfg["csv"],
            "output": str(run_dir),
            "result": f"test macro Acc={tm['accuracy']:.4f}, F1={tm['macro_f1']:.4f}, AUC={tm['macro_auc']:.4f}\n"
                      + format_metrics_table(results["test"], tasks),
            "conclusion": f"特征缓存 {feat_path}；最佳 epoch={best_epoch}；详细指标见 metrics.json。",
        },
    )
    logger.info(f"done. results={run_dir}")
    logger.info(f"figures={fig_dir}")


if __name__ == "__main__":
    main()
