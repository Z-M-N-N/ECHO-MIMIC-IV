"""Train single-view / multi-view feature-fusion MLP classifiers on cached features.

    python train.py --config config.yaml --device cuda

For the configured task it runs A4C, A2C and A4C+A2C (concat/mean/attention),
using identical samples, split and seed, then writes a comparison table.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "__init__.py").exists())
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common.metrics import format_metrics_table, masked_classification_metrics  # noqa: E402
from common.plots import (  # noqa: E402
    plot_confusion_matrix, plot_loss_curve, plot_pred_distribution, plot_roc_curve, plot_val_metric,
)
from common.utils import (  # noqa: E402
    AverageMeter, EarlyStopping, append_experiment_log, dump_json, environment_snapshot, git_info,
    load_config, make_run_dir, resolve_device, save_config, seed_everything, setup_logger, worker_init_fn,
)
from dataset import (  # noqa: E402
    FeatureDataset, align_features, build_alignment, collate_features, load_feature_cache,
)
from model import FusionClassifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen multi-view feature-fusion classification")
    p.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.yaml"))
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--task", default=None, help="override config task, e.g. cls_rv_function")
    p.add_argument("--groups", default=None, help="comma-separated group names; default = all in config")
    return p.parse_args()


def predict_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    with torch.inference_mode():
        for batch in loader:
            views = [x.to(device) for x in batch["features"]]
            logits = model(views)
            probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            labels.append(batch["labels"].numpy())
    return np.concatenate(probs, axis=0), np.concatenate(labels, axis=0)


def train_group(
    cfg: dict, group: dict, features: dict, labels: np.ndarray, splits: np.ndarray,
    task: str, num_classes: int, device, logger, table: pd.DataFrame,
) -> dict:
    views = list(group["views"])
    fusion = group["fusion"]
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    tr_cfg = cfg["train"]

    X = {v: features[v] for v in views}
    tr_idx = np.where(splits == "train")[0]
    va_idx = np.where(splits == "val")[0]
    te_idx = np.where(splits == "test")[0]
    train_ds = FeatureDataset({v: X[v][tr_idx] for v in views}, labels[tr_idx])
    val_ds = FeatureDataset({v: X[v][va_idx] for v in views}, labels[va_idx])
    test_ds = FeatureDataset({v: X[v][te_idx] for v in views}, labels[te_idx])
    bs = int(tr_cfg.get("batch_size", 128))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, collate_fn=collate_features)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0, collate_fn=collate_features)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0, collate_fn=collate_features)

    in_dim = int(X[views[0]].shape[1])
    model = FusionClassifier(
        in_dim=in_dim, num_classes=num_classes, num_views=len(views), fusion=fusion,
        hidden_dims=tuple(cfg["head"].get("hidden_dims", [512, 256])),
        dropout=float(cfg["head"].get("dropout", 0.2)),
        activation=str(cfg["head"].get("activation", "gelu")),
        input_norm=bool(cfg["head"].get("input_norm", True)),
    ).to(device)

    counts = np.bincount(labels[tr_idx], minlength=num_classes).astype(np.float64)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device) if tr_cfg.get("class_weight", "balanced") == "balanced" else None
    logger.info(f"[{group['name']}] views={views} fusion={fusion} dim={model.fused_dim} "
                f"train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)} class_weights={weights.round(3).tolist()}")

    opt = torch.optim.AdamW(model.parameters(), lr=float(tr_cfg.get("lr", 1e-3)), weight_decay=float(tr_cfg.get("weight_decay", 1e-4)))
    epochs = int(tr_cfg.get("epochs", 200))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    stopper = EarlyStopping(patience=int(tr_cfg.get("patience", 25)), min_delta=float(tr_cfg.get("min_delta", 1e-4)), mode="max")
    focal_gamma = float(tr_cfg.get("focal_gamma", 0.0))
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_accuracy": [], "lr": []}
    best_state, best_epoch, best_f1 = None, 0, float("-inf")

    for epoch in range(1, epochs + 1):
        model.train()
        meter = AverageMeter()
        for batch in train_loader:
            xs = [x.to(device) for x in batch["features"]]
            y = batch["labels"].to(device)
            logits = model(xs)
            ce = F.cross_entropy(logits, y, weight=class_weights, reduction="none")
            if focal_gamma > 0:
                pt = torch.exp(-F.cross_entropy(logits, y, reduction="none"))
                ce = (1.0 - pt) ** focal_gamma * ce
            loss = ce.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(tr_cfg.get("grad_clip", 5.0)) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr_cfg.get("grad_clip", 5.0)))
            opt.step()
            meter.update(float(loss.detach()), len(y))
        sched.step()

        val_probs, val_y = predict_probs(model, val_loader, device)
        val_metrics = masked_classification_metrics(val_y[:, None], {task: val_probs}, [task], {task: num_classes})
        val_f1 = float(val_metrics["macro"]["macro_f1"])
        val_acc = float(val_metrics["macro"]["accuracy"])
        with torch.inference_mode():
            vp = torch.as_tensor(val_probs, device=device)
            vy = torch.as_tensor(val_y, device=device)
            val_loss = float(F.cross_entropy(vp.log(), vy, weight=class_weights).item())
        history["train_loss"].append(meter.avg)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_f1)
        history["val_accuracy"].append(val_acc)
        history["lr"].append(float(opt.param_groups[0]["lr"]))
        logger.info(f"[{group['name']}] epoch {epoch:3d}/{epochs} | train {meter.avg:.5f} | val {val_loss:.5f} | "
                    f"val_F1m {val_f1:.4f} | val_Acc {val_acc:.4f}")

        if stopper.step(val_f1, epoch):
            best_epoch, best_f1 = epoch, val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"  new best val macro-F1 {val_f1:.4f} @ epoch {epoch}")
        if stopper.should_stop:
            logger.info(f"  early stopping at epoch {epoch} (patience={stopper.patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info(f"[{group['name']}] restored best epoch {best_epoch} (val F1 {best_f1:.4f})")

    # Evaluate all splits and save artifacts
    suffix = str(cfg.get("output", {}).get("name_suffix", ""))
    run_name = f"{task}_{group['name']}{suffix}"
    run_dir = Path(cfg["output"]["root"]) / "results/qwen_feature_fusion" / run_name
    fig_dir = Path(cfg["output"]["root"]) / "figures/qwen_feature_fusion" / task / f"{group['name']}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    all_probs = {}
    for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0, collate_fn=collate_features)
        probs, y = predict_probs(model, loader, device)
        all_probs[name] = probs
        results[name] = masked_classification_metrics(y[:, None], {task: probs}, [task], {task: num_classes})
        logger.info(f"[{run_name} / {name}]\n" + format_metrics_table(results[name], [task]))

    plot_loss_curve(history, fig_dir / "loss_curve.png", best_epoch=best_epoch)
    plot_val_metric(history, "val_macro_f1", "macro-F1", "Validation macro-F1", fig_dir / "val_macro_f1.png")
    for split_name in ("val", "test"):
        suffix = "" if split_name == "test" else "_val"
        ds = val_ds if split_name == "val" else test_ds
        y = ds.labels.numpy()
        m = results[split_name]["per_task"][task]
        plot_confusion_matrix(np.asarray(m["confusion_matrix"]), ["0", "1"], task,
                              fig_dir / f"confusion_matrix{suffix}_{task}.png")
        plot_roc_curve(y, all_probs[split_name], task, fig_dir / f"roc_curve{suffix}_{task}.png", ["0", "1"])
        if split_name == "test":
            plot_pred_distribution(y, all_probs[split_name], task, fig_dir / f"pred_dist_{task}.png")

    torch.save({"state_dict": model.state_dict(), "views": views, "fusion": fusion,
                "in_dim": in_dim, "num_classes": num_classes, "task": task},
               run_dir / "mlp_head.pt")
    save_config(cfg, run_dir / "config.yaml")
    dump_json({"best_epoch": best_epoch, "best_val_macro_f1": best_f1,
               "history": history, "metrics": results, "views": views, "fusion": fusion},
              run_dir / "metrics.json")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    # Long-format predictions with both class probabilities
    rows = []
    for split_name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        idx = {"train": tr_idx, "val": va_idx, "test": te_idx}[split_name]
        for i, row_i in enumerate(idx):
            rows.append({"sample_idx": int(row_i), "subject_id": int(table.iloc[row_i]["subject_id"]),
                         "study_id": int(table.iloc[row_i]["study_id"]),
                         "split": split_name, "task": task,
                         "y_true": int(labels[row_i]), "y_pred": int(all_probs[split_name][i].argmax()),
                         "prob_0": float(all_probs[split_name][i, 0]),
                         "prob_1": float(all_probs[split_name][i, 1])})
    pd.DataFrame(rows).to_csv(run_dir / "predictions.csv", index=False)

    return {"group": group["name"], "views": "+".join(views), "fusion": fusion,
            "best_epoch": best_epoch, "best_val_f1": best_f1, "metrics": results}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(args.seed if args.seed is not None else cfg.get("seed", 42)))
    device = resolve_device(args.device)
    logger = setup_logger("qwen.feature_fusion.train")
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    if args.task:
        cfg["task"] = args.task

    task = cfg["task"]
    label_view = cfg.get("label_view", "A4C")
    view_csvs = {v: spec["csv"] for v, spec in cfg["views"].items()}
    table = build_alignment(cfg["views"][label_view]["csv"], view_csvs, task, label_view=label_view)
    labels = table["label"].to_numpy(dtype=np.int64)
    splits = table["split"].to_numpy()
    num_classes = int(cfg.get("num_classes", {}).get(task, int(labels.max()) + 1))
    logger.info(f"task={task} samples={len(table)} split={pd.Series(splits).value_counts().to_dict()}")

    feature_dir = Path(cfg["feature"]["dir"]).expanduser()
    tag = cfg["feature"]["tag"]
    caches = {v: load_feature_cache(feature_dir / v / f"{tag}.npz") for v in cfg["views"]}
    cached_ids = set.intersection(*[set(c["study_id"].astype(int)) for c in caches.values()])
    before = len(table)
    table = table[table["study_id"].astype(int).isin(cached_ids)].reset_index(drop=True)
    table.loc[:, "sample_idx"] = np.arange(len(table), dtype=np.int64)
    labels = table["label"].to_numpy(dtype=np.int64)
    splits = table["split"].to_numpy()
    if len(table) != before:
        logger.info(f"restricted to studies present in all feature caches: {before} -> {len(table)}")
    raw = {v: align_features(caches[v], table["study_id"], v) for v in cfg["views"]}
    # Per-view z-score using train statistics
    tr_idx = np.where(splits == "train")[0]
    features, stats = {}, {}
    for v, x in raw.items():
        mu = x[tr_idx].mean(axis=0)
        sd = x[tr_idx].std(axis=0)
        sd[sd < 1e-6] = 1.0
        features[v] = ((x - mu) / sd).astype(np.float32)
        stats[v] = {"mean": mu.tolist(), "std": sd.tolist(), "dim": int(x.shape[1])}
    dump_json(stats, Path(cfg["output"]["root"]) / "results/qwen_feature_fusion" /
          f"feature_stats_{task}{cfg.get('output', {}).get('name_suffix', '')}.json")

    groups = cfg["groups"]
    if args.groups:
        wanted = {g.strip() for g in args.groups.split(",")}
        groups = [g for g in groups if g["name"] in wanted]
    summaries = []
    for group in groups:
        summaries.append(train_group(cfg, group, features, labels, splits, task, num_classes, device, logger, table))

    # Comparison table
    comp_rows = []
    base = {s["group"]: s for s in summaries}
    for s in summaries:
        t = s["metrics"]["test"]["per_task"][task]
        row = {"group": s["group"], "views": s["views"], "fusion": s["fusion"],
               "best_epoch": s["best_epoch"],
               "test_acc": t["accuracy"], "test_macro_f1": t["macro_f1"], "test_macro_auc": t["macro_auc"],
               "test_sensitivity": t["sensitivity"], "test_specificity": t["specificity"],
               "test_precision_macro": t["macro_precision"], "test_recall_macro": t["macro_recall"]}
        for bname in dict.fromkeys(("A4C", "A2C", cfg.get("label_view", "A4C"))):
            if bname in base:
                bt = base[bname]["metrics"]["test"]["per_task"][task]
                row[f"d_acc_vs_{bname}"] = row["test_acc"] - bt["accuracy"]
                row[f"d_f1_vs_{bname}"] = row["test_macro_f1"] - bt["macro_f1"]
                row[f"d_auc_vs_{bname}"] = row["test_macro_auc"] - bt["macro_auc"]
        comp_rows.append(row)
    comp = pd.DataFrame(comp_rows)
    comp_dir = Path(cfg["output"]["root"]) / "results/qwen_feature_fusion"
    comp_dir.mkdir(parents=True, exist_ok=True)
    suffix = str(cfg.get("output", {}).get("name_suffix", ""))
    comp.to_csv(comp_dir / f"comparison_{task}{suffix}.csv", index=False)
    with open(comp_dir / f"comparison_{task}{suffix}.md", "w", encoding="utf-8") as f:
        f.write(f"# {task}: single-view vs feature fusion\n\n")
        f.write(comp.to_string(index=False))
        f.write("\n")

    append_experiment_log(
        ROOT, f"Qwen 多视角特征融合分类（{task}）",
        {
            "purpose": "A4C / A2C 单视角与 A4C+A2C 特征级融合（concat/mean/attention）的对比原型。",
            "command": " ".join(sys.argv),
            "params": f"seed={cfg.get('seed')}, frames={cfg['encoder']['num_frames']}, "
                      f"image_size={cfg['encoder']['image_size']}, task={task}, "
                      f"samples={len(table)}",
            "input": f"{cfg['views']['A4C']['csv']}; {cfg['views']['A2C']['csv']}",
            "output": str(comp_dir),
            "result": comp.to_string(index=False),
            "conclusion": "融合是否优于单视角见 comparison 表；所有实验使用相同 study 子集与划分。",
        },
    )
    logger.info(f"comparison saved: {comp_dir / f'comparison_{task}{suffix}.csv'}")


if __name__ == "__main__":
    main()
