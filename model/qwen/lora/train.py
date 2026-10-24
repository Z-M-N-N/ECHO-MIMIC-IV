"""Qwen2.5-VL vision tower + LoRA + MLP classifier, trained end-to-end.

    python train.py --config config.yaml --device cuda

Only LoRA adapters (attention qkv/proj) and the MLP classification head are
trainable; every other vision-tower parameter is frozen.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "__init__.py").exists())
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common.data import (  # noqa: E402
    CLASS_NAMES, TASK_COLS, TASK_NUM_CLASSES, VideoClassificationDataset, class_stats, collate_batch,
    load_records, save_split_manifest,
)
from common.metrics import format_metrics_table, masked_classification_metrics  # noqa: E402
from common.plots import plot_confusion_matrix, plot_loss_curve, plot_pred_distribution, plot_roc_curve, plot_val_metric  # noqa: E402
from common.trainer import masked_ce_loss, save_predictions  # noqa: E402
from common.utils import (  # noqa: E402
    AverageMeter, EarlyStopping, append_experiment_log, dump_json, environment_snapshot, git_info,
    link_latest_files, load_config, make_run_dir, resolve_device, save_config, seed_everything, setup_logger,
    worker_init_fn,
)
from model import QwenLoraClassifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen2.5-VL LoRA + MLP classification")
    p.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.yaml"))
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--max-train-samples", type=int, default=None, help="debug subset of train split")
    p.add_argument("--max-videos", type=int, default=None, help="debug: truncate every split")
    return p.parse_args()


def make_loader(cfg, df, tasks, device, train: bool, batch_size: int, max_samples=None):
    enc = cfg["encoder"]
    if max_samples:
        df = df.iloc[: int(max_samples)].reset_index(drop=True)
    ds = VideoClassificationDataset(df, tasks, num_frames=enc["num_frames"], image_size=enc["image_size"])
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=int(enc.get("num_workers", 8)),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_batch,
        persistent_workers=False,
        worker_init_fn=worker_init_fn,
    ), ds


@torch.no_grad()
def evaluate(model, loader, tasks, num_classes, device, amp_dtype, class_weights, collect_features=False):
    model.eval()
    probs = {t: [] for t in tasks}
    labels, feats = [], []
    total_loss, total_n = 0.0, 0
    meta_all = {k: [] for k in ("row_id", "subject_id", "study_id", "measurement_id", "video_path", "split")}
    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits, f = model(frames, return_features=True)
        loss = masked_ce_loss(logits, y, tasks, class_weights=class_weights)
        total_loss += float(loss) * len(y)
        total_n += len(y)
        for t in tasks:
            probs[t].append(torch.softmax(logits[t].float(), dim=1).cpu().numpy())
        labels.append(y.cpu().numpy())
        if collect_features:
            feats.append(f.float().cpu().numpy().astype(np.float16))
        for k in meta_all:
            meta_all[k].extend(batch["meta"][k])
    return {
        "probs": {t: np.concatenate(v, axis=0) for t, v in probs.items()},
        "labels": np.concatenate(labels, axis=0),
        "features": np.concatenate(feats, axis=0) if collect_features else None,
        "loss": total_loss / max(total_n, 1),
        "meta": meta_all,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    seed_everything(seed)
    device = resolve_device(args.device)
    logger = setup_logger("qwen.lora.train")

    tasks = list(cfg.get("tasks") or TASK_COLS)
    cfg["tasks"] = tasks
    cfg["num_classes"] = {
        t: int(cfg.get("num_classes", {}).get(t, TASK_NUM_CLASSES[t])) for t in tasks
    }
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)

    run_name = args.run_name or cfg.get("output", {}).get("run_name", "qwen_lora")
    run_dir = make_run_dir(cfg["output"]["root"], "results/qwen_lora", run_name)
    fig_dir = Path(cfg["output"]["root"]) / "figures/qwen_lora" / run_dir.name
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    logger.info(f"run dir: {run_dir}")

    df_all = load_records(cfg["csv"], tasks)
    save_split_manifest(df_all, run_dir / "split_manifest.csv")
    stats = class_stats(df_all, tasks)
    dump_json(stats, run_dir / "label_stats.json")

    tr_cfg = cfg["train"]
    bs = int(tr_cfg.get("batch_size", 2))
    eval_bs = int(tr_cfg.get("eval_batch_size", max(bs, 4)))
    splits = {s: df_all[df_all["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    if args.max_videos:
        splits = {k: v.iloc[: args.max_videos].reset_index(drop=True) for k, v in splits.items()}
    train_loader, train_ds = make_loader(cfg, splits["train"], tasks, device, True, bs, args.max_train_samples)
    val_loader, _ = make_loader(cfg, splits["val"], tasks, device, False, eval_bs)
    test_loader, _ = make_loader(cfg, splits["test"], tasks, device, False, eval_bs)
    logger.info(f"split sizes: train={len(train_ds)} val={len(splits['val'])} test={len(splits['test'])}")

    # --- model + optimizer --------------------------------------------------
    model = QwenLoraClassifier(cfg, device=device)
    summary = model.trainable_summary()
    logger.info(f"trainable summary: {json.dumps(summary)}")
    dump_json(summary, run_dir / "trainable_params.json")
    logger.info(f"vision load info: {model.encoder.load_info}")

    use_weights = str(tr_cfg.get("class_weight", "balanced")).lower() == "balanced"
    class_weights = {
        t: torch.tensor(stats[t]["class_weights"], dtype=torch.float32, device=device) for t in tasks
    } if use_weights else None

    lora_params = [p for p in model.encoder.visual.parameters() if p.requires_grad]
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": float(tr_cfg.get("lr_lora", 2e-4))},
            {"params": head_params, "lr": float(tr_cfg.get("lr_head", 1e-3))},
        ],
        weight_decay=float(tr_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(tr_cfg.get("epochs", 12))
    grad_accum = int(tr_cfg.get("grad_accum", 1))
    total_steps = max(1, int(np.ceil(len(train_loader) / grad_accum))) * epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps, eta_min=float(tr_cfg.get("min_lr_ratio", 0.05)) * float(tr_cfg.get("lr_lora", 2e-4))
    )
    stopper = EarlyStopping(patience=int(tr_cfg.get("patience", 3)), min_delta=float(tr_cfg.get("min_delta", 1e-4)), mode="max")
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(str(cfg["encoder"].get("dtype", "bfloat16")))
    focal_gamma = float(tr_cfg.get("focal_gamma", 0.0))
    kind = str(tr_cfg.get("loss", "ce")).lower()

    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_accuracy": [], "lr": []}
    best_epoch, best_score, best_lora, best_head = 0, float("-inf"), None, None

    for epoch in range(1, epochs + 1):
        model.train()
        meter = AverageMeter()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, 1):
            frames = batch["frames"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                logits = model(frames)
            loss = masked_ce_loss(
                logits, y, tasks, class_weights=class_weights,
                focal_gamma=focal_gamma if kind == "focal" else 0.0,
            )
            (loss / grad_accum).backward()
            meter.update(float(loss.detach()), frames.shape[0])

            if step % grad_accum == 0 or step == len(train_loader):
                if float(tr_cfg.get("grad_clip", 1.0)) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        float(tr_cfg.get("grad_clip", 1.0)),
                    )
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if step % int(tr_cfg.get("log_every", 20)) == 0:
                logger.info(f"epoch {epoch} step {step}/{len(train_loader)} | loss {meter.avg:.5f} | "
                            f"{meter.count / max(time.time() - t0, 1e-6):.2f} samples/s")

        val_out = evaluate(model, val_loader, tasks, cfg["num_classes"], device, amp_dtype, class_weights)
        val_metrics = masked_classification_metrics(val_out["labels"], val_out["probs"], tasks, cfg["num_classes"])
        val_f1 = float(val_metrics["macro"]["macro_f1"])
        val_acc = float(val_metrics["macro"]["accuracy"])
        history["train_loss"].append(meter.avg)
        history["val_loss"].append(val_out["loss"])
        history["val_macro_f1"].append(val_f1)
        history["val_accuracy"].append(val_acc)
        history["lr"].append(float(opt.param_groups[0]["lr"]))
        logger.info(f"epoch {epoch:3d}/{epochs} | train {meter.avg:.5f} | val {val_out['loss']:.5f} | "
                    f"val_F1m {val_f1:.4f} | val_Acc {val_acc:.4f} | lr {opt.param_groups[0]['lr']:.2e} | "
                    f"{time.time() - t0:.0f}s")

        if stopper.step(val_f1, epoch):
            best_epoch, best_score = epoch, val_f1
            best_lora = {k: v.detach().cpu().clone() for k, v in model.encoder.visual.state_dict().items() if "lora_" in k}
            best_head = {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}
            logger.info(f"  new best val macro-F1 {val_f1:.4f} @ epoch {epoch}")
        logger.info(f"  best {stopper.best:.4f} @ {best_epoch}")
        if stopper.should_stop:
            logger.info(f"early stopping at epoch {epoch} (patience={stopper.patience})")
            break

    if best_lora is not None:
        state = model.encoder.visual.state_dict()
        state.update(best_lora)
        model.encoder.visual.load_state_dict(state)
        model.head.load_state_dict(best_head)
        logger.info(f"restored best weights from epoch {best_epoch}")

    # --- final evaluation + pooled features --------------------------------
    results, all_probs, feature_bank = {}, {}, {"features": [], "labels": [], "split": [], "meta": {}}
    for name, loader, idx_df in (("train", train_loader, splits["train"]),
                                 ("val", val_loader, splits["val"]),
                                 ("test", test_loader, splits["test"])):
        out = evaluate(model, loader, tasks, cfg["num_classes"], device, amp_dtype, class_weights, collect_features=True)
        all_probs[name] = out["probs"]
        results[name] = masked_classification_metrics(out["labels"], out["probs"], tasks, cfg["num_classes"])
        logger.info(f"[{name}]\n" + format_metrics_table(results[name], tasks))
        feature_bank["features"].append(out["features"])
        feature_bank["labels"].append(out["labels"])
        feature_bank["split"].extend([name] * len(out["features"]))
        for k, v in out["meta"].items():
            feature_bank["meta"].setdefault(k, []).extend(v)

    # --- figures ------------------------------------------------------------
    plot_loss_curve(history, fig_dir / "loss_curve.png", best_epoch=best_epoch)
    plot_val_metric(history, "val_macro_f1", "macro-F1", "Validation macro-F1", fig_dir / "val_macro_f1.png")
    for name, idx_df in (("val", splits["val"]), ("test", splits["test"])):
        suffix = "" if name == "test" else "_val"
        for j, task in enumerate(tasks):
            m = results[name]["per_task"][task]
            plot_confusion_matrix(np.asarray(m["confusion_matrix"]), CLASS_NAMES[task], task,
                                  fig_dir / f"confusion_matrix{suffix}_{task}.png")
            plot_roc_curve(idx_df[task].to_numpy(dtype=np.float64), all_probs[name][task], task,
                           fig_dir / f"roc_curve{suffix}_{task}.png", CLASS_NAMES[task])
            if name == "test":
                plot_pred_distribution(idx_df[task].to_numpy(dtype=np.float64), all_probs[name][task], task,
                                       fig_dir / f"pred_dist_{task}.png")

    # --- artifacts ----------------------------------------------------------
    adapter_dir = run_dir / "adapter"
    model.encoder.visual.save_pretrained(adapter_dir)
    torch.save({"state_dict": model.head.state_dict(), "in_dim": model.head.in_dim,
                "num_classes": cfg["num_classes"], "tasks": tasks, "stats": stats,
                "config": cfg.get("head", {})}, run_dir / "mlp_head.pt")
    with open(run_dir / "lora_config.json", "w", encoding="utf-8") as f:
        json.dump({"lora": cfg.get("lora", {}), "trainable": summary,
                   "best_epoch": best_epoch, "best_val_macro_f1": best_score}, f, indent=2)

    feature_path = Path(cfg["output"]["root"]) / "features/qwen_lora" / f"{run_dir.name}.npz"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    meta = feature_bank["meta"]
    np.savez_compressed(
        feature_path,
        features=np.concatenate(feature_bank["features"], axis=0),
        labels=np.concatenate(feature_bank["labels"], axis=0).astype(np.int8),
        row_id=np.asarray(meta["row_id"], dtype=np.int64),
        subject_id=np.asarray(meta["subject_id"], dtype=np.int64),
        study_id=np.asarray(meta["study_id"], dtype=np.int64),
        measurement_id=np.asarray([np.nan if m is None else m for m in meta["measurement_id"]], dtype=np.float64),
        video_path=np.asarray(meta["video_path"]),
        split=np.asarray(feature_bank["split"]),
    )
    dump_json(
        {
            "tag": run_dir.name,
            "kind": "qwen_lora",
            "task_type": "classification",
            "run_dir": str(run_dir),
            "adapter_dir": str(adapter_dir),
            "num_frames": cfg["encoder"]["num_frames"],
            "image_size": cfg["encoder"]["image_size"],
            "pooling": cfg["encoder"]["pooling"],
            "feature_dim": int(feature_bank["features"][0].shape[1]),
            "dtype": "float16",
            "n_samples": int(sum(len(x) for x in feature_bank["features"])),
            "tasks": tasks,
            "num_classes": cfg["num_classes"],
            "class_names": {t: CLASS_NAMES[t] for t in tasks},
            "trainable": summary,
            "git": git_info(ROOT),
            "environment": environment_snapshot(),
            "created": datetime.now().isoformat(timespec="seconds"),
        },
        feature_path.with_suffix(".json"),
    )

    pred_df = pd.concat(
        [save_predictions(splits[name].reset_index(drop=True), all_probs[name], tasks,
                          run_dir / f"predictions_{name}.csv", split_label=name)
         for name in ("train", "val", "test")],
        ignore_index=True,
    )
    pred_df.to_csv(run_dir / "predictions.csv", index=False)

    metrics_out = {
        "subproject": "qwen_lora",
        "task_type": "classification",
        "vision_tower_only": True,
        "language_model_loaded": False,
        "adapter": str(adapter_dir),
        "feature_cache": str(feature_path),
        "feature_dim": int(feature_bank["features"][0].shape[1]),
        "num_frames": cfg["encoder"]["num_frames"],
        "image_size": cfg["encoder"]["image_size"],
        "pooling": cfg["encoder"]["pooling"],
        "lora": cfg.get("lora", {}),
        "trainable": summary,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_score,
        "label_stats": stats,
        "metrics": results,
        "history": history,
        "git": git_info(ROOT),
        "environment": environment_snapshot(),
    }
    dump_json(metrics_out, run_dir / "metrics.json")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    link_latest_files(run_dir, ["metrics.json", "predictions.csv", "config.yaml", "mlp_head.pt",
                                "label_stats.json", "lora_config.json"])

    tm = results["test"]["macro"]
    append_experiment_log(
        ROOT,
        f"Qwen2.5-VL 视觉塔 LoRA + MLP 分类（{run_name}）",
        {
            "purpose": "仅加载 Qwen2.5-VL 视觉塔，在注意力 qkv/proj 注入 LoRA，与 MLP 分类头联合训练。",
            "command": " ".join(sys.argv),
            "params": f"seed={seed}, frames={cfg['encoder']['num_frames']}, size={cfg['encoder']['image_size']}, "
                      f"lora_r={cfg['lora']['r']}, alpha={cfg['lora']['alpha']}, "
                      f"lr_lora={cfg['train']['lr_lora']}, lr_head={cfg['train']['lr_head']}, "
                      f"bs={cfg['train']['batch_size']}x{cfg['train'].get('grad_accum', 1)}, "
                      f"epochs={cfg['train']['epochs']}, best_epoch={best_epoch}, trainable={summary['trainable_params']}",
            "input": cfg["csv"],
            "output": str(run_dir),
            "result": f"test macro Acc={tm['accuracy']:.4f}, F1={tm['macro_f1']:.4f}, AUC={tm['macro_auc']:.4f}\n"
                      + format_metrics_table(results["test"], tasks),
            "conclusion": f"adapter -> {adapter_dir}；特征 -> {feature_path}；最佳 epoch={best_epoch}。",
        },
    )
    logger.info(f"done. results={run_dir}")
    logger.info(f"figures={fig_dir}")
    logger.info(f"adapter={adapter_dir}")


if __name__ == "__main__":
    main()
