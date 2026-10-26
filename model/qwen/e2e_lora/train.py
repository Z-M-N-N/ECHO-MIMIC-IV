"""End-to-end Qwen2.5-VL + LLM LoRA + MLP for A4C/A2C/joint classification."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "__init__.py").exists())
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common.metrics import format_metrics_table, masked_classification_metrics  # noqa: E402
from common.plots import (  # noqa: E402
    plot_confusion_matrix, plot_loss_curve, plot_pred_distribution, plot_roc_curve, plot_val_metric,
)
from common.utils import (  # noqa: E402
    AverageMeter, EarlyStopping, append_experiment_log, dump_json, environment_snapshot, git_info,
    load_config, resolve_device, save_config, seed_everything, setup_logger,
)
from dataset import E2EDataset, build_alignment  # noqa: E402
from lora_utils import load_lora_state_dict, lora_state_dict  # noqa: E402
from model import QwenE2EClassifier, build_optimizer  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Qwen2.5-VL e2e LoRA classification")
    p.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.yaml"))
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--task", default=None)
    p.add_argument("--groups", default=None, help="comma-separated group names")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    return p.parse_args()


def build_inputs(processor, cfg, items, views):
    prompt = cfg["prompt"]
    texts, videos = [], []
    for item in items:
        content = []
        for view, frames in zip(views, item["frames"]):
            imgs = [Image.fromarray(f.permute(1, 2, 0).numpy()) for f in frames]
            videos.append(imgs)
            content.append({"type": "video", "video": imgs})
            content.append({"type": "text", "text": f"{prompt['view_names'][view]}视角。"})
        content.append({"type": "text", "text": prompt["question"] + prompt.get("anchor_text", "答案：")})
        msgs = [{"role": "user", "content": content}]
        texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    return processor(text=texts, videos=videos, return_tensors="pt", padding=True)


def to_device(inputs, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}


@torch.no_grad()
def evaluate(model, processor, cfg, loader, views, device, task, num_classes, max_samples=None):
    model.eval()
    probs, labels, ids = [], [], []
    for step, items in enumerate(loader):
        if max_samples and step * len(items) >= max_samples:
            break
        inputs = to_device(build_inputs(processor, cfg, items, views), device)
        logits = model(inputs)
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        labels.extend([it["label"] for it in items])
        ids.extend([it["study_id"] for it in items])
    p = np.concatenate(probs, 0)
    y = np.asarray(labels, dtype=np.int64)
    m = masked_classification_metrics(y[:, None], {task: p}, [task], {task: num_classes})
    return p, y, np.asarray(ids), m


def train_group(cfg, group, table, processor, task, num_classes, device, logger, args):
    views = list(group["views"])
    tr_cfg = cfg["train"]
    seed_everything(int(cfg.get("seed", 42)))
    run_name = f"{task}_{group['name']}"
    run_dir = Path(cfg["output"]["root"]) / "results/qwen_e2e_lora" / run_name
    fig_dir = Path(cfg["output"]["root"]) / "figures/qwen_e2e_lora" / run_name
    ckpt_dir = Path(cfg["output"]["root"]) / "checkpoints/qwen_e2e_lora" / run_name
    for d in (run_dir, fig_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    splits = table["split"].to_numpy()
    idx_tr = np.where(splits == "train")[0]
    idx_va = np.where(splits == "val")[0]
    idx_te = np.where(splits == "test")[0]
    if args.max_train_samples:
        idx_tr = idx_tr[: args.max_train_samples]
    if args.max_eval_samples:
        idx_va = idx_va[: args.max_eval_samples]
        idx_te = idx_te[: args.max_eval_samples]
    logger.info(f"[{run_name}] samples train={len(idx_tr)} val={len(idx_va)} test={len(idx_te)} views={views}")

    def make_ds(idx):
        return E2EDataset(table.iloc[idx].reset_index(drop=True), views,
                          num_frames=cfg["frame"]["num_frames"], image_size=cfg["frame"]["image_size"])

    train_ds, val_ds, test_ds = make_ds(idx_tr), make_ds(idx_va), make_ds(idx_te)
    bs = int(cfg["train"].get("batch_size_joint" if len(views) > 1 else "batch_size_single", 1))
    eval_bs = int(cfg["train"].get("eval_batch_size", 2))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=int(cfg["train"].get("num_workers", 4)),
                              collate_fn=lambda x: x, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=eval_bs, shuffle=False, num_workers=int(cfg["train"].get("num_workers", 4)),
                            collate_fn=lambda x: x)
    test_loader = DataLoader(test_ds, batch_size=eval_bs, shuffle=False, num_workers=int(cfg["train"].get("num_workers", 4)),
                             collate_fn=lambda x: x)

    model = QwenE2EClassifier(cfg, device=device).to(device)
    summary = model.trainable_summary()
    logger.info(f"[{run_name}] trainable: {summary}")
    dump_json(summary, run_dir / "trainable_params.json")

    counts = np.bincount(table.iloc[idx_tr]["label"].to_numpy(), minlength=num_classes).astype(np.float64)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device) if tr_cfg.get("class_weight", "balanced") == "balanced" else None
    opt, lora_params, head_params = build_optimizer(model, cfg)
    epochs = int(args.epochs or tr_cfg["epochs"])
    accum = max(1, int(tr_cfg.get("grad_accum", 8)) // bs)
    total_steps = max(1, int(np.ceil(len(train_loader) / accum))) * epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    stopper = EarlyStopping(patience=int(tr_cfg.get("patience", 2)), min_delta=float(tr_cfg.get("min_delta", 1e-4)), mode="max")
    focal = float(tr_cfg.get("focal_gamma", 0.0))

    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_accuracy": [], "lr": []}
    best_epoch, best_f1, best_lora, best_head = 0, float("-inf"), None, None

    for epoch in range(1, epochs + 1):
        model.train()
        meter = AverageMeter()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        for step, items in enumerate(train_loader, 1):
            inputs = to_device(build_inputs(processor, cfg, items, views), device)
            logits = model(inputs)
            y = torch.tensor([it["label"] for it in items], device=device, dtype=torch.long)
            ce = F.cross_entropy(logits, y, weight=class_weights, reduction="none")
            if focal > 0:
                pt = torch.exp(-F.cross_entropy(logits, y, reduction="none"))
                ce = (1.0 - pt) ** focal * ce
            loss = ce.mean()
            (loss / accum).backward()
            meter.update(float(loss.detach()), len(items))
            if step % accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                               float(tr_cfg.get("grad_clip", 1.0)))
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if step % int(tr_cfg.get("log_every", 20)) == 0:
                logger.info(f"[{run_name}] epoch {epoch} step {step}/{len(train_loader)} | loss {meter.avg:.5f} | "
                            f"{meter.count / max(time.time() - t0, 1e-6):.3f} samples/s | {time.time()-t0:.0f}s")

        val_probs, val_y, _, val_m = evaluate(model, processor, cfg, val_loader, views, device, task, num_classes)
        val_f1 = float(val_m["macro"]["macro_f1"]); val_acc = float(val_m["macro"]["accuracy"])
        vp = torch.as_tensor(val_probs, device=device)
        vy = torch.as_tensor(val_y, device=device)
        val_loss = float(F.cross_entropy(vp.log(), vy, weight=class_weights).item())
        history["train_loss"].append(meter.avg); history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_f1); history["val_accuracy"].append(val_acc)
        history["lr"].append(float(opt.param_groups[0]["lr"]))
        logger.info(f"[{run_name}] epoch {epoch:2d}/{epochs} | train {meter.avg:.5f} | val {val_loss:.5f} | "
                    f"val_F1m {val_f1:.4f} | val_Acc {val_acc:.4f} | {time.time()-t0:.0f}s")
        if stopper.step(val_f1, epoch):
            best_epoch, best_f1 = epoch, val_f1
            best_lora = lora_state_dict(model.qwen)
            best_head = {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}
            logger.info(f"  new best val macro-F1 {val_f1:.4f} @ epoch {epoch}")
        if stopper.should_stop:
            logger.info(f"  early stopping at epoch {epoch}")
            break

    if best_lora is not None:
        load_lora_state_dict(model.qwen, best_lora)
        model.head.load_state_dict(best_head)
        logger.info(f"[{run_name}] restored best epoch {best_epoch}")

    results, preds_all = {}, {}
    train_eval_max = int(tr_cfg.get("eval_train_samples", 0)) or None
    for name, ds, loader in (("train", train_ds, train_loader), ("val", val_ds, val_loader), ("test", test_ds, test_loader)):
        probs, y, ids, m = evaluate(model, processor, cfg, loader, views, device, task, num_classes,
                                     max_samples=train_eval_max if name == "train" else None)
        results[name] = m; preds_all[name] = (probs, y, ids)
        logger.info(f"[{run_name} / {name}]\n" + format_metrics_table(m, [task]))

    plot_loss_curve(history, fig_dir / "loss_curve.png", best_epoch=best_epoch)
    plot_val_metric(history, "val_macro_f1", "macro-F1", "Validation macro-F1", fig_dir / "val_macro_f1.png")
    for split_name in ("val", "test"):
        probs, y, _ = preds_all[split_name]
        suffix = "" if split_name == "test" else "_val"
        m = results[split_name]["per_task"][task]
        plot_confusion_matrix(np.asarray(m["confusion_matrix"]), ["0", "1"], task,
                              fig_dir / f"confusion_matrix{suffix}_{task}.png")
        plot_roc_curve(y, probs, task, fig_dir / f"roc_curve{suffix}_{task}.png", ["0", "1"])
        if split_name == "test":
            plot_pred_distribution(y, probs, task, fig_dir / f"pred_dist_{task}.png")

    torch.save(best_lora, ckpt_dir / "lora_adapter.pt")
    torch.save({"state_dict": best_head, "in_dim": model.head.in_dim, "out_dim": model.head.out_dim},
               ckpt_dir / "mlp_head.pt")
    with open(ckpt_dir / "lora_config.json", "w", encoding="utf-8") as f:
        json.dump({"lora": cfg["lora"], "views": views, "trainable": summary,
                   "best_epoch": best_epoch, "best_val_macro_f1": best_f1}, f, indent=2)
    save_config(cfg, run_dir / "config.yaml")
    dump_json({"task": task, "views": views, "trainable": summary, "best_epoch": best_epoch,
               "best_val_macro_f1": best_f1, "metrics": results,
               "frame": cfg["frame"], "prompt": cfg["prompt"],
               "git": git_info(ROOT), "environment": environment_snapshot()}, run_dir / "metrics.json")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    with open(run_dir / "train_log.txt", "w", encoding="utf-8") as f:
        f.write(f"best_epoch={best_epoch} best_val_macro_f1={best_f1}\n")
        f.write(pd.DataFrame(history).to_string(index=False))

    rows = []
    for split_name in ("train", "val", "test"):
        probs, y, ids = preds_all[split_name]
        for i in range(len(y)):
            rows.append({"sample_idx": int(i), "study_id": int(ids[i]), "split": split_name, "task": task,
                         "y_true": int(y[i]), "y_pred": int(probs[i].argmax()),
                         "prob_0": float(probs[i, 0]), "prob_1": float(probs[i, 1])})
    pd.DataFrame(rows).to_csv(run_dir / "predictions.csv", index=False)

    return {"group": group["name"], "views": views, "best_epoch": best_epoch,
            "best_val_f1": best_f1, "metrics": results, "trainable": summary}


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    device = resolve_device(args.device)
    logger = setup_logger("qwen.e2e_lora.train")
    if args.task:
        cfg["task"] = args.task
    task = cfg["task"]
    num_classes = int(cfg.get("num_classes", {}).get(task, 2))

    table = build_alignment(cfg["views"]["A4C"]["csv"], cfg["views"]["A2C"]["csv"], task)
    logger.info(f"task={task} samples={len(table)} split={table.split.value_counts().to_dict()}")
    processor = AutoProcessor.from_pretrained(cfg["model"]["path"])

    groups = cfg["groups"]
    if args.groups:
        wanted = {g.strip() for g in args.groups.split(",")}
        groups = [g for g in groups if g["name"] in wanted]
    summaries = []
    for group in groups:
        summaries.append(train_group(cfg, group, table, processor, task, num_classes, device, logger, args))

    comp_rows = []
    base = {s["group"]: s for s in summaries}
    for s in summaries:
        t = s["metrics"]["test"]["per_task"][task]
        row = {"group": s["group"], "views": "+".join(s["views"]), "best_epoch": s["best_epoch"],
               "test_acc": t["accuracy"], "test_macro_f1": t["macro_f1"], "test_macro_auc": t["macro_auc"],
               "test_sensitivity": t["sensitivity"], "test_specificity": t["specificity"],
               "trainable_params": s["trainable"]["trainable"], "trainable_ratio": s["trainable"]["ratio"]}
        for bname in ("A4C", "A2C"):
            if bname in base:
                bt = base[bname]["metrics"]["test"]["per_task"][task]
                row[f"d_acc_vs_{bname}"] = row["test_acc"] - bt["accuracy"]
                row[f"d_f1_vs_{bname}"] = row["test_macro_f1"] - bt["macro_f1"]
                row[f"d_auc_vs_{bname}"] = row["test_macro_auc"] - bt["macro_auc"]
        comp_rows.append(row)
    comp = pd.DataFrame(comp_rows)
    out_dir = Path(cfg["output"]["root"]) / "results/qwen_e2e_lora"
    comp.to_csv(out_dir / f"comparison_{task}.csv", index=False)
    with open(out_dir / f"comparison_{task}.md", "w", encoding="utf-8") as f:
        f.write(f"# {task}: single-view vs joint e2e LoRA\n\n")
        f.write(comp.to_string(index=False))
    append_experiment_log(ROOT, f"Qwen 端到端 LoRA 分类（{task}）", {
        "purpose": "A4C/A2C 单视角与 A4C+A2C joint 的端到端 Qwen2.5-VL LoRA 对比。",
        "command": " ".join(sys.argv),
        "params": f"seed={cfg.get('seed')}, frames={cfg['frame']['num_frames']}, size={cfg['frame']['image_size']}, "
                  f"lora_r={cfg['lora']['r']}, lr_lora={cfg['train']['lr_lora']}, lr_head={cfg['train']['lr_head']}",
        "input": f"{cfg['views']['A4C']['csv']}; {cfg['views']['A2C']['csv']}",
        "output": str(out_dir),
        "result": comp.to_string(index=False),
        "conclusion": "单/双视角对比见 comparison 表；所有组共享样本、划分、prompt 与 LoRA 配置。",
    })
    logger.info(f"comparison saved: {out_dir / f'comparison_{task}.csv'}")


if __name__ == "__main__":
    main()
