"""Training / inference helpers for frozen-feature classification."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .metrics import masked_classification_metrics
from .mlp import MLPClassifier
from .utils import EarlyStopping


def masked_ce_loss(
    logits: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    tasks: Sequence[str],
    class_weights: Mapping[str, torch.Tensor] | None = None,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    """Mean of per-task masked cross-entropy (tasks with no labels are skipped)."""
    losses = []
    for j, task in enumerate(tasks):
        y = labels[:, j]
        valid = y >= 0
        if not valid.any():
            continue
        z = logits[task][valid].float()  # fp32 loss even under bf16 autocast
        t = y[valid]
        w = class_weights.get(task) if class_weights else None
        if w is not None:
            w = w.to(dtype=z.dtype)
        ce = F.cross_entropy(z, t, weight=w, reduction="none")
        if focal_gamma and focal_gamma > 0:
            pt = torch.exp(-F.cross_entropy(z, t, reduction="none"))
            ce = (1.0 - pt) ** float(focal_gamma) * ce
        losses.append(ce.mean())
    if not losses:
        return next(iter(logits.values())).sum() * 0.0
    return torch.stack(losses).mean()


@torch.no_grad()
def predict_proba(
    model: MLPClassifier,
    features: torch.Tensor,
    tasks: Sequence[str],
    device="cpu",
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    model.eval()
    out = {t: [] for t in tasks}
    for i in range(0, features.shape[0], batch_size):
        xb = features[i : i + batch_size].to(device)
        logits = model(xb)
        for t in tasks:
            out[t].append(torch.softmax(logits[t].float(), dim=1).cpu().numpy())
    return {t: (np.concatenate(v, axis=0) if v else np.zeros((0, model.num_classes[t]), dtype=np.float32)) for t, v in out.items()}


def fit_classifier(
    features_train: np.ndarray,
    labels_train: np.ndarray,
    features_val: np.ndarray | None,
    labels_val: np.ndarray | None,
    label_stats: Mapping[str, dict],
    tasks: Sequence[str],
    cfg: dict,
    seed: int,
    device: torch.device,
    logger,
) -> dict:
    """Train a multi-task MLP with masked CE, early stopping on val macro-F1."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    tasks = list(tasks)
    num_classes = {t: int(label_stats[t]["num_classes"]) for t in tasks}

    Xtr = torch.as_tensor(np.asarray(features_train), dtype=torch.float32)
    Ytr = torch.as_tensor(np.asarray(labels_train), dtype=torch.int64)
    Xva = torch.as_tensor(np.asarray(features_val), dtype=torch.float32) if features_val is not None else None
    Yva = torch.as_tensor(np.asarray(labels_val), dtype=torch.int64) if labels_val is not None else None

    model = MLPClassifier(
        in_dim=Xtr.shape[1],
        num_classes=num_classes,
        hidden_dims=tuple(cfg.get("hidden_dims", [512, 256])),
        dropout=float(cfg.get("dropout", 0.2)),
        activation=str(cfg.get("activation", "gelu")),
        input_norm=bool(cfg.get("input_norm", True)),
    ).to(device)

    use_weights = str(cfg.get("class_weight", "balanced")).lower() == "balanced"
    class_weights = None
    if use_weights:
        class_weights = {
            t: torch.tensor(label_stats[t]["class_weights"], dtype=torch.float32, device=device) for t in tasks
        }
    focal_gamma = float(cfg.get("focal_gamma", 0.0))
    kind = str(cfg.get("loss", "ce")).lower()

    opt = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.get("lr", 1e-3)), weight_decay=float(cfg.get("weight_decay", 1e-4))
    )
    epochs = int(cfg.get("epochs", 200))
    batch_size = int(cfg.get("batch_size", 128))
    grad_clip = float(cfg.get("grad_clip", 5.0))
    scheduler_name = str(cfg.get("lr_scheduler", "cosine")).lower()
    steps_per_epoch = max(1, int(np.ceil(len(Xtr) / batch_size)))
    if scheduler_name == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
    elif scheduler_name == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=8)
    else:
        sched = None

    stopper = EarlyStopping(
        patience=int(cfg.get("patience", 25)), min_delta=float(cfg.get("min_delta", 1e-4)), mode="max"
    )
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_accuracy": [], "lr": []}
    best_state, best_epoch, best_f1 = None, 0, float("-inf")

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        total, count = 0.0, 0
        for i in range(0, len(Xtr), batch_size):
            idx = perm[i : i + batch_size]
            xb = Xtr[idx].to(device, non_blocking=True)
            yb = Ytr[idx].to(device, non_blocking=True)
            logits = model(xb)
            loss = masked_ce_loss(
                logits, yb, tasks, class_weights=class_weights,
                focal_gamma=focal_gamma if kind == "focal" else 0.0,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            if sched is not None and scheduler_name == "cosine":
                sched.step()
            total += float(loss.detach()) * len(idx)
            count += len(idx)
        train_loss = total / max(count, 1)

        val_loss, val_f1, val_acc = float("nan"), float("nan"), float("nan")
        if Xva is not None and Yva is not None and len(Xva):
            probs = predict_proba(model, Xva, tasks, device=device, batch_size=512)
            yva_np = Yva.numpy()
            with torch.no_grad():
                logits_va = model(Xva.to(device))
                val_loss = float(masked_ce_loss(logits_va, Yva.to(device), tasks, class_weights=class_weights))
            metrics = masked_classification_metrics(yva_np, probs, tasks, num_classes)
            val_f1 = float(metrics["macro"]["macro_f1"])
            val_acc = float(metrics["macro"]["accuracy"])
            if sched is not None and scheduler_name == "plateau":
                sched.step(val_f1)

        monitor = val_f1 if np.isfinite(val_f1) else (-val_loss if np.isfinite(val_loss) else -train_loss)
        improved = stopper.step(monitor, epoch)
        if improved:
            best_epoch, best_f1 = epoch, monitor
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_f1)
        history["val_accuracy"].append(val_acc)
        history["lr"].append(float(opt.param_groups[0]["lr"]))
        logger.info(
            f"epoch {epoch:3d}/{epochs} | train_loss {train_loss:.5f} | val_loss {val_loss:.5f} | "
            f"val_F1m {val_f1:.4f} | val_Acc {val_acc:.4f} | best {stopper.best:.4f} @ {best_epoch}"
        )
        if stopper.should_stop:
            logger.info(f"early stopping at epoch {epoch} (patience={stopper.patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "class_weights": {t: label_stats[t]["class_weights"] for t in tasks},
    }


def save_predictions(
    df: pd.DataFrame,
    probs: Mapping[str, np.ndarray],
    tasks: Sequence[str],
    path: str | Path,
    split_label: str | None = None,
) -> pd.DataFrame:
    """Long-format predictions: one row per (sample, task) with all class probs."""
    path = Path(path)
    rows = []
    for i in range(len(df)):
        rec = df.iloc[i]
        for task in tasks:
            p = probs[task][i]
            truth = rec[task]
            label = None if pd.isna(truth) else int(truth)
            row = {
                "row_id": int(rec["row_id"]),
                "subject_id": int(rec["subject_id"]),
                "study_id": int(rec["study_id"]),
                "measurement_id": None if pd.isna(rec.get("measurement_id")) else int(rec["measurement_id"]),
                "video_path": rec["video_path"],
                "split": split_label or rec["split"],
                "task": task,
                "y_true": label,
                "y_pred": int(np.argmax(p)),
            }
            for c, v in enumerate(p):
                row[f"prob_{c}"] = float(v)
            rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["y_true"] = out["y_true"].astype("Int64")
        out["y_pred"] = out["y_pred"].astype("Int64")
    else:
        out = pd.DataFrame(columns=["row_id", "subject_id", "study_id", "measurement_id",
                                    "video_path", "split", "task", "y_true", "y_pred"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out
