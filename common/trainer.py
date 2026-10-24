"""MLP regression training on frozen features (shared by the probe projects)."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .mlp import MLPHead
from .utils import EarlyStopping


def standardize_targets(y: np.ndarray, stats: dict, tasks: Sequence[str]) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).copy()
    for j, task in enumerate(tasks):
        y[:, j] = (y[:, j] - stats[task]["mean"]) / stats[task]["std"]
    return y.astype(np.float32)


def inverse_standardize(y: np.ndarray, stats: dict, tasks: Sequence[str]) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).copy()
    for j, task in enumerate(tasks):
        y[:, j] = y[:, j] * stats[task]["std"] + stats[task]["mean"]
    return y.astype(np.float32)


def masked_task_loss(
    pred: torch.Tensor, target: torch.Tensor, kind: str = "mse", huber_delta: float = 1.0
) -> torch.Tensor:
    """Average of per-task masked losses (tasks without labels in the batch are skipped)."""
    valid = torch.isfinite(target)
    losses = []
    for j in range(target.shape[1]):
        m = valid[:, j]
        if not m.any():
            continue
        p, t = pred[m, j], target[m, j]
        if kind == "huber":
            losses.append(F.huber_loss(p, t, delta=huber_delta, reduction="mean"))
        else:
            losses.append(F.mse_loss(p, t, reduction="mean"))
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


@torch.no_grad()
def predict_mlp(head: MLPHead, features: torch.Tensor, batch_size: int = 512, device="cpu") -> np.ndarray:
    head.eval()
    outs = []
    for i in range(0, features.shape[0], batch_size):
        outs.append(head(features[i : i + batch_size].to(device)).float().cpu())
    return torch.cat(outs, dim=0).numpy() if outs else np.zeros((0, head.out_dim), dtype=np.float32)


def fit_mlp(
    features_train: np.ndarray,
    labels_train: np.ndarray,
    features_val: np.ndarray | None,
    labels_val: np.ndarray | None,
    stats: dict,
    tasks: Sequence[str],
    cfg: dict,
    seed: int,
    device: torch.device,
    logger,
) -> dict:
    """Train the MLP head with masked loss + early stopping.

    ``labels_*`` are in the original units (NaNs allowed).  Loss and early
    stopping use z-scored targets; returned predictions are in original units.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr = torch.as_tensor(np.asarray(features_train), dtype=torch.float32)
    Ytr = torch.as_tensor(standardize_targets(labels_train, stats, tasks), dtype=torch.float32)
    Xva = torch.as_tensor(np.asarray(features_val), dtype=torch.float32) if features_val is not None else None
    Yva = (
        torch.as_tensor(standardize_targets(labels_val, stats, tasks), dtype=torch.float32)
        if labels_val is not None
        else None
    )

    head = MLPHead(
        in_dim=Xtr.shape[1],
        out_dim=len(tasks),
        hidden_dims=tuple(cfg.get("hidden_dims", [512, 256])),
        dropout=float(cfg.get("dropout", 0.2)),
        activation=str(cfg.get("activation", "gelu")),
        input_norm=bool(cfg.get("input_norm", True)),
    ).to(device)

    opt = torch.optim.AdamW(
        head.parameters(), lr=float(cfg.get("lr", 1e-3)), weight_decay=float(cfg.get("weight_decay", 1e-4))
    )
    epochs = int(cfg.get("epochs", 200))
    batch_size = int(cfg.get("batch_size", 128))
    kind = str(cfg.get("loss", "mse")).lower()
    delta = float(cfg.get("huber_delta", 1.0))
    grad_clip = float(cfg.get("grad_clip", 5.0))
    scheduler_name = str(cfg.get("lr_scheduler", "cosine")).lower()

    steps_per_epoch = max(1, int(np.ceil(len(Xtr) / batch_size)))
    if scheduler_name == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
    elif scheduler_name == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
    else:
        sched = None

    stopper = EarlyStopping(patience=int(cfg.get("patience", 25)), min_delta=float(cfg.get("min_delta", 1e-5)))
    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_state = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(len(Xtr))
        total, count = 0.0, 0
        for i in range(0, len(Xtr), batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = Xtr[idx].to(device, non_blocking=True), Ytr[idx].to(device, non_blocking=True)
            pred = head(xb)
            loss = masked_task_loss(pred, yb, kind=kind, huber_delta=delta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
            opt.step()
            if sched is not None and scheduler_name == "cosine":
                sched.step()
            total += float(loss.detach()) * len(idx)
            count += len(idx)
        train_loss = total / max(count, 1)

        val_loss = float("nan")
        if Xva is not None and Yva is not None and len(Xva):
            head.eval()
            with torch.no_grad():
                vp = predict_mlp(head, Xva, batch_size=512, device=device)
                vp_t = torch.as_tensor(vp, dtype=torch.float32)
                val_loss = float(masked_task_loss(vp_t, Yva, kind=kind, huber_delta=delta))
            if sched is not None and scheduler_name == "plateau":
                sched.step(val_loss)

        monitor = val_loss if np.isfinite(val_loss) else train_loss
        improved = stopper.step(monitor, epoch)
        if improved:
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(float(opt.param_groups[0]["lr"]))

        logger.info(
            f"epoch {epoch:3d}/{epochs} | train {train_loss:.5f} | val {val_loss:.5f} | "
            f"lr {opt.param_groups[0]['lr']:.2e} | best {stopper.best:.5f} @ {best_epoch}"
        )
        if stopper.should_stop:
            logger.info(f"early stopping at epoch {epoch} (patience={stopper.patience})")
            break

    if best_state is not None:
        head.load_state_dict(best_state)
    return {"head": head, "history": history, "best_epoch": best_epoch, "best_val_loss": float(stopper.best or float("nan"))}


def save_predictions(
    df: pd.DataFrame,
    preds: np.ndarray,
    tasks: Sequence[str],
    path: str | Path,
    split_label: str | None = None,
) -> pd.DataFrame:
    """Long-format predictions joined with metadata and ground truth."""
    rows = []
    for i in range(len(df)):
        rec = df.iloc[i]
        for j, task in enumerate(tasks):
            truth = rec[task]
            rows.append(
                {
                    "row_id": int(rec["row_id"]),
                    "subject_id": int(rec["subject_id"]),
                    "study_id": int(rec["study_id"]),
                    "measurement_id": None if pd.isna(rec.get("measurement_id")) else int(rec["measurement_id"]),
                    "video_path": rec["video_path"],
                    "split": split_label or rec["split"],
                    "task": task,
                    "y_true": None if pd.isna(truth) else float(truth),
                    "y_pred": float(preds[i, j]),
                }
            )
    out = pd.DataFrame(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out
