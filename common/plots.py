"""Figure helpers: loss/validation curves, confusion matrix, ROC, score distribution."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

DPI = 160


def _save(fig, path: Path, also_pdf: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    if also_pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(history: dict, path: str | Path, best_epoch: int | None = None) -> None:
    path = Path(path)
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(epochs, history["train_loss"], label="train", lw=1.8)
    if len(history.get("val_loss", [])) == len(epochs):
        ax.plot(epochs, history["val_loss"], label="val", lw=1.8)
    if best_epoch is not None and best_epoch >= 1:
        ax.axvline(best_epoch, color="grey", ls=":", lw=1.2, label=f"best epoch {best_epoch}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked cross-entropy")
    ax.set_title("Training / validation loss")
    ax.grid(alpha=0.3)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    _save(fig, path)


def plot_val_metric(history: dict, key: str, ylabel: str, title: str, path: str | Path) -> None:
    values = history.get(key, [])
    if not values:
        return
    path = Path(path)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(np.arange(1, len(values) + 1), values, lw=1.8, color="tab:green")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    _save(fig, path)


def plot_confusion_matrix(
    cm: np.ndarray, class_names: list[str], task: str, path: str | Path
) -> None:
    path = Path(path)
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix: {task}")
    total = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = 100.0 * cm[i, j] / total if total else 0.0
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, path)


def plot_roc_curve(
    y_true: np.ndarray, probs: np.ndarray, task: str, path: str | Path, class_names: list[str] | None = None
) -> dict:
    """Binary: single ROC for class 1.  Multiclass: one-vs-rest ROC per class."""
    path = Path(path)
    valid = y_true >= 0
    yt = y_true[valid].astype(int)
    pr = probs[valid]
    num_classes = pr.shape[1]
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    out = {}
    if yt.size and len(np.unique(yt)) >= 2:
        if num_classes == 2:
            fpr, tpr, _ = roc_curve(yt, pr[:, 1])
            a = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=1.9, label=f"class 1 (AUC={a:.3f})")
            out["class_1"] = float(a)
        else:
            for c in range(num_classes):
                yc = (yt == c).astype(int)
                if len(np.unique(yc)) < 2:
                    continue
                fpr, tpr, _ = roc_curve(yc, pr[:, c])
                a = auc(fpr, tpr)
                name = class_names[c] if class_names else str(c)
                ax.plot(fpr, tpr, lw=1.7, label=f"{name} (AUC={a:.3f})")
                out[name] = float(a)
        ax.plot([0, 1], [0, 1], "k--", lw=1.1)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="lower right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "single class / no data", transform=ax.transAxes, ha="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC: {task}")
    ax.grid(alpha=0.3)
    _save(fig, path)
    return out


def plot_pred_distribution(y_true: np.ndarray, probs: np.ndarray, task: str, path: str | Path) -> None:
    """Binary: histogram of P(class 1) split by true class.  Multiclass: max-prob histogram."""
    path = Path(path)
    valid = y_true >= 0
    yt = y_true[valid].astype(int)
    pr = probs[valid]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    if pr.shape[1] == 2:
        bins = np.linspace(0, 1, 21)
        for c, color in ((0, "tab:blue"), (1, "tab:orange")):
            vals = pr[yt == c, 1]
            if vals.size:
                ax.hist(vals, bins=bins, alpha=0.55, color=color, label=f"true {c} (n={vals.size})")
        ax.axvline(0.5, color="grey", ls=":", lw=1.2)
        ax.set_xlabel("predicted P(class 1)")
    else:
        conf = pr.max(axis=1)
        ax.hist(conf, bins=np.linspace(0, 1, 21), color="tab:purple", alpha=0.75)
        ax.set_xlabel("max predicted probability")
    ax.set_ylabel("count")
    ax.set_title(f"Prediction distribution: {task}")
    ax.grid(alpha=0.3)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
    _save(fig, path)
