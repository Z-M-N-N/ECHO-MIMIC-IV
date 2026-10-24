"""Figure helpers: loss curves and prediction-vs-truth scatter plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
    has_val = len(history.get("val_loss", [])) == len(epochs)
    if has_val:
        ax.plot(epochs, history["val_loss"], label="val", lw=1.8)
    if best_epoch is not None and best_epoch >= 1:
        ax.axvline(best_epoch, color="grey", ls=":", lw=1.2, label=f"best epoch {best_epoch}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked MSE loss (z-score space)")
    ax.set_title("Training / validation loss")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, path)


def plot_pred_vs_true(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str,
    path: str | Path,
    metrics: dict | None = None,
) -> dict:
    """Scatter of predictions vs truth with y=x and least-squares fit lines."""
    path = Path(path)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = np.asarray(y_true)[mask].astype(float)
    yp = np.asarray(y_pred)[mask].astype(float)
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    if yt.size:
        ax.scatter(yt, yp, s=10, alpha=0.45, edgecolors="none", label=f"samples (n={yt.size})")
        lo = float(min(yt.min(), yp.min()))
        hi = float(max(yt.max(), yp.max()))
        pad = 0.03 * (hi - lo if hi > lo else 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.4, label="ideal y = x")
        if yt.size > 1 and np.std(yt) > 1e-9:
            slope, intercept = np.polyfit(yt, yp, 1)
            xs = np.linspace(lo - pad, hi + pad, 100)
            ax.plot(xs, slope * xs + intercept, color="crimson", lw=1.6,
                    label=f"fit: y = {slope:.3f}x {intercept:+.3f}")
        else:
            slope, intercept = float("nan"), float("nan")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
    else:
        slope = intercept = float("nan")
        ax.text(0.5, 0.5, "no labelled samples", transform=ax.transAxes, ha="center")

    ax.set_xlabel(f"{task} (true)")
    ax.set_ylabel(f"{task} (predicted)")
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    if metrics is not None:
        txt = (
            f"n = {metrics.get('n', 0)}\n"
            f"MAE = {metrics.get('mae', float('nan')):.4g}\n"
            f"RMSE = {metrics.get('rmse', float('nan')):.4g}\n"
            f"R$^2$ = {metrics.get('r2', float('nan')):.4g}\n"
            f"slope = {slope:.3f}\n"
            f"intercept = {intercept:.3f}"
        )
        ax.text(
            0.98, 0.02, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="0.7"),
        )
    _save(fig, path)
    return {"slope": float(slope), "intercept": float(intercept), "n": int(yt.size)}
