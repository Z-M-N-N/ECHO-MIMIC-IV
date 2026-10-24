"""NaN-aware per-task regression metrics (MAE / RMSE / R2)."""
from __future__ import annotations

from typing import Sequence

import numpy as np

METRIC_KEYS = ("n", "mae", "rmse", "r2")


def _task_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    n = int(err.size)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if n > 1 and ss_tot > 1e-12 else float("nan")
    return {"n": n, "mae": mae, "rmse": rmse, "r2": r2}


def masked_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, tasks: Sequence[str]
) -> dict[str, dict]:
    """Return ``{"per_task": {...}, "macro": {...}}`` ignoring NaNs per task."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.shape[1] != len(tasks):
        raise ValueError("number of columns does not match number of tasks")

    per_task = {}
    for j, task in enumerate(tasks):
        valid = np.isfinite(y_true[:, j]) & np.isfinite(y_pred[:, j])
        if valid.sum() == 0:
            per_task[task] = {"n": 0, "mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
            continue
        per_task[task] = _task_metrics(y_true[valid, j], y_pred[valid, j])

    macro = {}
    for key in ("mae", "rmse", "r2"):
        vals = [per_task[t][key] for t in tasks if np.isfinite(per_task[t][key])]
        macro[key] = float(np.mean(vals)) if vals else float("nan")
    macro["n_tasks"] = len(tasks)
    return {"per_task": per_task, "macro": macro}


def format_metrics_table(metrics: dict[str, dict], tasks: Sequence[str]) -> str:
    lines = [f"{'task':22s} {'n':>6s} {'MAE':>10s} {'RMSE':>10s} {'R2':>8s}"]
    for t in tasks:
        m = metrics["per_task"][t]
        lines.append(
            f"{t:22s} {m['n']:6d} {m['mae']:10.4f} {m['rmse']:10.4f} "
            f"{(m['r2'] if np.isfinite(m['r2']) else float('nan')):8.4f}"
        )
    m = metrics["macro"]
    lines.append(f"{'macro':22s} {'':>6s} {m['mae']:10.4f} {m['rmse']:10.4f} {m['r2']:8.4f}")
    return "\n".join(lines)
