"""NaN-aware (missing-label aware) classification metrics.

Per task we report Accuracy, macro-F1, macro-AUC (OVR), macro Precision/Recall,
per-class precision/recall/F1, and for binary tasks the positive-class
(``class 1``) Sensitivity / Specificity.  A confusion matrix is stored as a
nested list for plotting.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_KEYS = ("n", "accuracy", "macro_f1", "macro_auc", "macro_precision", "macro_recall")


def _safe_auc(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    present = np.unique(y_true)
    try:
        if num_classes == 2:
            if len(present) < 2:
                return float("nan")
            return float(roc_auc_score(y_true, probs[:, 1]))
        if len(present) < num_classes:
            return float("nan")
        return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=list(range(num_classes))))
    except ValueError:
        return float("nan")


def _binary_sens_spec(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return float(sens), float(spec)


def _task_metrics(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> dict:
    valid = y_true >= 0
    n = int(valid.sum())
    if n == 0:
        return {
            "n": 0, "accuracy": float("nan"), "macro_f1": float("nan"), "macro_auc": float("nan"),
            "macro_precision": float("nan"), "macro_recall": float("nan"),
            "sensitivity": float("nan"), "specificity": float("nan"),
            "per_class": {}, "confusion_matrix": [[0] * num_classes for _ in range(num_classes)],
        }

    yt = y_true[valid].astype(int)
    pr = probs[valid].astype(np.float64)
    yp = pr.argmax(axis=1)

    acc = float(accuracy_score(yt, yp))
    f1_macro = float(f1_score(yt, yp, average="macro", labels=list(range(num_classes)), zero_division=0))
    prec_macro = float(precision_score(yt, yp, average="macro", labels=list(range(num_classes)), zero_division=0))
    rec_macro = float(recall_score(yt, yp, average="macro", labels=list(range(num_classes)), zero_division=0))
    auc = _safe_auc(yt, pr, num_classes)

    p_cls, r_cls, f_cls, s_cls = precision_recall_fscore_support(
        yt, yp, labels=list(range(num_classes)), zero_division=0
    )
    per_class = {
        str(c): {
            "precision": float(p_cls[c]), "recall": float(r_cls[c]), "f1": float(f_cls[c]),
            "support": int(s_cls[c]),
        }
        for c in range(num_classes)
    }
    cm = confusion_matrix(yt, yp, labels=list(range(num_classes)))
    sens = spec = float("nan")
    if num_classes == 2:
        sens, spec = _binary_sens_spec(yt, yp)

    return {
        "n": n,
        "accuracy": acc,
        "macro_f1": f1_macro,
        "macro_auc": auc,
        "macro_precision": prec_macro,
        "macro_recall": rec_macro,
        "sensitivity": sens,
        "specificity": spec,
        "per_class": per_class,
        "confusion_matrix": cm.astype(int).tolist(),
    }


def masked_classification_metrics(
    y_true: np.ndarray,
    probs: Mapping[str, np.ndarray],
    tasks: Sequence[str],
    num_classes: Mapping[str, int] | None = None,
) -> dict[str, dict]:
    """``y_true``: ``[N,K]`` int with -1 missing; ``probs``: task -> ``[N,C]``."""
    y_true = np.asarray(y_true)
    if y_true.ndim != 2 or y_true.shape[1] != len(tasks):
        raise ValueError(f"unexpected y_true shape {y_true.shape}")
    num_classes = num_classes or {t: int(probs[t].shape[1]) for t in tasks}

    per_task = {}
    for j, task in enumerate(tasks):
        p = np.asarray(probs[task])
        if p.shape[0] != y_true.shape[0]:
            raise ValueError(f"{task}: probs rows {p.shape[0]} != labels {y_true.shape[0]}")
        per_task[task] = _task_metrics(y_true[:, j], p, int(num_classes[task]))

    macro = {}
    for key in ("accuracy", "macro_f1", "macro_auc", "macro_precision", "macro_recall"):
        vals = [per_task[t][key] for t in tasks if np.isfinite(per_task[t][key])]
        macro[key] = float(np.mean(vals)) if vals else float("nan")
    macro["n_tasks"] = len(tasks)
    return {"per_task": per_task, "macro": macro}


def format_metrics_table(metrics: dict[str, dict], tasks: Sequence[str]) -> str:
    lines = [f"{'task':20s} {'n':>5s} {'Acc':>7s} {'F1m':>7s} {'AUCm':>7s} {'Pm':>7s} {'Rm':>7s} {'Sens':>7s} {'Spec':>7s}"]
    for t in tasks:
        m = metrics["per_task"][t]
        lines.append(
            f"{t:20s} {m['n']:5d} {m['accuracy']:7.4f} {m['macro_f1']:7.4f} {m['macro_auc']:7.4f} "
            f"{m['macro_precision']:7.4f} {m['macro_recall']:7.4f} {m['sensitivity']:7.4f} {m['specificity']:7.4f}"
        )
    a = metrics["macro"]
    lines.append(
        f"{'macro':20s} {'':>5s} {a['accuracy']:7.4f} {a['macro_f1']:7.4f} {a['macro_auc']:7.4f} "
        f"{a['macro_precision']:7.4f} {a['macro_recall']:7.4f}"
    )
    return "\n".join(lines)
