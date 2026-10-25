"""Build the cross-task overall comparison figure from comparison_*.csv."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "out/results/qwen_feature_fusion"
FIG = ROOT / "out/figures/qwen_feature_fusion"

TASKS = ["cls_la_size", "cls_rv_function", "cls_ra_size"]
TASK_LABEL = {"cls_la_size": "LA size", "cls_rv_function": "RV function", "cls_ra_size": "RA size"}
GROUP_ORDER = ["A4C", "A2C", "A4C+A2C", "A4C+A2C+Apical_Doppler",
               "A4C+A2C+Apical_Doppler_mean", "A4C+A2C+Apical_Doppler_attention"]
GROUP_LABEL = {
    "A4C": "A4C",
    "A2C": "A2C",
    "A4C+A2C": "A4C+A2C",
    "A4C+A2C+Apical_Doppler": "3V concat",
    "A4C+A2C+Apical_Doppler_mean": "3V mean",
    "A4C+A2C+Apical_Doppler_attention": "3V attention",
}
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def load() -> pd.DataFrame:
    frames = []
    for task in TASKS:
        p3 = RES / f"comparison_{task}_3view.csv"
        p2 = RES / f"comparison_{task}.csv"
        path = p3 if p3.exists() else p2
        if not path.exists():
            continue
        d = pd.read_csv(path)
        d["task"] = task
        d["task_label"] = TASK_LABEL.get(task, task)
        frames.append(d)
    if not frames:
        raise FileNotFoundError("no comparison_*.csv found")
    d = pd.concat(frames, ignore_index=True)
    d["group_label"] = d["group"].map(GROUP_LABEL).fillna(d["group"])
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(FIG))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load()
    df.to_csv(out_dir / "overall_comparison.csv", index=False)
    with open(out_dir / "overall_comparison.md", "w", encoding="utf-8") as f:
        f.write("# Overall comparison (test)\n\n")
        f.write(df[["task", "group", "test_acc", "test_macro_f1", "test_macro_auc",
                    "test_sensitivity", "test_specificity"]].to_string(index=False))
        f.write("\n")

    groups = [g for g in GROUP_ORDER if g in set(df["group"])]
    tasks = [t for t in TASKS if t in set(df["task"])]
    metrics = [("test_acc", "Accuracy"), ("test_macro_f1", "macro-F1"), ("test_macro_auc", "macro-AUC")]

    # ---- grouped bar chart ----
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), sharey=False)
    x = np.arange(len(tasks))
    width = 0.13
    for ax, (metric, title) in zip(axes, metrics):
        vals = df.pivot_table(index="task", columns="group", values=metric, aggfunc="first").reindex(tasks)
        for j, g in enumerate(groups):
            y = vals[g].to_numpy() if g in vals else np.full(len(tasks), np.nan)
            ax.bar(x + (j - (len(groups) - 1) / 2) * width, y, width=width * 0.92,
                   label=GROUP_LABEL[g], color=COLORS[j % len(COLORS)])
        ax.set_xticks(x)
        ax.set_xticklabels([TASK_LABEL[t] for t in tasks], fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        lo, hi = np.nanmin(vals.to_numpy()), np.nanmax(vals.to_numpy())
        pad = max(0.015, (hi - lo) * 0.18)
        ax.set_ylim(max(0, lo - pad), min(1, hi + pad))
        ax.set_ylabel(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(groups), frameon=False, fontsize=9)
    fig.suptitle("Qwen vision features: single-view vs multi-view feature fusion (test)", fontsize=13)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(out_dir / "overall_comparison.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "overall_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- heatmaps (F1 / AUC) ----
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 3.8))
    for ax, (metric, title) in zip(axes, [("test_macro_f1", "macro-F1"), ("test_macro_auc", "macro-AUC")]):
        mat = df.pivot_table(index="task", columns="group", values=metric, aggfunc="first").reindex(tasks)[groups]
        im = ax.imshow(mat.to_numpy(), cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([GROUP_LABEL[g] for g in groups], rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels([TASK_LABEL[t] for t in tasks], fontsize=10)
        vmin, vmax = np.nanmin(mat.to_numpy()), np.nanmax(mat.to_numpy())
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.iloc[i, j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if (v - vmin) > 0.6 * (vmax - vmin) else "black")
        ax.set_title(f"Test {title}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(out_dir / "overall_comparison_heatmap.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "overall_comparison_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_dir/'overall_comparison.png'}")
    print(f"saved: {out_dir/'overall_comparison_heatmap.png'}")
    print(f"saved: {out_dir/'overall_comparison.csv'}")


if __name__ == "__main__":
    main()
