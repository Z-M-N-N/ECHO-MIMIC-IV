"""Bar chart: e2e LoRA vs frozen-feature fusion on cls_la_size."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
E2E = ROOT / "out/results/qwen_e2e_lora/comparison_cls_la_size.csv"
FROZEN = Path("/tmp/echo_logs/archive_mimic_feature_fusion_2026-10-25/results/"
              "qwen_feature_fusion/comparison_cls_la_size.csv")
OUT = ROOT / "out/figures/qwen_e2e_lora"

GROUPS = ["A4C", "A2C", "A4C+A2C"]
LABELS = {"A4C": "A4C", "A2C": "A2C", "A4C+A2C": "A4C+A2C"}
METRICS = [("test_acc", "Accuracy"), ("test_macro_f1", "macro-F1"), ("test_macro_auc", "macro-AUC")]


def load(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    return d[d["group"].isin(GROUPS)].set_index("group").reindex(GROUPS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", default=str(E2E))
    ap.add_argument("--frozen", default=str(FROZEN))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    e2e = load(Path(args.e2e))
    frozen = load(Path(args.frozen))

    # merged table
    rows = []
    for g in GROUPS:
        for name, d in (("e2e_lora", e2e), ("frozen_feature_fusion", frozen)):
            r = {"group": g, "method": name}
            for m, _ in METRICS:
                r[m] = float(d.loc[g, m])
            rows.append(r)
    merged = pd.DataFrame(rows)
    merged.to_csv(out_dir / "comparison_e2e_vs_frozen_cls_la_size.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    x = np.arange(len(GROUPS)); w = 0.36
    colors = {"e2e_lora": "#4C72B0", "frozen_feature_fusion": "#DD8452"}
    for ax, (metric, title) in zip(axes, METRICS):
        for j, (name, d) in enumerate((("e2e_lora", e2e), ("frozen_feature_fusion", frozen))):
            vals = d[metric].to_numpy(dtype=float)
            bars = ax.bar(x + (j - 0.5) * w, vals, width=w * 0.92, label=name, color=colors[name])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([LABELS[g] for g in GROUPS])
        ax.set_title(title); ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0.55, 0.80)
        ax.set_ylabel(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, ["e2e LoRA (16f@224)", "frozen features + MLP"], loc="lower center",
               ncol=2, frameon=False)
    fig.suptitle("cls_la_size: A4C / A2C / joint — e2e LoRA vs frozen feature fusion (test)", fontsize=13)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(out_dir / "comparison_e2e_vs_frozen_cls_la_size.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "comparison_e2e_vs_frozen_cls_la_size.pdf", bbox_inches="tight")
    plt.close(fig)
    print(out_dir / "comparison_e2e_vs_frozen_cls_la_size.png")
    print(out_dir / "comparison_e2e_vs_frozen_cls_la_size.csv")


if __name__ == "__main__":
    main()
