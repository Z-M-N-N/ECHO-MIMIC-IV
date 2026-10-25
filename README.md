# ECHO-MIMIC-IV — Qwen 多视角特征融合分类原型

本阶段按 `task.txt` 重构为 **Qwen2.5-VL-3B 视觉塔（冻结）+ 特征级融合 + MLP**：

- 视角：**A4C**（心尖四腔）与 **A2C**（心尖两腔）
- 任务：`cls_la_size`（左房大小分级，二分类；可在 config 切换 `cls_rv_function` / `cls_ra_size`）
- 对照：A4C 单视角、A2C 单视角、A4C+A2C 融合（`concat` / `mean` / `attention`）
- 视觉塔只加载 `visual.*`（390 个张量，约 0.67B），**不加载语言模型、不微调**

## 数据与对齐

- `data/A4C/classification.csv`：6766 studies，含标签 `cls_rv_function` / `cls_la_size` / `cls_ra_size`
- `data/A2C/classification.csv`：6566 studies，**没有分类标签**
- 标签按 `study_id` 从 A4C 传给 A2C；为保证严格可比：
  - 只保留 **A4C、A2C 都存在** 且 **当前任务标签非空** 的 study；
  - 三组实验使用完全相同的样本、划分与 seed；
  - A4C/A2C 的 `split` 已核对完全一致。
- 默认任务 `cls_la_size` 对齐后：**5424** samples（train 4319 / val 551 / test 554，类别约 53/47%）。

## 目录

```text
model/qwen/feature_fusion/
├── extract_features.py   # 按视角提取 Qwen 视觉特征并缓存
├── train.py              # 读取缓存特征，训练单/双视角 MLP 并输出对比
├── dataset.py            # CSV 对齐、缓存读取、Dataset/Collate
├── model.py              # single / concat / mean / attention 融合 + MLP
├── make_overall_figure.py # 跨任务总览图（读 comparison_*.csv）
└── config.yaml

out/features/qwen_feature_fusion/{A4C,A2C}/<tag>.npz
out/results/qwen_feature_fusion/<task>_<group>/{metrics.json,predictions.csv,config.yaml,...}
out/figures/qwen_feature_fusion/<task>/<group>/{loss_curve,confusion_matrix_<task>,roc_curve_<task>,pred_dist_<task>}
out/figures/qwen_feature_fusion/overall_comparison.{png,pdf,csv,md}   # 跨任务总览
out/figures/qwen_feature_fusion/overall_comparison_heatmap.{png,pdf}
out/results/qwen_feature_fusion/comparison_<task>.csv/.md
```

## 运行

```bash
source /home/mzhao/miniconda3/etc/profile.d/conda.sh && conda activate Qwen_back
cd /home/mzhao/ECHO-MIMIC-IV

# 1) 特征提取（A4C + A2C；只需一次）
CUDA_VISIBLE_DEVICES=<卡号> python -u model/qwen/feature_fusion/extract_features.py \
  --config model/qwen/feature_fusion/config.yaml --view all --device cuda

# 2) 训练三组/五组对照
CUDA_VISIBLE_DEVICES=<卡号> python -u model/qwen/feature_fusion/train.py \
  --config model/qwen/feature_fusion/config.yaml --device cuda
```

- `--view A4C|A2C|all`、`--overwrite`、`--max-videos N`（调试）
- `train.py --groups A4C,A2C,A4C+A2C` 可只跑指定组
- `task: cls_rv_function|cls_la_size|cls_ra_size` 可切换任务（同一套视图对齐逻辑）

## 特征与融合

- 每视角 16 帧 @224 → Qwen 视觉塔 → mean-pooled **2048-D** 特征，缓存为 float16。
- 训练前按 **train split 的均值/方差** 对每个视角做 z-score（统计量存 `feature_stats.json`）。
- 融合：
  - `concat`：4096-D 拼接；
  - `mean`：两视角均值；
  - `attention`：逐视角打分 + softmax 加权（可学习）。
- MLP：LayerNorm → 512 → 256 → 2，GELU + Dropout(0.2)。
- AdamW + cosine，class weight balanced，早停 val macro-F1（patience 25），固定 seed。

## 评估

- Accuracy、macro-F1、macro-AUC（OVR），二分类另报 Precision/Recall、Sensitivity/Specificity；
- 混淆矩阵、ROC、预测分布图；
- `comparison_<task>.csv` 汇总各组指标，并给出相对 A4C / A2C 的 ΔAcc、ΔF1、ΔAUC。

## 图表组织

- 每个实验组的图放在 `out/figures/qwen_feature_fusion/<task>/<group>/`；
- 跨任务总览由 `make_overall_figure.py` 生成：

```bash
python model/qwen/feature_fusion/make_overall_figure.py
```

输出：
- `overall_comparison.png/.pdf`：Acc / macro-F1 / macro-AUC 三面板分组柱状图；
- `overall_comparison_heatmap.png/.pdf`：task × group 的 F1/AUC 热力图；
- `overall_comparison.csv/.md`：汇总表。

## 说明

- 旧的 VideoMAE / Qwen LoRA / 多任务 linear_probe 代码与产物已归档到
  `/tmp/echo_logs/archive_mimic_multitask_2026-10-25/`，当前 `out/` 只保留本原型结构。
- `data/` 下所有视角数据与原始 CSV 全部保留。
