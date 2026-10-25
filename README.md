# ECHO-MIMIC-IV 分类任务

基于 `data/classification.csv` 的 3 个二分类任务，构建三个可独立运行的子项目：

| 子项目 | 目录 | 视觉骨干 | 训练参数 | 特征 |
|---|---|---|---|---|
| A. Qwen 冻结视觉塔 + MLP | `model/qwen/linear_probe/` | Qwen2.5-VL-3B `visual.*`，冻结 | 仅 MLP | 一次性缓存 |
| B. Qwen LoRA + MLP | `model/qwen/lora/` | Qwen2.5-VL-3B `visual.*` | LoRA(qkv/proj) + MLP | 不缓存，端到端；结束后保存 pooled 特征 |
| C. VideoMAE 冻结 + MLP | `model/videomae/` | VideoMAE-base，冻结 | 仅 MLP | 一次性缓存 |

> Qwen 两个子项目**只加载视觉塔**（`config.json -> vision_config` + safetensors 中 `visual.*` 权重，390 个张量，约 0.67B 参数），不构建/不加载语言模型。

## 数据

- `data/classification.csv`：6766 行，每行一个视频；`split` 固定为 train/val/test（5396/679/691）。
- 3 个二分类任务（正类均为 class 1）：

| 任务 | train 0/1 | val 0/1 | test 0/1 | 说明 |
|---|---:|---:|---:|---|
| `cls_rv_function` | 4043 / 1042 | 504 / 144 | 515 / 132 | 明显不平衡（1 类 ~20%） |
| `cls_la_size` | 2443 / 2094 | 298 / 276 | 299 / 283 | 基本平衡 |
| `cls_ra_size` | 2403 / 1979 | 287 / 261 | 300 / 244 | 基本平衡 |

- 标签缺失编码为 `-1`，**多任务 + 逐任务 mask**：缺失标签不参与 loss/指标，但该视频仍可用于其它任务。
- 类别不平衡：默认按 train split 计算 **balanced class weight**（`class_weight: balanced`），也可设 `none` 或改用 focal loss。
- 各流程使用同一 CSV 划分；每次运行把 `split_manifest.csv` 写入结果目录。
- 视频均为 224×224、约 64 帧；decord 均匀采样固定帧数（可配置）。

## 环境

```bash
source /home/mzhao/miniconda3/etc/profile.d/conda.sh && conda activate Qwen_back
```

已验证：Python 3.10 / torch 2.5.1+cu124 / transformers 4.57.0 / peft 0.20.0 / decord 0.6.0 / scikit-learn 1.7.2。

GPU 任务按项目规则需用户指定卡号：

```bash
CUDA_VISIBLE_DEVICES=<卡号> python train.py --config config.yaml --device cuda
```

## 快速开始

### A. Qwen 冻结视觉塔 + MLP

当前 `config.yaml` 使用 **16 帧**（8/16/32 帧对比结论：16 帧最优，32 帧退化）。

```bash
cd model/qwen/linear_probe
python extract_features.py --config config.yaml --device cuda   # 可选；train.py 会自动补
python train.py --config config.yaml --device cuda
```

### B. Qwen LoRA + MLP

```bash
cd model/qwen/lora
python train.py --config config.yaml --device cuda
```

### C. VideoMAE 冻结 + MLP

```bash
cd model/videomae
python extract_features.py --config config.yaml --device cuda   # 可选；train.py 会自动补
python train.py --config config.yaml --device cuda
```

常用参数：`--seed`、`--epochs`、`--device cpu|cuda|auto`、`--run-name`、`--max-videos N`（调试）、
`--overwrite-features`、`--reuse-features`；提取脚本还支持 `--num-shards/--shard-id/--merge` 多卡分片。

CPU 小样本流程验证（不产生正式结果）：

```bash
python train.py --config config.yaml --device cpu --max-videos 24 --epochs 1 --run-name smoke
```

## 输出结构

```text
out/
├── features/
│   ├── qwen_linear_probe/<tag>.npz + .json     # features[N,D], labels[N,K] (-1=缺失), row_id, split
│   ├── qwen_lora/<run>.npz + .json             # 每个 run 的 pooled 特征（train/val/test）
│   └── videomae/<tag>.npz + .json
├── results/<子项目>/
│   ├── <时间戳>_<run_name>/                     # 每次运行独立目录，不覆盖
│   │   ├── metrics.json                        # 逐任务 Acc/macro-F1/macro-AUC、P/R、Sens/Spec、混淆矩阵
│   │   ├── predictions.csv                     # 长表：split/task/y_true/y_pred/prob_0/prob_1
│   │   ├── config.yaml / split_manifest.csv / label_stats.json / history.csv
│   │   ├── mlp_head.pt
│   │   └── adapter/                            # 仅 LoRA 子项目
│   ├── latest -> <最近一次运行>
│   └── metrics.json / predictions.csv / config.yaml -> latest/*
└── figures/<子项目>/<时间戳>_<run_name>/
    ├── loss_curve.png/.pdf
    ├── val_macro_f1.png
    ├── confusion_matrix_<task>.png/.pdf         # test；val 为 confusion_matrix_val_<task>
    ├── roc_curve_<task>.png/.pdf                # test；val 为 roc_curve_val_<task>
    └── pred_dist_<task>.png/.pdf
```

## 评估规范

- 每个任务分别报告 **Accuracy / macro-F1 / macro-AUC（OVR）**，并附 macro Precision/Recall；
  二分类额外报告正类（class 1）的 **Sensitivity / Specificity**、每类 P/R/F1 与混淆矩阵。
- 早停基于 **验证集 macro-F1**（跨任务平均），同时记录 val loss。
- test 只在最终评估使用，不参与调参。
- 固定随机种子（Python/NumPy/PyTorch/CUDA）；运行目录保存完整 config、环境快照与 git commit。
- 所有运行自动追加 `/home/mzhao/ECHO-MIMIC-IV/EXPERIMENT_LOG.md`。

## 日志与运行控制

```bash
LOG=/tmp/echo_logs/logs_<任务名>.log
screen -dmS <任务名> bash -lc "python -u train.py --config config.yaml --device cuda > \"$LOG\" 2>&1"

tail -f /tmp/echo_logs/logs_<任务名>.log
grep -aE "epoch |\[val\]|\[test\]|early stopping|done|Traceback" /tmp/echo_logs/logs_<任务名>.log | tail -n 40
```

## 关键实现说明

- **Qwen 预处理**：逐帧 resize 到 28 的倍数 → [0,1] → Qwen mean/std → 按 `(t, gh, gw, mh, mw, c, tp, p, p)`
  打包 patch；已与 transformers `Qwen2VLVideoProcessor` 数值对齐（最大误差 2.4e-7）。
- **LoRA**：仅注入视觉注意力 `attn.qkv` / `attn.proj`（r/alpha/dropout/层数可配），冻结其余全部视觉参数；
  与 MLP 分类头联合训练，保存 `adapter/`（PEFT 格式）与 `mlp_head.pt`。
- **VideoMAE**：`VideoMAEForPreTraining` + `videomae.*` 权重加载（0 missing / 0 unexpected），输入 [0,1]，
  取 `last_hidden_state` 做 mean pooling（也支持 max / mean_std）。
- **权重路径**：task.txt 写的是 `/home/mzhao/模型权重/VideoMAE/VideoMAE_pth/pytorch_model.bin`，
  实际文件位于 `/home/mzhao/模型权重/VideoMAE/pytorch_model.bin`，config 已指向实际路径。
- **分类头**：LayerNorm → Linear → GELU → Dropout（2 层共享 trunk）→ 每个任务独立 Linear 分类头。
- 历史回归版本及其产物已归档到 `/tmp/echo_logs/archive_regression_out_2026-10-25/`，不在当前项目 `out/` 中。
