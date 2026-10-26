# ECHO-MIMIC-IV — Qwen2.5-VL 端到端 LoRA 分类原型

当前阶段按 `task.txt` 使用 **Qwen2.5-VL-3B 端到端前向**：

- 输入：超声视频采样帧（16 帧 @224×224）+ 任务 prompt + 视角标识
- 表征：LLM 第 9 层、视觉 token 的 mean-pooled hidden state（2048 维）
- 训练：LLM LoRA（q/k/v/o + gate/up/down，r=8/alpha=16）+ 全参数 MLP 分类头
- 视觉塔冻结；任务：`cls_la_size`
- 对照：A4C 单视角 / A2C 单视角 / A4C+A2C joint（两段视频一次前向）

## 数据

- `data/A4C/classification.csv`、`data/A2C/classification.csv`
- A2C 无分类标签，按 `study_id` 从 A4C 关联；严格保留双视角齐全且标签非空的 study
- 对齐后 5424 条（train 4319 / val 551 / test 554）

## 结构

```text
model/qwen/e2e_lora/
├── dataset.py       # CSV 对齐、视频解码、单/双视角样本
├── model.py         # Qwen2.5-VL + LoRA + 表征提取 + MLP
├── lora_utils.py    # LoRA 注入/保存/可训练参数统计
├── train.py         # 三组端到端训练、早停、评估、作图
└── config.yaml

out/results/qwen_e2e_lora/<task>_<group>/{metrics.json,predictions.csv,config.yaml,train_log.txt}
out/figures/qwen_e2e_lora/<task>_<group>/{loss_curve,confusion_matrix,roc_curve,pred_dist}
out/checkpoints/qwen_e2e_lora/<task>_<group>/{lora_adapter.pt,mlp_head.pt,lora_config.json}
```

## 运行

```bash
source /home/mzhao/miniconda3/etc/profile.d/conda.sh && conda activate Qwen_back
cd /home/mzhao/ECHO-MIMIC-IV
CUDA_VISIBLE_DEVICES=3 python -u model/qwen/e2e_lora/train.py \
  --config model/qwen/e2e_lora/config.yaml --device cuda
```

可选：`--task cls_rv_function`、`--groups A4C,A2C`、`--max-train-samples N`（调试）。

## 关键配置

- 帧：16 @ 224×224；prompt 模板与锚点文本在 `config.yaml`
- 表征：`representation.layer=9`（-1/最后层在诊断中几乎不可分）；`pooling=visual_mean`
- LoRA：r=8, alpha=16, dropout=0.05, 252 个模块；可训练 16.15M / 3.77B = 0.43%
- 训练：bs2、grad_accum 8→4、lr_lora 5e-5、lr_head 1e-3、2 epochs、早停 val macro-F1
- 未使用梯度检查点（实测约 1.6x 加速，显存约 16GB）

## 当前结果（cls_la_size, test）

| 组 | Acc | macro-F1 | macro-AUC |
|---|---:|---:|---:|
| A4C | 0.6155 | 0.6108 | 0.6334 |
| A2C | **0.6372** | **0.6366** | **0.6845** |
| A4C+A2C joint | 0.6119 | 0.6100 | 0.6764 |

> 端到端 LoRA 在当前 2 epoch 设置下低于上一阶段冻结特征融合（macro-F1 0.65–0.69），
> 训练 loss 仍在下降，属于欠训练；详见 `EXPERIMENT_LOG.md` 与 `CLEANUP.md`。
