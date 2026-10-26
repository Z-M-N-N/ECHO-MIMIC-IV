# CLEANUP — qwen_e2e_lora

日期：2026-10-25

| 对象 | 处理 | 原因 |
|---|---|---|
| `model/qwen/feature_fusion/` | 归档到 `archive/feature_fusion/` | 旧“冻结视觉塔 + 特征缓存/融合”方案，不参与当前端到端 LoRA 任务 |
| `out/features/qwen_feature_fusion/` | 归档到 `/tmp/echo_logs/archive_mimic_feature_fusion_2026-10-25/features/` | 旧特征缓存，当前任务不使用 |
| `out/results/qwen_feature_fusion/` | 归档到 `/tmp/echo_logs/archive_mimic_feature_fusion_2026-10-25/results/` | 旧实验结果，避免与新任务混淆 |
| `out/figures/qwen_feature_fusion/` | 归档到 `/tmp/echo_logs/archive_mimic_feature_fusion_2026-10-25/figures/` | 旧图表，避免与新任务混淆 |

保留：`data/` 全部原始标注与路径；模型权重目录未修改。
当前 `out/` 仅保留 `qwen_e2e_lora/` 的新目录结构。

## 训练过程中的中间尝试

| 对象 | 处理 | 原因 |
|---|---|---|
| `/tmp/echo_logs/archive_e2e_pool_last/` | 归档 | 表征使用最后 token / 全 token mean 的失败尝试（val F1≈0.34） |
| `/tmp/echo_logs/archive_e2e_debug/`、`archive_e2e_debug2/` | 归档 | 诊断/中断的 A4C 试跑，未用于最终结果 |

最终采用：LLM 第 9 层、`visual_mean` pooling；结果见 `out/results/qwen_e2e_lora/`。
