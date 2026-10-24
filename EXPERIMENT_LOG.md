
## [2026-10-24 23:55] 构建验证（CPU smoke tests）
- 目的：验证三个子项目的代码路径（数据读取→特征/前向→MLP/LoRA 训练→指标与图）。
- 执行命令：见 /tmp/echo_logs/logs_smoke_*.log（均使用 --max-videos 小样本、CPU）。
- 关键参数：小样本、1-2 epoch，仅验证流程，不代表正式结果。
- 输入：data/regression.csv 前若干行。
- 输出：产物已归档至 /tmp/echo_logs/smoke_artifacts/。
- 结果/指标：流程全部跑通；无正式指标。
- 结论与备注：正式实验需在用户指定 GPU 上重跑。

## [2026-10-24 23:56] VideoMAE 冻结特征 + MLP 回归（videomae_mlp）

- 目的：VideoMAE 冻结 backbone 提取特征，训练 MLP 多任务回归头，报告各任务 MAE/RMSE/R2。
- 执行命令：model/videomae/train.py --config model/videomae/config.yaml --device cuda
- 关键参数：seed=42, frames=16, size=224, pooling=mean, hidden=[512, 256], lr=0.001, bs=128, epochs=200, best_epoch=19
- 输入：/home/mzhao/ECHO-MIMIC-IV/data/regression.csv
- 输出：/home/mzhao/ECHO-MIMIC-IV/out/results/videomae/2026-10-24_23-55-23_videomae_mlp
- 结果/指标：
```
test macro MAE=10.9750, RMSE=13.9550, R2=0.1996
task                        n        MAE       RMSE       R2
reg_lvef                  459     8.4403    11.0191   0.3177
reg_la_4ch_length         605     0.6275     0.7901   0.2812
reg_ra_length             606     0.6380     0.7949   0.2503
reg_la_vol                504    18.8383    23.9162   0.2468
reg_biplane_lvedv         126    23.3489    28.4977  -0.0511
reg_biplane_lvesv         113    13.9567    18.7117   0.1527
macro                            10.9750    13.9550   0.1996
```
- 结论与备注：特征缓存 /home/mzhao/ECHO-MIMIC-IV/out/features/videomae/videomae_16f_224px_mean.npz；最佳 epoch=19；详细指标见 metrics.json。

## [2026-10-25 00:00] Qwen2.5-VL 冻结视觉塔 + MLP 回归（qwen_linear_probe）

- 目的：仅加载 Qwen2.5-VL 视觉塔并冻结，缓存 pooled 特征后训练 MLP 多任务回归头。
- 执行命令：model/qwen/linear_probe/train.py --config model/qwen/linear_probe/config.yaml --device cuda
- 关键参数：seed=42, frames=8, size=224, pooling=mean, hidden=[512, 256], lr=0.001, bs=128, epochs=200, best_epoch=96
- 输入：/home/mzhao/ECHO-MIMIC-IV/data/regression.csv
- 输出：/home/mzhao/ECHO-MIMIC-IV/out/results/qwen_linear_probe/2026-10-24_23-55-24_qwen_linear_probe
- 结果/指标：
```
test macro MAE=11.5941, RMSE=15.6709, R2=0.0073
task                        n        MAE       RMSE       R2
reg_lvef                  459     9.5713    12.7128   0.0919
reg_la_4ch_length         605     0.6699     0.8506   0.1669
reg_ra_length             606     0.6694     0.8494   0.1441
reg_la_vol                504    18.9352    25.2982   0.1572
reg_biplane_lvedv         126    24.5715    32.6239  -0.3775
reg_biplane_lvesv         113    15.1473    21.6907  -0.1386
macro                            11.5941    15.6709   0.0073
```
- 结论与备注：特征缓存 /home/mzhao/ECHO-MIMIC-IV/out/features/qwen_linear_probe/qwen2vl3b_8f_224px_mean_frozen.npz；LM 未加载；最佳 epoch=96。
