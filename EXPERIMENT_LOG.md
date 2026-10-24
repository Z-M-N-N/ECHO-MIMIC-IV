
## [2026-10-25 00:45] 项目切换为分类任务 + 构建验证
- 目的：按新版 task.txt 将三个子项目从回归改为 classification.csv 的 3 个二分类任务。
- 执行命令：VideoMAE/Qwen-LP/Qwen-LoRA 的 `train.py --device cpu --max-videos 8~24 --epochs 1~2`（小样本流程验证）。
- 关键参数：balanced class weight、masked CE、早停 val macro-F1；LoRA 仅 qkv/proj。
- 输入：/home/mzhao/ECHO-MIMIC-IV/data/classification.csv。
- 输出：验证产物在 /tmp/echo_logs/smoke_cls_out/；旧回归产物已归档至 /tmp/echo_logs/archive_regression_out_2026-10-25/。
- 结果/指标：三个流程均跑通，metrics.json / predictions.csv / 混淆矩阵 / ROC / 预测分布图均正常生成；无正式指标。
- 结论与备注：正式训练需在用户指定 GPU 上运行。

## [2026-10-25 00:50] VideoMAE 冻结特征 + MLP 分类（videomae_cls）

- 目的：VideoMAE 冻结 backbone 提取特征，训练 MLP 多任务分类头；报告 Accuracy/macro-F1/macro-AUC 与混淆矩阵。
- 执行命令：model/videomae/train.py --config model/videomae/config.yaml --device cuda
- 关键参数：seed=42, frames=16, size=224, pooling=mean, hidden=[512, 256], lr=0.001, bs=128, epochs=200, class_weight=balanced, best_epoch=40
- 输入：/home/mzhao/ECHO-MIMIC-IV/data/classification.csv
- 输出：/home/mzhao/ECHO-MIMIC-IV/out/results/videomae/2026-10-25_00-48-40_videomae_cls
- 结果/指标：
```
test macro Acc=0.6774, F1=0.6522, AUC=0.7469
task                     n     Acc     F1m    AUCm      Pm      Rm    Sens    Spec
cls_rv_function        647  0.7342  0.6644  0.7941  0.6554  0.7147  0.6818  0.7476
cls_la_size            582  0.6546  0.6531  0.7175  0.6549  0.6533  0.6042  0.7023
cls_ra_size            544  0.6434  0.6390  0.7289  0.6393  0.6388  0.5943  0.6833
macro                       0.6774  0.6522  0.7469  0.6499  0.6689
```
- 结论与备注：特征缓存 /home/mzhao/ECHO-MIMIC-IV/out/features/videomae/videomae_16f_224px_mean.npz；最佳 epoch=40；详细指标见 metrics.json。

## [2026-10-25 00:53] Qwen2.5-VL 冻结视觉塔 + MLP 分类（qwen_linear_probe_cls）

- 目的：仅加载 Qwen2.5-VL 视觉塔并冻结，缓存 pooled 特征后训练 MLP 多任务分类头；报告 Accuracy/macro-F1/macro-AUC 与混淆矩阵。
- 执行命令：model/qwen/linear_probe/train.py --config model/qwen/linear_probe/config.yaml --device cuda
- 关键参数：seed=42, frames=8, size=224, pooling=mean, hidden=[512, 256], lr=0.001, bs=128, epochs=200, class_weight=balanced, best_epoch=50
- 输入：/home/mzhao/ECHO-MIMIC-IV/data/classification.csv
- 输出：/home/mzhao/ECHO-MIMIC-IV/out/results/qwen_linear_probe/2026-10-25_00-48-41_qwen_linear_probe_cls
- 结果/指标：
```
test macro Acc=0.6296, F1=0.6010, AUC=0.6689
task                     n     Acc     F1m    AUCm      Pm      Rm    Sens    Spec
cls_rv_function        647  0.6646  0.5788  0.6696  0.5804  0.6118  0.5227  0.7010
cls_la_size            582  0.6048  0.6047  0.6588  0.6059  0.6056  0.6360  0.5753
cls_ra_size            544  0.6195  0.6195  0.6785  0.6270  0.6267  0.6967  0.5567
macro                       0.6296  0.6010  0.6689  0.6044  0.6147
```
- 结论与备注：特征缓存 /home/mzhao/ECHO-MIMIC-IV/out/features/qwen_linear_probe/qwen2vl3b_8f_224px_mean_frozen.npz；最佳 epoch=50；详细指标见 metrics.json。
