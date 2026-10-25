
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

## [2026-10-25 01:27] Qwen2.5-VL 冻结视觉塔 + MLP 分类（qwen_linear_probe_16f）

- 目的：仅加载 Qwen2.5-VL 视觉塔并冻结，缓存 pooled 特征后训练 MLP 多任务分类头；报告 Accuracy/macro-F1/macro-AUC 与混淆矩阵。
- 执行命令：model/qwen/linear_probe/train.py --config model/qwen/linear_probe/config_16f.yaml --device cuda
- 关键参数：seed=42, frames=16, size=224, pooling=mean, hidden=[512, 256], lr=0.001, bs=128, epochs=200, class_weight=balanced, best_epoch=59
- 输入：/home/mzhao/ECHO-MIMIC-IV/data/classification.csv
- 输出：/home/mzhao/ECHO-MIMIC-IV/out/results/qwen_linear_probe/2026-10-25_01-20-31_qwen_linear_probe_16f
- 结果/指标：
```
test macro Acc=0.6684, F1=0.6292, AUC=0.6955
task                     n     Acc     F1m    AUCm      Pm      Rm    Sens    Spec
cls_rv_function        647  0.7419  0.6334  0.7078  0.6261  0.6463  0.4848  0.8078
cls_la_size            582  0.6512  0.6485  0.7135  0.6525  0.6493  0.5795  0.7191
cls_ra_size            544  0.6121  0.6058  0.6653  0.6068  0.6055  0.5410  0.6700
macro                       0.6684  0.6292  0.6955  0.6285  0.6337
```
- 结论与备注：特征缓存 /home/mzhao/ECHO-MIMIC-IV/out/features/qwen_linear_probe/qwen2vl3b_16f_224px_mean_frozen.npz；最佳 epoch=59；详细指标见 metrics.json。

## [2026-10-25 04:30] Qwen2.5-VL 视觉塔 LoRA + MLP 分类（qwen_lora_cls_16f）

- 目的：仅加载 Qwen2.5-VL 视觉塔，在注意力 qkv/proj 注入 LoRA，与 MLP 分类头联合训练。
- 执行命令：model/qwen/lora/train.py --config model/qwen/lora/config.yaml --device cuda
- 关键参数：seed=42, frames=16, size=224, lora_r=8, alpha=16, lr_lora=0.0001, lr_head=0.0003, bs=4x2, epochs=4, best_epoch=3, trainable=3152134
- 输入：/home/mzhao/ECHO-MIMIC-IV/data/classification.csv
- 输出：/home/mzhao/ECHO-MIMIC-IV/out/results/qwen_lora/2026-10-25_03-21-31_qwen_lora_cls_16f
- 结果/指标：
```
test macro Acc=0.6826, F1=0.6473, AUC=0.7250
task                     n     Acc     F1m    AUCm      Pm      Rm    Sens    Spec
cls_rv_function        647  0.7372  0.6345  0.7373  0.6263  0.6518  0.5076  0.7961
cls_la_size            582  0.6615  0.6593  0.7250  0.6626  0.6598  0.5972  0.7224
cls_ra_size            544  0.6489  0.6480  0.7126  0.6491  0.6507  0.6680  0.6333
macro                       0.6826  0.6473  0.7250  0.6460  0.6541
```
- 结论与备注：adapter -> /home/mzhao/ECHO-MIMIC-IV/out/results/qwen_lora/2026-10-25_03-21-31_qwen_lora_cls_16f/adapter；特征 -> /home/mzhao/ECHO-MIMIC-IV/out/features/qwen_lora/2026-10-25_03-21-31_qwen_lora_cls_16f.npz；最佳 epoch=3。

## [2026-10-25 17:14] VideoMAE 冻结特征 + MLP 分类（videomae_apical_doppler）

- 目的：VideoMAE 冻结 backbone 提取特征，训练 MLP 多任务分类头；报告 Accuracy/macro-F1/macro-AUC 与混淆矩阵。
- 执行命令：model/videomae/train.py --config model/videomae/config_apical_doppler.yaml --device cuda
- 关键参数：seed=42, frames=16, size=224, pooling=mean, hidden=[512, 256], lr=0.001, bs=128, epochs=200, class_weight=balanced, best_epoch=47
- 输入：/home/mzhao/ECHO-MIMIC-IV/data/Apical_Doppler/classification.csv
- 输出：/home/mzhao/ECHO-MIMIC-IV/out/results/videomae/2026-10-25_17-11-42_videomae_apical_doppler
- 结果/指标：
```
test macro Acc=0.7682, F1=0.7201, AUC=0.8189
task                     n     Acc     F1m    AUCm      Pm      Rm    Sens    Spec
cls_mitral_regurg      673  0.7801  0.7368  0.8287  0.7238  0.7715  0.7545  0.7885
cls_tricuspid_regurg   665  0.7564  0.7034  0.8091  0.6919  0.7424  0.7161  0.7686
macro                       0.7682  0.7201  0.8189  0.7078  0.7569
```
- 结论与备注：特征缓存 /home/mzhao/ECHO-MIMIC-IV/out/features/videomae/apical_doppler_16f_224px_mean.npz；最佳 epoch=47；详细指标见 metrics.json。

## [2026-10-25 17:26] Qwen2.5-VL 冻结视觉塔 + MLP 分类（qwen_linear_probe_apical_doppler）

- 目的：仅加载 Qwen2.5-VL 视觉塔并冻结，缓存 pooled 特征后训练 MLP 多任务分类头；报告 Accuracy/macro-F1/macro-AUC 与混淆矩阵。
- 执行命令：model/qwen/linear_probe/train.py --config model/qwen/linear_probe/config_apical_doppler.yaml --device cuda
- 关键参数：seed=42, frames=16, size=224, pooling=mean, hidden=[512, 256], lr=0.001, bs=128, epochs=200, class_weight=balanced, best_epoch=52
- 输入：/home/mzhao/ECHO-MIMIC-IV/data/Apical_Doppler/classification.csv
- 输出：/home/mzhao/ECHO-MIMIC-IV/out/results/qwen_linear_probe/2026-10-25_17-11-42_qwen_linear_probe_apical_doppler
- 结果/指标：
```
test macro Acc=0.7504, F1=0.6730, AUC=0.7572
task                     n     Acc     F1m    AUCm      Pm      Rm    Sens    Spec
cls_mitral_regurg      673  0.7489  0.6642  0.7435  0.6639  0.6645  0.4970  0.8320
cls_tricuspid_regurg   665  0.7519  0.6818  0.7709  0.6721  0.7013  0.6065  0.7961
macro                       0.7504  0.6730  0.7572  0.6680  0.6829
```
- 结论与备注：特征缓存 /home/mzhao/ECHO-MIMIC-IV/out/features/qwen_linear_probe/apical_doppler_qwen16f_224px_mean.npz；最佳 epoch=52；详细指标见 metrics.json。

## [2026-10-25 17:36] VideoMAE 冻结特征 + MLP 分类（videomae_dpl）

- 目的：VideoMAE 冻结 backbone 提取特征，训练 MLP 多任务分类头；报告 Accuracy/macro-F1/macro-AUC 与混淆矩阵。
- 执行命令：model/videomae/train.py --config model/videomae/config_dpl.yaml --device cuda
- 关键参数：seed=42, frames=16, size=224, pooling=mean, hidden=[512, 256], lr=0.001, bs=128, epochs=200, class_weight=balanced, best_epoch=26
- 输入：/home/mzhao/ECHO-MIMIC-IV/data/Doppler_Parasternal_Long/classification.csv
- 输出：/home/mzhao/ECHO-MIMIC-IV/out/results/videomae/2026-10-25_17-35-11_videomae_dpl
- 结果/指标：
```
test macro Acc=0.8000, F1=0.5289, AUC=0.5838
task                     n     Acc     F1m    AUCm      Pm      Rm    Sens    Spec
cls_aortic_regurg      290  0.8000  0.5289  0.5838  0.5408  0.5273  0.1395  0.9150
macro                       0.8000  0.5289  0.5838  0.5408  0.5273
```
- 结论与备注：特征缓存 /home/mzhao/ECHO-MIMIC-IV/out/features/videomae/dpl_16f_224px_mean.npz；最佳 epoch=26；详细指标见 metrics.json。

## [2026-10-25 17:41] Qwen2.5-VL 冻结视觉塔 + MLP 分类（qwen_linear_probe_dpl）

- 目的：仅加载 Qwen2.5-VL 视觉塔并冻结，缓存 pooled 特征后训练 MLP 多任务分类头；报告 Accuracy/macro-F1/macro-AUC 与混淆矩阵。
- 执行命令：model/qwen/linear_probe/train.py --config model/qwen/linear_probe/config_dpl.yaml --device cuda
- 关键参数：seed=42, frames=16, size=224, pooling=mean, hidden=[512, 256], lr=0.001, bs=128, epochs=200, class_weight=balanced, best_epoch=35
- 输入：/home/mzhao/ECHO-MIMIC-IV/data/Doppler_Parasternal_Long/classification.csv
- 输出：/home/mzhao/ECHO-MIMIC-IV/out/results/qwen_linear_probe/2026-10-25_17-35-12_qwen_linear_probe_dpl
- 结果/指标：
```
test macro Acc=0.7828, F1=0.5480, AUC=0.6690
task                     n     Acc     F1m    AUCm      Pm      Rm    Sens    Spec
cls_aortic_regurg      290  0.7828  0.5480  0.6690  0.5510  0.5459  0.2093  0.8826
macro                       0.7828  0.5480  0.6690  0.5510  0.5459
```
- 结论与备注：特征缓存 /home/mzhao/ECHO-MIMIC-IV/out/features/qwen_linear_probe/dpl_qwen16f_224px_mean.npz；最佳 epoch=35；详细指标见 metrics.json。
