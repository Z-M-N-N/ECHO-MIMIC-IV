## [2026-10-25 18:16] 重构为 Qwen A4C/A2C 特征融合原型

- 目的：按新版 task.txt 搭建冻结 Qwen 视觉塔 + A4C/A2C 特征级融合 + MLP 的单任务分类原型。
- 执行命令：`extract_features.py --view all` 与 `train.py` 的 CPU 小样本/合成特征流程验证。
- 关键参数：task=cls_la_size；A4C/A2C 按 study_id 对齐；concat/mean/attention 三种融合；seed=42。
- 输入：/home/mzhao/ECHO-MIMIC-IV/data/A4C/classification.csv、data/A2C/classification.csv。
- 输出：out/features/qwen_feature_fusion/{A4C,A2C}/、out/results/qwen_feature_fusion/、out/figures/qwen_feature_fusion/。
- 结果/指标：数据对齐 5424 样本（4319/551/554）；特征提取与 5 组训练/对比表生成流程已验证（合成特征）。
- 结论与备注：正式实验需在 GPU 上提取 A4C/A2C 特征后运行；旧代码与旧产物已归档。

## [2026-10-25 18:45] Qwen 多视角特征融合分类（cls_la_size）

- 目的：A4C / A2C 单视角与 A4C+A2C 特征级融合（concat/mean/attention）的对比原型。
- 执行命令：model/qwen/feature_fusion/train.py --config model/qwen/feature_fusion/config.yaml --device cuda
- 关键参数：seed=42, frames=16, image_size=224, task=cls_la_size, samples=5424
- 输入：/home/mzhao/ECHO-MIMIC-IV/data/A4C/classification.csv; /home/mzhao/ECHO-MIMIC-IV/data/A2C/classification.csv
- 输出：/home/mzhao/ECHO-MIMIC-IV/out/results/qwen_feature_fusion
- 结果/指标：
```
group   views    fusion  best_epoch  test_acc  test_macro_f1  test_macro_auc  test_sensitivity  test_specificity  test_precision_macro  test_recall_macro  d_acc_vs_A4C  d_f1_vs_A4C  d_auc_vs_A4C  d_acc_vs_A2C  d_f1_vs_A2C  d_auc_vs_A2C
              A4C     A4C    single           8  0.648014       0.647047        0.707344          0.600000          0.695341              0.649079           0.647670      0.000000     0.000000      0.000000      0.021661     0.022958      0.002418
              A2C     A2C    single          64  0.626354       0.624089        0.704927          0.552727          0.698925              0.628638           0.625826     -0.021661    -0.022958     -0.002418      0.000000     0.000000      0.000000
          A4C+A2C A4C+A2C    concat           2  0.689531       0.689272        0.744203          0.723636          0.655914              0.690570           0.689775      0.041516     0.042224      0.036859      0.063177     0.065182      0.039277
     A4C+A2C_mean A4C+A2C      mean          29  0.667870       0.665564        0.731841          0.589091          0.745520              0.671627           0.667305      0.019856     0.018517      0.024497      0.041516     0.041475      0.026914
A4C+A2C_attention A4C+A2C attention           3  0.658845       0.658354        0.727103          0.701818          0.616487              0.660249           0.659153      0.010830     0.011307      0.019759      0.032491     0.034265      0.022177
```
- 结论与备注：融合是否优于单视角见 comparison 表；所有实验使用相同 study 子集与划分。

## [2026-10-25 20:46] Qwen 多视角特征融合分类（cls_rv_function）

- 目的：A4C / A2C 单视角与 A4C+A2C 特征级融合（concat/mean/attention）的对比原型。
- 执行命令：train.py --config config.yaml --device cuda --task cls_rv_function
- 关键参数：seed=42, frames=16, image_size=224, task=cls_rv_function, samples=6007
- 输入：/home/mzhao/ECHO-MIMIC-IV/data/A4C/classification.csv; /home/mzhao/ECHO-MIMIC-IV/data/A2C/classification.csv
- 输出：/home/mzhao/ECHO-MIMIC-IV/out/results/qwen_feature_fusion
- 结果/指标：
```
group   views    fusion  best_epoch  test_acc  test_macro_f1  test_macro_auc  test_sensitivity  test_specificity  test_precision_macro  test_recall_macro  d_acc_vs_A4C  d_f1_vs_A4C  d_auc_vs_A4C  d_acc_vs_A2C  d_f1_vs_A2C  d_auc_vs_A2C
              A4C     A4C    single          29  0.737624       0.622157        0.691485          0.444444          0.814583              0.617182           0.629514      0.000000     0.000000      0.000000     -0.009901     0.002095      0.014484
              A2C     A2C    single          28  0.747525       0.620063        0.677001          0.404762          0.837500              0.619058           0.621131      0.009901    -0.002095     -0.014484      0.000000     0.000000      0.000000
          A4C+A2C A4C+A2C    concat          25  0.770627       0.633119        0.710268          0.380952          0.872917              0.641713           0.626935      0.033003     0.010962      0.018783      0.023102     0.013056      0.033267
     A4C+A2C_mean A4C+A2C      mean          15  0.773927       0.663643        0.711095          0.484127          0.850000              0.660613           0.667063      0.036304     0.041485      0.019610      0.026403     0.043580      0.034094
A4C+A2C_attention A4C+A2C attention          28  0.727723       0.601562        0.691865          0.396825          0.814583              0.598486           0.605704     -0.009901    -0.020595      0.000380     -0.019802    -0.018501      0.014864
```
- 结论与备注：融合是否优于单视角见 comparison 表；所有实验使用相同 study 子集与划分。

## [2026-10-25 20:48] Qwen 多视角特征融合分类（cls_ra_size）

- 目的：A4C / A2C 单视角与 A4C+A2C 特征级融合（concat/mean/attention）的对比原型。
- 执行命令：train.py --config config.yaml --device cuda --task cls_ra_size
- 关键参数：seed=42, frames=16, image_size=224, task=cls_ra_size, samples=5243
- 输入：/home/mzhao/ECHO-MIMIC-IV/data/A4C/classification.csv; /home/mzhao/ECHO-MIMIC-IV/data/A2C/classification.csv
- 输出：/home/mzhao/ECHO-MIMIC-IV/out/results/qwen_feature_fusion
- 结果/指标：
```
group   views    fusion  best_epoch  test_acc  test_macro_f1  test_macro_auc  test_sensitivity  test_specificity  test_precision_macro  test_recall_macro  d_acc_vs_A4C  d_f1_vs_A4C  d_auc_vs_A4C  d_acc_vs_A2C  d_f1_vs_A2C  d_auc_vs_A2C
              A4C     A4C    single          35  0.640385       0.636892        0.680801          0.597458          0.676056              0.637070           0.636757      0.000000     0.000000      0.000000      0.030769     0.028042      0.047117
              A2C     A2C    single           3  0.609615       0.608850        0.633683          0.622881          0.598592              0.609795           0.610736     -0.030769    -0.028042     -0.047117      0.000000     0.000000      0.000000
          A4C+A2C A4C+A2C    concat          13  0.665385       0.662745        0.695050          0.635593          0.690141              0.662644           0.662867      0.025000     0.025854      0.014249      0.055769     0.053895      0.061366
     A4C+A2C_mean A4C+A2C      mean           7  0.640385       0.637423        0.674534          0.605932          0.669014              0.637377           0.637473      0.000000     0.000531     -0.006266      0.030769     0.028572      0.040851
A4C+A2C_attention A4C+A2C attention          10  0.621154       0.618784        0.683397          0.597458          0.640845              0.618643           0.619151     -0.019231    -0.018108      0.002596      0.011538     0.009934      0.049714
```
- 结论与备注：融合是否优于单视角见 comparison 表；所有实验使用相同 study 子集与划分。

## [2026-10-25 21:15] Qwen 多视角特征融合分类（cls_la_size）

- 目的：A4C / A2C 单视角与 A4C+A2C 特征级融合（concat/mean/attention）的对比原型。
- 执行命令：train.py --config config_3view.yaml --device cuda --task cls_la_size
- 关键参数：seed=42, frames=16, image_size=224, task=cls_la_size, samples=5420
- 输入：/home/mzhao/ECHO-MIMIC-IV/data/A4C/classification.csv; /home/mzhao/ECHO-MIMIC-IV/data/A2C/classification.csv
- 输出：/home/mzhao/ECHO-MIMIC-IV/out/results/qwen_feature_fusion
- 结果/指标：
```
group                  views    fusion  best_epoch  test_acc  test_macro_f1  test_macro_auc  test_sensitivity  test_specificity  test_precision_macro  test_recall_macro  d_acc_vs_A4C  d_f1_vs_A4C  d_auc_vs_A4C  d_acc_vs_A2C  d_f1_vs_A2C  d_auc_vs_A2C
                             A4C                    A4C    single           5  0.649819       0.649815        0.704751          0.658182          0.641577              0.649903           0.649879      0.000000     0.000000      0.000000     -0.003610    -0.003610     -0.027788
                             A2C                    A2C    single           8  0.653430       0.653425        0.732538          0.661818          0.645161              0.653514           0.653490      0.003610     0.003610      0.027788      0.000000     0.000000      0.000000
                         A4C+A2C                A4C+A2C    concat           7  0.646209       0.646136        0.705324          0.665455          0.627240              0.646531           0.646347     -0.003610    -0.003679      0.000573     -0.007220    -0.007289     -0.027214
          A4C+A2C+Apical_Doppler A4C+A2C+Apical_Doppler    concat           2  0.646209       0.645303        0.731900          0.701818          0.591398              0.648340           0.646608     -0.003610    -0.004512      0.027149     -0.007220    -0.008122     -0.000639
     A4C+A2C+Apical_Doppler_mean A4C+A2C+Apical_Doppler      mean           4  0.662455       0.658843        0.747279          0.563636          0.759857              0.668376           0.661746      0.012635     0.009028      0.042529      0.009025     0.005418      0.014741
A4C+A2C+Apical_Doppler_attention A4C+A2C+Apical_Doppler attention           3  0.682310       0.682244        0.725670          0.672727          0.691756              0.682318           0.682242      0.032491     0.032429      0.020919      0.028881     0.028819     -0.006869
```
- 结论与备注：融合是否优于单视角见 comparison 表；所有实验使用相同 study 子集与划分。

## [2026-10-25 21:18] Qwen 多视角特征融合分类（cls_rv_function）

- 目的：A4C / A2C 单视角与 A4C+A2C 特征级融合（concat/mean/attention）的对比原型。
- 执行命令：train.py --config config_3view.yaml --device cuda --task cls_rv_function
- 关键参数：seed=42, frames=16, image_size=224, task=cls_rv_function, samples=5987
- 输入：/home/mzhao/ECHO-MIMIC-IV/data/A4C/classification.csv; /home/mzhao/ECHO-MIMIC-IV/data/A2C/classification.csv
- 输出：/home/mzhao/ECHO-MIMIC-IV/out/results/qwen_feature_fusion
- 结果/指标：
```
group                  views    fusion  best_epoch  test_acc  test_macro_f1  test_macro_auc  test_sensitivity  test_specificity  test_precision_macro  test_recall_macro  d_acc_vs_A4C  d_f1_vs_A4C  d_auc_vs_A4C  d_acc_vs_A2C  d_f1_vs_A2C  d_auc_vs_A2C
                             A4C                    A4C    single          20  0.740433       0.608371        0.698605             0.384          0.834034              0.607753           0.609017      0.000000     0.000000      0.000000      0.044925     0.000404     -0.012101
                             A2C                    A2C    single           2  0.695507       0.607967        0.710706             0.536          0.737395              0.603575           0.636697     -0.044925    -0.000404      0.012101      0.000000     0.000000      0.000000
                         A4C+A2C                A4C+A2C    concat          15  0.770383       0.645188        0.716303             0.424          0.861345              0.648000           0.642672      0.029950     0.036817      0.017697      0.074875     0.037221      0.005597
          A4C+A2C+Apical_Doppler A4C+A2C+Apical_Doppler    concat          21  0.787022       0.643918        0.717613             0.368          0.897059              0.664042           0.632529      0.046589     0.035547      0.019008      0.091514     0.035951      0.006908
     A4C+A2C+Apical_Doppler_mean A4C+A2C+Apical_Doppler      mean          20  0.763727       0.643517        0.741395             0.440          0.848739              0.642696           0.644370      0.023295     0.035146      0.042790      0.068220     0.035550      0.030689
A4C+A2C+Apical_Doppler_attention A4C+A2C+Apical_Doppler attention          64  0.777038       0.613412        0.691126             0.304          0.901261              0.639227           0.602630      0.036606     0.005041     -0.007479      0.081531     0.005445     -0.019580
```
- 结论与备注：融合是否优于单视角见 comparison 表；所有实验使用相同 study 子集与划分。

## [2026-10-25 21:20] Qwen 多视角特征融合分类（cls_ra_size）

- 目的：A4C / A2C 单视角与 A4C+A2C 特征级融合（concat/mean/attention）的对比原型。
- 执行命令：train.py --config config_3view.yaml --device cuda --task cls_ra_size
- 关键参数：seed=42, frames=16, image_size=224, task=cls_ra_size, samples=5241
- 输入：/home/mzhao/ECHO-MIMIC-IV/data/A4C/classification.csv; /home/mzhao/ECHO-MIMIC-IV/data/A2C/classification.csv
- 输出：/home/mzhao/ECHO-MIMIC-IV/out/results/qwen_feature_fusion
- 结果/指标：
```
group                  views    fusion  best_epoch  test_acc  test_macro_f1  test_macro_auc  test_sensitivity  test_specificity  test_precision_macro  test_recall_macro  d_acc_vs_A4C  d_f1_vs_A4C  d_auc_vs_A4C  d_acc_vs_A2C  d_f1_vs_A2C  d_auc_vs_A2C
                             A4C                    A4C    single           3  0.611538       0.610062        0.647589          0.605932          0.616197              0.610223           0.611065      0.000000     0.000000      0.000000      0.003846     0.005464      0.003297
                             A2C                    A2C    single           8  0.607692       0.604598        0.644292          0.572034          0.637324              0.604535           0.604679     -0.003846    -0.005464     -0.003297      0.000000     0.000000      0.000000
                         A4C+A2C                A4C+A2C    concat          11  0.628846       0.625521        0.662151          0.588983          0.661972              0.625569           0.625477      0.017308     0.015459      0.014562      0.021154     0.020923      0.017859
          A4C+A2C+Apical_Doppler A4C+A2C+Apical_Doppler    concat           7  0.650000       0.646734        0.699361          0.610169          0.683099              0.646853           0.646634      0.038462     0.036672      0.051772      0.042308     0.042136      0.055070
     A4C+A2C+Apical_Doppler_mean A4C+A2C+Apical_Doppler      mean           8  0.638462       0.635354        0.702196          0.601695          0.669014              0.635354           0.635354      0.026923     0.025293      0.054607      0.030769     0.030756      0.057905
A4C+A2C+Apical_Doppler_attention A4C+A2C+Apical_Doppler attention           2  0.611538       0.611331        0.626611          0.648305          0.580986              0.613911           0.614646      0.000000     0.001270     -0.020978      0.003846     0.006733     -0.017680
```
- 结论与备注：融合是否优于单视角见 comparison 表；所有实验使用相同 study 子集与划分。

## [2026-10-25 22:20] 图表按任务归档 + 跨任务总览

- 目的：将三任务/多实验组的图按任务归档，并生成统一对比图。
- 执行命令：`python model/qwen/feature_fusion/make_overall_figure.py`
- 关键参数：使用 `comparison_<task>_3view.csv`（含 A4C/A2C/2-view/3-view 全部组）。
- 输入：out/results/qwen_feature_fusion/comparison_*.csv
- 输出：out/figures/qwen_feature_fusion/<task>/<group>/、overall_comparison.{png,pdf,csv,md}、overall_comparison_heatmap.{png,pdf}
- 结果/指标：3 个任务 × 6 组共 18 条 test 指标汇总；原 231 张图已归入各自任务目录。
- 结论与备注：后续训练图目录改为 `figures/qwen_feature_fusion/<task>/<group>/`。

## [2026-10-25 22:31] Qwen 多视角特征融合分类（cls_mitral_regurg）

- 目的：A4C / A2C 单视角与 A4C+A2C 特征级融合（concat/mean/attention）的对比原型。
- 执行命令：train.py --config config_doppler.yaml --device cuda --task cls_mitral_regurg
- 关键参数：seed=42, frames=16, image_size=224, task=cls_mitral_regurg, samples=5862
- 输入：/home/mzhao/ECHO-MIMIC-IV/data/A4C/classification.csv; /home/mzhao/ECHO-MIMIC-IV/data/A2C/classification.csv
- 输出：/home/mzhao/ECHO-MIMIC-IV/out/results/qwen_feature_fusion
- 结果/指标：
```
group                  views    fusion  best_epoch  test_acc  test_macro_f1  test_macro_auc  test_sensitivity  test_specificity  test_precision_macro  test_recall_macro  d_acc_vs_A4C  d_f1_vs_A4C  d_auc_vs_A4C  d_acc_vs_A2C  d_f1_vs_A2C  d_auc_vs_A2C
                  Apical_Doppler         Apical_Doppler    single          33  0.731023       0.641364        0.728506          0.457516          0.823400              0.642325           0.640458      0.039604     0.021554      0.047151      0.044554     0.007660      0.010965
                             A4C                    A4C    single           6  0.691419       0.619810        0.681355          0.509804          0.752759              0.615119           0.631282      0.000000     0.000000      0.000000      0.004950    -0.013894     -0.036186
                             A2C                    A2C    single           2  0.686469       0.633704        0.717540          0.607843          0.713024              0.630191           0.660434     -0.004950     0.013894      0.036186      0.000000     0.000000      0.000000
              Apical_Doppler+A4C     Apical_Doppler+A4C    concat          13  0.712871       0.632061        0.710903          0.483660          0.790287              0.628546           0.636974      0.021452     0.012251      0.029549      0.026403    -0.001643     -0.006637
              Apical_Doppler+A2C     Apical_Doppler+A2C    concat          23  0.731023       0.644472        0.726161          0.470588          0.818985              0.644164           0.644786      0.039604     0.024662      0.044807      0.044554     0.010768      0.008621
          Apical_Doppler+A4C+A2C Apical_Doppler+A4C+A2C    concat          25  0.752475       0.638764        0.727308          0.379085          0.878587              0.660288           0.628836      0.061056     0.018954      0.045954      0.066007     0.005060      0.009768
     Apical_Doppler+A4C+A2C_mean Apical_Doppler+A4C+A2C      mean          50  0.750825       0.661734        0.741044          0.470588          0.845475              0.666237           0.658031      0.059406     0.041924      0.059689      0.064356     0.028030      0.023503
Apical_Doppler+A4C+A2C_attention Apical_Doppler+A4C+A2C attention          19  0.721122       0.600487        0.670577          0.339869          0.849890              0.612757           0.594879      0.029703    -0.019323     -0.010778      0.034653    -0.033217     -0.046964
```
- 结论与备注：融合是否优于单视角见 comparison 表；所有实验使用相同 study 子集与划分。

## [2026-10-25 22:33] Qwen 多视角特征融合分类（cls_tricuspid_regurg）

- 目的：A4C / A2C 单视角与 A4C+A2C 特征级融合（concat/mean/attention）的对比原型。
- 执行命令：train.py --config config_doppler.yaml --device cuda --task cls_tricuspid_regurg
- 关键参数：seed=42, frames=16, image_size=224, task=cls_tricuspid_regurg, samples=5921
- 输入：/home/mzhao/ECHO-MIMIC-IV/data/A4C/classification.csv; /home/mzhao/ECHO-MIMIC-IV/data/A2C/classification.csv
- 输出：/home/mzhao/ECHO-MIMIC-IV/out/results/qwen_feature_fusion
- 结果/指标：
```
group                  views    fusion  best_epoch  test_acc  test_macro_f1  test_macro_auc  test_sensitivity  test_specificity  test_precision_macro  test_recall_macro  d_acc_vs_A4C  d_f1_vs_A4C  d_auc_vs_A4C  d_acc_vs_A2C  d_f1_vs_A2C  d_auc_vs_A2C
                  Apical_Doppler         Apical_Doppler    single          10  0.732888       0.661553        0.746734          0.573427          0.782895              0.653553           0.678161      0.053422     0.050356      0.057938      0.050083     0.037621      0.058459
                             A4C                    A4C    single           7  0.679466       0.611197        0.688796          0.545455          0.721491              0.607757           0.633473      0.000000     0.000000      0.000000     -0.003339    -0.012736      0.000521
                             A2C                    A2C    single           7  0.682805       0.623933        0.688274          0.601399          0.708333              0.621347           0.654866      0.003339     0.012736     -0.000521      0.000000     0.000000      0.000000
              Apical_Doppler+A4C     Apical_Doppler+A4C    concat           7  0.724541       0.654303        0.765857          0.573427          0.771930              0.646580           0.672678      0.045075     0.043107      0.077061      0.041736     0.030371      0.077583
              Apical_Doppler+A2C     Apical_Doppler+A2C    concat           3  0.739566       0.679679        0.778233          0.643357          0.769737              0.670070           0.706547      0.060100     0.068482      0.089437      0.056761     0.055746      0.089958
          Apical_Doppler+A4C+A2C Apical_Doppler+A4C+A2C    concat           2  0.742905       0.692259        0.791345          0.706294          0.754386              0.682685           0.730340      0.063439     0.081062      0.102549      0.060100     0.068326      0.103070
     Apical_Doppler+A4C+A2C_mean Apical_Doppler+A4C+A2C      mean           7  0.714524       0.637580        0.730033          0.531469          0.771930              0.631159           0.651699      0.035058     0.026383      0.041237      0.031720     0.013647      0.041759
Apical_Doppler+A4C+A2C_attention Apical_Doppler+A4C+A2C attention           2  0.682805       0.625102        0.714912          0.608392          0.706140              0.622759           0.657266      0.003339     0.013905      0.026116      0.000000     0.001169      0.026638
```
- 结论与备注：融合是否优于单视角见 comparison 表；所有实验使用相同 study 子集与划分。
