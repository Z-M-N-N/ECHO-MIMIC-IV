# cls_la_size: single-view vs joint e2e LoRA

  group   views  best_epoch  test_acc  test_macro_f1  test_macro_auc  test_sensitivity  test_specificity  trainable_params  trainable_ratio  d_acc_vs_A4C  d_f1_vs_A4C  d_auc_vs_A4C  d_acc_vs_A2C  d_f1_vs_A2C  d_auc_vs_A2C
    A4C     A4C           2  0.615523       0.610805        0.633392          0.509091          0.720430          16151810         0.004283      0.000000     0.000000      0.000000     -0.021661    -0.025753     -0.051144
    A2C     A2C           2  0.637184       0.636558        0.684536          0.600000          0.673835          16151810         0.004283      0.021661     0.025753      0.051144      0.000000     0.000000      0.000000
A4C+A2C A4C+A2C           2  0.611913       0.609981        0.676377          0.545455          0.677419          16151810         0.004283     -0.003610    -0.000824      0.042985     -0.025271    -0.026577     -0.008159