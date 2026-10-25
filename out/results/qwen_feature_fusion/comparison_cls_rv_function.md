# cls_rv_function: single-view vs feature fusion

            group   views    fusion  best_epoch  test_acc  test_macro_f1  test_macro_auc  test_sensitivity  test_specificity  test_precision_macro  test_recall_macro  d_acc_vs_A4C  d_f1_vs_A4C  d_auc_vs_A4C  d_acc_vs_A2C  d_f1_vs_A2C  d_auc_vs_A2C
              A4C     A4C    single          29  0.737624       0.622157        0.691485          0.444444          0.814583              0.617182           0.629514      0.000000     0.000000      0.000000     -0.009901     0.002095      0.014484
              A2C     A2C    single          28  0.747525       0.620063        0.677001          0.404762          0.837500              0.619058           0.621131      0.009901    -0.002095     -0.014484      0.000000     0.000000      0.000000
          A4C+A2C A4C+A2C    concat          25  0.770627       0.633119        0.710268          0.380952          0.872917              0.641713           0.626935      0.033003     0.010962      0.018783      0.023102     0.013056      0.033267
     A4C+A2C_mean A4C+A2C      mean          15  0.773927       0.663643        0.711095          0.484127          0.850000              0.660613           0.667063      0.036304     0.041485      0.019610      0.026403     0.043580      0.034094
A4C+A2C_attention A4C+A2C attention          28  0.727723       0.601562        0.691865          0.396825          0.814583              0.598486           0.605704     -0.009901    -0.020595      0.000380     -0.019802    -0.018501      0.014864
