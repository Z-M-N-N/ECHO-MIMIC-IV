# cls_la_size: single-view vs feature fusion

            group   views    fusion  best_epoch  test_acc  test_macro_f1  test_macro_auc  test_sensitivity  test_specificity  test_precision_macro  test_recall_macro  d_acc_vs_A4C  d_f1_vs_A4C  d_auc_vs_A4C  d_acc_vs_A2C  d_f1_vs_A2C  d_auc_vs_A2C
              A4C     A4C    single           8  0.648014       0.647047        0.707344          0.600000          0.695341              0.649079           0.647670      0.000000     0.000000      0.000000      0.021661     0.022958      0.002418
              A2C     A2C    single          64  0.626354       0.624089        0.704927          0.552727          0.698925              0.628638           0.625826     -0.021661    -0.022958     -0.002418      0.000000     0.000000      0.000000
          A4C+A2C A4C+A2C    concat           2  0.689531       0.689272        0.744203          0.723636          0.655914              0.690570           0.689775      0.041516     0.042224      0.036859      0.063177     0.065182      0.039277
     A4C+A2C_mean A4C+A2C      mean          29  0.667870       0.665564        0.731841          0.589091          0.745520              0.671627           0.667305      0.019856     0.018517      0.024497      0.041516     0.041475      0.026914
A4C+A2C_attention A4C+A2C attention           3  0.658845       0.658354        0.727103          0.701818          0.616487              0.660249           0.659153      0.010830     0.011307      0.019759      0.032491     0.034265      0.022177
