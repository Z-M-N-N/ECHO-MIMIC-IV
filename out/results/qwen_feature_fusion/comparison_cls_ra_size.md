# cls_ra_size: single-view vs feature fusion

            group   views    fusion  best_epoch  test_acc  test_macro_f1  test_macro_auc  test_sensitivity  test_specificity  test_precision_macro  test_recall_macro  d_acc_vs_A4C  d_f1_vs_A4C  d_auc_vs_A4C  d_acc_vs_A2C  d_f1_vs_A2C  d_auc_vs_A2C
              A4C     A4C    single          35  0.640385       0.636892        0.680801          0.597458          0.676056              0.637070           0.636757      0.000000     0.000000      0.000000      0.030769     0.028042      0.047117
              A2C     A2C    single           3  0.609615       0.608850        0.633683          0.622881          0.598592              0.609795           0.610736     -0.030769    -0.028042     -0.047117      0.000000     0.000000      0.000000
          A4C+A2C A4C+A2C    concat          13  0.665385       0.662745        0.695050          0.635593          0.690141              0.662644           0.662867      0.025000     0.025854      0.014249      0.055769     0.053895      0.061366
     A4C+A2C_mean A4C+A2C      mean           7  0.640385       0.637423        0.674534          0.605932          0.669014              0.637377           0.637473      0.000000     0.000531     -0.006266      0.030769     0.028572      0.040851
A4C+A2C_attention A4C+A2C attention          10  0.621154       0.618784        0.683397          0.597458          0.640845              0.618643           0.619151     -0.019231    -0.018108      0.002596      0.011538     0.009934      0.049714
