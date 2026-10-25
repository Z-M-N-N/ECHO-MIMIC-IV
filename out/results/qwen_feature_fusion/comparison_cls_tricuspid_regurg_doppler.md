# cls_tricuspid_regurg: view/fusion comparison (Apical Doppler labels)

                           group  best_epoch  test_acc  test_macro_f1  test_macro_auc  test_sensitivity  test_specificity  d_acc_vs_Apical_Doppler  d_f1_vs_Apical_Doppler  d_auc_vs_Apical_Doppler
                  Apical_Doppler          10    0.7329         0.6616          0.7467            0.5734            0.7829                   0.0000                  0.0000                   0.0000
                             A4C           7    0.6795         0.6112          0.6888            0.5455            0.7215                  -0.0534                 -0.0504                  -0.0579
                             A2C           7    0.6828         0.6239          0.6883            0.6014            0.7083                  -0.0501                 -0.0376                  -0.0585
              Apical_Doppler+A4C           7    0.7245         0.6543          0.7659            0.5734            0.7719                  -0.0083                 -0.0072                   0.0191
              Apical_Doppler+A2C           3    0.7396         0.6797          0.7782            0.6434            0.7697                   0.0067                  0.0181                   0.0315
          Apical_Doppler+A4C+A2C           2    0.7429         0.6923          0.7913            0.7063            0.7544                   0.0100                  0.0307                   0.0446
     Apical_Doppler+A4C+A2C_mean           7    0.7145         0.6376          0.7300            0.5315            0.7719                  -0.0184                 -0.0240                  -0.0167
Apical_Doppler+A4C+A2C_attention           2    0.6828         0.6251          0.7149            0.6084            0.7061                  -0.0501                 -0.0365                  -0.0318