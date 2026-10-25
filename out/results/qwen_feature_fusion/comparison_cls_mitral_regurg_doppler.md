# cls_mitral_regurg: view/fusion comparison (Apical Doppler labels)

                           group  best_epoch  test_acc  test_macro_f1  test_macro_auc  test_sensitivity  test_specificity  d_acc_vs_Apical_Doppler  d_f1_vs_Apical_Doppler  d_auc_vs_Apical_Doppler
                  Apical_Doppler          33    0.7310         0.6414          0.7285            0.4575            0.8234                   0.0000                  0.0000                   0.0000
                             A4C           6    0.6914         0.6198          0.6814            0.5098            0.7528                  -0.0396                 -0.0216                  -0.0472
                             A2C           2    0.6865         0.6337          0.7175            0.6078            0.7130                  -0.0446                 -0.0077                  -0.0110
              Apical_Doppler+A4C          13    0.7129         0.6321          0.7109            0.4837            0.7903                  -0.0182                 -0.0093                  -0.0176
              Apical_Doppler+A2C          23    0.7310         0.6445          0.7262            0.4706            0.8190                   0.0000                  0.0031                  -0.0023
          Apical_Doppler+A4C+A2C          25    0.7525         0.6388          0.7273            0.3791            0.8786                   0.0215                 -0.0026                  -0.0012
     Apical_Doppler+A4C+A2C_mean          50    0.7508         0.6617          0.7410            0.4706            0.8455                   0.0198                  0.0204                   0.0125
Apical_Doppler+A4C+A2C_attention          19    0.7211         0.6005          0.6706            0.3399            0.8499                  -0.0099                 -0.0409                  -0.0579