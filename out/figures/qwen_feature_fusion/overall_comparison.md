# Overall comparison (test)

           task                            group  test_acc  test_macro_f1  test_macro_auc  test_sensitivity  test_specificity
    cls_la_size                              A4C  0.649819       0.649815        0.704751          0.658182          0.641577
    cls_la_size                              A2C  0.653430       0.653425        0.732538          0.661818          0.645161
    cls_la_size                          A4C+A2C  0.646209       0.646136        0.705324          0.665455          0.627240
    cls_la_size           A4C+A2C+Apical_Doppler  0.646209       0.645303        0.731900          0.701818          0.591398
    cls_la_size      A4C+A2C+Apical_Doppler_mean  0.662455       0.658843        0.747279          0.563636          0.759857
    cls_la_size A4C+A2C+Apical_Doppler_attention  0.682310       0.682244        0.725670          0.672727          0.691756
cls_rv_function                              A4C  0.740433       0.608371        0.698605          0.384000          0.834034
cls_rv_function                              A2C  0.695507       0.607967        0.710706          0.536000          0.737395
cls_rv_function                          A4C+A2C  0.770383       0.645188        0.716303          0.424000          0.861345
cls_rv_function           A4C+A2C+Apical_Doppler  0.787022       0.643918        0.717613          0.368000          0.897059
cls_rv_function      A4C+A2C+Apical_Doppler_mean  0.763727       0.643517        0.741395          0.440000          0.848739
cls_rv_function A4C+A2C+Apical_Doppler_attention  0.777038       0.613412        0.691126          0.304000          0.901261
    cls_ra_size                              A4C  0.611538       0.610062        0.647589          0.605932          0.616197
    cls_ra_size                              A2C  0.607692       0.604598        0.644292          0.572034          0.637324
    cls_ra_size                          A4C+A2C  0.628846       0.625521        0.662151          0.588983          0.661972
    cls_ra_size           A4C+A2C+Apical_Doppler  0.650000       0.646734        0.699361          0.610169          0.683099
    cls_ra_size      A4C+A2C+Apical_Doppler_mean  0.638462       0.635354        0.702196          0.601695          0.669014
    cls_ra_size A4C+A2C+Apical_Doppler_attention  0.611538       0.611331        0.626611          0.648305          0.580986
