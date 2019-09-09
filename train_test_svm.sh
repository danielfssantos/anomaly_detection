#!/bin/bash
#!/usr/bin/env python

:<<END
################################### TRAIN SVM ##############################################
python3 main_script_anomaly_detection.py  --mode='train_svm' --svm-params-path='SVMParams' --gen-dataset=1\
                                                                  --c=256.0 --g=1.0 --norm-type='min_max_norm' --data-save-path='TrainData'
END

################################### TEST SVM ##############################################
python3 main_script_anomaly_detection.py  --mode='test_svm' --svm-params-path='SVMParams' --norm-type='min_max_norm'
