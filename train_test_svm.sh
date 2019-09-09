#!/bin/bash
#!/usr/bin/env python


################################### TRAIN SVM ##############################################
python3 main_script_anomaly_detection.py  --mode='train_svm_aug' --svm-params-path='SVMParamsAug' --gen-dataset=1\
                                                                  --c=256.0 --g=1.0 --norm-type='min_max_norm' --sample-ites=50\
                                                                  --data-save-path='TrainData/KDDTrain+_20Percent/'\
                                                                  --train-file-name='discriminative_kdd_nsl_processed_aug.npz'\
                                                                  --svm-params-path='./SVMParams'



################################### TEST SVM ##############################################
python3 main_script_anomaly_detection.py  --mode='test_svm_aug' --svm-params-path='SVMParams' --norm-type='min_max_norm'\
                                                                  --data-save-path='TrainData/KDDTrain+_20Percent/'\
                                                                  --train-file-name='discriminative_kdd_nsl_processed_aug.npz'
