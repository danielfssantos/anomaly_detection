#!/bin/bash
#!/usr/bin/env python

:<<END
################################### TRAIN SVM WITHOUT AUGMENTATION #########################
python3 main_script_anomaly_detection.py  --mode='train_svm' --svm-params-path='SVMParams' --gen-dataset=1\
                                                                  --c=256.0 --g=1.0 --norm-type='min_max_norm'\
                                                                  --data-save-path='TrainData/KDDTrain+_20Percent/'\
                                                                  --train-file-name='discriminative_kdd_nsl_processed.npz'



################################### TEST NOT AUGMENTED SVM ##############################################
python3 main_script_anomaly_detection.py  --mode='test_svm' --svm-params-path='SVMParams' --norm-type='min_max_norm'\
                                                                  --data-save-path='TrainData/KDDTrain+_20Percent/'\
                                                                  --train-file-name='discriminative_kdd_nsl_processed.npz'
END


################################### TRAIN SVM WITH AUGMENTATION #########################
python3 main_script_anomaly_detection.py  --mode='train_svm_aug' --svm-params-path='SVMParamsAug' --gen-dataset=1\
                                                                  --c=256.0 --g=1.0 --norm-type='min_max_norm' --sample-ites=10\
                                                                  --data-save-path='TrainData/KDDTrain+_20Percent/' --batch-sz=500\
                                                                  --train-file-name='discriminative_kdd_nsl_processed_aug.npz'\
                                                                  --use-oc-svm=0


################################### TEST AUGMENTED SVM ##############################################
python3 main_script_anomaly_detection.py  --mode='test_svm_aug' --svm-params-path='SVMParamsAug' --norm-type='min_max_norm'\
                                                                  --data-save-path='TrainData/KDDTrain+_20Percent/'\
                                                                  --train-file-name='discriminative_kdd_nsl_processed_aug.npz'
