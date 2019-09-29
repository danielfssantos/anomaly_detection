#!/bin/bash
#!/usr/bin/env python

:<<END
################################### TRAIN SVM WITHOUT AUGMENTATION #########################
python3 main_script_anomaly_detection.py  --mode='train_svm' --svm-params-path='SVMParams/KDDTrain+_20Percent' --gen-dataset=1\
                                                                  --c=256.0 --g=1.0 --norm-type='min_max_norm'\
                                                                  --data-save-path='TrainData/KDDTrain+_20Percent/'\
                                                                  --train-file-name='discriminative_kdd_nsl_processed.npz'



################################### TEST NOT AUGMENTED SVM ##############################################
python3 main_script_anomaly_detection.py  --mode='test_svm' --svm-params-path='SVMParams/KDDTrain+_20Percent' --norm-type='min_max_norm'\
                                                                  --data-save-path='TrainData/KDDTrain+_20Percent/'\
                                                                  --train-file-name='discriminative_kdd_nsl_processed.npz'
                                                                  --svm-analysis-path='./SVMAnalysis/Original'
END

:<<END
################################### TRAIN SVM WITH RBM AUGMENTATION #########################
python3 main_script_anomaly_detection.py  --mode='train_svm_aug_rbm' --svm-params-path='SVMParamsAug/RBM' --gen-dataset=1\
                                                                  --c=256.0 --g=1.0 --norm-type='min_max_norm' --sample-ites=100\
                                                                  --data-save-path='TrainData/KDDTrain+_20Percent/' --batch-sz=500\
                                                                  --train-file-name='discriminative_kdd_nsl_processed_aug_rbm.npz'\
                                                                  --use-oc-svm=0 --data-sampler-train-type='bbrbm_pcd'\
                                                                  --data-sampler-params-path='./RBMParams/KDDTrain+_20Percent'


################################### TEST AUGMENTED SVM ##############################################
python3 main_script_anomaly_detection.py  --mode='test_svm_aug_rbm' --svm-params-path='SVMParamsAug/RBM' --norm-type='min_max_norm'\
                                                                  --data-save-path='TrainData/KDDTrain+_20Percent/'\
                                                                  --train-file-name='discriminative_kdd_nsl_processed_aug_rbm.npz'\
                                                                  --svm-analysis-path='./SVMAnalysis/RBMAugmented'
END


################################### TRAIN SVM WITH GAN AUGMENTATION #########################
python3 main_script_anomaly_detection.py  --mode='train_svm_aug_gan' --svm-params-path='SVMParamsAug/GAN' --gen-dataset=1\
                                                                  --c=256.0 --g=1.0 --norm-type='min_max_norm' \
                                                                  --data-save-path='TrainData/KDDTrain+_20Percent/' --batch-sz=128\
                                                                  --train-file-name='discriminative_kdd_nsl_processed_aug_gan.npz'\
                                                                  --use-oc-svm=0 --data-sampler-train-type='gan'\
                                                                  --data-sampler-params-path='./GANParams/KDDTrain+_20Percent'


################################### TEST AUGMENTED SVM ##############################################
python3 main_script_anomaly_detection.py  --mode='test_svm_aug_gan' --svm-params-path='SVMParamsAug/GAN' --norm-type='min_max_norm'\
                                                                  --data-save-path='TrainData/KDDTrain+_20Percent/'\
                                                                  --train-file-name='discriminative_kdd_nsl_processed_aug_gan.npz'\
                                                                  --svm-analysis-path='./SVMAnalysis/GANAugmented'
