#!/bin/bash
#!/usr/bin/env python

:<<END
################################### TRAIN SVM WITHOUT AUGMENTATION #########################
python3 main_script_anomaly_detection.py  --mode='train_svm'\
                                          --svm-params-path='SVMParams/KDDTrain+_20Percent/Original/KDDfeatures'\
                                          --gen-dataset=1\
                                          --c=256.0 --g=1.0 --norm-type='min_max_norm'\
                                          --data-save-path='TrainData/KDDTrain+_20Percent/'\
                                          --train-file-name='discriminative_kdd_nsl_processed.npz'


################################### TEST NOT AUGMENTED SVM ##############################################
python3 main_script_anomaly_detection.py  --mode='test_svm'\
                                          --svm-params-path='SVMParams/KDDTrain+_20Percent/Original/KDDfeatures'\
                                          --norm-type='min_max_norm'\
                                          --data-save-path='TrainData/KDDTrain+_20Percent/'\
                                          --train-file-name='discriminative_kdd_nsl_processed.npz'
                                          --svm-analysis-path='./SVMAnalysis/Original/KDDfeatures'
END


# --c=2 --g=4 for pcd bbrbm (10 neurons)
# --c=0.5 --g=2 for pcd bbrbm (20 neurons)
# --c=0.25 --g=2 for pcd bbrbm (30 neurons)
# --c=1 --g=2 for pcd bbrbm (40 neurons)
# --c=0.5 --g=4 for pcd bbrbm (50 neurons)
# --c=0.25 --g=2 for pcd bbrbm (60 neurons)
# --c=1 --g=1 for pcd bbrbm (70 neurons)
# --c=1 --g=0.5 for pcd bbrbm (80 neurons)
# --c=0.5 --g=0.5 for pcd bbrbm (90 neurons)


# --c=0.25 --g=0.25 for cd bbrbm (100 neurons)
# --c=0.25 --g=2.0 for pcd bbrbm (100 neurons)

:<<END
################################### TRAIN SVM WITHOUT AUGMENTATION RBM features#########################
python3 main_script_anomaly_detection.py  --mode='train_svm_rbm_features'\
                                          --svm-params-path='SVMParams/KDDTrain+_20Percent/Original/RBMPCDfeatures/10neurons'\
                                          --data-sampler-params-path='./RBMParams/KDDTrain+_20Percent/10neurons'\
                                          --gen-dataset=1 --rbm-train-type='bbrbm_pcd'\
                                          --c=2 --g=4 --norm-type='min_max_norm'\
                                          --data-save-path='TrainData/KDDTrain+_20Percent/'\
                                          --train-file-name='discriminative_kdd_nsl_processed_features.npz'\
                                          --sample-ites=1
END


################################### TEST NOT AUGMENTED SVM RBM FEATURES ##############################################
python3 main_script_anomaly_detection.py  --mode='test_svm_rbm_features'\
                                          --svm-params-path='SVMParams/KDDTrain+_20Percent/Original/RBMPCDfeatures/10neurons'\
                                          --data-sampler-params-path='./RBMParams/KDDTrain+_20Percent/10neurons'\
                                          --svm-analysis-path='./SVMAnalysis/Original/RBMPCDfeatures/10neurons'\
                                          --rbm-train-type='bbrbm_pcd'\
                                          --data-save-path='TrainData/KDDTrain+_20Percent/'\
                                          --train-file-name='discriminative_kdd_nsl_processed_features.npz'\
                                          --sample-ites=1


:<<END
################################### TRAIN SVM WITH RBM AUGMENTATION #########################
python3 main_script_anomaly_detection.py  --mode='train_svm_aug_rbm' --svm-params-path='SVMParamsAug/RBM'\
                                          --gen-dataset=1 --c=32.0 --g=.5 --norm-type='min_max_norm' --sample-ites=100\
                                          --data-save-path='TrainData/KDDTrain+_20Percent/' --batch-sz=500\
                                          --train-file-name='discriminative_kdd_nsl_processed_aug_rbm.npz'\
                                          --use-oc-svm=0 --data-sampler-train-type='bbrbm_pcd'\
                                          --data-sampler-params-path='./RBMParams/KDDTrain+_20Percent'


################################### TEST AUGMENTED SVM ##############################################
python3 main_script_anomaly_detection.py  --mode='test_svm_aug_rbm' --svm-params-path='SVMParamsAug/RBM' --norm-type='min_max_norm'\
                                                                  --data-save-path='TrainData/KDDTrain+_20Percent/'\
                                                                  --train-file-name='discriminative_kdd_nsl_processed_aug_rbm.npz'\
                                                                  --svm-analysis-path='./SVMAnalysis/RBMAugmented2'

END

:<<END
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
END