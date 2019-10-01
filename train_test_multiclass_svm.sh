#!/bin/bash
#!/usr/bin/env python


################################### TRAIN SVM WITHOUT AUGMENTATION #########################
python3 main_script_multiclass_svm_analysis.py  --mode='train_svm'\
                                                                          --svm-params-path='MultiSVMParams/Original'\
                                                                          --c=8 --g=.5 --norm-type='min_max_norm'\
                                                                          --data-save-path='TrainData/KDDTrain+_20Percent/'\
                                                                          --train-file-name='discriminative_kdd_nsl_processed.npz'


################################### TEST NOT AUGMENTED SVM ##############################################
python3 main_script_multiclass_svm_analysis.py  --mode='test_svm'\
                                                                          --svm-params-path='MultiSVMParams/Original'\
                                                                          --svm-analysis-path='MultiSVMAnalysis/Original'\
                                                                          --norm-type='min_max_norm'\
                                                                          --data-save-path='TrainData/KDDTrain+_20Percent/'\
                                                                          --train-file-name='discriminative_kdd_nsl_processed.npz'


################################### TRAIN SVM WITH RBM AUGMENTATION #########################
python3 main_script_multiclass_svm_analysis.py  --mode='train_svm_aug'\
                                                                          --svm-params-path='MultiSVMParams/RBM'\
                                                                          --c=1024 --g=1.0 --norm-type='min_max_norm'\
                                                                          --data-save-path='TrainData/KDDTrain+_20Percent/'\
                                                                          --train-file-name='discriminative_kdd_nsl_processed_aug_rbm.npz'


################################### TEST RBM AUGMENTED SVM ##############################################
python3 main_script_multiclass_svm_analysis.py  --mode='test_svm_aug' --svm-params-path='MultiSVMParams/RBM'\
                                                                          --svm-analysis-path='MultiSVMAnalysis/RBM'\
                                                                          --norm-type='min_max_norm'\
                                                                          --data-save-path='TrainData/KDDTrain+_20Percent/'\
                                                                          --train-file-name='discriminative_kdd_nsl_processed_aug_rbm.npz'


################################### TRAIN SVM WITH GAN AUGMENTATION #########################
python3 main_script_multiclass_svm_analysis.py  --mode='train_svm_aug_gan'\
                                                                          --svm-params-path='MultiSVMParams/GAN'\
                                                                          --c=512 --g=.5 --norm-type='min_max_norm'\
                                                                          --data-save-path='TrainData/KDDTrain+_20Percent/'\
                                                                          --train-file-name='discriminative_kdd_nsl_processed_aug_gan.npz'


################################### TEST GAN AUGMENTED SVM ##############################################
python3 main_script_multiclass_svm_analysis.py  --mode='test_svm_aug_gan'\
                                                                          --svm-params-path='MultiSVMParams/GAN'\
                                                                          --svm-analysis-path='MultiSVMAnalysis/GAN'\
                                                                          --norm-type='min_max_norm'\
                                                                          --data-save-path='TrainData/KDDTrain+_20Percent/'\
                                                                          --train-file-name='discriminative_kdd_nsl_processed_aug_gan.npz'

