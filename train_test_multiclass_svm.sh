#!/bin/bash
#!/usr/bin/env python

:<<END
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
END

:<<END
################################### TRAIN SVM WITH RBM AUGMENTATION #########################
python3 main_script_multiclass_svm_analysis.py  --mode='train_svm_aug'\
                                                                          --svm-params-path='MultiSVMParams/RBM2'\
                                                                          --c=32 --g=.5 --norm-type='min_max_norm'\
                                                                          --data-save-path='TrainData/KDDTrain+_20Percent/'\
                                                                          --train-file-name='discriminative_kdd_nsl_processed_aug_rbm.npz'


################################### TEST RBM AUGMENTED SVM ##############################################
python3 main_script_multiclass_svm_analysis.py  --mode='test_svm_aug' --svm-params-path='MultiSVMParams/RBM2'\
                                                                          --svm-analysis-path='MultiSVMAnalysis/RBM2'\
                                                                          --norm-type='min_max_norm'\
                                                                          --data-save-path='TrainData/KDDTrain+_20Percent/'\
                                                                          --train-file-name='discriminative_kdd_nsl_processed_aug_rbm.npz'
END

:<<END
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
END

:<<END
################################### TRAIN SVM WITH RBM FEATURES #########################
python3 main_script_multiclass_svm_analysis.py  --mode='train_svm_features'\
                                                                          --svm-params-path='MultiSVMParams/RBMfeatures'\
                                                                          --c=2 --g=.5 --norm-type='min_max_norm'\
                                                                          --data-save-path='TrainData/KDDTrain+_20Percent/'\
                                                                          --train-file-name='discriminative_kdd_nsl_processed_features.npz'
END

################################### TEST RBM FEATURES ##############################################
python3 main_script_multiclass_svm_analysis.py  --mode='test_svm_features_rbm' --svm-params-path='MultiSVMParams/RBMfeatures'\
                                                                          --svm-analysis-path='MultiSVMAnalysis/RBMfeatures'\
                                                                          --norm-type='min_max_norm'\
                                                                          --rbm-train-type='bbrbm_pcd'\
                                                                          --sample-ites=1\
                                                                          --data-save-path='TrainData/KDDTrain+_20Percent/'\
                                                                          --train-file-name='discriminative_kdd_nsl_processed_features.npz'
