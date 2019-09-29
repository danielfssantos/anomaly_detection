#!/bin/bash
#!/usr/bin/env python


################################### TRAIN SVM WITHOUT AUGMENTATION #########################
python3 main_script_multiclass_svm_analysis.py  --mode='train_svm' --svm-params-path='MultiSVMParams'\
                                                                          --c=8 --g=.5 --norm-type='min_max_norm'\
                                                                          --data-save-path='TrainData/KDDTrain+_20Percent/'\
                                                                          --train-file-name='discriminative_kdd_nsl_processed.npz'



################################### TEST NOT AUGMENTED SVM ##############################################
python3 main_script_multiclass_svm_analysis.py  --mode='test_svm' --svm-params-path='MultiSVMParams' --norm-type='min_max_norm'\
                                                                  --data-save-path='TrainData/KDDTrain+_20Percent/'\
                                                                  --train-file-name='discriminative_kdd_nsl_processed.npz'
:<<END


################################### TRAIN SVM WITH AUGMENTATION #########################
python3 main_script_multiclass_svm_analysis.py  --mode='train_svm_aug' --svm-params-path='MultiSVMParams'\
                                                                          --c=1024 --g=1.0 --norm-type='min_max_norm'\
                                                                          --data-save-path='TrainData/KDDTrain+_20Percent/'\
                                                                          --train-file-name='discriminative_kdd_nsl_processed_aug.npz'



################################### TEST AUGMENTED SVM ##############################################
python3 main_script_multiclass_svm_analysis.py  --mode='test_svm_aug' --svm-params-path='MultiSVMParams' --norm-type='min_max_norm'\
                                                                  --data-save-path='TrainData/KDDTrain+_20Percent/'\
                                                                  --train-file-name='discriminative_kdd_nsl_processed_aug.npz'
END

