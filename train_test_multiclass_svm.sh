#!/bin/bash
#!/usr/bin/env python

:<<END
################################### TRAIN SVM WITHOUT AUGMENTATION #########################
python3 main_script_multiclass_svm_analysis.py  --mode='train_svm' --svm-params-path='MultiSVMParams' --gen-dataset=1\
                                                                          --c=1024 --g=2.0 --norm-type='min_max_norm'\
                                                                          --data-save-path='TrainData/KDDTrain+_20Percent_MultiSVMAnalisys/'\
                                                                          --train-file-name='discriminative_kdd_nsl_processed.npz'
END


################################### TEST NOT AUGMENTED SVM ##############################################
python3 main_script_multiclass_svm_analysis.py  --mode='test_svm' --svm-params-path='MultiSVMParams' --norm-type='min_max_norm'\
                                                                  --data-save-path='TrainData/KDDTrain+_20Percent_MultiSVMAnalisys/'\
                                                                  --train-file-name='discriminative_kdd_nsl_processed.npz' --gen-dataset=0



