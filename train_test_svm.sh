#!/bin/bash
#!/usr/bin/env python

for DATASET in 'KDDTrain+' 'KDDTrain+_20Percent'; do
       ################################### TRAIN SVM WITHOUT PROJECTED DATA ##############################################
       python3 svm_detector.py  --mode='train_svm'\
                                                 --svm-params-path='SVMParams/'$DATASET'/WithoutProjection'\
                                                 --gen-dataset=1\
                                                 --norm-type=''\
                                                 --c=64 --g=0.00006\
                                                 --data-save-path='TrainData/'$DATASET'/'\
                                                 --train-file-name='discriminative_kdd_nsl_processed.npz'\
                                                 --train-data-file-path='./NSL_KDD_Dataset/'$DATASET'.txt'
       ################################### TEST SVM WITHOUT PROJECTED DATA ##############################################
       python3 svm_detector.py  --mode='test_svm'\
                                                 --svm-params-path='SVMParams/'$DATASET'/WithoutProjection'\
                                                 --norm-type=''\
                                                 --data-save-path='TrainData/'$DATASET'/'\
                                                 --train-file-name='discriminative_kdd_nsl_processed.npz'\
                                                 --svm-analysis-path='./SVMAnalysis/'$DATASET'/WithoutProjection'\
                                                 --train-data-file-path='./NSL_KDD_Dataset/'$DATASET'.txt'
       ################ FULL EXPERIMENT USING RBMs 10 20 38 80 100 ####################
       for n in 10 20 38 80 100; do
              if [ $n -eq 10 ]; then
                     c_val=2048
                     g_val=0.125
              elif [ $n -eq 20 ]; then
                     c_val=16384
                     g_val=0.01562
              elif [ $n -eq 38 ]; then
                     c_val=512
                     g_val=0.25
              elif [ $n -eq 80 ]; then
                     c_val=4096
                     g_val=0.0625
              elif [ $n -eq 100 ]; then
                     c_val=256
                     g_val=0.5
              fi
              ################################### TRAIN SVM WITH PROJECTED DATA ##############################################
              python3 svm_detector.py  --mode='train_svm_rbm_proj'\
                                                        --gen-dataset=1 --c=$c_val --g=$g_val\
                                                        --svm-params-path='SVMParams/'$DATASET'/WithProjection/RBM'$n'neurons'\
                                                        --data-sampler-params-path='./RBMParams/'$DATASET'/'$n'neurons'\
                                                        --train-file-name=$n'_projected_kdd_nsl.npz'\
                                                        --data-save-path='TrainData/'$DATASET'/RBMProjected'\
                                                        --train-data-file-path='./NSL_KDD_Dataset/'$DATASET'.txt'
              ################################### TEST SVM WITH PROJECTED DATA ##############################################
              python3 svm_detector.py  --mode='test_svm_rbm_proj'\
                                                        --gen-dataset=0\
                                                        --svm-params-path='SVMParams/'$DATASET'/WithProjection/RBM'$n'neurons'\
                                                        --data-sampler-params-path='./RBMParams/'$DATASET'/'$n'neurons'\
                                                        --svm-analysis-path='./SVMAnalysis/'$DATASET'/WithProjection/RBM'$n'neurons'\
                                                        --train-file-name=$n'_projected_kdd_nsl.npz'\
                                                        --data-save-path='TrainData/'$DATASET'/RBMProjected'\
                                                        --train-data-file-path='./NSL_KDD_Dataset/'$DATASET'.txt'
       done
done


