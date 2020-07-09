#!/bin/bash
#!/usr/bin/env python

for DATASET in 'KDDTrain+' 'KDDTrain+_20Percent'; do
       ################################### TRAIN RANDOM FOREST WITHOUT PROJECTED DATA ##############################################
       python3 randforest_detector.py  --mode='train_randforest'\
                                                 --randforest-params-path='Results/RANDFORESTParams/'$DATASET'/WithoutProjection'\
                                                 --gen-dataset=1\
                                                 --n-estimators=1400 --max-features='auto'\
                                                 --max-depth=100\
                                                 --data-save-path='Results/TrainData/'$DATASET'/'\
                                                 --train-file-name='discriminative_kdd_nsl_processed.npz'\
                                                 --train-data-file-path='./NSL_KDD_Dataset/'$DATASET'.txt'
       ################################### TEST RANDOM FOREST WITHOUT PROJECTED DATA ##############################################
       python3 randforest_detector.py  --mode='test_randforest'\
                                                 --randforest-params-path='Results/RANDFORESTParams/'$DATASET'/WithoutProjection'\
                                                 --data-save-path='Results/TrainData/'$DATASET'/'\
                                                 --train-file-name='discriminative_kdd_nsl_processed.npz'\
                                                 --randforest-analysis-path='./Results/RANDFORESTAnalysis/'$DATASET'/WithoutProjection'\
                                                 --train-data-file-path='./NSL_KDD_Dataset/'$DATASET'.txt'
       ################ FULL EXPERIMENT USING RBMs 10 20 38 80 100 ####################
       for n in 10 20 38 80 100; do
              if [ $n -eq 10 ]; then
                     n_est=1000
                     total_feat='auto'
                     depth=300
              elif [ $n -eq 20 ]; then
                     n_est=1400
                     total_feat='auto'
                     depth=100
              elif [ $n -eq 38 ]; then
                     n_est=600
                     total_feat='sqrt'
                     depth=420
              elif [ $n -eq 80 ]; then
                     n_est=1800
                     total_feat='auto'
                     depth=420
              elif [ $n -eq 100 ]; then
                     n_est=1000
                     total_feat='auto'
                     depth=300
              fi
       ################################### TRAIN RANDOM FOREST WITH PROJECTED DATA ##############################################
       python3 randforest_detector.py  --mode='train_randforest_rbm_proj'\
                                                 --gen-dataset=1\
                                                 --n-estimators=$n_est --max-features=$total_feat\
                                                 --max-depth=$depth\
                                                 --randforest-params-path='Results/RANDFORESTParams/'$DATASET'/WithProjection/RBM'$n'neurons'\
                                                 --data-sampler-params-path='./Results/RBMParams/'$DATASET'/'$n'neurons'\
                                                 --train-file-name=$n'_projected_kdd_nsl.npz'\
                                                 --data-save-path='Results/TrainData/'$DATASET'/RBMProjected'\
                                                 --train-data-file-path='./NSL_KDD_Dataset/'$DATASET'.txt'

       ################################### TEST RANDOM FOREST WITH PROJECTED DATA ##############################################
       python3 randforest_detector.py  --mode='test_randforest_rbm_proj'\
                                                 --gen-dataset=0\
                                                 --randforest-params-path='Results/RANDFORESTParams/'$DATASET'/WithProjection/RBM'$n'neurons'\
                                                 --data-sampler-params-path='./Results/RBMParams/'$DATASET'/'$n'neurons'\
                                                 --randforest-analysis-path='./Results/RANDFORESTAnalysis/'$DATASET'/WithProjection/RBM'$n'neurons'\
                                                 --train-file-name=$n'_projected_kdd_nsl.npz'\
                                                 --data-save-path='Results/TrainData/'$DATASET'/RBMProjected'\
                                                 --train-data-file-path='./NSL_KDD_Dataset/'$DATASET'.txt'
       done
done