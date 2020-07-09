#!/bin/bash
#!/usr/bin/env python

for DATASET in 'KDDTrain+' 'KDDTrain+_20Percent'; do
       ################################### TRAIN DTREE WITHOUT PROJECTED DATA ##############################################
       python3 dtree_detector.py --mode='train_dtree'\
                                                 --dtree-params-path='Results/DTREEParams/'$DATASET'/WithoutProjection'\
                                                 --gen-dataset=1\
                                                 --max-features='sqrt'\
                                                 --max-depth=100\
                                                 --data-save-path='Results/TrainData/'$DATASET'/'\
                                                 --train-file-name='discriminative_kdd_nsl_processed.npz'\
                                                 --train-data-file-path='./NSL_KDD_Dataset/'$DATASET'.txt'
       ################################### TEST DTREE WITHOUT PROJECTED DATA ##############################################
       python3 dtree_detector.py  --mode='test_dtree'\
                                                 --dtree-params-path='Results/DTREEParams/'$DATASET'/WithoutProjection'\
                                                 --data-save-path='Results/TrainData/'$DATASET'/'\
                                                 --train-file-name='discriminative_kdd_nsl_processed.npz'\
                                                 --dtree-analysis-path='./Results/DTREEAnalysis/'$DATASET'/WithoutProjection'\
                                                 --train-data-file-path='./NSL_KDD_Dataset/'$DATASET'.txt'
       ################ FULL EXPERIMENT USING RBMs 10 20 38 80 100 ####################
       for n in 10 20 38 80 100; do
              if [ $n -eq 10 ]; then
                     total_feat='sqrt'
                     depth=420
              elif [ $n -eq 20 ]; then
                     total_feat='auto'
                     depth=140
              elif [ $n -eq 38 ]; then
                     total_feat='sqrt'
                     depth=140
              elif [ $n -eq 80 ]; then
                     total_feat='sqrt'
                     depth=500
              elif [ $n -eq 100 ]; then
                     total_feat='sqrt'
                     depth=100
              fi
              ################################### TRAIN DTREE WITH PROJECTED DATA ##############################################
              python3 dtree_detector.py  --mode='train_dtree_rbm_proj'\
                                                        --gen-dataset=1\
                                                        --max-features=$total_feat\
                                                        --max-depth=$depth\
                                                        --dtree-params-path='Results/DTREEParams/'$DATASET'/WithProjection/RBM'$n'neurons'\
                                                        --data-sampler-params-path='./Results/RBMParams/'$DATASET'/'$n'neurons'\
                                                        --train-file-name=$n'_projected_kdd_nsl.npz'\
                                                        --data-save-path='Results/TrainData/'$DATASET'/RBMProjected'\
                                                        --train-data-file-path='./NSL_KDD_Dataset/'$DATASET'.txt'
              ################################### TEST DTREE WITH PROJECTED DATA ##############################################
              python3 dtree_detector.py  --mode='test_dtree_rbm_proj'\
                                                        --gen-dataset=0\
                                                        --dtree-params-path='Results/DTREEParams/'$DATASET'/WithProjection/RBM'$n'neurons'\
                                                        --data-sampler-params-path='./RBMParams/'$DATASET'/'$n'neurons'\
                                                        --dtree-analysis-path='./Results/DTREEAnalysis/'$DATASET'/WithProjection/RBM'$n'neurons'\
                                                        --train-file-name=$n'_projected_kdd_nsl.npz'\
                                                        --data-save-path='Results/TrainData/'$DATASET'/RBMProjected'\
                                                        --train-data-file-path='./NSL_KDD_Dataset/'$DATASET'.txt'
       done
done
