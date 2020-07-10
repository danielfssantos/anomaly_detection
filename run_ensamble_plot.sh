#!/bin/bash
#!/usr/bin/env python

for DATASET in 'KDDTrain+' 'KDDTrain+_20Percent'; do
    ######################## GEN KDEs #################################
    python3 ensamble_results.py  --mode='kde' --exp-ids 10 20 38 80 100\
                                 --samples-path='./Results/SamplingAnalysis/'$DATASET
    ######################## GEN PCA 2D PROJECTIONS ########################
    python3 ensamble_results.py  --mode='scatter_pca' --exp-ids 10 20 38 80 100\
                                 --samples-path='./Results/SamplingAnalysis/'$DATASET
    ######################## GEN ROCs ##################################
    for tech in 'SVM' 'DTREE' 'RANDFOREST'; do
        python3 ensamble_results.py  --mode='roc' --exp-ids 10 20 38 80 100\
                                     --results-path='./Results/'$tech'Analysis/'$DATASET
    done
done
