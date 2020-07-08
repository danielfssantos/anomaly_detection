#!/bin/bash
#!/usr/bin/env python

# Script that train pcd_rbms for each one of the data types [normal, dos, probe, u2r, r2l]
for DATASET in 'KDDTrain+' 'KDDTrain+_20Percent'; do
   for n in 10 20 38 80 100; do
         RBM_PARAMS_PATH='./RBMParams/'$DATASET'/'$n'neurons'
         RBM_ANALYSIS_PATH='./SamplingAnalysis/'$DATASET'/RBM'$n'neurons'
         if [ $n -eq 10 ]; then
            gen_data=1
         else
            gen_data=0
         fi
         # TRAIN RBMs FOR EACH ONE OF THE ATTACK TYPES
         for attk in 'normal' 'dos' 'probe' 'u2r' 'r2l'; do
            if [ $attk = 'normal' ]; then
               epw=0.05
               ephb=0.05
               epvb=0.05
               batch_sz=100
             elif [ $attk = 'dos' ]; then
               epw=0.001
               ephb=0.001
               epvb=0.001
               batch_sz=100
            elif [ $attk = 'u2r' ]; then
               epw=0.005
               ephb=0.005
               epvb=0.005
               batch_sz=1
            elif [ $attk = 'r2l' ]; then
               epw=0.005
               ephb=0.005
               epvb=0.005
               batch_sz=25
             elif [ $attk = 'probe' ]; then
               epw=0.001
               ephb=0.001
               epvb=0.001
               batch_sz=100
             fi
            ################################### TRAIN BBRBM ##############################################
            python3 generative_rbm.py  --mode='train' --rbm-train-type='bbrbm_pcd_'$attk --num-epochs=300\
                                                   --cd-steps=3 --num-hid-nodes=$n --rbm-params-path=$RBM_PARAMS_PATH\
                                                   --gen-dataset=$gen_data --sample-visdata=0 --epsilonw=$epw --epsilonhb=$ephb --epsilonvb=$epvb\
                                                   --batch-sz=$batch_sz --weightcost=0.0 --initialmomentum=0.0 --finalmomentum=0.0\
                                                   --train-data-file-path='./NSL_KDD_Dataset/'$DATASET'.txt'\
                                                   --train-data-save-path='./TrainData/'$DATASET
            ################################### EVALUATE BBRBM SAMPLES ###############################################
            python3 generative_rbm.py  --mode='gen_samples' --gen-dataset=0 --rbm-train-type='bbrbm_pcd_'$attk\
                                                   --batch-sz=$batch_sz --sample-visdata=0 --sample-ites=10 --sample-data-repetitions=10\
                                                   --rbm-params-path=$RBM_PARAMS_PATH --results-samples-path=$RBM_ANALYSIS_PATH/$attk\
                                                   --train-data-file-path='./NSL_KDD_Dataset/'$DATASET'.txt'\
                                                   --train-data-save-path='./TrainData/'$DATASET
            ################################### EVALUATE BBRBM FEATURES###############################################
            python3 generative_rbm.py  --mode='gen_features' --gen-dataset=0 --rbm-train-type='bbrbm_pcd_'$attk\
                                                   --batch-sz=$batch_sz --sample-visdata=0 --sample-ites=1 --sample-data-repetitions=10\
                                                   --rbm-params-path=$RBM_PARAMS_PATH --results-features-path=$RBM_ANALYSIS_PATH/$attk\
                                                   --train-data-file-path='./NSL_KDD_Dataset/'$DATASET'.txt'\
                                                   --train-data-save-path='./TrainData/'$DATASET
         done
         ############################### GENERATE PROJECTION MATRIX #######################
         python3 main_script_generative_rbm.py  --mode='gen_proj_matrix' --rbm-train-type='bbrbm_pcd'\
                                                   --rbm-params-path=$RBM_PARAMS_PATH
   done
done

