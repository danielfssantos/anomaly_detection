#!/bin/bash
#!/usr/bin/env python

RBM_PARAMS_PATH='./RBMParams/KDDTrain+_20Percent/10neurons'
RBM_ANALYSIS_PATH='./SamplingAnalysis/KDDTrain+_20Percent/RBM/10neurons'
NEURONS=10

:<<END
################################### NOT ATTACK ###############################################
################################### TRAIN BBRBM ##############################################
python3 main_script_generative_rbm.py  --mode='train' --rbm-train-type='bbrbm_pcd_normal' --num-epochs=300\
                                       --cd-steps=1 --num-hid-nodes=$NEURONS --rbm-params-path=$RBM_PARAMS_PATH\
                                       --gen-dataset=0 --sample-visdata=0 --epsilonw=0.01 --epsilonhb=0.01 --epsilonvb=0.01\
                                       --batch-sz=100 --weightcost=0.0 --initialmomentum=0.0 --finalmomentum=0.0
END


################################### TEST BBRBM ###############################################
python3 main_script_generative_rbm.py  --mode='gen_features' --gen-dataset=0 --rbm-train-type='bbrbm_pcd_normal'\
                                       --batch-sz=100 --sample-visdata=0 --sample-ites=500 --sample-data-repetitions=10\
                                       --rbm-params-path=$RBM_PARAMS_PATH --results-features-path=$RBM_ANALYSIS_PATH


:<<END
################################### DOS ATTACK ###############################################
################################### TRAIN BBRBM ##############################################
python3 main_script_generative_rbm.py  --mode='train' --rbm-train-type='bbrbm_pcd_dos' --num-epochs=300\
                                       --cd-steps=1 --num-hid-nodes=$NEURONS --rbm-params-path=$RBM_PARAMS_PATH\
                                       --gen-dataset=0 --sample-visdata=0 --epsilonw=0.001 --epsilonhb=0.001 --epsilonvb=0.001\
                                       --batch-sz=100 --weightcost=0.0 --initialmomentum=0.0 --finalmomentum=0.0
END


################################### TEST BBRBM ###############################################
python3 main_script_generative_rbm.py  --mode='gen_features' --gen-dataset=0 --rbm-train-type='bbrbm_pcd_dos'\
                                       --batch-sz=100 --sample-visdata=0 --sample-ites=500 --sample-data-repetitions=10\
                                       --rbm-params-path=$RBM_PARAMS_PATH --results-features-path=$RBM_ANALYSIS_PATH

:<<END
#################################### U2R ATTACK ##############################################
################################### TRAIN BBRBM  #############################################
python3 main_script_generative_rbm.py  --mode='train' --rbm-train-type='bbrbm_pcd_u2r' --num-epochs=300\
                                       --cd-steps=1 --num-hid-nodes=$NEURONS --rbm-params-path=$RBM_PARAMS_PATH\
                                       --gen-dataset=1 --sample-visdata=0 --epsilonw=0.005 --epsilonhb=0.005 --epsilonvb=0.005\
                                       --batch-sz=5 --weightcost=0.0 --initialmomentum=0.0 --finalmomentum=0.0

END


################################### TEST BBRBM ################################################
python3 main_script_generative_rbm.py  --mode='gen_features' --gen-dataset=0 --rbm-train-type='bbrbm_pcd_u2r'\
                                       --batch-sz=5 --sample-visdata=0 --sample-ites=500 --sample-data-repetitions=10\
                                       --rbm-params-path=$RBM_PARAMS_PATH --results-features-path=$RBM_ANALYSIS_PATH

:<<END
#################################### R2L ATTACK ###############################################
################################### TRAIN BBRBM  ##############################################
python3 main_script_generative_rbm.py  --mode='train' --rbm-train-type='bbrbm_pcd_r2l' --num-epochs=300\
                                       --cd-steps=1 --num-hid-nodes=$NEURONS --rbm-params-path=$RBM_PARAMS_PATH\
                                       --gen-dataset=0 --sample-visdata=0 --epsilonw=0.005 --epsilonhb=0.005 --epsilonvb=0.005\
                                       --batch-sz=100 --weightcost=0.0 --initialmomentum=0.0 --finalmomentum=0.0
END


################################### TEST BBRBM ################################################
python3 main_script_generative_rbm.py  --mode='gen_features' --gen-dataset=0 --rbm-train-type='bbrbm_pcd_r2l'\
                                       --batch-sz=100 --sample-visdata=0 --sample-ites=500 --sample-data-repetitions=10\
                                       --rbm-params-path=$RBM_PARAMS_PATH --results-features-path=$RBM_ANALYSIS_PATH


:<<END
#################################### PROBE ATTACK #############################################
#################################### TRAIN BBRBM  #############################################
python3 main_script_generative_rbm.py  --mode='train' --rbm-train-type='bbrbm_pcd_probe' --num-epochs=300\
                                       --cd-steps=1 --num-hid-nodes=$NEURONS --rbm-params-path=$RBM_PARAMS_PATH\
                                       --gen-dataset=0 --sample-visdata=0 --epsilonw=0.005 --epsilonhb=0.005 --epsilonvb=0.005\
                                       --batch-sz=100 --weightcost=0.0 --initialmomentum=0.0 --finalmomentum=0.0
END


#################################### TEST BBRBM ################################################
python3 main_script_generative_rbm.py  --mode='gen_features' --gen-dataset=0 --rbm-train-type='bbrbm_pcd_probe'\
                                       --batch-sz=100 --sample-visdata=0 --sample-ites=100 --sample-data-repetitions=10\
                                       --rbm-params-path=$RBM_PARAMS_PATH --results-features-path=$RBM_ANALYSIS_PATH
