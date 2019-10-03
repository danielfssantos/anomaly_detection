#!/bin/bash
#!/usr/bin/env python

################################### NOT ATTACK ###############################################
################################### FEATURES BBRBM ###############################################
python3 main_script_generative_rbm.py  --mode='gen_features' --gen-dataset=0 --rbm-train-type='bbrbm_pcd_normal'\
                                       --batch-sz=100 --sample-visdata=0 --sample-ites=100 --sample-data-repetitions=10\
                                       --results-features-path='./SamplingAnalysis/KDDTrain+_20Percent/RBM/Features'

:<<END
################################### DOS ATTACK ###############################################
################################### FEATURES BBRBM ###############################################
python3 main_script_generative_rbm.py  --mode='gen_features' --gen-dataset=0 --rbm-train-type='bbrbm_pcd_dos'\
                                       --batch-sz=100 --sample-visdata=0 --sample-ites=500 --sample-data-repetitions=10


#################################### U2R ATTACK ##############################################
################################### FEATURES BBRBM ################################################
python3 main_script_generative_rbm.py  --mode='gen_features' --gen-dataset=0 --rbm-train-type='bbrbm_pcd_u2r'\
                                       --batch-sz=5 --sample-visdata=0 --sample-ites=500 --sample-data-repetitions=10


#################################### R2L ATTACK ###############################################
################################### FEATURES BBRBM ################################################
python3 main_script_generative_rbm.py  --mode='gen_features' --gen-dataset=0 --rbm-train-type='bbrbm_pcd_r2l'\
                                       --batch-sz=100 --sample-visdata=0 --sample-ites=500 --sample-data-repetitions=10



#################################### PROBE ATTACK #############################################
#################################### FEATURES BBRBM ################################################
python3 main_script_generative_rbm.py  --mode='gen_features' --gen-dataset=0 --rbm-train-type='bbrbm_pcd_probe'\
                                       --batch-sz=100 --sample-visdata=0 --sample-ites=100 --sample-data-repetitions=10
END