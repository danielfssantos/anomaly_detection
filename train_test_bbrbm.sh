#!/bin/bash
#!/usr/bin/env python

################################### NOT ATTACK ###############################################
:<<END
################################### TRAIN BBRBM ##############################################
python3 main_script_generative_rbm.py  --mode='train' --train-type='bbrbm_pcd_normal' --num-epochs=300 --cd-steps=1 --num-hid-nodes=500\
                                       --gen-dataset=1 --sample-visdata=0 --epsilonw=0.01 --epsilonhb=0.01 --epsilonvb=0.01\
                                       --batch-sz=100 --weightcost=0.0 --initialmomentum=0.0 --finalmomentum=0.0
END

################################### TEST BBRBM ###############################################
python3 main_script_generative_rbm.py  --mode='test' --gen-dataset=0 --train-type='bbrbm_pcd_normal'\
                                       --batch-sz=100 --sample-visdata=0 --sample-ites=500 --sample-data-repetitions=10


################################### DOS ATTACK ###############################################
:<<END
################################### TRAIN BBRBM ##############################################
python3 main_script_generative_rbm.py  --mode='train' --train-type='bbrbm_pcd_dos' --num-epochs=300 --cd-steps=1 --num-hid-nodes=500\
                                       --gen-dataset=0 --sample-visdata=0 --epsilonw=0.001 --epsilonhb=0.001 --epsilonvb=0.001\
                                       --batch-sz=100 --weightcost=0.0 --initialmomentum=0.0 --finalmomentum=0.0
END
################################### TEST BBRBM ###############################################
python3 main_script_generative_rbm.py  --mode='test' --gen-dataset=0 --train-type='bbrbm_pcd_dos'\
                                       --batch-sz=100 --sample-visdata=0 --sample-ites=500 --sample-data-repetitions=10



#################################### U2R ATTACK ##############################################
:<<END
################################### TRAIN BBRBM  #############################################
python3 main_script_generative_rbm.py  --mode='train' --train-type='bbrbm_pcd_u2r' --num-epochs=300 --cd-steps=1 --num-hid-nodes=500\
                                       --gen-dataset=0 --sample-visdata=0 --epsilonw=0.005 --epsilonhb=0.005 --epsilonvb=0.005\
                                       --batch-sz=5 --weightcost=0.0 --initialmomentum=0.0 --finalmomentum=0.0
END
################################### TEST BBRBM ################################################
python3 main_script_generative_rbm.py  --mode='test' --gen-dataset=0 --train-type='bbrbm_pcd_u2r'\
                                       --batch-sz=50 --sample-visdata=0 --sample-ites=500 --sample-data-repetitions=10



#################################### R2L ATTACK ###############################################
:<<END
################################### TRAIN BBRBM  ##############################################
python3 main_script_generative_rbm.py  --mode='train' --train-type='bbrbm_pcd_r2l' --num-epochs=300 --cd-steps=1 --num-hid-nodes=500\
                                       --gen-dataset=0 --sample-visdata=0 --epsilonw=0.005 --epsilonhb=0.005 --epsilonvb=0.005\
                                       --batch-sz=100 --weightcost=0.0 --initialmomentum=0.0 --finalmomentum=0.0
END
################################### TEST BBRBM ################################################
python3 main_script_generative_rbm.py  --mode='test' --gen-dataset=0 --train-type='bbrbm_pcd_r2l'\
                                       --batch-sz=100 --sample-visdata=0 --sample-ites=500 --sample-data-repetitions=10



#################################### PROBE ATTACK #############################################
:<<END
#################################### TRAIN BBRBM  #############################################
python3 main_script_generative_rbm.py  --mode='train' --train-type='bbrbm_pcd_probe' --num-epochs=300 --cd-steps=1 --num-hid-nodes=500\
                                       --gen-dataset=0 --sample-visdata=0 --epsilonw=0.005 --epsilonhb=0.005 --epsilonvb=0.005\
                                       --batch-sz=100 --weightcost=0.0 --initialmomentum=0.0 --finalmomentum=0.0
END

#################################### TEST BBRBM ################################################
python3 main_script_generative_rbm.py  --mode='test' --gen-dataset=0 --train-type='bbrbm_pcd_probe'\
                                       --batch-sz=100 --sample-visdata=0 --sample-ites=100 --sample-data-repetitions=10
