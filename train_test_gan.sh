#!/bin/bash
#!/usr/bin/env python


################################### NOT ATTACK ###############################################
################################### TRAIN GAN ##############################################
python3 main_script_gan.py  --mode='train' --gan-train-type='gan_normal' --num-epochs=300\
                                            --gen-dataset=0 --lrn-rate=0.0001 --batch-sz=128 --verbose=3\
                                            --data-sampler-params-path='./GANParams/KDDTrain+_20Percent/normal'

################################### TEST GAN ###############################################
python3 main_script_gan.py  --mode='test' --gen-dataset=0 --gan-train-type='gan_normal'\
                                           --batch-sz=128 --sample-data-repetitions=10\
                                           --data-sampler-params-path='./GANParams/KDDTrain+_20Percent/normal'


################################### DOS ATTACK ###############################################
################################### TRAIN GAN ##############################################
python3 main_script_gan.py  --mode='train' --gan-train-type='gan_dos' --num-epochs=300\
                                            --gen-dataset=0 --lrn-rate=0.0001 --batch-sz=16 --verbose=3\
                                            --data-sampler-params-path='./GANParams/KDDTrain+_20Percent/dos'

################################### TEST GAN ###############################################
python3 main_script_gan.py  --mode='test' --gen-dataset=0 --gan-train-type='gan_dos'\
                                           --batch-sz=128 --sample-data-repetitions=10\
                                           --data-sampler-params-path='./GANParams/KDDTrain+_20Percent/dos'



################################### U2R ATTACK ###############################################
################################### TRAIN GAN ##############################################
python3 main_script_gan.py  --mode='train' --gan-train-type='gan_u2r' --num-epochs=300\
                                            --gen-dataset=0 --lrn-rate=0.0001 --batch-sz=8 --verbose=3\
                                            --data-sampler-params-path='./GANParams/KDDTrain+_20Percent/u2r'
################################### TEST GAN ###############################################
python3 main_script_gan.py  --mode='test' --gen-dataset=0 --gan-train-type='gan_u2r'\
                                           --batch-sz=128 --sample-data-repetitions=10\
                                           --data-sampler-params-path='./GANParams/KDDTrain+_20Percent/u2r'



################################### R2L ATTACK ###############################################
################################### TRAIN GAN ##############################################
python3 main_script_gan.py  --mode='train' --gan-train-type='gan_r2l' --num-epochs=300\
                                            --gen-dataset=0 --lrn-rate=0.0001 --batch-sz=16 --verbose=3\
                                            --data-sampler-params-path='./GANParams/KDDTrain+_20Percent/r2l'

################################### TEST GAN ###############################################
python3 main_script_gan.py  --mode='test' --gen-dataset=0 --gan-train-type='gan_r2l'\
                                           --batch-sz=128 --sample-data-repetitions=10\
                                           --data-sampler-params-path='./GANParams/KDDTrain+_20Percent/r2l'



################################### PROBE ATTACK ###############################################
################################### TRAIN GAN ##############################################
python3 main_script_gan.py  --mode='train' --gan-train-type='gan_probe' --num-epochs=300\
                                            --gen-dataset=0 --lrn-rate=0.0001 --batch-sz=16 --verbose=3\
                                            --data-sampler-params-path='./GANParams/KDDTrain+_20Percent/probe'

################################### TEST GAN ###############################################
python3 main_script_gan.py  --mode='test' --gen-dataset=0 --gan-train-type='gan_probe'\
                                           --batch-sz=128 --sample-data-repetitions=10\
                                           --data-sampler-params-path='./GANParams/KDDTrain+_20Percent/probe'
