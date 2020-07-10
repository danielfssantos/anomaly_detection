#!/bin/bash
#!/usr/bin/env python


# Train and generate RBM results
printf "\nRunning run_train_test_bbrbm.sh...\n\n"
eval ./train_test_bbrbm.sh


# Train and generate SVM results
printf "\nRunning run_train_test_svm.sh...\n\n"
eval ./train_test_svm.sh


# Train and generate DTREE results
printf "Running run_train_test_dtree.sh...\n\n"
eval ./train_test_dtree.sh


# Train and generate RANDFOREST results
printf "Running run_train_test_randforest.sh...\n\n"
eval ./train_test_randforest.sh


# Plot KDEs, PCA projections and ROCs
printf "Running run_ensamble_plot.sh...\n\n"
eval ./run_ensamble_plot.sh