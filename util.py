import numpy as np
import math, os
from operator import itemgetter
# https://seaborn.pydata.org/tutorial/distributions.html
import seaborn as sns
import matplotlib.pyplot as plt
import os, sys, operator
# Download libsvm from:
#             http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/libsvm.cgi?+http://www.csie.ntu.edu.tw/~cjlin/libsvm+tar.gz
# Change the svm path according to your respective libsvm path location
# Obs: Remember to compyle the lib first
sys.path.append('/home/daniel/Documents/DeepLearningOpenCV/libsvm-3.23/python')
from svmutil import *
from rbm import RBM
from sklearn.metrics.pairwise import euclidean_distances

def plot_kde_distributions(x, x_sampled, attack_type, plot_path):
    sns.kdeplot(np.mean(x_sampled, axis=0),  shade=True, label="BBRBM Sampled Data", color="g");
    ax = sns.kdeplot(np.mean(x, axis=0),  shade=True, label="True Data", color="r");
    ax.set_title('{:s} Distributions'.format(attack_type))
    plt.savefig(os.path.join(plot_path, attack_type + 'Distributions.png'))
    return

def z_norm(data):
    mean = np.mean(data, axis=0, keepdims=True)
    std = (np.std(data, axis=0, keepdims=True) + 1e-8)
    return ((data - mean)/std, mean, std)

def min_max_norm(data):
    return (data - np.amin(data, axis=0, keepdims=True))/ \
            (1e-8 + np.amax(data, axis=0, keepdims=True) - np.amin(data, axis=0, keepdims=True))

def matrix_to_batches(input_data, batch_sz=100):
	numcases = math.ceil(input_data.shape[0]/batch_sz)
	if input_data.shape[0] % batch_sz:
		multiple_div = False
	else:
		multiple_div = True
	batchdata = []
	for i in range(numcases):
		if i == numcases - 1 and not multiple_div:
			batchdata.append(input_data[i * batch_sz :, :])
		else:
			batchdata.append(input_data[i * batch_sz : i * batch_sz + batch_sz, :])
	return batchdata

def load_nsl_kdd_dataset(data_file_path):
    data = []
    labels = []

    # Read the dataset.txt file
    with open(data_file_path, 'r') as f:
        train_file_content = f.readlines()
    train_file_lines = [line.replace('\n', '').split() for line in train_file_content]
    for line in train_file_lines:
        line = line[0].split(',')
        line.pop() # Remove the sample difficult level information
        del(line[1 : 4]) # Ignore not numeric features
        # Use attacks_dict to generate sample label in hot vector representation
        if line[-1] == 'normal':
            labels.append(-1)
        else:
            labels.append(1)
        line.pop() # Remove label data information
        # Convert remaining line data to numeric float
        data.append(list(map(float, line)))
    return (np.array(data), np.array(labels))

def load_nsl_kdd_splitted_dataset(data_file_path, attacks_file_path):
    # Train data formats
    normal_data = []
    dos_data = []
    u2r_data = []
    r2l_data = []
    probe_data = []
    # Read the attacks dictionary
    with open(attacks_file_path, 'r') as f:
        attacks_file_content = f.readlines()
    attacks_file_lines = [line.replace('\n', '').split() for line in attacks_file_content]
    if attacks_file_lines[-1] == []:
        attacks_file_lines.pop()
    attacks_dict = dict(attacks_file_lines)
    # Read the dataset.txt file
    with open(data_file_path, 'r') as f:
        train_file_content = f.readlines()
    train_file_lines = [line.replace('\n', '').split() for line in train_file_content]
    for line in train_file_lines:
        line = line[0].split(',')
        line.pop() # Remove the sample difficult level information
        del(line[1 : 4]) # Ignore not numeric features
        # Use attacks_dict to generate sample label in hot vector representation
        # and convert remaining line data to numeric float.
        if line[-1] == 'normal':
            line.pop() # Remove label data information
            normal_data.append(list(map(float, line)))
        elif attacks_dict[line[-1]] == 'u2r':
            line.pop()
            u2r_data.append(list(map(float, line)))
        elif attacks_dict[line[-1]] == 'r2l':
            line.pop()
            r2l_data.append(list(map(float, line)))
        elif attacks_dict[line[-1]] == 'dos':
            line.pop()
            dos_data.append(list(map(float, line)))
        elif attacks_dict[line[-1]] == 'probe':
            line.pop()
            probe_data.append(list(map(float, line)))
    return [np.array(normal_data), np.array(u2r_data), np.array(r2l_data),\
                np.array(dos_data), np.array(probe_data)]

# Code from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/#roc_curve_for_binary_svm
def generate_roc(deci, label, roc_path):
    #count of postive and negative labels
    db = []
    pos, neg = 0, 0
    for i in range(len(label)):
        if label[i]>0:
            pos+=1
        else:
            neg+=1
        db.append([deci[i], label[i]])

    #sorting by decision value
    db = sorted(db, key=itemgetter(0), reverse=True)

    #calculate ROC
    xy_arr = []
    tp, fp = 0., 0.         #assure float division
    for i in range(len(db)):
        if db[i][1]>0:      #positive
            tp+=1
        else:
            fp+=1
        xy_arr.append([fp/neg,tp/pos])

    #area under curve
    auc = 0.
    prev_x = 0
    for x,y in xy_arr:
        if x != prev_x:
            auc += (x - prev_x) * y
            prev_x = x

    #also write to file
    xy_arr = np.array(xy_arr)
    plt.plot(xy_arr[:, 0], xy_arr[:, 1], '-r')
    plt.title('ROC curve of NSL-KDD Test+ AUC: {:.4f}'.format(auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim(left=0, right=1)
    plt.ylim(bottom=0, top=1)
    plt.savefig(roc_path)

def generate_error_bars(qnt_attacks, labels, pred_labels, bars_path):
    prev_qnt = 0
    attack_names = {0 : 'normal', 1 : 'u2r', 2 : 'r2l', 3 : 'dos', 4 : 'probe'}
    wrong_results = {}

    for i, qnt in enumerate(qnt_attacks):
        wrong_results[attack_names[i]] = np.sum(labels[prev_qnt : prev_qnt + qnt] != np.array(pred_labels[prev_qnt : prev_qnt + qnt]))
        wrong_results[attack_names[i]] /= qnt
        prev_qnt = qnt

    wrong_results = sorted(wrong_results.items(), key=operator.itemgetter(1))
    names = [result[0] for result in wrong_results]
    values = [result[1] for result in wrong_results]
    #names = list(wrong_results.keys())
    #values = list(wrong_results.values())
    fig, ax = plt.subplots()
    plt.bar(range(5), values)
    plt.xticks(range(5), names)
    plt.ylabel('Percentage of missclassification')
    plt.savefig(bars_path)

def train_load_oc_svm(datum_train, attack_names, svm_model_path):
    svm_model = svm_load_model(os.path.join(svm_model_path, 'oc_svm_model_aug.txt'))
    if svm_model is None:
        print('Training OC-SVM for class {:s}'.format(attack_names[0]))
        label = np.ones((len(datum_train[0]),))
        svm_model = svm_train(label, np.vstack(datum_train[0]), '-s 2 -t 2  -n 0.5' )
        svm_save_model(os.path.join(svm_model_path, 'oc_svm_model_aug.txt'), svm_model)
    #pred_labels, evals, deci  = svm_predict(labels_test, data_test, svm_model, '-q')
    #print('Test Acc: {:.4f}'.format(evals[0]))
    return svm_model


def augment_dataset(args):
    datum_train = load_nsl_kdd_splitted_dataset(args.train_data_file_path, args.metadata_file_path)
    args.num_vis_nodes = datum_train[0].shape[1]
    attack_names = {0 : 'normal', 1 : 'u2r', 2 : 'r2l', 3 : 'dos', 4 : 'probe'}

    if args.use_oc_svm:
        oc_svm_model = train_load_oc_svm(datum_train, attack_names, args.svm_params_path)
        # Remove from normal class all samples wrong classified by the oc_svm
        #pred_labels, evals, deci  = svm_predict(np.ones((len(datum_train[0]),)), np.vstack(datum_train[0]), oc_svm_model, '-q')
        #datum_train[0] = [datum_train[0][i] for i in range(len(datum_train[0])) if pred_labels[i] == 1]

    attacks_biggest_size = len(datum_train[0])
    batch_sz = args.batch_sz
    sampled_datum_train = []
    rbm_train_type = args.rbm_train_type
    for i in range(len(datum_train)):
        if i > 0:
            args.rbm_train_type = rbm_train_type + '_' + attack_names[i]
            # Instatiate BBRBM
            rbm_model = RBM(args)
            rbm_model.load(args.rbm_params_path)
            print('Sampling data from {:s} BBRBM\n'.format(attack_names[i]))
            attack_train_data = np.array(datum_train[i])
            qnt_to_sample = attacks_biggest_size - attack_train_data.shape[0]
            qnt_valid_samples = 0
            sampled_data_train = np.array([])
            while(qnt_valid_samples < qnt_to_sample):
                if args.rbm_train_type.find('bbrbm') != -1:
                    sampled_data = np.random.randint(low=0, high=2, size=(batch_sz, rbm_model.numdims))
                elif args.rbm_train_type.find('bbrbm') != -1:
                    sampled_data = np.random.rand(batch_sz, rbm_model.numdims)
                elif args.rbm_train_type.find('gbrbm') != -1:
                    sampled_data = np.random.randn(batch_sz, rbm_model.numdims)
                sampled_data = rbm_model.sample_data(sampled_data, ites=args.sample_ites)
                if args.use_oc_svm:
                    if i == 0:
                        pred_labels, evals, deci  = svm_predict(np.ones((batch_sz,)), sampled_data, oc_svm_model, '-q')
                        pred_labels = np.array(pred_labels)
                        sampled_data = sampled_data[np.where(pred_labels == 1)[0], :]
                    else:
                        pred_labels, evals, deci  = svm_predict(-1 * np.ones((batch_sz,)), sampled_data, oc_svm_model, '-q')
                        pred_labels = np.array(pred_labels)
                        sampled_data = sampled_data[np.where(pred_labels != 1)[0], :]
                if sampled_data.size:
                    if not sampled_data_train.size:
                        sampled_data_train = sampled_data
                    else:
                        sampled_data_train = np.concatenate((sampled_data_train, sampled_data), axis=0)
                    if sampled_data_train.size > 1:
                        # Use pairwise euclidian distance to avoid redundancies in sampled data
                        samples_distances = np.mean(euclidean_distances(sampled_data_train), axis=0)
                        sampled_data_train = sampled_data_train[np.where(samples_distances >= 0.8)[0], :]
                    qnt_valid_samples = sampled_data_train.shape[0]
                    print('{:d} collected samples from {:d}'.format(qnt_valid_samples, qnt_to_sample))
            if sampled_data_train.size:
                sampled_datum_train.append(np.vstack(sampled_data_train)[0 : qnt_to_sample, :])

    sampled_data_train = np.vstack(sampled_datum_train).reshape(-1, args.num_vis_nodes)

    data_train = np.vstack(datum_train)
    data_train = np.concatenate((data_train, sampled_data_train), axis=0)
    labels_train = -1 * np.ones((len(datum_train[0]),))
    labels_train = np.concatenate( (labels_train, np.ones( (data_train.shape[0] - len(datum_train[0]), ) ) ), axis=0)
    return data_train, labels_train



