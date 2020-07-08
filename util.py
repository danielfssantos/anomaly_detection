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
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV

def plot_kde_distributions(x, x_sampled, attack_type, plot_path=''):
    plt.clf()
    #sns.kdeplot(np.mean(x_sampled, axis=0),  shade=True, label=sampler + " Sampled Data", color="g");
    real_mean = np.mean(x, axis=0)
    sampled_mean = np.mean(x_sampled, axis=0)
    sns.kdeplot(sampled_mean,  shade=True, label="Sampled Data", color="g");
    ax = sns.kdeplot(real_mean,  shade=True, label="Real Data", color="r");
    ax.set_title('{:s} Distributions'.format(attack_type))
    if plot_path != '':
        plt.savefig(os.path.join(plot_path, attack_type + 'Distributions.png'))
    else:
        plt.pause(0.0001)
        plt.show(block=False)
    np.save(os.path.join(plot_path,'real_' + attack_type + '_data' '.npy'), real_mean)
    np.save(os.path.join(plot_path,'sampled_' + attack_type + '_data' '.npy'), sampled_mean)
    return

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

def load_splitted_nsl_kdd_dataset(data_file_path, attacks_file_path):
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
    np.save(os.path.join(roc_path, 'roc.npy'), xy_arr)
    plt.plot(xy_arr[:, 0], xy_arr[:, 1], '-r')
    plt.title('ROC curve of NSL-KDD Test+ AUC: {:.4f}'.format(auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim(left=0, right=1)
    plt.ylim(bottom=0, top=1)
    plt.savefig(os.path.join(roc_path, 'roc.png'))

def generate_roc2(deci, label, roc_path):
    fpr, tpr, _ = roc_curve(label, deci)
    roc_auc = auc(fpr, tpr)
    #also write to file
    xy_arr = np.concatenate((fpr[:, np.newaxis], tpr[:, np.newaxis]), axis=1)
    np.save(os.path.join(roc_path, 'roc.npy'), xy_arr)
    plt.plot(xy_arr[:, 0], xy_arr[:, 1], '-r')
    plt.title('ROC curve of NSL-KDD Test+ AUC: {:.4f}'.format(roc_auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim(left=0, right=1)
    plt.ylim(bottom=0, top=1)
    plt.savefig(os.path.join(roc_path, 'roc.png'))


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


def select_valid_samples(datum):
    normal_data = datum[0]
    attack_dict = {0:'normal', 1:'u2r', 2:'r2l', 3:'dos', 4:'probe'}

    normal_data_mean_dists_tmp = np.mean(euclidean_distances(normal_data), axis=0)
    sorted_idxs = normal_data_mean_dists_tmp.argsort()
    qnt_attack_data = 0
    for data in datum[1 :]:
        qnt_attack_data += len(data)

    new_datum = [normal_data[sorted_idxs[0 : qnt_attack_data], :]]
    print('Attack {:s} Selected {:d} from {:d} samples'.format(attack_dict[0],
                                                                qnt_attack_data,
                                                                len(datum[0])))
    #print(np.mean(euclidean_distances(normal_data), axis=0))
    for i in range(1, len(datum)):
        if i == 1:
            new_datum.append(datum[i])
            continue
        attack_data = datum[i]
        normal_attack_mean_dists = np.mean(euclidean_distances(normal_data, attack_data), axis=0)
        #print(normal_attack_mean_dists)
        #input()
        #continue
        #selected_idxs = np.where(normal_attack_mean_dists > 16000)[0]
        selected_idxs = np.where(normal_attack_mean_dists > 0.75)[0]
        print('Attack {:s} Selected {:d} from {:d} samples'.format(attack_dict[i],
                                                                                                attack_data[selected_idxs, :].shape[0],
                                                                                                attack_data.shape[0]))
        new_datum.append(attack_data[selected_idxs, :])
        #print([attack_data[selected_idxs, :].shape[0], attack_data.shape[0]])
        '''
        # Sort distances in descending order
        normal_attack_mean_dists_tmp = np.amax(normal_attack_mean_dists) - normal_attack_mean_dists
        sorted_idxs = normal_attack_mean_dists_tmp.argsort()
        print(normal_attack_mean_dists[sorted_idxs])
        '''
    return new_datum



def gen_features_dataset(args, data_sampler_model):
    if args.mode.find('train') != -1:
        datum_train = load_splitted_nsl_kdd_dataset(args.train_data_file_path, args.metadata_file_path)
    else:
        datum_train = load_splitted_nsl_kdd_dataset(args.test_data_file_path, args.metadata_file_path)
    args.num_vis_nodes = datum_train[0].shape[1]
    attack_names = {0 : 'normal', 1 : 'u2r', 2 : 'r2l', 3 : 'dos', 4 : 'probe'}
    attacks_biggest_size =  len(datum_train[0])
    batch_sz = args.batch_sz
    data_train = []
    datum_features = []
    for i in range(len(datum_train)):
        rbm_train_type_aux = args.rbm_train_type
        if args.mode.find('rbm') != -1:
            args.rbm_train_type = args.rbm_train_type + '_' + attack_names[i]
            data_sampler_model.load(args.data_sampler_params_path)
            print('Sampling data from {:s} BBRBM\n'.format(attack_names[i]))
            args.rbm_train_type = rbm_train_type_aux
        else:
            data_sampler_model.load(args.data_sampler_params_path + '/' + attack_names[i])
            print('Sampling data from {:s} GAN\n'.format(attack_names[i]))
        current_class_features = []
        for j in range(math.ceil(len(datum_train[i])/args.batch_sz)):
            if j != math.ceil(len(datum_train[i])/args.batch_sz) - 1:
                batchdata = datum_train[i][j * args.batch_sz : (j + 1) * args.batch_sz, :]
            else:
                batchdata = datum_train[i][j * args.batch_sz :, :]
            batchdata = min_max_norm(batchdata)
            features = data_sampler_model.sample_data(batchdata,
                                                                                ites=args.sample_ites,
                                                                                return_type='features')
            data_train.append(features)
            current_class_features.append(features)
        datum_features.append(np.vstack(current_class_features))

    data_train = np.vstack(data_train)
    labels_train = -1 * np.ones((len(datum_train[0]),))
    labels_train = np.concatenate( (labels_train, np.ones( (data_train.shape[0] - len(datum_train[0]), ) ) ), axis=0)
    return (data_train, labels_train, datum_features)


def proj_datasets(args, proj_matrix):
    data_train, labels_train = load_nsl_kdd_dataset(args.train_data_file_path)
    data_test, labels_test = load_nsl_kdd_dataset(args.test_data_file_path)

    datum_train = load_splitted_nsl_kdd_dataset(args.train_data_file_path, args.metadata_file_path)
    for i in range(len(datum_train)):
        datum_train[i] = preprocessing.normalize(datum_train[i], norm='l2')
        #print(len(datum_train[i]))

    datum_test = load_splitted_nsl_kdd_dataset(args.test_data_file_path, args.metadata_file_path)
    for i in range(len(datum_test)):
        datum_test[i] = preprocessing.normalize(datum_test[i], norm='l2')

    # Project Train Data
    projected_data = []
    num_batches = math.ceil(data_train.shape[0]/args.batch_sz)
    for j in range(0, num_batches):
        if j == num_batches - 1:
            curr_batchdata = data_train[j * args.batch_sz :, :]
        else:
            curr_batchdata = data_train[j * args.batch_sz : (j + 1) * args.batch_sz, :]
        curr_batchdata = preprocessing.normalize(curr_batchdata, norm='l2')
        curr_batchdata = np.concatenate((curr_batchdata, np.ones((curr_batchdata.shape[0], 1))), axis=1)
        curr_batchdata = np.dot(curr_batchdata, proj_matrix)
        projected_data.append(curr_batchdata)
    data_train = np.vstack(projected_data)

    # Project Test Data
    projected_data = []
    num_batches = math.ceil(data_test.shape[0]/args.batch_sz)
    for j in range(0, num_batches):
        if j == num_batches - 1:
            curr_batchdata = data_test[j * args.batch_sz :, :]
        else:
            curr_batchdata = data_test[j * args.batch_sz : (j + 1) * args.batch_sz, :]
        curr_batchdata = preprocessing.normalize(curr_batchdata, norm='l2')
        curr_batchdata = np.concatenate((curr_batchdata, np.ones((curr_batchdata.shape[0], 1))), axis=1)
        curr_batchdata = np.dot(curr_batchdata, proj_matrix)
        projected_data.append(curr_batchdata)
    data_test = np.vstack(projected_data)
    return (data_train, data_test, labels_train, labels_test, datum_train, datum_test)

def random_search_cv(model, params, data_train, labels_train, n_iter=100):
    # Random search of parameters
    rfc_random = RandomizedSearchCV(estimator = model,
                                    param_distributions = params,
                                    n_iter = n_iter, cv = 3, verbose=2,
                                    random_state=42, n_jobs = -1)
    # Fit the model
    rfc_random.fit(data_train, labels_train)
    # print results
    print(rfc_random.best_params_)
    return rfc_random.best_params_