import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV


def plot_kde_distributions(x, x_sampled, attack_type, plot_path=''):
    """
    Use KDE procedure to estimate the real and sampled data distributions
    and plot them using seaborn. The plots will be saved in plot_path folder.
    """
    plt.clf()
    real_mean = np.mean(x, axis=0)
    sampled_mean = np.mean(x_sampled, axis=0)
    sns.kdeplot(sampled_mean,  shade=True, label="Sampled Data", color="g")
    ax = sns.kdeplot(real_mean,  shade=True, label="Real Data", color="r")
    ax.set_title('{:s} Distributions'.format(attack_type))
    if plot_path != '':
        plt.savefig(os.path.join(plot_path, attack_type + 'Distributions.png'))
    else:
        plt.pause(0.0001)
        plt.show(block=False)
    np.save(os.path.join(plot_path,
                         'real_' + attack_type + '_data' '.npy'),
            real_mean)
    np.save(os.path.join(plot_path,
            'sampled_' + attack_type + '_data' '.npy'),
            sampled_mean)
    return


def load_nsl_kdd_dataset(data_file_path):
    """
    Load KDD dataset into RAM inside numpy float arrays.
    At same time label the samples according to their respectively classes
    been them [normal == -1] or [attack == 1].
    """
    data = []
    labels = []
    # Read the dataset.txt file
    with open(data_file_path, 'r') as f:
        train_file_content = f.readlines()
    train_file_lines = [line.replace('\n', '').split() for line in train_file_content]
    for line in train_file_lines:
        line = line[0].split(',')
        line.pop()  # Remove the sample difficult level information
        del(line[1: 4])  # Ignore not numeric features
        # Use attacks_dict to generate sample label in hot vector representation
        if line[-1] == 'normal':
            labels.append(-1)
        else:
            labels.append(1)
        line.pop()  # Remove label data information
        # Convert remaining line data to numeric float
        data.append(list(map(float, line)))
    return (np.array(data), np.array(labels))


def load_splitted_nsl_kdd_dataset(data_file_path, attacks_file_path):
    """
    Load KDD dataset into RAM inside numpy float arrays.
    Similar to load_nsl_kdd_dataset, except for splitting
    the normal and attack samples in different numpy arrays.
    Also in this case the labels are not returned.
    """
    # Train data formats
    normal_data = []
    dos_data = []
    u2r_data = []
    r2l_data = []
    probe_data = []
    # Read the attacks dictionary
    with open(attacks_file_path, 'r') as f:
        attacks_file_content = f.readlines()
    attacks_file_lines = [line.replace('\n', '').split()
                          for line in attacks_file_content]
    if attacks_file_lines[-1] == []:
        attacks_file_lines.pop()
    attacks_dict = dict(attacks_file_lines)
    # Read the dataset.txt file
    with open(data_file_path, 'r') as f:
        train_file_content = f.readlines()
    train_file_lines = [line.replace('\n', '').split()
                        for line in train_file_content]
    for line in train_file_lines:
        line = line[0].split(',')
        line.pop()  # Remove the sample difficult level information
        del(line[1: 4])  # Ignore not numeric features
        # Use attacks_dict to generate sample label in hot vector representation
        # and convert remaining line data to numeric float.
        if line[-1] == 'normal':
            line.pop()  # Remove label data information
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
    return [np.array(normal_data),
            np.array(u2r_data),
            np.array(r2l_data),
            np.array(dos_data),
            np.array(probe_data)]


def generate_roc(deci, label, roc_path):
    """
    Compute ROC curve and AUC.
    Save the plot in roc_path folder.
    """
    fpr, tpr, _ = roc_curve(label, deci)
    roc_auc = auc(fpr, tpr)
    xy_arr = np.concatenate((fpr[:, np.newaxis], tpr[:, np.newaxis]), axis=1)
    np.save(os.path.join(roc_path, 'roc.npy'), xy_arr)
    plt.plot(xy_arr[:, 0], xy_arr[:, 1], '-r')
    plt.title('ROC curve of NSL-KDD Test+ AUC: {:.4f}'.format(roc_auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim(left=0, right=1)
    plt.ylim(bottom=0, top=1)
    plt.savefig(os.path.join(roc_path, 'roc.png'))


def proj_datasets(args, proj_matrix):
    '''Project KDD train and test data into pcd trained RBM feature space.'''
    batch_sz = 100
    data_train, labels_train = load_nsl_kdd_dataset(args.train_data_file_path)
    data_test, labels_test = load_nsl_kdd_dataset(args.test_data_file_path)
    datum_train = load_splitted_nsl_kdd_dataset(args.train_data_file_path,
                                                args.metadata_file_path)
    for i in range(len(datum_train)):
        datum_train[i] = preprocessing.normalize(datum_train[i], norm='l2')
    datum_test = load_splitted_nsl_kdd_dataset(args.test_data_file_path,
                                               args.metadata_file_path)
    for i in range(len(datum_test)):
        datum_test[i] = preprocessing.normalize(datum_test[i], norm='l2')
    # Project Train Data
    projected_data = []
    num_batches = int(np.ceil(data_train.shape[0]/batch_sz))
    for j in range(0, num_batches):
        if j == num_batches - 1:
            curr_batchdata = data_train[j * batch_sz:, :]
        else:
            curr_batchdata = data_train[j * batch_sz: (j + 1) * batch_sz, :]
        curr_batchdata = preprocessing.normalize(curr_batchdata, norm='l2')
        curr_batchdata = np.concatenate((curr_batchdata,
                                         np.ones((curr_batchdata.shape[0], 1))),
                                        axis=1)
        curr_batchdata = np.dot(curr_batchdata, proj_matrix)
        projected_data.append(curr_batchdata)
    data_train = np.vstack(projected_data)
    # Project Test Data
    projected_data = []
    num_batches = int(np.ceil(data_test.shape[0]/batch_sz))
    for j in range(0, num_batches):
        if j == num_batches - 1:
            curr_batchdata = data_test[j * batch_sz:, :]
        else:
            curr_batchdata = data_test[j * batch_sz: (j + 1) * batch_sz, :]
        curr_batchdata = preprocessing.normalize(curr_batchdata, norm='l2')
        curr_batchdata = np.concatenate((curr_batchdata,
                                         np.ones((curr_batchdata.shape[0], 1))),
                                        axis=1)
        curr_batchdata = np.dot(curr_batchdata, proj_matrix)
        projected_data.append(curr_batchdata)
    data_test = np.vstack(projected_data)
    return (data_train, data_test,
            labels_train, labels_test,
            datum_train, datum_test)


def random_search_cv(model, params, data_train, labels_train, n_iter=100):
    '''
    Use Random Search to find the optimum DTREE and
    RANDOM FOREST train parameters configuration.
    '''
    # Random search of parameters
    rfc_random = RandomizedSearchCV(estimator=model,
                                    param_distributions=params,
                                    n_iter=n_iter, cv=3, verbose=2,
                                    random_state=42, n_jobs=-1)
    # Fit the model
    rfc_random.fit(data_train, labels_train)
    print(rfc_random.best_params_)
    return rfc_random.best_params_
