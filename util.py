import numpy as np
import math, os
from operator import itemgetter
# https://seaborn.pydata.org/tutorial/distributions.html
import seaborn as sns
import matplotlib.pyplot as plt

def plot_kde_distributions(x, x_sampled, attack_type):
    sns.kdeplot(np.mean(x_sampled, axis=0),  shade=True, label="BBRBM Sampled Data", color="g");
    ax = sns.kdeplot(np.mean(x, axis=0),  shade=True, label="True Data", color="r");
    ax.set_title('{:s} Distributions'.format(attack_type))
    plt.savefig(os.path.join('SamplingAnalysis', attack_type + 'Distributions.png'))
    #plt.show()

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
    plt.savefig(os.path.join(roc_path, 'roc.png'))


