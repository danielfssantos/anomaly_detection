import os, sys
import numpy as np
import argparse
from util import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore', DeprecationWarning)

sys.path.append('/home/daniel/Documents/DeepLearningOpenCV/libsvm-3.23/python')
from svmutil import *

def main(args):
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
    if args.mode.find('train') != -1 and args.gen_dataset:
        data_train, labels_train = load_nsl_kdd_dataset(args.train_data_file_path)
        data_test, labels_test = load_nsl_kdd_dataset(args.test_data_file_path)
        #datum = load_nsl_kdd_dataset(args.train_data_file_path)
        # Save processed data
        os.system('mkdir -p ' + args.data_save_path)
        np.savez_compressed(os.path.join(args.data_save_path, 'kdd_nsl_processed.npz'), data_train,
                                            labels_train, data_test, labels_test)
    elif args.mode.find('train') != -1 and not args.gen_dataset:
        npzfiles = np.load(os.path.join(args.data_save_path, 'kdd_nsl_processed.npz'), allow_pickle=True)
        data_train = npzfiles['arr_0']; labels_train = npzfiles['arr_1']
        data_test = npzfiles['arr_2']; labels_test = npzfiles['arr_3']
    # Normalize data according to norm_type
    if args.norm_type == 'z_norm':
        data_train, mean, std = z_norm(data_train)
        data_test -= mean
        data_test /= std
    if args.norm_type == 'min_max_norm':
        data_train = min_max_norm(data_train)
        data_test = min_max_norm(data_test)
    # Instantiate svm model and train it using rbf kernel
    if args.mode == 'train_svm_cross':
        idx_rand = np.random.permutation(data_train.shape[0])
        data_train_cross_val = data_train[idx_rand, :]
        labels_train_cross_val = labels_train[idx_rand]
        data_train_cross_val = data_train_cross_val[0 : data_train_cross_val.shape[0]//2, :]
        labels_train_cross_val = labels_train_cross_val[0 : labels_train_cross_val.shape[0]//2]
        # Use svm cross validation technique to find best Cost and RBF kernel deviation values
        best_c = -1
        best_g = -1
        best_acc = 0
        for i in range(-5, 15, 1):
            for j in range(3, -15, -1):
                new_c = 2**i; new_g = 2**j
                cv_acc = svm_train(np.squeeze(labels_train_cross_val), data_train_cross_val, '-c {:.5f} -g {:.5f} -v 10 -q'.format(new_c, new_g))
                if cv_acc > best_acc:
                    best_acc = cv_acc
                    best_c = new_c
                    best_g = new_g
                    print('\nbest_c[2^{:d}]: {:.5f} best_g[2^{:d}]: {:.5f} best_acc: {:.5f}\n'.format(i, best_c, j, best_g, best_acc))
    elif args.mode == 'train_svm':
        idx_rand = np.random.permutation(data_train.shape[0])
        svm_model = svm_train(np.squeeze(labels_train), data_train, '-c {:.5f} -g {:.5f} -q'.format(args.c, args.g))
        # Test prediction
        prob_test_labels, _, _ = svm_predict(np.squeeze(labels_test), data_test, svm_model, '-q')
        test_acc, _, _ = evaluations(np.squeeze(labels_test), prob_test_labels)
        print('Test Acc: {:.2f}'.format(test_acc))
        #print('best_acc: {:.4f} best_c: {:.4f} best_g: {:.4f}'.format(best_acc, best_c, best_g))
    # Use the trained RBM model to sample --batch-sz elements from --sample-ites
    # number of sampling iterations.
    #elif args.mode == 'test':


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_file_path', type=str, default='./NSL_KDD_Dataset/KDDTrain+_20Percent.txt',
                        help='path to the train data .csv file')
    parser.add_argument('--test_data_file_path', type=str, default='./NSL_KDD_Dataset/KDDTest+.txt',
                        help='path to the test data .csv file')
    parser.add_argument('--metadata-file-path', type=str, default='./NSL_KDD_Dataset/training_attack_types.txt',
                        help='path to the test data .csv file')
    parser.add_argument('--rbm-params-path', type=str, default='./Params',
                        help='path location to save RBM trained weights')
    parser.add_argument('--data-save-path', type=str, default='./TrainData',
                        help='train and test datasets save location')
    parser.add_argument('--mode', type=str, default='train_svm',
                        help='train svm but first use cross validation technique to infer the optimum train parameter values')
    parser.add_argument('--sample-data-repetitions', type=int, default=100,
                        help='number of times to execute the sampling process')
    parser.add_argument('--gen-dataset', type=int, default=1,
                        help='generate or just load (1/0) the train and test NSL-KDD datasets')
    parser.add_argument('--norm-type', type=str, default='min_max_norm',
                        help='normalization type that will be applied to the train and test data')
    parser.add_argument('--train-type', type=str, default='gbrbm_pcd_dos',
                        help='[ bbrbm_cd_attack_type, bbrbm_pcd_attack_type,\
                                    gbrbm_cd_attack_type, gbrbm_pcd_attack_type ] train types\
                                    where attack_type in [ normal, dos, u2r, l2r, probe ]')
    parser.add_argument('--sample-visdata', type=int, default=0,
                        help='sample or not (1/0) visible data during gibbs sampling')
    parser.add_argument('--num-hid-nodes', type=int, default=1000,
                        help='maximum quantity of hidden layer nodes')
    parser.add_argument('--cd-steps', type=int, default=1,
                        help='number of CD iteartions')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='total of training epochs')
    parser.add_argument('--c', type=float, default=256,#bernoulli-bernoulli 0.005
                        help='svm optimization cost')
    parser.add_argument('--g', type=float, default=1.0,#bernoulli-bernoulli 0.005
                        help='svm optimization RBF standart deviation')
    parser.add_argument('--epsilonw', type=float, default=0.0005,#bernoulli-bernoulli 0.005
                        help='weight matrix learning rate value')
    parser.add_argument('--epsilonvb', type=float, default=0.0005,#bernoulli-bernoulli 0.005
                        help='visible layer biases learning rate')
    parser.add_argument('--epsilonhb', type=float, default=0.0005,#bernoulli-bernoulli 0.005
                        help='hidden layer biases learning rate')
    parser.add_argument('--sample-ites', type=int, default=200,
                        help='number of sample iterations')
    parser.add_argument('--batch-sz', type=int, default=100,
                        help='maximum quantity of samples per batch')
    parser.add_argument('--weightcost', type=float, default=2e-4,
                        help='controls the weight decay velocity')
    parser.add_argument('--initialmomentum', type=float, default=.5,
                        help='momentum value to start the training process')
    parser.add_argument('--finalmomentum', type=float, default=.9,
                        help='momentum value to end the training process')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())

