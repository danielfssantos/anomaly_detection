import os, sys
import numpy as np
import argparse
from util import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore', DeprecationWarning)

# Download libsvm from:
#             http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/libsvm.cgi?+http://www.csie.ntu.edu.tw/~cjlin/libsvm+tar.gz
# Change the svm path according to your respective libsvm path location
# Obs: Remember to compyle the lib first
sys.path.append('/home/daniel/Documents/DeepLearningOpenCV/libsvm-3.23/python')
from svmutil import *
from rbm import RBM

def main(args):
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
    if args.mode.find('train') != -1 and args.gen_dataset:
        if args.mode.find('aug') != -1: # Augmenting data
            datum_train = load_nsl_kdd_splitted_dataset(args.train_data_file_path, args.metadata_file_path)
            # Instatiate BBRBM
            args.num_vis_nodes = datum_train[0].shape[1]
            attack_names = {0 : 'normal', 1 : 'u2r', 2 : 'r2l', 3 : 'dos', 4 : 'probe'}
            attacks_biggest_size = len(datum_train[0])
            print('Augmenting NLS-KDD dataset using Normal data size as reference')
            batch_sz = 100
            sampled_data_train = []
            rbm_train_type = args.rbm_train_type
            for i in range(len(datum_train)):
                # Sample biggest_size - len(datum_train[i]) times
                # for u2r, l2r and probe attacks
                if i not in [0, 1, 4]:
                    args.rbm_train_type = rbm_train_type + '_' + attack_names[i]
                    rbm_model = RBM(args)
                    rbm_model.load(args.rbm_params_path)
                    print('Sampling data from {:s} BBRBM\n'.format(attack_names[i]))
                    attack_train_data = np.array(datum_train[i])
                    qnt_to_sample = attacks_biggest_size - attack_train_data.shape[0]
                    for j in range(0, qnt_to_sample, batch_sz):
                        if args.rbm_train_type.find('bbrbm') != -1:
                            sampled_data = np.random.randint(low=0, high=2, size=(batch_sz, rbm_model.numdims))
                        elif args.rbm_train_type.find('bbrbm') != -1:
                            sampled_data = np.random.rand(batch_sz, rbm_model.numdims)
                        elif args.rbm_train_type.find('gbrbm') != -1:
                            sampled_data = np.random.randn(batch_sz, rbm_model.numdims)
                        sampled_data_train.append(rbm_model.sample_data(sampled_data, ites=args.sample_ites))
            sampled_data_train = np.vstack(sampled_data_train)
            data_train = np.vstack(datum_train)
            data_train = np.concatenate((data_train, sampled_data_train), axis=0)
            labels_train = -1 * np.ones((len(datum_train[0]),))
            labels_train = np.concatenate( (labels_train, np.ones( (data_train.shape[0] - len(datum_train[0]), ) ) ), axis=0)
        else:
            data_train, labels_train = load_nsl_kdd_dataset(args.train_data_file_path)
        data_test, labels_test = load_nsl_kdd_dataset(args.test_data_file_path)
        #datum = load_nsl_kdd_dataset(args.train_data_file_path)
        # Save processed data
        os.system('mkdir -p ' + args.data_save_path)
        np.savez_compressed(os.path.join(args.data_save_path, args.train_file_name),\
                                         data_train, labels_train, data_test, labels_test)
    else:
        npzfiles = np.load(os.path.join(args.data_save_path, args.train_file_name), allow_pickle=True)
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
    # Test SVM rbf kernel
    if args.mode.find('train_svm') != -1:
        if args.mode == 'train_svm_cross':
            print('Searching for best c and g values...')
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
            args.c = best_c
            args.g = best_g
        else:
            svm_model = svm_train(np.squeeze(labels_train), data_train, '-c {:.5f} -g {:.5f} -q'.format(args.c, args.g))
        os.system('mkdir -p ' + args.svm_params_path)
        if args.mode.find('aug') != -1:
            svm_save_model(os.path.join(args.svm_params_path, 'svm_model_aug.txt'), svm_model)
        else:
            svm_save_model(os.path.join(args.svm_params_path, 'svm_model.txt'), svm_model)
    # Test SVM
    elif args.mode.find('test_svm') != -1:
        os.system('mkdir -p SVMAnalysis')
        print('Testing SVM...')
        if args.mode.find('aug') != -1:
            svm_model = svm_load_model(os.path.join(args.svm_params_path, 'svm_model_aug.txt'))
        else:
            svm_model = svm_load_model(os.path.join(args.svm_params_path, 'svm_model.txt'))
        labels_test = np.squeeze(labels_test)
        pred_labels, evals, deci  = svm_predict(labels_test, data_test, svm_model, '-q')
        print('Test Acc: {:.4f}'.format(evals[0]))
        if args.mode.find('aug') != -1:
            np.savetxt('SVMAnalysis/acc_aug.txt', np.array(evals[0]).reshape(1,), fmt='%.4f')
        else:
            np.savetxt('SVMAnalysis/acc.txt', np.array(evals[0]).reshape(1,), fmt='%.4f')
        #Plot ROC curve and Misclassification bars graph
        labels = svm_model.get_labels()
        deci = [labels[0]*val[0] for val in deci]
        # Use datum_test to measure misclassification percentage
        datum_test = load_nsl_kdd_splitted_dataset(args.test_data_file_path, args.metadata_file_path)
        qnt_attacks = []
        for d in datum_test:
            qnt_attacks.append(len(d))
        print('Saving ROC under SVMAnalysis folder')
        if args.mode.find('aug') != -1:
            generate_roc(deci, labels_test, 'SVMAnalysis/roc_aug.png')
            generate_error_bars(qnt_attacks, labels_test, pred_labels, 'SVMAnalysis/misclass_aug.png')
        else:
            generate_roc(deci, labels_test, 'SVMAnalysis/roc.png')
            generate_error_bars(qnt_attacks, labels_test, pred_labels, 'SVMAnalysis/misclass.png')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data-file-path', type=str, default='./NSL_KDD_Dataset/KDDTrain+_20Percent.txt',
                        help='path to the train data .csv file')
    parser.add_argument('--test-data-file-path', type=str, default='./NSL_KDD_Dataset/KDDTest+.txt',
                        help='path to the test data .csv file')
    parser.add_argument('--metadata-file-path', type=str, default='./NSL_KDD_Dataset/training_attack_types.txt',
                        help='path to the test data .csv file')
    parser.add_argument('--train-file-name', type=str, default='discriminative_kdd_nsl_processed_aug.npz',
                        help='path to the train data .csv file')
    parser.add_argument('--svm-params-path', type=str, default='./SVMParams',
                        help='path location to save RBM trained weights')
    parser.add_argument('--rbm-params-path', type=str, default='./RBMParams/KDDTrain+_20Percent',
                        help='path location to save RBM trained weights')
    parser.add_argument('--rbm-train-type', type=str, default='bbrbm_pcd',
                        help='[ bbrbm_cd_attack_type, bbrbm_pcd_attack_type,\
                                    gbrbm_cd_attack_type, gbrbm_pcd_attack_type ] train types\
                                    where attack_type in [ normal, dos, u2r, l2r, probe ]')
    parser.add_argument('--data-save-path', type=str, default='./TrainData/KDDTrain+_20Percent/Augmented',
                        help='train and test datasets save location')
    parser.add_argument('--datum-save-path', type=str, default='./TrainData/KDDTrain+_20Percent',
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

