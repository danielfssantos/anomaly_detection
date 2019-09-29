import argparse
from util import *
import warnings, os
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore', DeprecationWarning)
from rbm import RBM
from gan import GAN

def main(args):
    os.system('cls' if os.name == 'nt' else 'clear')
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
    if args.mode.find('train') != -1 and args.gen_dataset:
        if args.mode.find('aug') != -1:
            print('Augmenting NLS-KDD dataset using Normal data size as reference')
            if args.mode.find('aug_rbm') != -1:
                data_sampler_model = RBM(args)
            elif args.mode.find('aug_gan') != -1:
                data_sampler_model = GAN(args)
            else:
                print('Inform a valid data sampler')
                exit(0)
            data_train, labels_train, aug_datum = augment_dataset(args, data_sampler_model)
        else:
            data_train, labels_train = load_nsl_kdd_dataset(args.train_data_file_path)
            datum_train = load_nsl_kdd_splitted_dataset(args.train_data_file_path, args.metadata_file_path)
        data_test, labels_test = load_nsl_kdd_dataset(args.test_data_file_path)
        #datum = load_nsl_kdd_dataset(args.train_data_file_path)
        # Save processed data
        os.system('mkdir -p ' + args.data_save_path)
        if args.mode.find('aug') != -1:
            np.savez_compressed(os.path.join(args.data_save_path, args.train_file_name),\
                                              data_train, labels_train, data_test, labels_test, aug_datum)
        else:
            np.savez_compressed(os.path.join(args.data_save_path, args.train_file_name),\
                                             data_train, labels_train, data_test, labels_test, datum_train)
            exit(0)
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
        if args.mode.find('cross') != -1:
            print('Searching for best c and g values...')
            idx_rand = np.random.permutation(data_train.shape[0])
            data_train_cross_val = data_train[idx_rand, :]
            labels_train_cross_val = labels_train[idx_rand]
            if args.mode.find('aug'):
                data_train_cross_val = data_train_cross_val[0 : data_train_cross_val.shape[0]//30, :]
                labels_train_cross_val = labels_train_cross_val[0 : labels_train_cross_val.shape[0]//30]
                #print(data_train_cross_val.shape)
                #input()
            else:
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
        #else:
        svm_model = svm_train(np.squeeze(labels_train), data_train, '-c {:.5f} -g {:.5f} -q'.format(args.c, args.g))
        #else:
        os.system('mkdir -p ' + args.svm_params_path)
        if args.mode.find('aug') != -1:
            svm_save_model(os.path.join(args.svm_params_path, 'svm_model_aug.txt'), svm_model)
        else:
            svm_save_model(os.path.join(args.svm_params_path, 'svm_model.txt'), svm_model)
    # Test SVM
    elif args.mode.find('test_svm') != -1:
        os.system('mkdir -p ' + args.svm_analysis_path)
        print('Testing SVM...')
        if args.mode.find('aug') != -1:
            svm_model = svm_load_model(os.path.join(args.svm_params_path, 'svm_model_aug.txt'))
        else:
            svm_model = svm_load_model(os.path.join(args.svm_params_path, 'svm_model.txt'))
        labels_test = np.squeeze(labels_test)
        pred_labels, evals, deci  = svm_predict(labels_test, data_test, svm_model, '-q')
        print('Test Acc: {:.4f}'.format(evals[0]))
        if args.mode.find('aug') != -1:
            np.savetxt(args.svm_analysis_path+'/acc_aug.txt', np.array(evals[0]).reshape(1,), fmt='%.4f')
        else:
            np.savetxt(args.svm_analysis_path+'/acc.txt', np.array(evals[0]).reshape(1,), fmt='%.4f')
        #Plot ROC curve and Misclassification bars graph
        labels = svm_model.get_labels()
        deci = [labels[0]*val[0] for val in deci]
        # Use datum_test to measure misclassification percentage
        datum_test = load_nsl_kdd_splitted_dataset(args.test_data_file_path, args.metadata_file_path)
        qnt_attacks = []
        for d in datum_test:
            qnt_attacks.append(len(d))
        print('Saving ROC under {:s} folder'.format(args.svm_analysis_path))
        if args.mode.find('aug') != -1:
            generate_roc(deci, labels_test, args.svm_analysis_path+'/roc_aug.png')
            generate_error_bars(qnt_attacks, labels_test, pred_labels, args.svm_analysis_path+'/misclass_aug.png')
        else:
            generate_roc(deci, labels_test, args.svm_analysis_path+'/roc.png')
            generate_error_bars(qnt_attacks, labels_test, pred_labels, args.svm_analysis_path+'/misclass.png')

def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--train-data-file-path', type=str, default='./NSL_KDD_Dataset/KDDTrain+_20Percent.txt',
      #                  help='path to the train data .csv file')
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
    parser.add_argument('--svm-analysis-path', type=str, default='./SVMAnalysis',
                        help='path location to save RBM trained weights')
    parser.add_argument('--data-sampler-params-path', type=str, default='./RBMParams/KDDTrain+_20Percent',
                        help='path location to save RBM trained weights')
    parser.add_argument('--data-sampler-train-type', type=str, default='bbrbm_pcd',
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
    parser.add_argument('--use-oc-svm', type=int, default=0,
                        help='use one class svm to improove data augmentation sample quality')
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
    parser.add_argument('--num-vis-nodes', type=int, default=38,
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

