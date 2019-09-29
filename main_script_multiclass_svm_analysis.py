import argparse
from util import *
import warnings
import xlwt
from xlwt import Workbook
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore', DeprecationWarning)

def main(args):
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
    npzfiles = np.load(os.path.join(args.data_save_path, args.train_file_name), allow_pickle=True)
    train_datum = npzfiles['arr_4']
    # Create train data and labels
    data_train = np.array([])
    labels_train = np.array([])
    if args.mode.find('cross') != -1:
        data_train_cross_val = np.array([])
        labels_train_cross_val = np.array([])
        max_cross_val_samples = 100
    for i in range(len(train_datum)):
        if data_train.size == 0:
            data_train = train_datum[i]
            labels_train = i * np.ones((len(train_datum[i]),))
            if args.mode.find('cross') != -1:
                idx_rand = np.random.permutation(train_datum[i].shape[0])
                if train_datum[i].shape[0] < max_cross_val_samples:
                    data_train_cross_val = train_datum[i][idx_rand[0 :], :]
                    labels_train_cross_val = i * np.ones((train_datum[i].shape[0],))
                else:
                    data_train_cross_val = train_datum[i][idx_rand[0 : max_cross_val_samples], :]
                    labels_train_cross_val = i * np.ones((max_cross_val_samples,))
        else:
            data_train = np.concatenate((data_train, train_datum[i]), axis=0)
            labels_train = np.concatenate((labels_train, i * np.ones((len(train_datum[i]),))), axis=0)
            if args.mode.find('cross') != -1:
                idx_rand = np.random.permutation(train_datum[i].shape[0])
                if train_datum[i].shape[0] < max_cross_val_samples:
                    data_train_cross_val =  np.concatenate((data_train_cross_val,\
                                                                                 train_datum[i][idx_rand[0 :], :]), axis=0)
                    labels_train_cross_val = np.concatenate((labels_train_cross_val,\
                                                                                   i * np.ones((train_datum[i].shape[0],))), axis=0)
                else:
                    data_train_cross_val =  np.concatenate((data_train_cross_val,\
                                                                                 train_datum[i][idx_rand[0 : max_cross_val_samples], :]), axis=0)
                    labels_train_cross_val = np.concatenate((labels_train_cross_val,\
                                                                                   i * np.ones((max_cross_val_samples,))), axis=0)
    # Create test data and labels
    datum_test = load_nsl_kdd_splitted_dataset(args.test_data_file_path, args.metadata_file_path)
    data_test = np.array([])
    labels_test = np.array([])
    #del datum_test[0]
    for i in range(len(datum_test)):
        if data_test.size == 0:
            data_test = datum_test[i]
            labels_test = i * np.ones((len(datum_test[i]),))
        else:
            data_test = np.concatenate((data_test, datum_test[i]), axis=0)
            labels_test = np.concatenate((labels_test, i * np.ones((len(datum_test[i]),))), axis=0)

    # Normalize data according to norm_type
    if args.norm_type == 'z_norm':
        data_train, mean, std = z_norm(data_train)
        data_test -= mean
        data_test /= std
    if args.norm_type == 'min_max_norm':
        data_train = min_max_norm(data_train)
        data_test = min_max_norm(data_test)
        if args.mode.find('cross') != -1:
            data_train_cross_val = min_max_norm(data_train_cross_val)

    # Test SVM rbf kernel
    if args.mode.find('train_svm') != -1:
        if args.mode.find('cross') != -1:
            print('Searching for best c and g values...')
            '''
            idx_rand = np.random.permutation(data_train.shape[0])
            data_train_cross_val = data_train[idx_rand, :]
            labels_train_cross_val = labels_train[idx_rand]
            data_train_cross_val = data_train_cross_val[0 : data_train_cross_val.shape[0]//10, :]
            labels_train_cross_val = labels_train_cross_val[0 : labels_train_cross_val.shape[0]//10]
            '''
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
            print('Cross validation done...')
            print('\nbest_c: {:.5f} best_g: {:.5f} best_acc: {:.5f}\n'.format(best_c, best_g, best_acc))
        else:
            os.system('mkdir -p ' + args.svm_params_path)
            svm_model = svm_train(np.squeeze(labels_train), data_train, '-c {:.5f} -g {:.5f} -q'.format(args.c, args.g))
            if args.mode.find('aug') != -1:
                svm_save_model(os.path.join(args.svm_params_path, 'svm_model_aug.txt'), svm_model)
            else:
                svm_save_model(os.path.join(args.svm_params_path, 'svm_model.txt'), svm_model)
            print('Train done...')
    # Test SVM
    elif args.mode.find('test_svm') != -1:
        os.system('mkdir -p MultiSVMAnalysis')
        print('Testing SVM...')
        if args.mode.find('aug') != -1:
            svm_model = svm_load_model(os.path.join(args.svm_params_path, 'svm_model_aug.txt'))
        else:
            svm_model = svm_load_model(os.path.join(args.svm_params_path, 'svm_model.txt'))
        labels_test = np.squeeze(labels_test)
        pred_labels, evals, deci  = svm_predict(labels_test, data_test, svm_model, '-q')
        print('Test Acc: {:.4f}'.format(evals[0]))
        if args.mode.find('aug') != -1:
            np.savetxt('MultiSVMAnalysis/acc_aug.txt', np.array(evals[0]).reshape(1,), fmt='%.4f')
        else:
            np.savetxt('MultiSVMAnalysis/acc.txt', np.array(evals[0]).reshape(1,), fmt='%.4f')
        #Generate confusion matrix
        attack_names = {0 : 'normal', 1 : 'u2r', 2 : 'r2l', 3 : 'dos', 4 : 'probe'}
        #attack_names = {0 : 'u2r', 1 : 'r2l', 2 : 'dos', 3 : 'probe'}

        # Save results into spread sheet
        # Workbook is created
        wb = Workbook()
        style_table_header = xlwt.easyxf('font: bold 1')
        style_float_num = xlwt.XFStyle()
        style_float_num.num_format_str = '0.00'
        # add_sheet is used to create sheet.
        sheet1 = wb.add_sheet('confusion_matrix')
        sheet1.write(0, 0, '', style_table_header)
        sheet1.write(1, 0, attack_names[0], style_table_header)
        sheet1.write(2, 0, attack_names[1], style_table_header)
        sheet1.write(3, 0, attack_names[2], style_table_header)
        sheet1.write(4, 0, attack_names[3], style_table_header)
        sheet1.write(5, 0, attack_names[4], style_table_header)
        sheet1.write(0, 1, 'Predicted '+attack_names[0], style_table_header)
        sheet1.write(0, 2, 'Predicted '+attack_names[1], style_table_header)
        sheet1.write(0, 3, 'Predicted '+attack_names[2], style_table_header)
        sheet1.write(0, 4, 'Predicted '+attack_names[3], style_table_header)
        sheet1.write(0, 5, 'Predicted '+attack_names[4], style_table_header)
        labels_test = labels_test.astype('int')
        pred_labels = np.array(pred_labels).astype('int')
        conf_matrix = np.zeros((5, 5))
        for i in range(5):
            pred_labels_current_class = pred_labels[np.where(labels_test == i)]
            for j in range(5):
                conf_matrix[i, j] = pred_labels_current_class[np.where(pred_labels_current_class == j)].size/pred_labels_current_class.size
                sheet1.write(i + 1, j + 1, '{:.2f}'.format(conf_matrix[i, j]), style_float_num)
        if args.mode.find('aug') != -1:
            wb.save(os.path.join('MultiSVMAnalysis', 'confusion_matrix_aug.xls'))
        else:
            wb.save(os.path.join('MultiSVMAnalysis', 'confusion_matrix.xls'))


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
    parser.add_argument('--use-oc-svm', type=int, default=0,
                        help='use one class svm to improove data augmentation sample quality')
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

