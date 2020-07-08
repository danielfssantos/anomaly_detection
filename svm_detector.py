import argparse
import warnings, os
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore', DeprecationWarning)
from rbm import RBM
from sklearn.metrics import classification_report
import xlsxwriter
from sklearn.preprocessing import StandardScaler
from util import *
sys.path.append('/home/daniel/Documents/DeepLearningOpenCV/libsvm-3.23/python')
from svmutil import *


def main(args):
    os.system('cls' if os.name == 'nt' else 'clear')
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
    if args.mode.find('train') != -1 and args.gen_dataset:
        if args.mode.find('features') != -1:
            data_sampler_model = RBM(args)
            data_train, labels_train, datum_features = gen_features_dataset(args, data_sampler_model)
        elif args.mode.find('proj') != -1:
            npz_files = np.load(os.path.join(args.data_sampler_params_path,
                                                            args.projection_matrix_name),
                                                            allow_pickle=True)
            rbm_proj_matrix = npz_files['arr_0']
            data_train, data_test, labels_train,\
            labels_test, datum_train, datum_test = proj_datasets(args, rbm_proj_matrix)
        else:
            data_train, labels_train = load_nsl_kdd_dataset(args.train_data_file_path)
            datum_train = load_splitted_nsl_kdd_dataset(args.train_data_file_path, args.metadata_file_path)
            data_test, labels_test = load_nsl_kdd_dataset(args.test_data_file_path)
            datum_test = load_splitted_nsl_kdd_dataset(args.test_data_file_path, args.metadata_file_path)
        if args.mode.find('proj') == -1:
            data_test, labels_test = load_nsl_kdd_dataset(args.test_data_file_path)

        # Save processed data
        os.makedirs(args.data_save_path, exist_ok=True)
        if args.mode.find('features') != -1:
            np.savez_compressed(os.path.join(args.data_save_path, args.train_file_name),\
                                             data_train, labels_train, data_test,
                                             labels_test, datum_features)
        elif args.mode.find('proj') != -1:
            np.savez_compressed(os.path.join(args.data_save_path, args.train_file_name),\
                                             data_train, labels_train, data_test,
                                             labels_test, datum_train, datum_test)
        else:
            np.savez_compressed(os.path.join(args.data_save_path, args.train_file_name),\
                                             data_train, labels_train, data_test, labels_test)
    elif args.mode.find('features') == -1:
        npzfiles = np.load(os.path.join(args.data_save_path, args.train_file_name), allow_pickle=True)
        data_train = npzfiles['arr_0']; labels_train = npzfiles['arr_1']
        data_test = npzfiles['arr_2']; labels_test = npzfiles['arr_3']

    # Train SVM rbf kernel
    if args.mode.find('train_svm') != -1:
        # If mode train_svm_cross the cross validation will be executed
        # over 3% of the data_train randomly chosen samples.
        if args.mode.find('cross') != -1:
            print('Searching for best c and g values...')
            idx_rand = np.random.permutation(data_train.shape[0])
            data_train_cross_val = data_train[idx_rand, :]
            labels_train_cross_val = labels_train[idx_rand]
            data_train_cross_val = data_train_cross_val[0 : math.ceil(data_train_cross_val.shape[0] * 0.03), :]
            labels_train_cross_val = labels_train_cross_val[0 : math.ceil(labels_train_cross_val.shape[0] * 0.03)]
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
            print('\nbest_c[: {:.5f} best_g: {:.5f} best_acc: {:.5f}\n'.format(best_c, best_g, best_acc))
            args.c = best_c
            args.g = best_g
        if args.mode.find('proj') != -1:
            print('\nTraining SVM using RBM projection...\n')
        else:
            print('\nTraining SVM without RBM projection...\n')
        svm_model = svm_train(np.squeeze(labels_train), data_train, '-c {:.5f} -g {:.5f} -q'.format(args.c, args.g))
        os.makedirs(args.svm_params_path, exist_ok=True)
        svm_save_model(os.path.join(args.svm_params_path, 'svm_model.txt'), svm_model)
    # Test SVM
    elif args.mode.find('test_svm') != -1:
        os.makedirs(args.svm_params_path, exist_ok=True)
        if args.mode.find('proj') != -1:
            print('\nTesting SVM using RBM projection...\n')
        else:
            print('\nTesting SVM without RBM projection...\n')
        svm_model = svm_load_model(os.path.join(args.svm_params_path, 'svm_model.txt'))
        if args.mode.find('features') != -1:
            data_sampler_model = RBM(args)
            data_test, labels_test, _ = gen_features_dataset(args, data_sampler_model)
        labels_test = np.squeeze(labels_test)
        pred_labels, evals, deci  = svm_predict(labels_test, data_test, svm_model, '-q')
        accuracy = evals[0]/100.0
        # Plot ROC curve and Misclassification bars graph
        labels = svm_model.get_labels()
        deci = [labels[0]*val[0] for val in deci]
        os.makedirs(args.svm_analysis_path, exist_ok=True)
        print('Saving ROC under {:s} folder'.format(args.svm_analysis_path))
        generate_roc2(deci, labels_test, args.svm_analysis_path)
        results_dict = classification_report(labels_test, pred_labels, output_dict=True)
        precision = (results_dict['-1']['precision']+results_dict['1']['precision'])/2
        recall = (results_dict['-1']['recall']+results_dict['1']['recall'])/2
        f_score = (results_dict['-1']['f1-score']+results_dict['1']['f1-score'])/2
        print('Accuracy {:.4f}\nPrecision: {:.4f}\nRecall: {:.4f}\nF1-Score: {:.4f}'.format(
                accuracy, precision, recall, f_score))
        # Save results into spread sheet
        workbook = xlsxwriter.Workbook(os.path.join(args.svm_analysis_path, 'results.xlsx'))
        worksheet = workbook.add_worksheet()
        bold = workbook.add_format({'bold': True})
        worksheet.write(0, 0, '')
        worksheet.write(1, 0, 'evals', bold)
        worksheet.write(0, 1, 'Accuracy', bold)
        worksheet.write(1, 1, '{:.4f}'.format(accuracy))
        worksheet.write(0, 2, 'Precision', bold)
        worksheet.write(1, 2, '{:.4f}'.format(precision))
        worksheet.write(0, 3, 'Recall', bold)
        worksheet.write(1, 3, '{:.4f}'.format(recall))
        worksheet.write(0, 4, 'F1-Score', bold)
        worksheet.write(1, 4, '{:.4f}'.format(f_score))
        workbook.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data-file-path', type=str, default='',
                        help='path to the train data .csv file')
    parser.add_argument('--test-data-file-path', type=str, default='./NSL_KDD_Dataset/KDDTest+.txt',
                        help='path to the test data .csv file')
    parser.add_argument('--metadata-file-path', type=str, default='./NSL_KDD_Dataset/training_attack_types.txt',
                        help='path to the test data .csv file')
    parser.add_argument('--train-file-name', type=str, default='',
                        help='path to the train data .csv file')
    parser.add_argument('--svm-params-path', type=str, default='./SVMParams',
                        help='path location to save RBM trained weights')
    parser.add_argument('--svm-analysis-path', type=str, default='./SVMAnalysis',
                        help='path location to save RBM trained weights')
    parser.add_argument('--data-sampler-params-path', type=str, default='./RBMParams/KDDTrain+_20Percent',
                        help='path location to save RBM trained weights')
    parser.add_argument('--data-save-path', type=str, default='',
                        help='train and test datasets save location')
    parser.add_argument('--datum-save-path', type=str, default='',
                        help='train and test datasets save location')
    parser.add_argument('--mode', type=str, default='train_svm',
                        help='train svm but first use cross validation technique to infer the optimum train parameter values')
    parser.add_argument('--gen-dataset', type=int, default=1,
                        help='generate or just load (1/0) the train and test NSL-KDD datasets')
    parser.add_argument('--c', type=float, default=256,
                        help='svm cost')
    parser.add_argument('--g', type=float, default=1.0,
                        help='svm RBF standart deviation')
    parser.add_argument('--projection-matrix-name', type=str, default='rbm_projection_matrix.npz',
                        help='normalization type that will be applied to the train and test data')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())

