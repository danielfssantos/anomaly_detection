import argparse
from util import load_nsl_kdd_dataset
from util import load_splitted_nsl_kdd_dataset
from util import proj_datasets
from util import random_search_cv
from util import generate_roc
import warnings
import numpy as np
import os
import pickle
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore', DeprecationWarning)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import xlsxwriter


def main(args):
    os.system('cls' if os.name == 'nt' else 'clear')
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

    if args.mode.find('train') != -1 and args.gen_dataset:
        if args.mode.find('proj') != -1:
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
        if args.mode.find('proj') != -1:
            np.savez_compressed(os.path.join(args.data_save_path,
                                             args.train_file_name),
                                data_train, labels_train, data_test,
                                labels_test, datum_train, datum_test)
        else:
            np.savez_compressed(os.path.join(args.data_save_path,
                                             args.train_file_name),
                                data_train, labels_train,
                                data_test, labels_test)
    else:
        npzfiles = np.load(os.path.join(args.data_save_path, args.train_file_name), allow_pickle=True)
        data_train = npzfiles['arr_0']
        labels_train = npzfiles['arr_1']
        data_test = npzfiles['arr_2']
        labels_test = npzfiles['arr_3']
    # Train DTREE classifier
    if args.mode.find('train_dtree') != -1:
        if args.mode.find('cross') != - 1:
            print('Find suitable parameters...')
            max_features = ['auto', 'sqrt']
            max_depth = [int(x) for x in np.linspace(100, 500, num=11)]
            max_depth.append(None)
            # create random grid
            params_to_tune = {
             'max_features': max_features,
             'max_depth': max_depth
             }
            dtree_model = DecisionTreeClassifier()
            idx_rand = np.random.permutation(data_train.shape[0])
            data_train_cross_val = data_train[idx_rand, :]
            labels_train_cross_val = labels_train[idx_rand]
            data_train_cross_val = data_train_cross_val[0: int(np.ceil(data_train_cross_val.shape[0] * 0.5)), :]
            labels_train_cross_val = labels_train_cross_val[0: int(np.ceil(labels_train_cross_val.shape[0] * 0.5))]
            best_params = random_search_cv(dtree_model,
                                           params_to_tune,
                                           data_train_cross_val,
                                           labels_train_cross_val,
                                           n_iter=20)
            args.max_features = best_params['max_features']
            args.max_depth = best_params['max_depth']
        if args.mode.find('proj') != -1:
            print('\nTraining DTREE with projected data...\n')
        else:
            print('\nTraining DTREE without projected data...\n')
        dtree_model = DecisionTreeClassifier(max_depth=args.max_depth,
                                             max_features=args.max_features)
        dtree_model = dtree_model.fit(data_train, labels_train)
        # Save model
        os.makedirs(args.dtree_params_path, exist_ok=True)
        pkl_file = open(os.path.join(args.dtree_params_path, 'dtree_model.pkl'), 'wb')
        pickle.dump(dtree_model, pkl_file)
        pkl_file.close()
    # Test DTREE classifier
    elif args.mode.find('test_dtree') != -1:
        os.makedirs(args.dtree_analysis_path, exist_ok=True)
        if args.mode.find('proj') != -1:
            print('\nTesting DTREE with projected data...\n')
        else:
            print('\nTesting DTREE without projected data...\n')
        pkl_file = open(os.path.join(args.dtree_params_path, 'dtree_model.pkl'), 'rb')
        dtree_model = pickle.load(pkl_file)
        pred_labels = dtree_model.predict(data_test)
        # Plot ROC curve and Misclassification bars graph
        print('Saving ROC under {:s} folder'.format(args.dtree_analysis_path))
        generate_roc(pred_labels, labels_test, args.dtree_analysis_path)
        results_dict = classification_report(labels_test, pred_labels, output_dict=True)
        accuracy = np.sum(pred_labels == labels_test)/pred_labels.shape[0]
        precision = (results_dict['-1']['precision']+results_dict['1']['precision'])/2
        recall = (results_dict['-1']['recall']+results_dict['1']['recall'])/2
        f_score = (results_dict['-1']['f1-score']+results_dict['1']['f1-score'])/2
        print('Accuracy {:.4f}\nPrecision: {:.4f}\nRecall: {:.4f}\nF1-Score: {:.4f}'.format(
                accuracy, precision, recall, f_score))
        # Save results into spread sheet
        # Workbook is created
        workbook = xlsxwriter.Workbook(os.path.join(args.dtree_analysis_path,
                                                    'results.xlsx'))
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
    parser.add_argument('--data-sampler-params-path', type=str, default='./RBMParams/KDDTrain+_20Percent',
                        help='path location to save RBM trained weights')
    parser.add_argument('--dtree-params-path', type=str, default='./DTREEParams',
                        help='path location to save RBM trained weights')
    parser.add_argument('--dtree-analysis-path', type=str, default='./DTREEAnalysis',
                        help='path location to save RBM trained weights')
    parser.add_argument('--data-save-path', type=str, default='',
                        help='train and test datasets save location')
    parser.add_argument('--datum-save-path', type=str, default='',
                        help='train and test datasets save location')
    parser.add_argument('--mode', type=str, default='train_dtree',
                        help='train svm but first use cross validation technique to infer the optimum train parameter values')
    parser.add_argument('--gen-dataset', type=int, default=1,
                        help='generate or just load (1/0) the train and test NSL-KDD datasets')
    parser.add_argument('--projection-matrix-name', type=str, default='rbm_projection_matrix.npz',
                        help='normalization type that will be applied to the train and test data')
    parser.add_argument('--max-depth', type=int, default=3,
                        help='momentum value to end the training process')
    parser.add_argument('--max-features', type=str, default='auto',
                        help='momentum value to end the training process')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
