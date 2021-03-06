import os
import numpy as np
import argparse
from rbm import RBM
from util import load_splitted_nsl_kdd_dataset
from util import plot_kde_distributions
from sklearn import preprocessing


def main(args):
    if args.mode in ['train', 'gen_samples', 'gen_features'] and args.gen_dataset:
        datum_train = load_splitted_nsl_kdd_dataset(args.train_data_file_path,
                                                    args.metadata_file_path)
        # Save processed data
        os.makedirs(args.train_data_save_path, exist_ok=True)
        np.savez_compressed(os.path.join(args.train_data_save_path,
                                         'generative_kdd_nsl_processed.npz'),
                            datum_train)
    elif args.mode.find('gen_proj_matrix') == -1:
        npzfiles = np.load(os.path.join(args.train_data_save_path,
                                        'generative_kdd_nsl_processed.npz'),
                           allow_pickle=True)
        datum_train = npzfiles['arr_0']
    if args.mode.find('gen_proj_matrix') == -1:
        # Chose the specific attack data to train the RBM
        attack_type = args.rbm_train_type.split('_')[-1]
        if attack_type == 'normal':
            data_train = datum_train[0]
            print('{:s} data {:d} samples'.format(attack_type, len(datum_train[0])))
        elif attack_type == 'u2r':
            data_train = datum_train[1]
            print('{:s} data {:d} samples'.format(attack_type, len(datum_train[1])))
        elif attack_type == 'r2l':
            data_train = datum_train[2]
            print('{:s} data {:d} samples'.format(attack_type, len(datum_train[2])))
        elif attack_type == 'dos':
            data_train = datum_train[3]
            print('{:s} data {:d} samples'.format(attack_type, len(datum_train[3])))
        elif attack_type == 'probe':
            data_train = datum_train[4]
            print('{:s} data {:d} samples'.format(attack_type, len(datum_train[4])))
        # Adjust dataset size to fit into batch_sz
        if data_train.shape[0] < args.batch_sz:
            args.batch_sz = data_train.shape[0]//5
        else:
            max_train_data = data_train.shape[0] -\
                            (data_train.shape[0] % args.batch_sz)
            data_train = data_train[0: max_train_data, :]
        # Normalize data according to train_type
        data_train = preprocessing.normalize(data_train, norm='l2')
    # Instantiate bbrbm model and train it using cd or pcd algorithm
    if args.mode == 'train':
        print('{:d} train data samples loaded from category {:s}'.format(data_train.shape[0], attack_type))
        args.num_vis_nodes = data_train.shape[1]
        rbm_model = RBM(args)
        rbm_model.train(data_train)
        rbm_model.save(args.rbm_params_path)
        print('RBM train done...')
    # Use the trained RBM model to sample --batch-sz elements
    # from --sample-ites number of sampling iterations.
    elif args.mode == 'gen_samples':
        os.makedirs(args.results_samples_path, exist_ok=True)
        np.savetxt(os.path.join(args.results_samples_path,
                                'original_'+attack_type+'_data.txt'),
                   data_train, fmt='%.4f')
        args.num_vis_nodes = data_train.shape[1]
        args.batch_sz = 100
        rbm_model = RBM(args)
        rbm_model.load(args.rbm_params_path)
        final_sampled_data = np.zeros((args.batch_sz, data_train.shape[1]))
        for i in range(args.sample_data_repetitions):
            print('Ite: {:d} from {:d}'.format(i + 1, args.sample_data_repetitions))
            print('Sampling data from {:s} using {:d} iterations\n'.format(args.rbm_train_type, args.sample_ites))
            if args.rbm_train_type.find('bbrbm') != -1 and args.sample_visdata:
                sampled_data = np.random.randint(low=0,
                                                 high=2,
                                                 size=(args.batch_sz,
                                                       rbm_model.numdims))
            elif args.rbm_train_type.find('bbrbm') != -1 and not args.sample_visdata:
                sampled_data = np.random.rand(args.batch_sz, rbm_model.numdims)
            final_sampled_data += rbm_model.sample_data(sampled_data,
                                                        ites=args.sample_ites)
        sampled_data = final_sampled_data/args.sample_data_repetitions
        np.savetxt(os.path.join(args.results_samples_path,
                                'sampled_'+attack_type+'_data.txt'),
                   sampled_data, fmt='%.4f')
        plot_kde_distributions(data_train,
                               sampled_data,
                               attack_type,
                               args.results_samples_path)
    elif args.mode == 'gen_features':
        os.makedirs(args.results_features_path, exist_ok=True)
        np.savetxt(os.path.join(args.results_features_path,
                                'original_' + attack_type + '_data.txt'),
                   data_train, fmt='%.4f')
        args.num_vis_nodes = data_train.shape[1]
        rbm_model = RBM(args)
        rbm_model.load(args.rbm_params_path)
        data_features = rbm_model.sample_data(data_train,
                                              ites=args.sample_ites,
                                              return_type='pos_features')
        np.savetxt(os.path.join(args.results_features_path,
                                'features_' + attack_type + '_data.txt'),
                   data_features, fmt='%.4f')
    elif args.mode == 'gen_proj_matrix':
        print('Generating projection matrix')
        proj_matrix = []
        attack_names = {0: 'normal', 1: 'u2r', 2: 'r2l', 3: 'dos', 4: 'probe'}
        rbm_train_type_aux = args.rbm_train_type
        for i in range(5):
            args.rbm_train_type = rbm_train_type_aux
            args.rbm_train_type = args.rbm_train_type + '_' + attack_names[i]
            rbm_model = RBM(args)
            rbm_model.load(args.rbm_params_path)
            proj_matrix.append(np.concatenate((rbm_model.vishid,
                                               rbm_model.hidbiases),
                                              axis=0))
        proj_matrix = np.hstack(proj_matrix)
        np.savez_compressed(os.path.join(args.rbm_params_path,
                                         'rbm_projection_matrix.npz'),
                            proj_matrix)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data-file-path', type=str, default='./NSL_KDD_Dataset/KDDTrain+_20Percent.txt',
                        help='path to the train data .csv file')
    parser.add_argument('--test-data-file-path', type=str, default='./NSL_KDD_Dataset/KDDTest+.txt',
                        help='path to the test data .csv file')
    parser.add_argument('--metadata-file-path', type=str, default='./NSL_KDD_Dataset/training_attack_types.txt',
                        help='path to the test data .csv file')
    parser.add_argument('--rbm-params-path', type=str, default='./RBMParams/KDDTrain+_20Percent',
                        help='path location to save RBM trained weights')
    parser.add_argument('--mlp-params-path', type=str, default='./RBMParams/KDDTrain+_20Percent',
                        help='path location to save RBM trained weights')
    parser.add_argument('--lrn-rate', type=float, default=0.001,
                        help='path location to save RBM trained weights')
    parser.add_argument('--train-data-save-path', type=str, default='./TrainData/KDDTrain+_20Percent',
                        help='train and test datasets save location')
    parser.add_argument('--results-samples-path', type=str, default='./SamplingAnalysis/KDDTrain+_20Percent',
                        help='path location to save RBM trained weights')
    parser.add_argument('--results-features-path', type=str, default='./SamplingAnalysis/KDDTrain+_20Percent',
                        help='path location to save RBM trained weights')
    parser.add_argument('--mode', type=str, default='train',
                        help='train/test rbm')
    parser.add_argument('--sample-data-repetitions', type=int, default=100,
                        help='number of times to execute the sampling process')
    parser.add_argument('--gen-dataset', type=int, default=0,
                        help='generate or just load (1/0) the train and test datasets')
    parser.add_argument('--rbm-train-type', type=str, default='bbrbm_pcd_dos',
                        help='[ bbrbm_pcd_attack_type ] train types\
                                where attack_type in [ normal, dos, u2r, l2r, probe ]')
    parser.add_argument('--sample-visdata', type=int, default=0,
                        help='sample or not (1/0) visible data during gibbs sampling')
    parser.add_argument('--num-hid-nodes', type=int, default=100,
                        help='maximum quantity of hidden layer nodes')
    parser.add_argument('--num-vis-nodes', type=int, default=38,
                        help='maximum quantity of hidden layer nodes')
    parser.add_argument('--cd-steps', type=int, default=1,
                        help='number of CD iteartions')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='total of training epochs')
    parser.add_argument('--epsilonw', type=float, default=0.0005,
                        help='weight matrix learning rate value')
    parser.add_argument('--epsilonvb', type=float, default=0.0005,
                        help='visible layer biases learning rate')
    parser.add_argument('--epsilonhb', type=float, default=0.0005,
                        help='hidden layer biases learning rate')
    parser.add_argument('--sample-ites', type=int, default=1,
                        help='number of sample iterations')
    parser.add_argument('--batch-sz', type=int, default=100,
                        help='maximum quantity of samples per batch')
    parser.add_argument('--weightcost', type=float, default=2e-4,
                        help='controls the weight decay velocity')
    parser.add_argument('--initialmomentum', type=float, default=.5,
                        help='momentum value to start the training process')
    parser.add_argument('--finalmomentum', type=float, default=.9,
                        help='momentum value to end the training process')
    parser.add_argument('--log-every', type=int, default=10,
                        help='save params every log-every epochs')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
