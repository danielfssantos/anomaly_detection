import os, sys
import numpy as np
import argparse
from gan import GAN
from util import *

def main(args):
    os.system('cls' if os.name == 'nt' else 'clear')
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
    if args.mode == 'train' and args.gen_dataset:
        datum_train = load_nsl_kdd_splitted_dataset(args.train_data_file_path, args.metadata_file_path)
        # Save processed data
        os.system('mkdir -p ' + args.train_data_save_path)
        np.savez_compressed(os.path.join(args.train_data_save_path, 'generative_kdd_nsl_processed.npz'), datum_train)
    elif args.mode in ['train', 'test']:
        npzfiles = np.load(os.path.join(args.train_data_save_path, 'generative_kdd_nsl_processed.npz'), allow_pickle=True)
        datum_train = npzfiles['arr_0']
    print('Selecting train valid samples')
    datum_train = select_valid_samples(datum_train)
    # Chose the specific kind of data to train the GAN
    attack_type = args.gan_train_type.split('_')[-1]
    if attack_type == 'normal':
        data_train = datum_train[0]
    elif attack_type == 'u2r':
        data_train = datum_train[1]
    elif attack_type == 'r2l':
        data_train = datum_train[2]
    elif attack_type == 'dos':
        data_train = datum_train[3]
    elif attack_type == 'probe':
        data_train = datum_train[4]
    # Adjust dataset size to fit into batch_sz
    if data_train.shape[0] < args.batch_sz:
        args.batch_sz = data_train.shape[0]//5
    max_train_data = data_train.shape[0] - (data_train.shape[0] % args.batch_sz)
    data_train = data_train[0 : max_train_data, :]
    # Normalize data according to train_type
    data_train = min_max_norm(data_train)
    # Instantiate gbrbm or bbrbm model and train it using cd or pcd algorithm
    if args.mode == 'train':
        print('{:d} train data samples loaded from category {:s}'.format(data_train.shape[0], attack_type))
        args.num_vis_nodes = data_train.shape[1]
        gan_model = GAN(args)
        gan_model.train(data_train, attack_type, verbose=args.verbose)
        gan_model.save(args.data_sampler_params_path)
        print('Train step done...')
    # Use the trained GAN model to sample --batch-sz elements from --sample-ites
    # number of sampling iterations.
    elif args.mode == 'test':
        os.system('mkdir -p ' + args.results_path)
        np.savetxt(os.path.join(args.results_path, 'original_'+attack_type+'_data.txt'), data_train, fmt='%.4f')
        args.num_vis_nodes = data_train.shape[1]
        gan_model = GAN(args)
        gan_model.load(args.data_sampler_params_path)
        final_sampled_data = np.zeros((args.batch_sz, data_train.shape[1]))
        for i in range(args.sample_data_repetitions):
            print('Ite: {:d} from {:d}'.format(i + 1, args.sample_data_repetitions))
            print('Sampling data from {:s}\n'.format(args.gan_train_type))
            sampled_data = np.random.randn(args.batch_sz, args.num_vis_nodes)
            final_sampled_data += gan_model.sample_data(sampled_data)
        sampled_data = final_sampled_data/args.sample_data_repetitions
        np.savetxt(os.path.join(args.results_path, 'sampled_'+attack_type+'_data.txt'), sampled_data, fmt='%.4f')
        plot_kde_distributions(data_train, sampled_data, attack_type, 'GAN', args.results_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data-file-path', type=str, default='./NSL_KDD_Dataset/KDDTrain+_20Percent.txt',
                        help='path to the train data .csv file')
    parser.add_argument('--test-data-file-path', type=str, default='./NSL_KDD_Dataset/KDDTest+.txt',
                        help='path to the test data .csv file')
    parser.add_argument('--metadata-file-path', type=str, default='./NSL_KDD_Dataset/training_attack_types.txt',
                        help='path to the test data .csv file')
    parser.add_argument('--data-sampler-params-path', type=str, default='./GANParams/KDDTrain+_20Percent',
                        help='path location to save GAN trained weights')
    parser.add_argument('--train-data-save-path', type=str, default='./TrainData/KDDTrain+_20Percent',
                        help='train and test datasets save location')
    parser.add_argument('--results-path', type=str, default='./SamplingAnalysis/KDDTrain+_20Percent/GAN',
                        help='path location to save GAN trained weights')
    parser.add_argument('--mode', type=str, default='train',
                        help='train/test gan')
    parser.add_argument('--sample-data-repetitions', type=int, default=100,
                        help='number of times to execute the sampling process')
    parser.add_argument('--gen-dataset', type=int, default=0,
                        help='generate or just load (1/0) the train and test datasets')
    parser.add_argument('--gan-train-type', type=str, default='gan_dos',
                        help='gan_attack_type train types\
                                    where attack_type in [ normal, dos, u2r, l2r, probe ]')
    parser.add_argument('--num-epochs', type=int, default=50,
                        help='total of training epochs')
    parser.add_argument('--lrn-rate', type=float, default=0.001,#bernoulli-bernoulli 0.005
                        help='weight matrix learning rate value')
    parser.add_argument('--batch-sz', type=int, default=1,
                        help='maximum quantity of samples per batch')
    parser.add_argument('--verbose', type=int, default=1,
                        help='level of information to display')
    parser.add_argument('--log-every', type=int, default=10,
                        help='save params every log-every epochs')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())