import os
import numpy as np
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def main(args):
    attacks_dict = {0: 'normal', 1: 'u2r', 2: 'r2l', 3: 'dos', 4: 'probe'}
    colors = ['r', 'g', 'dimgrey', 'gold', 'k', 'blue']
    if args.mode == 'kde':
        out_path = os.path.join(args.samples_path, 'RBMSamplesDistributions')
        os.makedirs(out_path, exist_ok=True)
        for j in range(5):
            real_data = np.load(os.path.join(args.samples_path,
                                             'RBM'+str(args.exp_ids[0])+\
                                             'neurons/'+attacks_dict[j]+'/real_'+\
                                             attacks_dict[j]+'_data.npy'))
            sns.kdeplot(real_data,  shade=True, label="Real Data", color=colors[0]);
            for pos, i in enumerate(args.exp_ids):
                sampled_data = np.load(os.path.join(args.samples_path,
                                                    'RBM'+str(i)+\
                                                    'neurons/'+attacks_dict[j]+'/sampled_'+\
                                                    attacks_dict[j]+'_data.npy'))
                ax = sns.kdeplot(sampled_data,  shade=True,
                                 label="Sampled Data {:d} Neurons".format(i),
                                 color=colors[pos + 1]);
            ax.set_title('{:s} Distributions'.format(attacks_dict[j]))
            plt.savefig(os.path.join(out_path, attacks_dict[j] + '_distributions.png'))
            ax.clear()
    elif args.mode == 'roc':
        xy_arr = np.load(os.path.join(args.results_path, 'WithoutProjection/roc.npy'))
        legend_terms = ['WithoutProjection']
        ax = plt.figure('ROCs')
        plt.plot(xy_arr[:, 0], xy_arr[:, 1], color=colors[0])
        for pos, i in enumerate(args.exp_ids):
            xy_arr = np.load(os.path.join(args.results_path,
                                          'WithProjection/RBM' + str(i)+\
                                          'neurons/roc.npy'))
            plt.plot(xy_arr[:, 0], xy_arr[:, 1], color=colors[pos + 1])
            legend_terms.append('RBM {:d} neurons'.format(i))
        plt.title('ROC curves comparison')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim(left=0, right=1)
        plt.ylim(bottom=0, top=1)
        plt.legend(legend_terms)
        plt.savefig(os.path.join(args.results_path, 'rocs.png'))
    elif args.mode == 'scatter_pca':
        colors = ['r', 'cyan', 'b', 'lime', 'yellow']
        markers = ['+', '>', 'd', 'o', 'x']
        pca_2D = PCA(n_components=2, svd_solver='full')
        # Use PCA to project the features generated by the RBMs
        for n in args.exp_ids:
            legend_terms = []
            plt.figure('PCA Projections RBM '+str(n))
            for j in range(5):
                data = np.loadtxt(os.path.join(args.samples_path,
                                               'RBM'+str(n)+\
                                               'neurons/'+attacks_dict[j]+'/features_'+\
                                               attacks_dict[j]+'_data.txt'))
                pca_2D.fit(data)
                proj_data = np.dot(data, pca_2D.components_.T)
                plt.scatter(proj_data[:, 0], proj_data[:, 1], s=25, c=colors[j], marker=markers[j])
                legend_terms.append(attacks_dict[j])
            plt.title('PCA Projections RBM Features '+str(n))
            plt.legend(legend_terms, loc='center left', bbox_to_anchor=(1.0, 0.5))
            plt.xlabel('First Component')
            plt.ylabel('Second Component')
            plt.savefig(os.path.join(args.samples_path, 'RBM'+str(n) + \
                                     'neurons/features_2Dpca_proj.png'), bbox_inches='tight')
        # Use PCA to project the original data
        legend_terms = []
        plt.figure('PCA Original Data')
        for j in range(5):
            data = np.loadtxt(os.path.join(args.samples_path,
                                           'RBM10neurons/'+attacks_dict[j]+'/original_'+\
                                           attacks_dict[j]+'_data.txt'))
            pca_2D.fit(data)
            proj_data = np.dot(data, pca_2D.components_.T)
            plt.scatter(proj_data[:, 0], proj_data[:, 1], s=25, c=colors[j], marker=markers[j])
            legend_terms.append(attacks_dict[j])
        plt.title('PCA Projections Original Data')
        plt.legend(legend_terms, loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.xlabel('First Component')
        plt.ylabel('Second Component')
        plt.savefig(os.path.join(args.samples_path, 'original_data_2Dpca_proj.png'), bbox_inches='tight')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples-path', type=str, default='./SamplingAnalysis/KDDTrain+',
                        help='path location to save RBM trained weights')
    parser.add_argument('--results-path', type=str, default='./SVMAnalysis/KDDTrain+',
                        help='path location to save RBM trained weights')
    parser.add_argument('--exp-ids', nargs='+', type=int, default=[10, 20, 38, 80, 100],
                        help='experiments ids')
    parser.add_argument('--mode', type=str, default='kde',
                        help='specify the kind of output graph')

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
