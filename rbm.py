import numpy as np
import os


class RBM():
    def __init__(self, params):
        # Initialize RBM train parameters
        self.numhid = params.num_hid_nodes
        self.numdims = params.num_vis_nodes
        self.batch_sz = params.batch_sz
        self.weightcost = params.weightcost
        self.initialmomentum = params.initialmomentum
        self.finalmomentum  = params.finalmomentum
        self.cd_steps = params.cd_steps     # Qnt. of contrastive divengence iterations
        self.maxepochs = params.num_epochs  # Qnt. of train epochs
        self.epsilonw = params.epsilonw    # Learning rate for weights
        self.epsilonvb = params.epsilonvb   # Learning rate for biases of visible units
        self.epsilonhb = params.epsilonhb   # Learning rate for biases of hidden units
        # Initializing weights and biases.
        self.vishid = np.sqrt(2.0 / (self.numdims + self.numhid)) \
                      * np.random.randn(self.numdims, self.numhid)  # xavier/glorot initialization
        self.hidbiases = np.zeros((1, self.numhid))
        self.visbiases = np.zeros((1, self.numdims))
        self.poshidprobs = np.zeros((self.numdims, self.numhid))
        self.neghidprobs = np.zeros((self.numdims, self.numhid))
        self.posprods = np.zeros((self.numdims, self.numhid))
        self.negprods = np.zeros((self.numdims, self.numhid))
        self.vishidinc = np.zeros((self.numdims, self.numhid))
        self.hidbiasinc = np.zeros((1, self.numhid))
        self.visbiasinc = np.zeros((1, self.numdims))
        self.lastposhidprobs = []
        self.params = params

    def sigmoid(self, x):
        return 1./(1 + np.exp(-x))

    def gibbs_step(self, visdata, return_features=False):
        numcases = visdata.shape[0]
        # Sample hidden units from visible units using Bernoulli method
        poshidprobs = self.sigmoid(np.dot(visdata, self.vishid) + self.hidbiases)
        poshidstates = poshidprobs > np.random.rand(numcases, self.numhid)
        negdataprob = np.dot(poshidstates, self.vishid.T) + self.visbiases
        # Sample visible units from hidden units using Bernoulli method
        negdataprob = self.sigmoid(negdataprob)
        if self.params.sample_visdata:
            negdata = negdataprob > np.random.rand(numcases, self.numdims)
        else:
            negdata = negdataprob
        neghidprobs = self.sigmoid(np.dot(negdata, self.vishid) + self.hidbiases)
        if return_features:
            return (negdata, poshidprobs, neghidprobs)
        else:
            return (negdata, neghidprobs)

    def sample_data(self,
                    input_data,
                    ites=1,
                    qnt_particles=100,
                    return_type='samples'):
        if return_type.find('features') != -1:
            return_features = True
        else:
            return_features = False
        for i in range(ites):
            output_tuple = self.gibbs_step(input_data, return_features)
            input_data = output_tuple[0]
        if return_features and return_type.find('neg'):
            return output_tuple[2]
        elif return_features and return_type.find('pos'):
            return output_tuple[1]
        else:
            return output_tuple[0]

    def save(self, save_path):
        os.system('mkdir -p ' + save_path)
        np.savez_compressed(os.path.join(save_path,
                                         self.params.rbm_train_type + '.npz'),
                            self.vishid, self.visbiases, self.hidbiases)
        return

    def load(self, load_path):
        npzfiles = np.load(os.path.join(load_path,
                                        self.params.rbm_train_type + '.npz'))
        self.vishid = npzfiles['arr_0']
        self.visbiases = npzfiles['arr_1']
        self.hidbiases = npzfiles['arr_2']
        self.numdims = self.visbiases.size
        self.numhid = self.hidbiases.size
        return

    def pcd_train(self, data_train):
        '''
        Method that given an input_data matrix of dims [total_samples, total_features]
        will perform the RBM generative train minimizing its Free Energy
        using Contrastive Divergence optimization technique
        '''
        # Start persistent markov chains (fantasy particles)
        if self.params.rbm_train_type.find('bbrbm') != -1 and self.params.sample_visdata:
            self.fantasy_particles = np.random.randint(low=0, high=2, size=(self.batch_sz, self.numdims))
        elif self.params.rbm_train_type.find('bbrbm') != -1 and not self.params.sample_visdata:
            self.fantasy_particles = np.random.rand(self.batch_sz, self.numdims)

        # Execute persistent contrastive divergence algorithm
        for epoch in range(self.maxepochs):
            errsum = 0
            data_train = data_train[np.random.permutation(data_train.shape[0]), :]
            databatches = data_train.reshape(-1, self.batch_sz, self.numdims)
            for i in range(databatches.shape[0]):
                batchdata = databatches[i, :]
                self.epsilonw = np.maximum(self.epsilonw/1.00015, 0.000010)
                self.epsilonvb = np.maximum(self.epsilonvb/1.00015, 0.000010)
                self.epsilonhb = np.maximum(self.epsilonhb/1.00015, 0.000010)
                # START POSITIVE PHASE
                poshidprobs = self.sigmoid(np.dot(batchdata, self.vishid) + self.hidbiases)
                posprods = np.dot(batchdata.T, poshidprobs)
                poshidact = np.sum(poshidprobs, axis=0)
                posvisact = np.sum(batchdata, axis=0)
                if len(self.lastposhidprobs) < databatches.shape[0]:
                    self.lastposhidprobs.append(poshidprobs)
                else:
                    self.lastposhidprobs[i] = poshidprobs
                # END OF POSITIVE PHASE
                # START NEGATIVE PHASE
                for i in range(self.cd_steps):
                    self.fantasy_particles, neghidprobs = self.gibbs_step(self.fantasy_particles)
                negprods = np.dot(self.fantasy_particles.T, neghidprobs)
                neghidact = np.sum(neghidprobs, axis=0)
                negvisact = np.sum(self.fantasy_particles, axis=0)
                # END OF NEGATIVE PHASE
                if epoch % 25 == 0 or epoch == self.maxepochs - 1:
                    negdata, _ = self.gibbs_step(batchdata)
                    err = np.sum((batchdata - negdata)**2)
                    errsum = err + errsum
                if epoch > 5:
                    momentum = self.finalmomentum
                else:
                    momentum = self.initialmomentum
                # UPDATE WEIGHTS AND BIASES
                self.vishidinc = momentum * self.vishidinc + self.epsilonw *\
                                 ((posprods - negprods)/self.batch_sz -\
                                 self.weightcost * self.vishid)
                self.visbiasinc = momentum * self.visbiasinc +\
                                 (self.epsilonvb / self.batch_sz)*\
                                 (posvisact - negvisact)
                self.hidbiasinc = momentum * self.hidbiasinc +\
                                 (self.epsilonhb / self.batch_sz) *\
                                 (poshidact - neghidact)
                self.vishid += self.vishidinc
                self.visbiases += self.visbiasinc
                self.hidbiases += self.hidbiasinc
                # END OF UPDATES
            if epoch % 25 == 0 or epoch == self.maxepochs - 1:
                print('epoch {:4d} epsilonw {:e} error {:e}'.format(epoch, self.epsilonw, errsum))

    def train(self, batchdata):
        print('Executing Persistent Contrastive Divergence Optimization...')
        self.pcd_train(batchdata)
