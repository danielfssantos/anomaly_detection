import numpy as np
import os

class RBM():
    def __init__(self, params):
        # Initialize RBM train parameters
        self.numhid = params.num_hid_nodes
        self.numdims = params.num_vis_nodes
        self.batch_sz = params.batch_sz
        self.weightcost  = params.weightcost
        self.initialmomentum = params.initialmomentum
        self.finalmomentum   = params.finalmomentum
        self.cd_steps = params.cd_steps     # Qnt. of contrastive divengence iterations
        self.maxepochs = params.num_epochs  # Qnt. of train epochs
        self.epsilonw  = params.epsilonw    # Learning rate for weights
        self.epsilonvb = params.epsilonvb   # Learning rate for biases of visible units
        self.epsilonhb = params.epsilonhb   # Learning rate for biases of hidden units
        # Initializing symmetric weights and biases.
        self.vishid = np.sqrt(2.0 / (self.numdims + self.numhid)) \
                      * np.random.randn(self.numdims, self.numhid) # xavier/glorot initialization
        #self.vishid = 0.1 * np.random.randn(self.numdims, self.numhid) # random initialization
        self.hidbiases  = np.zeros((1, self.numhid))
        self.visbiases  = np.zeros((1, self.numdims))
        self.poshidprobs = np.zeros((self.numdims, self.numhid))
        self.neghidprobs = np.zeros((self.numdims, self.numhid))
        self.posprods = np.zeros((self.numdims, self.numhid))
        self.negprods = np.zeros((self.numdims, self.numhid))
        self.vishidinc  = np.zeros((self.numdims, self.numhid))
        self.hidbiasinc = np.zeros((1, self.numhid))
        self.visbiasinc = np.zeros((1, self.numdims))
        self.lastposhidprobs = []
        if params.train_type.find('gbrbm') != -1:
            self.visdata_sampler = 'gaussian_distribution'
        elif params.train_type.find('bbrbm') != -1:
            self.visdata_sampler = 'uniform_distribution'
        self.params = params

    def sigmoid(self, x):
        return 1./(1 + np.exp(-x))

    def gibbs_step(self, visdata):
        numcases = visdata.shape[0]
        # Sample hidden units from visible units using Bernoulli method
        hidprobs = self.sigmoid(np.dot(visdata, self.vishid) + self.hidbiases)
        poshidstates = hidprobs > np.random.rand(numcases, self.numhid)
        negdataprob = np.dot(poshidstates, self.vishid.T) + self.visbiases
        # Sample visible units from hidden units using Gaussian method
        if self.visdata_sampler == 'gaussian_distribution':
            if self.params.sample_visdata:
                negdata = negdataprob + np.random.randn(*(negdataprob.shape))
            else:
                negdata = negdataprob
        # Sample visible units from hidden units using Bernoulli method
        elif self.visdata_sampler == 'uniform_distribution':
            negdataprob = self.sigmoid(negdataprob)
            if self.params.sample_visdata:
                negdata = negdataprob > np.random.rand(numcases, self.numdims)
            else:
                negdata = negdataprob
        neghidprobs = self.sigmoid(np.dot(negdata, self.vishid) + self.hidbiases)
        return negdata, neghidprobs

    def sample_data(self, input_data, ites=200, qnt_particles=100):
        for i in range(ites):
            input_data, _ = self.gibbs_step(input_data)
        return input_data

    def save(self, save_path):
        os.system('mkdir -p ' + save_path)
        np.savez_compressed(os.path.join(save_path, self.params.train_type + '.npz'), self.vishid, self.visbiases, self.hidbiases)
        return

    def load(self, load_path):
        npzfiles = np.load(os.path.join(load_path, self.params.train_type + '.npz'))
        self.vishid = npzfiles['arr_0']
        self.visbiases = npzfiles['arr_1']
        self.hidbiases = npzfiles['arr_2']
        self.numdims = self.visbiases.size
        self.numhid = self.hidbiases.size
        return

    def cd_train(self, data_train):
        '''
        Method that given an input_data matrix of dims [total_samples, total_features]
        will perform the GRBM generative train minimizing its Free Energy
        using Contrastive Divergence optimization technique
        '''
        for epoch in range(self.maxepochs):
            errsum = 0
            data_train = data_train[np.random.permutation(data_train.shape[0]), :]
            databatches = data_train.reshape(-1, self.batch_sz, self.numdims)
            for i in range(databatches.shape[0]):
                batchdata = databatches[i, :]
                ###### START POSITIVE PHASE ######
                #numcases = batchdata.shape[0]
                poshidprobs = self.sigmoid(np.dot(batchdata, self.vishid) + self.hidbiases)
                if len(self.lastposhidprobs) < len(databatches):
                    self.lastposhidprobs.append(poshidprobs)
                else:
                    self.lastposhidprobs[i] = poshidprobs
                posprods = np.dot(batchdata.T, poshidprobs)
                poshidact = np.sum(poshidprobs, axis=0)
                posvisact = np.sum(batchdata, axis=0)
                ###### END OF POSITIVE PHASE  #####
                poshidstates = poshidprobs > np.random.rand(self.batch_sz, self.numhid)
                ###### START NEGATIVE PHASE ######
                negdata = batchdata
                for i in range(self.cd_steps):
                    negdata, neghidprobs = self.gibbs_step(negdata)
                negprods  = np.dot(negdata.T, neghidprobs)
                neghidact = np.sum(neghidprobs, axis=0)
                negvisact = np.sum(negdata, axis=0)
                ###### END OF NEGATIVE PHASE ######
                if epoch % 25 == 0 or epoch == self.maxepochs - 1:
                    err = np.sum((batchdata-negdata)**2)
                    errsum = err + errsum;
                if epoch > 5:
                    momentum = self.finalmomentum
                else:
                    momentum = self.initialmomentum
                ####### UPDATE WEIGHTS AND BIASES #######
                self.vishidinc = momentum*self.vishidinc + self.epsilonw * ((posprods-negprods)/self.batch_sz - self.weightcost * self.vishid)
                self.visbiasinc = momentum*self.visbiasinc + (self.epsilonvb/self.batch_sz)*(posvisact-negvisact)
                self.hidbiasinc = momentum*self.hidbiasinc + (self.epsilonhb/self.batch_sz)*(poshidact-neghidact)
                self.vishid += self.vishidinc
                self.visbiases += self.visbiasinc
                self.hidbiases += self.hidbiasinc
                ####### END OF UPDATES ######
            if epoch % 25 == 0 or epoch == self.maxepochs - 1:
                print('epoch {:4d} error {:e}'.format(epoch, errsum))

    def pcd_train(self, data_train):
        '''
        Method that given an input_data matrix of dims [total_samples, total_features]
        will perform the RBM generative train minimizing its Free Energy
        using Contrastive Divergence optimization technique
        '''
        # Start persistent markov chains (fantasy particles)
        if self.params.train_type.find('gbrbm') != -1:
            self.fantasy_particles = np.random.randn(self.batch_sz, self.numdims)
        elif self.params.train_type.find('bbrbm') != -1 and self.params.sample_visdata: 
            self.fantasy_particles = np.random.randint(low=0, high=2, size=(self.batch_sz, self.numdims))
        elif self.params.train_type.find('bbrbm') != -1 and not self.params.sample_visdata:
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
                ###### START POSITIVE PHASE ######
                poshidprobs = self.sigmoid(np.dot(batchdata, self.vishid) + self.hidbiases)
                posprods = np.dot(batchdata.T, poshidprobs)
                poshidact = np.sum(poshidprobs, axis=0)
                posvisact = np.sum(batchdata, axis=0)
                if len(self.lastposhidprobs) < databatches.shape[0]:
                    self.lastposhidprobs.append(poshidprobs)
                else:
                    self.lastposhidprobs[i] = poshidprobs
                ###### END OF POSITIVE PHASE  #####
                ###### START NEGATIVE PHASE ######
                for i in range(self.cd_steps):
                    self.fantasy_particles, neghidprobs = self.gibbs_step(self.fantasy_particles)
                negprods  = np.dot(self.fantasy_particles.T, neghidprobs)
                neghidact = np.sum(neghidprobs, axis=0)
                negvisact = np.sum(self.fantasy_particles, axis=0)
                ###### END OF NEGATIVE PHASE ######
                if epoch % 25 == 0 or epoch == self.maxepochs - 1:
                    negdata, _ = self.gibbs_step(batchdata)
                    err = np.sum((batchdata - negdata)**2)
                    errsum = err + errsum;
                if epoch > 5:
                    momentum = self.finalmomentum
                else:
                    momentum = self.initialmomentum
                ####### UPDATE WEIGHTS AND BIASES #######
                self.vishidinc = momentum*self.vishidinc + self.epsilonw * ((posprods-negprods)/self.batch_sz - self.weightcost * self.vishid)
                self.visbiasinc = momentum*self.visbiasinc + (self.epsilonvb/self.batch_sz)*(posvisact-negvisact)
                self.hidbiasinc = momentum*self.hidbiasinc + (self.epsilonhb/self.batch_sz)*(poshidact-neghidact)
                self.vishid += self.vishidinc
                self.visbiases += self.visbiasinc
                self.hidbiases += self.hidbiasinc
                ####### END OF UPDATES ######
            if epoch % 25 == 0 or epoch == self.maxepochs - 1:
                print('epoch {:4d} epsilonw {:e} error {:e}'.format(epoch, self.epsilonw, errsum))

    def train(self, batchdata):
        if self.params.train_type.find('_cd') != -1:
            print('Executing Contrastive Divergence Optimization...')
            self.cd_train(batchdata)
        elif self.params.train_type.find('_pcd') != -1:
            print('Executing Persistent Contrastive Divergence Optimization...')
            self.pcd_train(batchdata)



