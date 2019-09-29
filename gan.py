import numpy as np
import tensorflow as tf
import os
from util import *
import time
import math

def optimizer(loss, var_list, ite, learning_rate):
    global_step = tf.Variable(ite, trainable=False)

    # Update mean and average values at training time (usefull to batch normalization)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimize = tf.train.AdamOptimizer(learning_rate).minimize(
            loss,
            global_step=global_step,
            var_list=var_list
        )
    return optimize

def log(x):
    '''
    Sometimes discriminiator outputs can reach values close to
    (or even slightly less than) zero due to numerical rounding.
    This just makes sure that we exclude those values so that we don't
    end up with NaNs during optimisation.
    '''
    return tf.log(tf.maximum(x, 1e-5))

class GAN(object):
    def __init__(self, params):

        tf.reset_default_graph()
        self.params = params
        # Create graph instance isolation
        self.model_graph = tf.Graph()
        # Tie session instance to the created graph scope
        self.session = tf.Session(graph=self.model_graph)
        self.model_path = self.params.data_sampler_params_path
        # Create network inside graph scope
        with self.model_graph.as_default():
            # Define GAN real input placeholder
            self.fake_x = tf.placeholder(tf.float32, shape=(None, self.params.num_vis_nodes))
            # Define GAN fake input placeholder
            self.real_x = tf.placeholder(tf.float32, shape=(None, self.params.num_vis_nodes))
            # Construct GAN
            self.generator = self.build_generator(self.fake_x)
            self.real_data_discriminator = self.build_discriminator(self.real_x)
            self.fake_data_discriminator = self.build_discriminator(self.generator,
                                                                                              reuse_params=True)
            # Setup of GAN optimization
            if self.params.mode == 'train':
                self.d_loss = tf.reduce_mean(-log(self.real_data_discriminator) -\
                                                             log(1 - self.fake_data_discriminator))
                self.g_loss = tf.reduce_mean(-log(self.fake_data_discriminator))
                # Access parameters from the Graph
                self.vars = tf.trainable_variables()
                self.d_params = [v for v in self.vars if v.name.startswith('DISCRIMINATOR/')]
                self.g_params = [v for v in self.vars if v.name.startswith('GENERATOR/')]
                # Associate discriminator and generator correspondent optimizers
                self.d_opt = optimizer(self.d_loss, self.d_params, 0, self.params.lrn_rate)
                self.g_opt = optimizer(self.g_loss, self.g_params, 0, self.params.lrn_rate)
                # Initialize all the network parameters with default values
                self.session.run(tf.local_variables_initializer())
                self.session.run(tf.global_variables_initializer())

    # This defines the GAN Generator architecture
    def build_generator(self, x):
        with tf.variable_scope('GENERATOR'):
            #x = tf.layers.dense(x, 1024, tf.nn.tanh)
            x = tf.layers.dense(x, 512, tf.nn.tanh)
            x = tf.layers.dense(x, 256, tf.nn.tanh)
            x = tf.layers.dense(x, 128, tf.nn.tanh)
            x = tf.layers.dense(x, self.params.num_vis_nodes, tf.nn.sigmoid)
        return x

    # This defines the GAN Discriminator architecture
    def build_discriminator(self, x, reuse_params=False):
        with tf.variable_scope('DISCRIMINATOR', reuse=reuse_params):
            x = tf.layers.dense(x, 512, tf.nn.tanh)
            x = tf.layers.dense(x, 512, tf.nn.tanh)
            x = tf.layers.dense(x, 512, tf.nn.tanh)
            x = tf.layers.dense(x, 1, tf.nn.sigmoid)
        return x

    # Save session in disk.
    def save(self, file_path):
        with self.model_graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.session, file_path)
        return

    # Restore session from disk.
    def load(self, file_path, epoch=-1, init_vars=False):
        with self.model_graph.as_default():
            if init_vars:
                saver = tf.train.Saver(var_list=self.net_params)
            else:
                saver = tf.train.Saver()
            if epoch > -1:
                file_path = file_path + '/Epoch_' + str(epoch) + '/save_files'
            else:
                directory_list = os.listdir(file_path)
                epoch = sorted([int(subdirname.split('_')[1]) for subdirname in directory_list])[-1]
                file_path = file_path + '/Epoch_' + str(epoch) + '/save_files'
            saver.restore(self.session, file_path)
        return

    # Close current session.
    def close(self):
        with self.model_graph.as_default():
            self.session.close()
        return

    def sample_data(self, fakedata):
        sampled_data = self.session.run(self.generator, {
            self.fake_x: fakedata
        })
        return sampled_data


    def train(self, realdata, attack_type, verbose=2):
        # Build the summary Tensor based on the TF collection of Summaries.
        #summary = tf.summary.merge_all()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        #summary_writer = tf.summary.FileWriter(self.params.log_dir_dae, session.graph)
        # Split validation_percent of the data to validate the network training
        if verbose > 1:
            print('Epoch\tloss_gen\tloss_disc')
        start_time = time.time()
        num_steps = math.ceil(realdata.shape[0]/self.params.batch_sz)
        g_losses = []
        d_losses = []
        for epoch in range(self.params.num_epochs):
            sorted_idx = np.random.permutation(realdata.shape[0])
            realdata = realdata[sorted_idx, :]
            g_loss_sum = 0
            d_loss_sum = 0
            for step in range(num_steps):
                if step == num_steps - 1:
                    batchrealdata = realdata[step * self.params.batch_sz : , :]
                else:
                    batchrealdata = realdata[step * self.params.batch_sz : (step + 1) * self.params.batch_sz, :]
                batchfakedata = np.random.randn(*(batchrealdata.shape))

                # Update Discriminator
                d_loss, _ = self.session.run([self.d_loss, self.d_opt], {
                self.real_x: batchrealdata,
                self.fake_x: batchfakedata
                })

                # Update generator
                batchfakedata = np.random.randn(*(batchrealdata.shape))
                g_loss, _ = self.session.run([self.g_loss, self.g_opt], {
                    self.fake_x: batchfakedata
                })

                # Visualize network parameters
                if verbose == 3 and step == num_steps // 2:
                    # Sample data with generator
                    batchfakedata = np.random.randn(*(batchrealdata.shape))
                    sampled_data = self.sample_data(batchfakedata)
                    plot_kde_distributions(realdata, sampled_data, attack_type, 'GAN', plot_path='')

                if verbose > 1:
                    g_loss_sum += g_loss
                    d_loss_sum += d_loss
            g_losses.append(g_loss_sum/num_steps)
            d_losses.append(d_loss_sum/num_steps)
            if(epoch % self.params.log_every == 0) or (epoch == self.params.num_epochs - 1):
                if verbose > 1:
                    print('{}:\t{:e}\t{:e}'.format(epoch,
                                                                    sum(g_losses)/len(g_losses),
                                                                    sum(d_losses)/len(d_losses)))
                if self.model_path:
                    folder_name = 'Epoch_' + str(epoch)
                    folder_path = os.path.join(self.model_path, folder_name)
                    os.makedirs(folder_path, exist_ok=True)
                    self.save(folder_path + '/save_files')
                g_losses = []
                d_losses = []
        if verbose > 1:
            print("--- %s seconds ---\n" % (time.time() - start_time))
        return


