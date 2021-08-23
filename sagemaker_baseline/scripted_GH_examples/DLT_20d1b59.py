# Standard Library
import argparse
import os
import time

# Third Party
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pandas as pd
from tensorflow.python.training.basic_session_run_hooks import _as_graph_element

# First Party
import smdebug.tensorflow as smd

CLASSIFICATION_KEY, REGRESSION_KEY, EPSILON = 'classification', 'regression', 1e-16


def load_mnist_dataset(mode='supervised'):
    """ Load the MNIST handwritten digits dataset.

    :param mode: 'supervised' or 'unsupervised' mode

    :return: train, validation, test data:
            for (X, y) if 'supervised',
            for (X) if 'unsupervised'
    """

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Training set
    trX = mnist.train.images
    trY = mnist.train.labels

    # Validation set
    vlX = mnist.validation.images
    vlY = mnist.validation.labels

    # Test set
    teX = mnist.test.images
    teY = mnist.test.labels

    if mode == 'supervised':
        return trX, trY, vlX, vlY, teX, teY

    elif mode == 'unsupervised':
        return trX, vlX, teX

def load_cifar10_dataset(cifar_dir, mode='supervised'):
    """ Load the cifar10 dataset.

    :param cifar_dir: path to the dataset directory (cPicle format from: https://www.cs.toronto.edu/~kriz/cifar.html)
    :param mode: 'supervised' or 'unsupervised' mode

    :return: train, test data:
            for (X, y) if 'supervised',
            for (X) if 'unsupervised'
    """

    # Training set
    trX = None
    trY = np.array([])

    # Test set
    teX = np.array([])
    teY = np.array([])

    for fn in os.listdir(cifar_dir):

        if not fn.startswith('batches') and not fn.startswith('readme'):
            fo = open(cifar_dir + fn, 'rb')
            data_batch = cPickle.load(fo)
            fo.close()

            if fn.startswith('data'):

                if trX is None:
                    trX = data_batch['data']
                    trY = data_batch['labels']
                else:
                    trX = np.concatenate((trX, data_batch['data']), axis=0)
                    trY = np.concatenate((trY, data_batch['labels']), axis=0)

            if fn.startswith('test'):
                teX = data_batch['data']
                teY = data_batch['labels']

    if mode == 'supervised':
        return trX, trY, teX, teY

    elif mode == 'unsupervised':
        return trX, teX

class DataLoaderFromArrays:

    def __init__(self, features, targets, homogeneous=True, 
                problem_type=CLASSIFICATION_KEY, shuffle=True, 
                one_hot=True, normalization=True, target_scaling=True, target_range=[0.0,1.0]):
        self.curr_pos = 0
        self.target_divider = None
        self.rows_count = features.shape[0]
        self.homogeneous = homogeneous
        self.to_shuffle = shuffle
        self.problem_type = problem_type
        self.normalization = normalization
        self.target_scaling = target_scaling
        self.target_range = target_range
        self.one_hot = one_hot
        self.features = self.normalize(features)
        self.targets = self.preprocess(targets)
        
    def next_batch(self, batch_size):
        if self.curr_pos+batch_size >= self.rows_count:
            batch = self.features[self.curr_pos:self.rows_count], self.targets[self.curr_pos:self.rows_count]
            self.curr_pos = 0
            if self.to_shuffle:
                self.shuffle()
            return batch
        batch = self.features[self.curr_pos:self.curr_pos+batch_size], self.targets[self.curr_pos:self.curr_pos+batch_size]
        self.curr_pos += batch_size
        return batch
    
    def shuffle(self):
        indices = np.arange(0, self.rows_count) 
        np.random.shuffle(indices) 
        self.features = self.features[indices]
        self.targets = self.targets[indices]

    def deprocess(self, perf):
        if self.problem_type == REGRESSION_KEY and self.target_scaling:
            deprocessed_perf = (perf/(self.target_range[1]-self.target_range[0]))*self.target_divider
            return deprocessed_perf
        else:
            return perf

    def preprocess(self, targets):
        if self.problem_type == REGRESSION_KEY and self.target_scaling:
            mi = targets.min(axis=0)
            divider = targets.max(axis=0) - mi
            if isinstance(divider, np.ndarray):
                divider[divider==0.0] = EPSILON
            else:
                divider = EPSILON if divider==0.0 else divider
            targets = self.target_range[0] + np.float32((targets-mi)/divider)*(self.target_range[1]-self.target_range[0])
            self.target_divider = divider
        elif self.one_hot:
            onehot_targets = np.zeros((self.rows_count, targets.max()+1))
            onehot_targets[np.arange(self.rows_count),targets] = 1
            targets = onehot_targets
        return targets

    def normalize(self, data):
        if not self.normalization:
            return data
        if self.homogeneous:
            mi = data.min()
            divider = data.max() - mi
            divider = EPSILON if divider==0.0 else divider
        else:
            mi = data.min(axis=0)
            divider = data.max(axis=0) - mi
            if isinstance(divider, np.ndarray):
                divider[divider==0.0] = EPSILON
            else:
                divider = EPSILON if divider==0.0 else divider
        return np.float32((data-mi)/divider)

    def get_epochs(self, batch_size):
        if self.rows_count % batch_size != 0:
            return self.rows_count // batch_size + 1  
        else:
            return self.rows_count // batch_size 
    
    def reset_cursor(self):
        self.curr_pos = 0

class ConvolutionalNetwork:

    """ Implementation of Convolutional Neural Networks using TensorFlow.
    The interface of the class is sklearn-like.
    """

    def __init__(self, layers, model_name='convnet', main_dir='convnet',
                 loss_func='cross_entropy', num_epochs=10, batch_size=10, dataset='mnist',
                 opt='gradient_descent', learning_rate=0.01, momentum=0.5, dropout=0.5, verbose=1):
        """
        :param layers: string used to build the model.
            This string is a comma-separate specification of the layers of the network.
            Supported values:
                conv2d-FX-FY-Z-S: 2d convolution with Z feature maps as output and FX x FY filters. S is the strides size
                maxpool-X: max pooling on the previous layer. X is the size of the max pooling
                full-X: fully connected layer with X units
                softmax: softmax layer
            For example:
                conv2d-5-5-32,maxpool-2,conv2d-5-5-64,maxpool-2,full-128,full-128,softmax
        :param loss_func: Loss function. ['mean_squared', 'cross_entropy']
        :param num_epochs: Number of epochs
        :param batch_size: Size of each mini-batch
        :param dataset: Which dataset to use. ['mnist', 'cifar10', 'custom']
        :param opt: Which tensorflow optimizer to use. ['gradient_descent', 'momentum', 'ada_grad', 'adam']
        :param learning_rate: Initial learning rate
        :param momentum: Momentum parameter
        :param dropout: Dropout parameter
        :param verbose: Level of verbosity. 0 - silent, 1 - print accuracy.
        """
        self.model_name = model_name
        self.main_dir = main_dir
        self.layers = layers
        self.loss_func = loss_func
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.dataset = dataset
        self.opt = opt
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dropout = dropout
        self.verbose = verbose

        self.input_data = None
        self.input_labels = None

        self.W_vars = None
        self.B_vars = None

        self.logits = None
        self.softmax_out = None
        self.train_step = None
        self.accuracy = None
        self.keep_prob = None

        self.tf_session = None
        self.tf_saver = None

    def build_model(self, n_features, n_classes, original_shape):

        """ Creates the computational graph of the model.
        :param n_features: Number of features.
        :param n_classes: number of classes.
        :param original_shape: original shape of the images.
        :return: self
        """

        self._create_placeholders(n_features, n_classes)
        self._create_layers(n_classes, original_shape)

        self._create_cost_function_node()
        self._create_train_step_node()

        self._create_test_node()

    def _create_placeholders(self, n_features, n_classes):

        """ Create the TensorFlow placeholders for the model.
        :param n_features: number of features of the first layer
        :param n_classes: number of classes
        :return: self
        """

        self.input_data = tf.placeholder('float', [None, n_features], name='x-input')
        self.input_labels = tf.placeholder('float', [None, n_classes], name='y-input')
        self.keep_prob = tf.placeholder('float', name='keep-probs')

    def _create_layers(self, n_classes, original_shape):

        """ Create the layers of the model from self.layers.
        :param n_classes: number of classes
        :param original_shape: original shape of the images. [width, height, channels]
        :return: self
        """

        next_layer_feed = tf.reshape(self.input_data, [-1, original_shape[0], original_shape[1], original_shape[2]])
        prev_output_dim = original_shape[2]
        first_full = True  # this flags indicates whether we are building the first fully connected layer

        self.W_vars = []
        self.B_vars = []

        for i, l in enumerate(self.layers.split(',')):

            node = l.split('-')
            node_type = node[0]

            if node_type == 'conv2d':

                # ################### #
                # Convolutional Layer #
                # ################### #

                # fx, fy = shape of the convolutional filter
                # feature_maps = number of output dimensions
                fx, fy, feature_maps, stride = int(node[1]), int(node[2]), int(node[3]), int(node[4])

                print('Building Convolutional layer with %d input channels and %d %dx%d filters with stride %d' %
                      (prev_output_dim, feature_maps, fx, fy, stride))

                # Create weights and biases
                W_conv = self.weight_variable([fx, fy, prev_output_dim, feature_maps])
                b_conv = self.bias_variable([feature_maps])
                self.W_vars.append(W_conv)
                self.B_vars.append(b_conv)

                # Convolution and Activation function
                h_conv = tf.nn.relu(self.conv2d(next_layer_feed, W_conv, stride) + b_conv, name='relu')

                # keep track of the number of output dims of the previous conv. layer
                prev_output_dim = feature_maps
                # output node of the last layer
                next_layer_feed = h_conv

            elif node_type == 'maxpool':

                # ################# #
                # Max Pooling Layer #
                # ################# #

                ksize = int(node[1])

                print('Building Max Pooling layer with size %d' % ksize)

                next_layer_feed = self.max_pool(next_layer_feed, ksize)

            elif node_type == 'full':

                # ####################### #
                # Densely Connected Layer #
                # ####################### #

                if first_full:  # first fully connected layer

                    dim = int(node[1])
                    shp = next_layer_feed.get_shape()
                    tmpx = shp[1].value
                    tmpy = shp[2].value
                    fanin = tmpx * tmpy * prev_output_dim

                    print('Building fully connected layer with %d in units and %d out units' %
                          (fanin, dim))

                    W_fc = self.weight_variable([fanin, dim])
                    b_fc = self.bias_variable([dim])
                    self.W_vars.append(W_fc)
                    self.B_vars.append(b_fc)

                    h_pool_flat = tf.reshape(next_layer_feed, [-1, fanin])
                    h_fc = tf.nn.relu(tf.matmul(h_pool_flat, W_fc) + b_fc, name='relu')
                    h_fc_drop = tf.nn.dropout(h_fc, self.keep_prob)

                    prev_output_dim = dim
                    next_layer_feed = h_fc_drop

                    first_full = False

                else:  # not first fully connected layer

                    dim = int(node[1])
                    W_fc = self.weight_variable([prev_output_dim, dim])
                    b_fc = self.bias_variable([dim])
                    self.W_vars.append(W_fc)
                    self.B_vars.append(b_fc)

                    h_fc = tf.nn.relu(tf.matmul(next_layer_feed, W_fc) + b_fc, name='relu')
                    h_fc_drop = tf.nn.dropout(h_fc, self.keep_prob)

                    prev_output_dim = dim
                    next_layer_feed = h_fc_drop

            elif node_type == 'softmax':

                # ############# #
                # Softmax Layer #
                # ############# #

                print('Building softmax layer with %d in units and %d out units' %
                      (prev_output_dim, n_classes))

                W_sm = self.weight_variable([prev_output_dim, n_classes])
                b_sm = self.bias_variable([n_classes])
                self.W_vars.append(W_sm)
                self.B_vars.append(b_sm)
                self.logits = tf.matmul(next_layer_feed, W_sm) + b_sm
                self.softmax_out = tf.nn.softmax(self.logits)

    def _create_cost_function_node(self):

        """ Create the cost function node.
        :return: self
        """

        with tf.name_scope("cost"):
            if self.loss_func == 'cross_entropy':
                self.cost = - tf.reduce_mean(self.input_labels * tf.log(tf.clip_by_value(self.logits, 1e-10, float('inf'))) +
                                        (1 - self.input_labels) * tf.log(tf.clip_by_value(1 - self.logits, 1e-10, float('inf'))))
                #self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.softmax_out, labels=self.input_labels))
            elif self.loss_func == 'mean_squared':
                self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.input_labels - self.softmax_out)))
            else:
                self.cost = None

    def _create_train_step_node(self):

        """ Create the training step node of the network.
        :return: self
        """

        with tf.name_scope("train"):
            if self.opt == 'gradient_descent':
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif self.opt == 'ada_grad':
                self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            elif self.opt == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.momentum)
            elif self.opt == 'adam':
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = hook.wrap_optimizer(self.optimizer)
        self.train_step = self.optimizer.minimize(self.cost)

    def _create_test_node(self):

        """ Create the test node of the network.
        :return: self
        """
        with tf.name_scope("test"):
            correct_prediction = tf.equal(tf.argmax(self.softmax_out, 1), tf.argmax(self.input_labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape=shape, stddev=0.1)
        return tf.Variable(initial, name='weight')

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name='bias')

    @staticmethod
    def conv2d(x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

    @staticmethod
    def max_pool(x, dim):
        return tf.nn.max_pool(x, ksize=[1, dim, dim, 1], strides=[1, dim, dim, 1], padding='SAME')

# #################### #
#   Flags definition   #
# #################### #
flags = tf.app.flags
FLAGS = flags.FLAGS

# Global configuration
flags.DEFINE_string('dataset', 'mnist', 'Which dataset to use. ["mnist", "cifar10", "custom"]')
flags.DEFINE_string('original_shape', '28,28,1', 'Original shape of the images in the dataset.')
flags.DEFINE_string('train_dataset', '', 'Path to train set .npy file.')
flags.DEFINE_string('train_labels', '', 'Path to train labels .npy file.')
flags.DEFINE_string('valid_dataset', '', 'Path to valid set .npy file.')
flags.DEFINE_string('valid_labels', '', 'Path to valid labels .npy file.')
flags.DEFINE_string('test_dataset', '', 'Path to test set .npy file.')
flags.DEFINE_string('test_labels', '', 'Path to test labels .npy file.')
flags.DEFINE_string('cifar_dir', '', 'Path to the cifar 10 dataset directory.')
flags.DEFINE_string('model_name', 'convnet', 'Model name.')
flags.DEFINE_string('model_dir', './model', 'Model DIR.')
flags.DEFINE_string('main_dir', 'convnet/', 'Directory to store data relative to the algorithm.')
flags.DEFINE_integer('verbose', 0, 'Level of verbosity. 0 - silent, 1 - print accuracy.')
flags.DEFINE_boolean('restore_previous_model', False, 'If true, restore previous model corresponding to model name.')
flags.DEFINE_integer('seed', -1, 'Seed for the random generators (>= 0). Useful for testing hyperparameters.')


# Convolutional Net parameters
flags.DEFINE_string('layers', 'conv2d-5-5-32-1,maxpool-2,conv2d-5-5-64-1,maxpool-2,full-1024,softmax', 'String representing the architecture of the network.')
flags.DEFINE_string('loss_func', 'cross_entropy', 'Loss function. ["mean_squared" or "cross_entropy"]')
flags.DEFINE_integer('num_epochs', 50, 'Number of epochs.')
flags.DEFINE_integer('batch_size', 50, 'Size of each mini-batch.')
flags.DEFINE_string('opt', 'adam', '["gradient_descent", "ada_grad", "momentum", "adam"]')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
flags.DEFINE_float('momentum', 0.5, 'Momentum parameter.')
flags.DEFINE_float('dropout', 0.5, 'Dropout parameter.')

assert FLAGS.dataset in ['mnist', 'cifar10', 'custom']
assert FLAGS.opt in ['gradient_descent', 'ada_grad', 'momentum', 'adam']

if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description="Train Lenet")
    #parser.add_argument("--model_dir", type=str, default="./model")
    #opt = parser.parse_args()

    if FLAGS.dataset == 'mnist':

        # ################# #
        #   MNIST Dataset   #
        # ################# #

        trX, trY, vlX, vlY, teX, teY = load_mnist_dataset(mode='supervised')

    elif FLAGS.dataset == 'cifar10':

        # ################### #
        #   Cifar10 Dataset   #
        # ################### #

        trX, trY, teX, teY = load_cifar10_dataset(FLAGS.cifar_dir, mode='supervised')
        vlX = teX[:5000]  # Validation set is the first half of the test set
        vlY = teY[:5000]

    elif FLAGS.dataset == 'custom':

        # ################## #
        #   Custom Dataset   #
        # ################## #

        def load_from_np(dataset_path):
            if dataset_path != '':
                return np.load(dataset_path)
            else:
                return None


        trX, trY = load_from_np(FLAGS.train_dataset), load_from_np(FLAGS.train_labels)
        vlX, vlY = load_from_np(FLAGS.valid_dataset), load_from_np(FLAGS.valid_labels)
        teX, teY = load_from_np(FLAGS.test_dataset), load_from_np(FLAGS.test_labels)

    else:
        trX, trY, vlX, vlY, teX, teY = None, None, None, None, None, None

    train_data_loader = DataLoaderFromArrays(trX, trY, shuffle=True, one_hot=False, normalization=False)
    test_data_loader = DataLoaderFromArrays(vlX, vlY, shuffle=True, one_hot=False, normalization=False)

    hook = smd.SessionHook.create_from_json_file()

    # Create the model object
    convnet = ConvolutionalNetwork(
        layers=FLAGS.layers, model_name=FLAGS.model_name, main_dir=FLAGS.main_dir,
        num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size, dataset=FLAGS.dataset, opt=FLAGS.opt,
        learning_rate=FLAGS.learning_rate, momentum=FLAGS.momentum, dropout=FLAGS.dropout, verbose=FLAGS.verbose
    )
    # Model Construction
    convnet.build_model(trX.shape[1], trY.shape[1], [int(i) for i in FLAGS.original_shape.split(',')])
    trainable_params = [var.name for var in tf.trainable_variables()]
    weights_names = [param for param in trainable_params if 'weight' in param]
    for w_name in weights_names:
        hook.add_to_collection("weights", _as_graph_element(w_name))
    biases_names = [param for param in trainable_params if 'bias' in param]
    for b_name in biases_names:
        hook.add_to_collection("biases", _as_graph_element(b_name))
    # start the training.
    train_epoch_iters = train_data_loader.get_epochs(FLAGS.batch_size)
    
    # Do not need to pass the hook to the session if in a zero script change environment
    # i.e. SageMaker/AWS Deep Learning Containers
    # as the framework will automatically do that there if the hook exists
    sess = tf.train.MonitoredSession()

    # use this session for running the tensorflow model
    hook.set_mode(smd.modes.TRAIN)
    for _ in range(FLAGS.num_epochs):
        for i in range(train_epoch_iters):
            batch_x, batch_y = train_data_loader.next_batch(FLAGS.batch_size)
            _, _loss = sess.run([convnet.train_step, convnet.cost], 
                                        feed_dict={convnet.input_data: batch_x, convnet.input_labels: batch_y, convnet.keep_prob:FLAGS.dropout})
            if i % 500 == 0:
                print(f"Step={i}, Loss={_loss}")
    
    test_epoch_iters = test_data_loader.get_epochs(FLAGS.batch_size)
    # set the mode for monitored session based runs
    # so smdebug can separate out steps by mode
    hook.set_mode(smd.modes.EVAL)
    for i in range(test_epoch_iters):
        batch_x, batch_y = test_data_loader.next_batch(FLAGS.batch_size)
        sess.run(convnet.cost, feed_dict={convnet.input_data: batch_x, convnet.input_labels: batch_y, convnet.keep_prob:1.0})