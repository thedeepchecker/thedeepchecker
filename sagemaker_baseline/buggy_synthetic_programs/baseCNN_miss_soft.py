# Standard Library
import argparse
import os
import time

# Third Party
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.training.basic_session_run_hooks import _as_graph_element

# First Party
import smdebug.tensorflow as smd

CLASSIFICATION_KEY, REGRESSION_KEY, EPSILON = 'classification', 'regression', 1e-16

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

class Model:

    def __init__(self, hook):
        self.features = tf.placeholder(tf.float32, [None, 28, 28])
        self.images = tf.reshape(self.features, [-1, 28, 28, 1])
        self.n_classes = 10
        self.labels = tf.placeholder(tf.float32, [None, self.n_classes])
        self.logits = self.build(self.images, self.n_classes, reuse=False, training=True)
        trainable_params = [var.name for var in tf.trainable_variables()]
        weights_names = [param for param in trainable_params if ('kernel' in param or 'weight' in param)]
        #weights_names = ['lenet/conv2d/kernel', 'lenet/conv2d_1/kernel', 'lenet/dense/kernel', 'lenet/dense_1/kernel']
        for w_name in weights_names:
            hook.add_to_collection("weights", _as_graph_element(w_name))
        biases_names = [param for param in trainable_params if 'bias' in param]
        #biases_names = ['lenet/conv2d/bias', 'lenet/conv2d_1/bias', 'lenet/dense/bias', 'lenet/dense_1/bias']
        for b_name in biases_names:
            hook.add_to_collection("biases", _as_graph_element(b_name))
        self.probabilities = self.logits
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.probabilities, 1)), tf.float32))
        self.reg_loss = tf.losses.get_regularization_loss()
        self.data_loss = self.cross_entropy(self.labels, self.probabilities)
        self.loss = self.data_loss+self.reg_loss
        hook.add_to_collection("losses", self.loss)
        optimizer = tf.train.GradientDescentOptimizer(0.1)
        optimizer = hook.wrap_optimizer(optimizer)
        self.train_op = optimizer.minimize(self.loss)
        
    def build(self, features, n_classes, reuse=False, l1=0.0, l2=1e-5, training=True):
        with tf.variable_scope('lenet', reuse=reuse):
            # Convolution Layer with 32 filters and a kernel size of 5
            conv1 = tf.layers.conv2d(inputs=features,
                                    filters=32,
                                    kernel_size=[5, 5], 
                                    padding='same',
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.initializers.he_normal(),
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1,l2))
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            # Convolution Layer with 64 filters and a kernel size of 3
            conv2 = tf.layers.conv2d(inputs=pool1, 
                                    filters=64, 
                                    kernel_size=[3, 3],
                                    padding='same', 
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.initializers.he_normal(),
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1,l2))
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            # Flatten the data to a 1-D vector for the fully connected layer
            pool2_flat = tf.layers.flatten(pool2)

            # Fully connected layer 
            dense = tf.layers.dense(inputs=pool2_flat, 
                                    units=512, 
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.initializers.he_normal(),
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1,l2))
            outputs = tf.layers.dense(inputs=dense,
                                      units=n_classes, 
                                      activation=None,
                                      kernel_regularizer=tf.keras.regularizers.l1_l2(l1,l2))
        return outputs

    def cross_entropy(self, labels, probabilities):
        epsilon = 1e-32
        #probabilities += epsilon
        probabilities = tf.clip_by_value(probabilities, epsilon, 1 - epsilon)
        return tf.reduce_mean(-tf.reduce_sum(labels * tf.log(probabilities), reduction_indices=[1]))

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Train Lenet")
    parser.add_argument("--model_dir", type=str, default="./model")
    opt = parser.parse_args()

    hook = smd.SessionHook.create_from_json_file()
    model = Model(hook)

    epoch = 50
    batch_size = 32

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    train_data_loader = DataLoaderFromArrays(x_train, y_train, shuffle=True, one_hot=True, normalization=True)
    test_data_loader = DataLoaderFromArrays(x_test, y_test, shuffle=True, one_hot=True, normalization=True)
    # start the training.
    train_epoch_iters = train_data_loader.get_epochs(batch_size)
    
    # Do not need to pass the hook to the session if in a zero script change environment
    # i.e. SageMaker/AWS Deep Learning Containers
    # as the framework will automatically do that there if the hook exists
    sess = tf.train.MonitoredSession()

    # use this session for running the tensorflow model
    hook.set_mode(smd.modes.TRAIN)
    for _ in range(epoch):
        for i in range(train_epoch_iters):
            batch_x, batch_y = train_data_loader.next_batch(batch_size)
            _, _loss = sess.run([model.train_op, model.loss], feed_dict={model.features: batch_x, model.labels: batch_y})
            if i % 500 == 0:
                print(f"Step={i}, Loss={_loss}")
    
    test_epoch_iters = test_data_loader.get_epochs(batch_size)
    # set the mode for monitored session based runs
    # so smdebug can separate out steps by mode
    hook.set_mode(smd.modes.EVAL)
    for i in range(test_epoch_iters):
        batch_x, batch_y = test_data_loader.next_batch(batch_size)
        sess.run(model.loss, feed_dict={model.features: batch_x, model.labels: batch_y})