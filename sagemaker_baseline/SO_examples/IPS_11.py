# Standard Library
import argparse
import os
import time
import random
# Third Party
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.training.basic_session_run_hooks import _as_graph_element

# First Party
import smdebug.tensorflow as smd

CLASSIFICATION_KEY, REGRESSION_KEY, EPSILON = 'classification', 'regression', 1e-16

tf.set_random_seed(20180130)

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

def conv_layer(input, size_in, size_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b, name='relu')
        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def fc_layer(input, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        activation = tf.nn.relu(tf.matmul(input, w) + b, name='relu')
        return activation

def mnist_model(learning_rate, hook):
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

    conv1 = conv_layer(x_image, 1, 32, "conv1")
    conv_out = conv_layer(conv1, 32, 64, "conv2")

    flattened = tf.reshape(conv_out, [-1, 7 * 7 * 64])

    fc1 = fc_layer(flattened, 7 * 7 * 64, 1024, name="fc1")
    logits = fc_layer(fc1, 1024, 10, name="fc2")
    probabilities = tf.nn.softmax(logits)

    weights_names = ['conv1/W:0', 'conv2/W:0','fc1/W:0', 'fc2/W:0']
    for w_name in weights_names:
        hook.add_to_collection("weights", _as_graph_element(w_name))
    biases_names = ['conv1/B:0', 'conv2/B:0', 'fc1/B:0', 'fc2/B:0']
    for b_name in biases_names:
        hook.add_to_collection("biases", _as_graph_element(b_name))

    with tf.name_scope("xent"):
        xent = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y), name="xent")
        hook.add_to_collection("losses", xent)
    
    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer = hook.wrap_optimizer(optimizer)
        train_step = optimizer.minimize(xent)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    epoch = 50
    batch_size = 32
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0],-1)
    x_test = x_test.reshape(x_test.shape[0],-1)
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
            _, _loss = sess.run([train_step, xent], 
                    feed_dict={x: batch_x, y: batch_y})
            if i % 500 == 0:
                print(f"Step={i}, Loss={_loss}")
    
    test_epoch_iters = test_data_loader.get_epochs(batch_size)
    # set the mode for monitored session based runs
    # so smdebug can separate out steps by mode
    hook.set_mode(smd.modes.EVAL)
    for i in range(test_epoch_iters):
        batch_x, batch_y = test_data_loader.next_batch(batch_size)
        sess.run(xent, 
                 feed_dict={x: batch_x, y: batch_y})

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Train Lenet")
    parser.add_argument("--model_dir", type=str, default="./model")
    opt = parser.parse_args()

    hook = smd.SessionHook.create_from_json_file()
    mnist_model(1e-3, hook)
