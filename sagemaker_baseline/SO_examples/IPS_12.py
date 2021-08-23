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
np.random.seed(20180130)
random.seed(20180130)

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

xx = np.random.normal(0, 3, 14)

unit_train_inputX = []
unit_train_inputY = []
for i in range(100):
    unit_train_inputX.append(np.random.normal(0, 0.1, 14))
    if random.uniform(0, 3) > 1:
        unit_train_inputY.append([0.])
        unit_train_inputX[-1] += float(random.randint(-1, 1))
    else:
        unit_train_inputY.append([1.])

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Train Lenet")
    parser.add_argument("--model_dir", type=str, default="./model")
    opt = parser.parse_args()

    hook = smd.SessionHook.create_from_json_file()
    X = tf.placeholder(tf.float32, shape=[None, 14], name="inputs")
    hook.add_to_collection("inputs", X)
    Y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')
    keep_prob = tf.placeholder(tf.float32, name='keep_p')

    #W1 = tf.Variable(tf.truncated_normal(shape=[14, 20], stddev=0.5), name="W_1")
    W1 = tf.Variable(tf.truncated_normal(shape=[14, 20], stddev=np.sqrt(2/14)), name="W_1")
    b1 = tf.Variable(tf.zeros([20]), name="B_1")
    l1 = tf.nn.relu(tf.matmul(X, W1) + b1, 'l1/relu')
    l1 = tf.nn.dropout(l1, rate=keep_prob)

    W2 = tf.Variable(tf.truncated_normal(shape=[20, 20], stddev=0.5), name="W_2")
    #W2 = tf.Variable(tf.truncated_normal(shape=[20, 20], stddev=np.sqrt(2/20)), name="W_2")
    b2 = tf.Variable(tf.zeros([20]), name="B_2")
    l2 = tf.nn.relu(tf.matmul(l1, W2) + b2, 'l2/relu')
    l2 = tf.nn.dropout(l2, rate=keep_prob)

    W3 = tf.Variable(tf.truncated_normal(shape=[20, 15], stddev=0.5), name="W_3")
    #W3 = tf.Variable(tf.truncated_normal(shape=[20, 15], stddev=np.sqrt(2/20)), name="W_3")
    b3 = tf.Variable(tf.zeros([15]), name="B_3")
    l3 = tf.nn.relu(tf.matmul(l2, W3) + b3, 'l3/relu')
    l3 = tf.nn.dropout(l3, rate=keep_prob)

    W5 = tf.Variable(tf.truncated_normal(shape=[15, 1], stddev=0.5), name="W_4")
    #W5 = tf.Variable(tf.truncated_normal(shape=[15, 1], stddev=np.sqrt(2/15)), name="W_4")
    b5 = tf.Variable(tf.zeros([1]), name="B_4")
    Yhat = tf.matmul(l3, W5) + b5
    Ypred = tf.nn.sigmoid(Yhat)

    weights_names = ['W_1','W_2','W_3','W_4']
    for w_name in weights_names:
        hook.add_to_collection("weights", _as_graph_element(w_name))
    biases_names = ['B_1','B_2','B_3','B_4']
    for b_name in biases_names:
        hook.add_to_collection("biases", _as_graph_element(b_name))

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Yhat, labels=Y))
    hook.add_to_collection("losses", loss)
    
    learning_rate = 0.005
    l2_weight = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate)
    optimizer = hook.wrap_optimizer(optimizer)
    train_step = optimizer.minimize(loss)

    correct_prediction = tf.equal(tf.greater(Y, 0.5), tf.greater(Yhat, 0.5))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    epoch = 50
    batch_size = 32
    unit_train_inputX = np.array(unit_train_inputX)
    unit_train_inputY = np.array(unit_train_inputY)
    train_data_loader = DataLoaderFromArrays(unit_train_inputX, unit_train_inputY, shuffle=True, one_hot=False, normalization=True, homogeneous=False)
    test_data_loader = DataLoaderFromArrays(unit_train_inputX, unit_train_inputY, shuffle=True, one_hot=False, normalization=True, homogeneous=False)
    
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
            _, _loss = sess.run([train_step, loss], 
                    feed_dict={X: batch_x, Y: batch_y, keep_prob:0.5})
            if i % 500 == 0:
                print(f"Step={i}, Loss={_loss}")
    
    test_epoch_iters = test_data_loader.get_epochs(batch_size)
    # set the mode for monitored session based runs
    # so smdebug can separate out steps by mode
    hook.set_mode(smd.modes.EVAL)
    for i in range(test_epoch_iters):
        batch_x, batch_y = test_data_loader.next_batch(batch_size)
        sess.run(loss, 
                 feed_dict={X: batch_x, Y: batch_y, keep_prob:1.0})
