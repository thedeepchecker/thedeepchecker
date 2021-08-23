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
import boto3
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
        self.features = tf.placeholder(tf.float32, [None, 9])
        self.out_dims = 1
        self.labels = tf.placeholder(tf.float32, [None, self.out_dims])
        self.outputs = self.build(self.features, self.out_dims, reuse=False, training=True)
        trainable_params = [var.name for var in tf.trainable_variables()]
        weights_names = [param for param in trainable_params if ('kernel' in param or 'weight' in param)]
        #weights_names = ['lenet/conv2d/kernel', 'lenet/conv2d_1/kernel', 'lenet/dense/kernel', 'lenet/dense_1/kernel']
        for w_name in weights_names:
            hook.add_to_collection("weights", _as_graph_element(w_name))
        biases_names = [param for param in trainable_params if 'bias' in param]
        #biases_names = ['lenet/conv2d/bias', 'lenet/conv2d_1/bias', 'lenet/dense/bias', 'lenet/dense_1/bias']
        for b_name in biases_names:
            hook.add_to_collection("biases", _as_graph_element(b_name))
        self.acc = tf.reduce_mean(tf.abs(self.labels - self.outputs))
        self.reg_loss = tf.losses.get_regularization_loss()
        self.data_loss = tf.losses.mean_squared_error(self.labels, self.outputs)
        self.loss = self.data_loss + self.reg_loss
        hook.add_to_collection("losses", self.loss)
        optimizer = tf.train.AdamOptimizer(0.001)
        optimizer = hook.wrap_optimizer(optimizer)
        self.train_op = optimizer.minimize(self.loss)

    def build(self, features, out_dims, reuse=False, l1=0.0, l2=1e-5, training=True):
        with tf.variable_scope('denseNN', reuse=reuse):
            # Fully connected layer 
            dense_0 = tf.layers.dense(inputs=features, 
                                    units=64, 
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.initializers.he_normal(),
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1,l2))
            dense_1 = tf.layers.dense(inputs=dense_0, 
                                    units=64, 
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.initializers.he_normal(),
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1,l2))
            outputs = tf.layers.dense(inputs=dense_1,
                                      units=out_dims, 
                                      activation=tf.nn.sigmoid,
                                      kernel_regularizer=tf.keras.regularizers.l1_l2(l1,l2))
        return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Lenet")
    parser.add_argument("--model_dir", type=str, default="./model")
    opt = parser.parse_args()

    hook = smd.SessionHook.create_from_json_file()
    model = Model(hook)

    epoch = 50
    batch_size = 32

    client = boto3.client('s3') #low-level functional API
    obj = client.get_object(Bucket='sagemaker-studio-wjai890dqql', Key='auto-mpg.csv')
    dataset = pd.read_csv(obj['Body'])
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()
    train_labels = train_features.pop('MPG')
    test_labels = test_features.pop('MPG')
    X_train, y_train = train_features.to_numpy(), train_labels.to_numpy().reshape(-1,1)
    X_test, y_test = test_features.to_numpy(), test_labels.to_numpy().reshape(-1,1)
    train_data_loader = DataLoaderFromArrays(X_train, y_train, problem_type='regression', shuffle=True, one_hot=False, normalization=True, homogeneous=False, target_range=[-1.0,1.0])
    test_data_loader =  DataLoaderFromArrays(X_test, y_test, problem_type='regression', shuffle=True, one_hot=False, normalization=True, homogeneous=False, target_range=[-1.0,1.0])
    
    # start the training.
    train_epoch_iters = train_data_loader.get_epochs(batch_size)
    
    # Do not need to pass the hook to the session if in a zero script change environment
    # i.e. SageMaker/AWS Deep Learning Containers
    # as the framework will automatically do that there if the hook exists
    sess = tf.train.MonitoredSession()