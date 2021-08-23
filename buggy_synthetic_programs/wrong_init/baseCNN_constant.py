'''
a common fault is to unbreak the neurons symmetry through constant weights.
'''
import random
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from deep_checker.checkers import DeepChecker
import deep_checker.interfaces as interfaces
import deep_checker.data as data
from deep_checker.settings import CLASSIFICATION_KEY, REGRESSION_KEY, EPSILON

class Model:

    def __init__(self):
        self.features = tf.placeholder(tf.float32, [None, 28, 28])
        self.images = tf.reshape(self.features, [-1, 28, 28, 1])
        self.n_classes = 10
        self.labels = tf.placeholder(tf.float32, [None, self.n_classes])
        self.logits = self.build(self.images, self.n_classes)
        self.probabilities = tf.nn.softmax(self.logits)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.logits, 1)), tf.float32))
        self.reg_loss = tf.losses.get_regularization_loss()
        self.loss = tf.losses.softmax_cross_entropy(self.labels, self.logits)
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(self.loss+self.reg_loss)

    def build(self, features, n_classes, l1=0.0, l2=1e-5):
        with tf.variable_scope('lenet'):
            # Convolution Layer with 32 filters and a kernel size of 5
            conv1 = tf.layers.conv2d(inputs=features,
                                    filters=32,
                                    kernel_size=[5, 5], 
                                    padding='same',
                                    activation=tf.nn.relu,
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1,l2),
                                    kernel_initializer=tf.initializers.constant(0.01))
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            # Convolution Layer with 64 filters and a kernel size of 3
            conv2 = tf.layers.conv2d(inputs=pool1, 
                                    filters=64, 
                                    kernel_size=[3, 3],
                                    padding='same', 
                                    activation=tf.nn.relu,
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1,l2),
                                    kernel_initializer=tf.initializers.constant(0.01))
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            # Flatten the data to a 1-D vector for the fully connected layer
            pool2_flat = tf.layers.flatten(pool2)

            # Fully connected layer 
            dense = tf.layers.dense(inputs=pool2_flat, 
                                    units=1024, 
                                    activation=tf.nn.relu,
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1,l2),
                                    kernel_initializer=tf.initializers.constant(0.01))

            outputs = tf.layers.dense(inputs=dense,
                                      units=n_classes, 
                                      activation=None,
                                      kernel_regularizer=tf.keras.regularizers.l1_l2(l1,l2),
                                      kernel_initializer=tf.initializers.constant(0.01))
        return outputs

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    data_loader_under_test = data.DataLoaderFromArrays(x_train, y_train, shuffle=True, one_hot=True, normalization=True)
    test_data_loader = data.DataLoaderFromArrays(x_test, y_test, shuffle=True, one_hot=True, normalization=True)
    model = Model()
    model_under_test =  interfaces.build_model_interface(problem_type='classification', 
                          test_mode=False,
                          features=model.features, 
                          targets=model.labels, 
                          train_op=model.train_op, 
                          loss=model.loss, 
                          reg_loss=model.reg_loss,
                          perf=model.acc, 
                          perf_dir='Up', 
                          outs_pre_act=model.logits, 
                          outs_post_act=model.probabilities)
    data_under_test = interfaces.build_data_interface(data_loader_under_test, test_data_loader, homogeneous=True)
    checker = DeepChecker(name='base_CNN_constant', data=data_under_test, model=model_under_test, buffer_scale=10)
    checker.run_full_checks()
