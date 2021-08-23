'''
As the by-default epsilon works well for the baseline neural network, we set intentionally a low epsilon 
for ADAM in order to mimic the situation where the by-default epsilon is too low and not adapted for the app.
'''
import numpy as np
import pandas as pd
import tensorflow as tf
from deep_checker.checkers import DeepChecker
import deep_checker.interfaces as interfaces
import deep_checker.data as data

class Model:

    def __init__(self):
        self.features = tf.placeholder(tf.float32, [None, 9])
        self.out_dims = 1
        self.labels = tf.placeholder(tf.float32, [None, self.out_dims])
        self.outputs = self.build(self.features, self.out_dims, reuse=False, training=True)
        self.acc = tf.reduce_mean(tf.abs(self.labels - self.outputs))
        self.reg_loss = tf.losses.get_regularization_loss()
        self.loss = tf.losses.mean_squared_error(self.labels, self.outputs)
        self.train_op = tf.train.AdamOptimizer(0.001, epsilon=1e-16).minimize((self.loss+self.reg_loss))

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
                                      activation=None,
                                      kernel_regularizer=tf.keras.regularizers.l1_l2(l1,l2))
        return outputs

if __name__ == "__main__":
    dataset = pd.read_csv('../../data/auto-mpg.csv')
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()
    train_labels = train_features.pop('MPG')
    test_labels = test_features.pop('MPG')
    X_train, y_train = train_features.to_numpy(), train_labels.to_numpy().reshape(-1,1)
    X_test, y_test = test_features.to_numpy(), test_labels.to_numpy().reshape(-1,1)
    data_loader_under_test = data.DataLoaderFromArrays(X_train, y_train, problem_type='regression', shuffle=True, one_hot=False, normalization=True, homogeneous=False)
    test_data_loader = data.DataLoaderFromArrays(X_test, y_test, problem_type='regression', shuffle=True, one_hot=False, normalization=True, homogeneous=False)
    model = Model()
    model_under_test =  interfaces.build_model_interface(problem_type='regression', 
                          test_mode=False,
                          features=model.features, 
                          targets=model.labels, 
                          train_op=model.train_op, 
                          loss=model.loss, 
                          reg_loss=model.reg_loss,
                          perf=model.acc, 
                          perf_dir='Down', 
                          outs_pre_act=model.outputs)
    data_under_test = interfaces.build_data_interface(data_loader_under_test, test_data_loader, homogeneous=False)
    checker = DeepChecker(name='regrNN_low_epsilon', data=data_under_test, model=model_under_test, buffer_scale=10)
    checker.run_full_checks(overfit_batch=8)