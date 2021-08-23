from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from tfcheck import DNNChecker
import interfaces
import data 
import tensorflow as tf

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.truncated_normal([784, 10]), name='W')
b = tf.Variable(tf.zeros([10]), name='B')
logits = tf.matmul(x, W) + b
y = tf.nn.softmax(logits)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))#tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=1)) #tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0],-1)
x_test = x_test.reshape(x_test.shape[0],-1)
data_loader_under_test = data.DataLoaderFromArrays(x_train, y_train, shuffle=True, one_hot=True, normalization=True)
test_data_loader = data.DataLoaderFromArrays(x_test, y_test, shuffle=True, one_hot=True, normalization=True)
model_under_test =  interfaces.build_model_interface(problem_type='classification', 
                                                        test_mode=False,
                                                        features=x, 
                                                        targets=y_, 
                                                        train_op=train_step, 
                                                        loss=cross_entropy, 
                                                        perf=accuracy, 
                                                        perf_dir='Up', 
                                                        outs_pre_act=logits, 
                                                        outs_post_act=y, 
                                                        weights=['W'], 
                                                        biases=['B'], 
                                                        activations=None, 
                                                        test_activations=None)
data_under_test = interfaces.build_data_interface(data_loader_under_test, test_data_loader, homogeneous=True)
checker = DNNChecker(name='IPS_14', data=data_under_test, model=model_under_test, buffer_scale=10)
checker.run_full_checks(overfit_iters=2000)

