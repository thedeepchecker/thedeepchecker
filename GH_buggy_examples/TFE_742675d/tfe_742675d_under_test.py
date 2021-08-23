'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import numpy as np
from tfcheck import DNNChecker
import interfaces
import data 

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes

# Create model

# Set model weights
W = tf.Variable(tf.zeros([784, 10]), name='w')
b = tf.Variable(tf.zeros([10]), name='b')
logits = tf.matmul(x, W) + b
# Construct model
activation = tf.nn.softmax(logits) # Softmax

# Minimize error using cross entropy
cost = -tf.reduce_sum(y*tf.log(activation)) # Cross entropy
#cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(activation), reduction_indices=1)) 
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # Gradient Descent


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
x_train, x_test = np.reshape(x_train,[len(x_train),784]),np.reshape(x_test,[len(x_test),784])
data_loader_under_test = data.DataLoaderFromArrays(x_train, y_train, shuffle=True, one_hot=True, normalization=True)
test_data_loader = data.DataLoaderFromArrays(x_test, y_test, shuffle=True, one_hot=True, normalization=True)

model_under_test =  interfaces.build_model_interface(problem_type='classification', 
                          test_mode=False,
                          features=x, 
                          targets=y, 
                          train_op=optimizer, 
                          loss=cost, 
                          reg_loss=None,
                          perf=cost, 
                          perf_dir='Down', 
                          outs_pre_act=logits, 
                          outs_post_act=activation,
                          weights=['w'], 
                          biases=['b'], 
                          activations=[], 
                          test_activations=[])
data_under_test = interfaces.build_data_interface(data_loader_under_test, test_data_loader, homogeneous=True)
checker = DNNChecker(name='tfe_742675d', data=data_under_test, model=model_under_test, buffer_scale=10)
checker.run_full_checks()