'''
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''
import tensorflow as tf
import numpy as np
from tfcheck import DNNChecker
import interfaces
import data 

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
x_train, x_test = np.reshape(x_train,[len(x_train),784]),np.reshape(x_test,[len(x_test),784])
#y_train, y_test = tf.one_hot(y_train,10),tf.one_hot(y_test,10)

learning_rate = 0.001
training_epochs = 25
batch_size = 100
total_batch = int(60000/batch_size)
display_step = 1

#tf Graph Input

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

W = tf.Variable(np.random.randn(784,10).astype(np.float32), name='w')
b = tf.Variable(np.random.randn(1,10).astype(np.float32), name='b')

logits = tf.add(tf.matmul(x,W),b)
pred = tf.nn.softmax(logits)

cost = tf.reduce_mean(-tf.reduce_sum(tf.multiply(y,tf.log(pred)),1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

data_loader_under_test = data.DataLoaderFromArrays(x_train, y_train, shuffle=True, one_hot=True, normalization=True)
test_data_loader = data.DataLoaderFromArrays(x_test, y_test, shuffle=True, one_hot=True, normalization=True)

model_under_test =  interfaces.build_model_interface(problem_type='classification', 
                          test_mode=False,
                          features=x, 
                          targets=y, 
                          train_op=optimizer, 
                          loss=cost, 
                          reg_loss=None,
                          perf=accuracy, 
                          perf_dir='Up', 
                          outs_pre_act=logits, 
                          outs_post_act=pred,
                          weights=['w'], 
                          biases=['b'], 
                          activations=[], 
                          test_activations=[])
data_under_test = interfaces.build_data_interface(data_loader_under_test, test_data_loader, homogeneous=True)
checker = DNNChecker(name='tfe_333', data=data_under_test, model=model_under_test, buffer_scale=10)
checker.run_full_checks()