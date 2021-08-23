import tensorflow as tf
import numpy as np
from tfcheck import DNNChecker
import interfaces
import data 

#assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)
# one hidden layer MLP

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

W_h1 = tf.Variable(tf.random_normal([784, 512]), name='W')
h1 = tf.nn.sigmoid(tf.matmul(x, W_h1))

W_out = tf.Variable(tf.random_normal([512, 10]), name='W')
y_ = tf.matmul(h1, W_out)

# cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(y_, y)
cross_entropy = tf.reduce_sum(- y * tf.log(y_) - (1 - y) * tf.log(1 - y_), 1)
loss = tf.reduce_mean(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

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
                                                        targets=y, 
                                                        train_op=train_step, 
                                                        loss=loss, 
                                                        perf=accuracy, 
                                                        perf_dir='Up', 
                                                        outs_pre_act=y_, 
                                                        outs_post_act=y_, 
                                                        weights=['W','W_1'], 
                                                        biases=None, 
                                                        activations=None, 
                                                        test_activations=None)
data_under_test = interfaces.build_data_interface(data_loader_under_test, test_data_loader, homogeneous=True)
checker = DNNChecker(name='IPS_7', data=data_under_test, model=model_under_test, buffer_scale=10)
checker.run_full_checks()