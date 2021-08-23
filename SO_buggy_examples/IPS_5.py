import tensorflow as tf
import numpy as np
import random
from tfcheck import DNNChecker
import interfaces
import data 

#assert tf.__version__ == "1.8.0"
tf.set_random_seed(20180130)
np.random.seed(20180130)
random.seed(20180130)

_w = random.uniform(-8, 8)
_b = random.uniform(-8, 8)
size = 8
X = [random.uniform(-8, 8) for _ in range(size)]
Y = [_w * x + _b + random.uniform(-1, 1) for x in X]
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.zeros([1]), name='w')
B = tf.Variable(tf.zeros([1]), name='b')
y_ = tf.multiply(x, W) + B
loss = tf.reduce_mean(tf.square(y_ - y))
mae = tf.reduce_mean(tf.abs(y_ - y))
optimizer = tf.train.GradientDescentOptimizer(1e-1).minimize(loss)

X = np.array([random.uniform(-8, 8) for _ in range(size)]).reshape(-1,1)
Y = np.array([_w * x + _b + random.uniform(-1, 1) for x in X]).reshape(-1,1)
data_loader_under_test = data.DataLoaderFromArrays(X, Y, shuffle=True, one_hot=False, normalization=False, problem_type='regression')
test_data_loader = data.DataLoaderFromArrays(X, Y, shuffle=True, one_hot=False, normalization=False, problem_type='regression')
model_under_test =  interfaces.build_model_interface(problem_type='regression', 
                                                        test_mode=False,
                                                        features=x, 
                                                        targets=y, 
                                                        train_op=optimizer, 
                                                        loss=loss, 
                                                        perf=mae, 
                                                        perf_dir='Down', 
                                                        outs_pre_act=y_, 
                                                        outs_post_act=None, 
                                                        weights=['w'], 
                                                        biases=['b'], 
                                                        activations=None, 
                                                        test_activations=None)
data_under_test = interfaces.build_data_interface(data_loader_under_test, test_data_loader, homogeneous=False)
checker = DNNChecker(name='IPS_5', data=data_under_test, model=model_under_test, buffer_scale=10)
checker.run_full_checks(overfit_batch=10)