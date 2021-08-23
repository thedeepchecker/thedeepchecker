import tensorflow as tf
import numpy as np
from tfcheck import DNNChecker
import interfaces
import data 

tf.set_random_seed(20180130)
np.random.seed(20180130)

T = 100
noise = 10 * np.random.random(size=(T, 1)).astype(np.float32)
x = np.array([np.arange(T), np.ones(T)]).T.astype(np.float32)
w = np.array([[2, 4]]).T.astype(np.float32)
y = x.dot(w) + noise
#w.dot(x)
X = tf.placeholder(tf.float32, [None, 2], name="X")
Y = tf.placeholder(tf.float32, [None, 1], name="Y")
W = tf.Variable(tf.zeros([2, 1]), name="W")
Yhat = tf.matmul(X, W)
MSE = (1. / (2 * T)) * tf.reduce_sum(tf.pow(Y - Yhat, 2))
MAE = tf.reduce_mean(tf.abs(Y - Yhat))
trainer = tf.train.GradientDescentOptimizer(0.5).minimize(MSE)

data_loader_under_test = data.DataLoaderFromArrays(x, y, shuffle=True, one_hot=False, normalization=False, problem_type='regression')
test_data_loader = data.DataLoaderFromArrays(x, y, shuffle=True, one_hot=False, normalization=False, problem_type='regression')
model_under_test =  interfaces.build_model_interface(problem_type='regression', 
                                                        test_mode=False,
                                                        features=X, 
                                                        targets=Y, 
                                                        train_op=trainer, 
                                                        loss=MSE, 
                                                        perf=MAE, 
                                                        perf_dir='Down', 
                                                        outs_pre_act=Yhat, 
                                                        outs_post_act=None, 
                                                        weights=['W'], 
                                                        biases=None, 
                                                        activations=None, 
                                                        test_activations=None)
data_under_test = interfaces.build_data_interface(data_loader_under_test, test_data_loader, homogeneous=False)
checker = DNNChecker(name='IPS_15', data=data_under_test, model=model_under_test, buffer_scale=10)
checker.run_full_checks()