import tensorflow as tf
import numpy as np
from tfcheck import DNNChecker
import interfaces
import data 
tf.set_random_seed(20180130)

# Model parameters
A = tf.Variable([0], dtype=tf.float32, name="W")
B = tf.Variable([0], dtype=tf.float32, name="W_1")
C = tf.Variable([0], dtype=tf.float32, name="B")
# Model input and output
x = tf.placeholder(tf.float32)
model = A * (x ** 2) + B * x + C
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(model - y))  # sum of the squares
mae = tf.reduce_mean(tf.abs(model - y))
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = np.array([0, 1, 2, 3])
y_train = np.array([0, 1, 4, 9])

data_loader_under_test = data.DataLoaderFromArrays(x_train, y_train, shuffle=True, one_hot=False, normalization=False, problem_type='regression')
test_data_loader = data.DataLoaderFromArrays(x_train, y_train, shuffle=True, one_hot=False, normalization=False, problem_type='regression')
model_under_test =  interfaces.build_model_interface(problem_type='regression', 
                                                        test_mode=False,
                                                        features=x, 
                                                        targets=y, 
                                                        train_op=train, 
                                                        loss=loss, 
                                                        perf=mae, 
                                                        perf_dir='Down', 
                                                        outs_pre_act=model, 
                                                        outs_post_act=None, 
                                                        weights=['W', 'W_1'], 
                                                        biases=['B'], 
                                                        activations=None, 
                                                        test_activations=None)
data_under_test = interfaces.build_data_interface(data_loader_under_test, test_data_loader, homogeneous=False)
checker = DNNChecker(name='IPS_17', data=data_under_test, model=model_under_test, buffer_scale=10)
checker.run_full_checks(overfit_batch=10)