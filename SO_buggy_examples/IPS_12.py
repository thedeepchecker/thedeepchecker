import tensorflow as tf
import random
import numpy as np
from tfcheck import DNNChecker
import interfaces
import data 

tf.set_random_seed(20180130)
np.random.seed(20180130)
random.seed(20180130)

X = tf.placeholder(tf.float32, shape=[None, 14], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')
keep_prob = tf.placeholder(tf.float32, name='keep_p')

#W1 = tf.Variable(tf.truncated_normal(shape=[14, 20], stddev=0.5), name="W_1")
W1 = tf.Variable(tf.truncated_normal(shape=[14, 20], stddev=np.sqrt(2/14)), name="W_1")
b1 = tf.Variable(tf.zeros([20]), name="B_1")
l1 = tf.nn.relu(tf.matmul(X, W1) + b1, 'l1/relu')
l1 = tf.nn.dropout(l1, rate=keep_prob)

W2 = tf.Variable(tf.truncated_normal(shape=[20, 20], stddev=0.5), name="W_2")
#W2 = tf.Variable(tf.truncated_normal(shape=[20, 20], stddev=np.sqrt(2/20)), name="W_2")
b2 = tf.Variable(tf.zeros([20]), name="B_2")
l2 = tf.nn.relu(tf.matmul(l1, W2) + b2, 'l2/relu')
l2 = tf.nn.dropout(l2, rate=keep_prob)

W3 = tf.Variable(tf.truncated_normal(shape=[20, 15], stddev=0.5), name="W_3")
#W3 = tf.Variable(tf.truncated_normal(shape=[20, 15], stddev=np.sqrt(2/20)), name="W_3")
b3 = tf.Variable(tf.zeros([15]), name="B_3")
l3 = tf.nn.relu(tf.matmul(l2, W3) + b3, 'l3/relu')
l3 = tf.nn.dropout(l3, rate=keep_prob)

W5 = tf.Variable(tf.truncated_normal(shape=[15, 1], stddev=0.5), name="W_4")
#W5 = tf.Variable(tf.truncated_normal(shape=[15, 1], stddev=np.sqrt(2/15)), name="W_4")
b5 = tf.Variable(tf.zeros([1]), name="B_4")
Yhat = tf.matmul(l3, W5) + b5
Ypred = tf.nn.sigmoid(Yhat)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Yhat, labels=Y))

learning_rate = 0.005
l2_weight = 0.001
learner = tf.train.AdamOptimizer(learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.greater(Y, 0.5), tf.greater(Yhat, 0.5))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

xx = np.random.normal(0, 3, 14)

unit_train_inputX = []
unit_train_inputY = []
for i in range(100):
    unit_train_inputX.append(np.random.normal(0, 0.1, 14))
    if random.uniform(0, 3) > 1:
        unit_train_inputY.append([0.])
        unit_train_inputX[-1] += float(random.randint(-1, 1))
    else:
        unit_train_inputY.append([1.])

unit_train_inputX = np.array(unit_train_inputX)
unit_train_inputY = np.array(unit_train_inputY)
data_loader_under_test = data.DataLoaderFromArrays(unit_train_inputX, unit_train_inputY, shuffle=True, one_hot=False, normalization=True, homogeneous=False)
test_data_loader = data.DataLoaderFromArrays(unit_train_inputX, unit_train_inputY, shuffle=True, one_hot=False, normalization=True, homogeneous=False)
model_under_test =  interfaces.build_model_interface(problem_type='classification', 
                                                        test_mode=False,
                                                        features=X, 
                                                        targets=Y, 
                                                        train_op=learner, 
                                                        loss=loss, 
                                                        perf=accuracy, 
                                                        perf_dir='Up', 
                                                        outs_pre_act=Yhat, 
                                                        outs_post_act=Ypred, 
                                                        weights=['W_1','W_2','W_3','W_4'], 
                                                        biases=['B_1','B_2','B_3','B_4'], 
                                                        activations=['l1/relu','l2/relu','l3/relu'], 
                                                        test_activations=None,
                                                        train_extra_feed_dict={keep_prob:0.5},
                                                        test_extra_feed_dict={keep_prob:1.0})
data_under_test = interfaces.build_data_interface(data_loader_under_test, test_data_loader, homogeneous=False)
checker = DNNChecker(name='IPS_12', data=data_under_test, model=model_under_test, buffer_scale=10)
checker.run_full_checks(overfit_batch=10)
