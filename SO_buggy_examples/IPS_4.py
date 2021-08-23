import tensorflow as tf
import numpy as np
from tfcheck import DNNChecker
import interfaces
import data 

tf.set_random_seed(20180130)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

# matrix = height * width
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])


# defining the neural network

def neural_network_model(data):
    hiddenLayer1 = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1]), name='W'),
                    'biases': tf.Variable(tf.random_normal([n_nodes_hl1]), name='b')}

    hiddenLayer2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2]), name='W'),
                    'biases': tf.Variable(tf.random_normal([n_nodes_hl2]), name='b')}

    hiddenLayer3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3]), name='W'),
                    'biases': tf.Variable(tf.random_normal([n_nodes_hl3]), name='b')}

    outputLayer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes]), name='W'),
                   'biases': tf.Variable(tf.random_normal([n_classes]), name='b')}

    l1 = tf.add(tf.matmul(data, hiddenLayer1['weights']), hiddenLayer1['biases'])
    l1 = tf.nn.relu(l1, name='l1/relu')

    l2 = tf.add(tf.matmul(l1, hiddenLayer2['weights']), hiddenLayer2['biases'])
    l2 = tf.nn.relu(l2, name='l2/relu')

    l3 = tf.add(tf.matmul(l2, hiddenLayer3['weights']), hiddenLayer3['biases'])
    l3 = tf.nn.relu(l3, name='l3/relu')
    output = tf.add(tf.matmul(l3, outputLayer['weights']), outputLayer['biases'])
    return output

# training the network
def train_neural_network(x):
    logits = neural_network_model(x)
    predictions = tf.nn.softmax(logits)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    optimizer = tf.train.AdamOptimizer(0.003).minimize(cost)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0],-1)
    x_test = x_test.reshape(x_test.shape[0],-1)
    data_loader_under_test = data.DataLoaderFromArrays(x_train, y_train, shuffle=True, one_hot=True, normalization=True)
    test_data_loader = data.DataLoaderFromArrays(x_test, y_test, shuffle=True, one_hot=True, normalization=True)
    model_under_test =  interfaces.build_model_interface(problem_type='classification', 
                                                            test_mode=False,
                                                            features=x, 
                                                            targets=y, 
                                                            train_op=optimizer, 
                                                            loss=cost, 
                                                            perf=accuracy, 
                                                            perf_dir='Up', 
                                                            outs_pre_act=logits, 
                                                            outs_post_act=predictions, 
                                                            weights=['W','W_1','W_2', 'W_3'], 
                                                            biases=['b','b_1','b_2', 'b_3'], 
                                                            activations=['l1/relu','l2/relu','l3/relu'], 
                                                            test_activations=None)
    data_under_test = interfaces.build_data_interface(data_loader_under_test, test_data_loader, homogeneous=True)
    checker = DNNChecker(name='IPS_4', data=data_under_test, model=model_under_test, buffer_scale=10)
    checker.run_full_checks()

train_neural_network(x)
