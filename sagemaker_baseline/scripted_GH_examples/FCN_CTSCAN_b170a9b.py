import numpy as np
import time
from datetime import datetime
# Standard Library
import argparse
import os
import time

# Third Party
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.training.basic_session_run_hooks import _as_graph_element

# First Party
import smdebug.tensorflow as smd

CLASSIFICATION_KEY, REGRESSION_KEY, EPSILON = 'classification', 'regression', 1e-16

device = '/gpu:0'
train_dir = '../data/trained/'
use_tboard = True

max_step = 1000000 
log_freq = 10
val_freq = 160

save_freq = 500
ckpt_name = 'model.ckpt'

rng_seed = 1311

class loss(object):
    def __init__(self): 
        self.use_tboard =  use_tboard
        self.lkey =  'loss' 
        self.reg_key = 'reg'

    def softmax_log_loss(self, X, target, target_weight=None, lm=1):
        xdev = X - tf.reduce_max(X, keep_dims=True, reduction_indices=[-1])
        lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), keep_dims=True, reduction_indices=[-1]))
        #lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), keep_dims=True, reduction_indices=[-1]))
        if (target_weight == None):
            target_weight=1
        l = -tf.reduce_mean(target_weight*target*lsm, name='softmax_log_loss')
        #l = -tf.reduce_sum(target_weight*target*lsm, name='softmax_log_loss')/tf.cast(tf.shape(X)[0], dtype= dtype)
        tf.add_to_collection(self.lkey, l)
        if (self.use_tboard):
            tf.summary.scalar('softmax_log_loss', l)
        return l

    def l2_loss(self, wkey, lm):
        all_var = tf.trainable_variables()
        for var in all_var:
            if (wkey in var.op.name):
                l = tf.multiply(tf.nn.l2_loss(var), lm, name='weight_loss')
                tf.add_to_collection(self.lkey, l)
                tf.add_to_collection(self.reg_key, l)
                if self.use_tboard:
                    tf.summary.scalar(var.op.name + '/weight_loss', l)
        return tf.add_n(tf.get_collection(self.reg_key), name='reg_loss')

    def total_loss(self):
        l = tf.add_n(tf.get_collection(self.lkey), name='total_loss')
        if self.use_tboard:
            tf.summary.scalar('total_loss', l)
        return l

class trainer(object):
    """trainer
    This class is for creating optimizer and computing gradients
    """
    def __init__(self, hook, loss_output):
        """__init__
        Initialize a trainer instance using global_cfg
        :param loss_output: the loss output, computed by calling total_loss in tfun/loss.py
        """
        self.hook = hook
        self.batch_size =  batch_size
        self.num_train =  num_train
        self.loss_output = loss_output
        self.use_tboard =  use_tboard
        self.optimizer = None
        self.grads = None
        #self.global_step = 0
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def create_sgd_optimizer(self, lr):
        """create_sgd_optimizer

        :param lr: learning rate
        """
        self.optimizer = tf.train.GradientDescentOptimizer(lr)
        
    def create_adam_optimizer(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        """create_adam_optimizer

        :param lr: learning rate
        :param beta1: beta1 in the paper
        :param beta2: beta2 in the paper
        :param eps: epsilon in the paper
        """
        self.optimizer = tf.train.AdamOptimizer(lr, beta1, beta2, eps)

    def get_trainer(self):
        """get_trainer
        Return the appply grad object so that a tf session could run on it
        """
        assert self.optimizer != None, "Please create an optimizer for trainer first before calling get_trainer()"

        self.optimizer = self.hook.wrap_optimizer(self.optimizer)
        
        # Create grad computation nodes & add them to summary
        self.grads = self.optimizer.compute_gradients(self.loss_output)
        if (self.use_tboard):
            for grad, var in self.grads:
                if grad != None:
                    tf.summary.histogram(var.op.name + '/grad', grad)

        # Add trainable variables to summary histogram
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Apply grad
        apply_grad_op = self.optimizer.apply_gradients(self.grads, global_step=self.global_step, name='train')

        return apply_grad_op 

def create_one_hot(target_vector, num_class, dtype=np.float32):
    """create_one_hot
    Generate one-hot 4D tensor from a target vector of length N (num sample)
    The one-hot tensor will have the shape of (N x 1 x 1 x num_class)

    :param target_vector: Index vector, values are ranged from 0 to num_class-1

    :param num_class: number of classes/labels
    :return: target vector as a 4D tensor
    """
    one_hot = np.eye(num_class+1, num_class, dtype=dtype)
    one_hot = one_hot[target_vector]
    result = np.reshape(one_hot, (target_vector.shape[0], 1, 1, num_class))
    
    return result

def create_var(name, shape=None, initializer=None, trainable=True):
    """create_var
    Create a tensor variable
    If GPU should be used, specify it with  device = '/gpu:0'
    :param name: name of the variable
    :param shape: the shape of the variable, tuple or list of int
    :param initializer: an tf initializer instance or a numpy array
    :param trainable: specify if the var should be trained in the main loop
    """ 
    with tf.device( device):
        dtype =  np.float32
        
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
        return var

def create_conv_layer(x, kernel_shape, use_bias, stride=[1,1,1,1], padding='SAME', activation=None, wkey='weight', initializer=tf.contrib.layers.xavier_initializer(), name=None):
    """create_conv_layer
    Create a 2D convlutional with optinal bias and activation function
    :param x: input, 4D tensor of shape [batch_size, h, w, in_dim]
    :param kernel_shape: shape of the conv2d kernel, [h, w, in_dim, out_dim]
    :param use_bias: boolean, specify if bias should be used here
    :param stride: stride of the convolution operator
    :param padding: 'SAME' or 'VALID'
    :param activation: activation function, can be None. If an activation functin is passed, it should only take on argument
    :param wkey: name of the kernel variable. Since, the loss.l2_loss is computed based on the name of the variable, you can use this to include the variable in l2_loss or not
    :param initializer: tensorflow intializer
    :param name: name of the operator
    """

    kernel = create_var(wkey, shape=kernel_shape, initializer=initializer)
    conv_result = tf.nn.conv2d(x, kernel, stride, padding=padding)
    if (use_bias):
        bias_shape = [1,1,1,kernel_shape[3]]
        bias = create_var('bias', shape=bias_shape, initializer=tf.constant_initializer(np.zeros(bias_shape)))
        conv_result = conv_result + bias
    if (activation != None):
        conv_result = activation(conv_result, name='relu')
    
    return conv_result

def create_linear_layer(x, w_shape, use_bias, activation=None, wkey='weight', initializer=tf.contrib.layers.xavier_initializer(), name=None):
    """create_linear_layer
    Create a linear layer with optional bias and activation function
    :param x: input, a tensor with num dim >= 2
    :param w_shape: shape of the weight, [in_dim, out_dim]
    :param use_bias: boolean, specify if bias should be used here
    :param activation: activation function, can be None. If an activation functin is passed, it should only take on argument
    :param wkey: name of the kernel variable. Since, the loss.l2_loss is computed based on the name of the variable, you can use this to include the variable in l2_loss or not
    :param initializer: tensorflow intializer
    :param name: name of the operator
    """
    # Preprocess x so that we could perform 2D matrix multiplication
    x_shape = tf.shape(x)
    x_reshape = tf.reshape(x, [x_shape[0], -1], name='reshape')
    
    bias_shape = (w_shape[1], )
    w = create_var('weight', shape=w_shape, initializer=initializer)
    b = create_var('bias', shape=bias_shape, initializer=tf.constant_initializer(np.zeros(bias_shape)))
    x = tf.nn.bias_add(tf.matmul(x_reshape, w), b, name=name)
    if (activation != None):
        x = activation(x, name='relu')
    return x

class DataLoaderFromArrays:

    def __init__(self, features, targets, homogeneous=True, 
                problem_type=CLASSIFICATION_KEY, shuffle=True, 
                one_hot=True, normalization=True, target_scaling=True, target_range=[0.0,1.0]):
        self.curr_pos = 0
        self.target_divider = None
        self.rows_count = features.shape[0]
        self.homogeneous = homogeneous
        self.to_shuffle = shuffle
        self.problem_type = problem_type
        self.normalization = normalization
        self.target_scaling = target_scaling
        self.target_range = target_range
        self.one_hot = one_hot
        self.features = self.normalize(features)
        self.targets = self.preprocess(targets)
        
    def next_batch(self, batch_size):
        if self.curr_pos+batch_size >= self.rows_count:
            batch = self.features[self.curr_pos:self.rows_count], self.targets[self.curr_pos:self.rows_count]
            self.curr_pos = 0
            if self.to_shuffle:
                self.shuffle()
            return batch
        batch = self.features[self.curr_pos:self.curr_pos+batch_size], self.targets[self.curr_pos:self.curr_pos+batch_size]
        self.curr_pos += batch_size
        return batch
    
    def shuffle(self):
        indices = np.arange(0, self.rows_count) 
        np.random.shuffle(indices) 
        self.features = self.features[indices]
        self.targets = self.targets[indices]

    def deprocess(self, perf):
        if self.problem_type == REGRESSION_KEY and self.target_scaling:
            deprocessed_perf = (perf/(self.target_range[1]-self.target_range[0]))*self.target_divider
            return deprocessed_perf
        else:
            return perf

    def preprocess(self, targets):
        if self.problem_type == REGRESSION_KEY and self.target_scaling:
            mi = targets.min(axis=0)
            divider = targets.max(axis=0) - mi
            if isinstance(divider, np.ndarray):
                divider[divider==0.0] = EPSILON
            else:
                divider = EPSILON if divider==0.0 else divider
            targets = self.target_range[0] + np.float32((targets-mi)/divider)*(self.target_range[1]-self.target_range[0])
            self.target_divider = divider
        elif self.one_hot:
            onehot_targets = np.zeros((self.rows_count, targets.max()+1))
            onehot_targets[np.arange(self.rows_count),targets] = 1
            targets = onehot_targets
        return targets

    def normalize(self, data):
        if not self.normalization:
            return data
        if self.homogeneous:
            mi = data.min()
            divider = data.max() - mi
            divider = EPSILON if divider==0.0 else divider
        else:
            mi = data.min(axis=0)
            divider = data.max(axis=0) - mi
            if isinstance(divider, np.ndarray):
                divider[divider==0.0] = EPSILON
            else:
                divider = EPSILON if divider==0.0 else divider
        return np.float32((data-mi)/divider)

    def get_epochs(self, batch_size):
        if self.rows_count % batch_size != 0:
            return self.rows_count // batch_size + 1  
        else:
            return self.rows_count // batch_size 
    
    def reset_cursor(self):
        self.curr_pos = 0

class logger_hook(tf.train.SessionRunHook):

    def __init__(self, loss):
        super(logger_hook, self).__init__()
        self.loss = loss

    def begin(self):
        self.step = -1
        self.t0 = time.time()
        
    def before_run(self, run_context):
        self.step += 1
        return tf.train.SessionRunArgs(self.loss)

    def after_run(self, run_context, run_values):
        if self.step %  log_freq == 0:
            t1 = time.time()
            duration = t1 - self.t0
            self.t0 = t1
            loss_value = run_values.results
            examples_per_sec =  log_freq *  batch_size / duration
            sec_per_batch = float(duration /  log_freq)
            format_str = ('%s: step %d, loss = %.3f (%.1f examples/sec; %.3f sec/batch')
            print(format_str % (datetime.now(), self.step, loss_value, examples_per_sec, sec_per_batch))
            
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Train Lenet")
    parser.add_argument("--model_dir", type=str, default="./model")
    opt = parser.parse_args()

    hook = smd.SessionHook.create_from_json_file()

    epoch = 50
    batch_size = 128

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    train_X, train_Y = (x_train, y_train)
    val_X, val_Y = (x_test[:len(x_test)//2], y_test[:len(y_test)//2])
    test_X, test_Y = (x_test[len(x_test)//2:], y_test[len(y_test)//2:])
    
    num_train = train_X.shape[0]
    num_val = val_X.shape[0]
    num_test = test_X.shape[0]

    train_X = np.reshape(train_X, ( num_train, 28, 28, 1))
    #train_Y = create_one_hot(train_Y, 10)

    val_X = np.reshape(val_X, ( num_val, 28, 28, 1))
    #val_Y = create_one_hot(val_Y, 10)

    test_X = np.reshape(test_X, ( num_test, 28, 28, 1))
    #test_Y = create_one_hot(test_Y, 10)
    
    # Create placeholder
    features = tf.placeholder(np.float32,  [None, 28, 28, 1], name='input_img')
    labels = tf.placeholder(np.float32, [None, 10], name='target')
    
    # Create the model 
    kernel_shape = (3,3,1,64) 
    with tf.variable_scope('conv1') as scope:
        x = create_conv_layer(features, kernel_shape, use_bias=True, activation=tf.nn.relu, name=scope.name)
    
    kernel_shape = (3,3,64,64)
    with tf.variable_scope('conv2') as scope:
        x = create_conv_layer(x, kernel_shape, use_bias=True, activation=tf.nn.relu, name=scope.name)
    
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding = 'VALID', name = 'pool1')
    
    kernel_shape = (3, 3, 64, 128) 
    with tf.variable_scope('conv3') as scope:
        x = create_conv_layer(x, kernel_shape, use_bias=True, activation=tf.nn.relu, name=scope.name)

    kernel_shape = (3, 3, 128, 128) 
    with tf.variable_scope('conv4') as scope:
        x = create_conv_layer(x, kernel_shape, use_bias=True, activation=tf.nn.relu, name=scope.name)
     
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding = 'VALID', name = 'pool2')

    kernel_shape = (3, 3, 128, 256)
    with tf.variable_scope('conv5') as scope:
        x = create_conv_layer(x, kernel_shape, use_bias=True, activation=tf.nn.relu, name=scope.name)
     
    kernel_shape = (3, 3, 256, 256) 
    with tf.variable_scope('conv6') as scope:
        x = create_conv_layer(x, kernel_shape, use_bias=True, activation=tf.nn.relu, name=scope.name)
     
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding = 'VALID', name = 'pool3')

    with tf.variable_scope('fc') as scope:
        x = create_linear_layer(x, (3*3*256, 10), use_bias=True, name=scope.name)
    softmax_out = tf.nn.softmax(x)
    trainable_params = [var.name for var in tf.trainable_variables()]
    weights_names = [param for param in trainable_params if 'weight' in param]
    for w_name in weights_names:
        hook.add_to_collection("weights", _as_graph_element(w_name))
    biases_names = [param for param in trainable_params if 'bias' in param]
    for b_name in biases_names:
        hook.add_to_collection("biases", _as_graph_element(b_name))

    x_shape = tf.shape(x) 
    x = tf.reshape(x, (x_shape[0], 1, 1, x_shape[1])) 
    l = loss()
    data_loss = l.softmax_log_loss(x, labels)
    reg_loss = l.l2_loss('weight', lm=0.0)
    total_loss = l.total_loss()
    hook.add_to_collection("losses", total_loss)
    t = trainer(hook, total_loss)
    t.create_adam_optimizer(0.001)
    train_step = t.get_trainer()

    train_data_loader = DataLoaderFromArrays(train_X, train_Y, shuffle=True, one_hot=True, normalization=True)
    test_data_loader = DataLoaderFromArrays(val_X, val_Y, shuffle=True, one_hot=True, normalization=True)
    # start the training.
    train_epoch_iters = train_data_loader.get_epochs(batch_size)
    
    # Do not need to pass the hook to the session if in a zero script change environment
    # i.e. SageMaker/AWS Deep Learning Containers
    # as the framework will automatically do that there if the hook exists
    sess = tf.train.MonitoredSession()

    # use this session for running the tensorflow model
    hook.set_mode(smd.modes.TRAIN)
    for _ in range(epoch):
        for i in range(train_epoch_iters):
            batch_x, batch_y = train_data_loader.next_batch(batch_size)
            _, _loss = sess.run([train_step, total_loss], feed_dict={features: batch_x, labels: batch_y})
            if i % 500 == 0:
                print(f"Step={i}, Loss={_loss}")
    
    test_epoch_iters = test_data_loader.get_epochs(batch_size)
    # set the mode for monitored session based runs
    # so smdebug can separate out steps by mode
    hook.set_mode(smd.modes.EVAL)
    for i in range(test_epoch_iters):
        batch_x, batch_y = test_data_loader.next_batch(batch_size)
        sess.run(total_loss, feed_dict={features: batch_x, labels: batch_y})
