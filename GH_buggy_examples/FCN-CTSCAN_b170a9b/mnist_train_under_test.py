from tfun.loss import loss
from tfun.trainer import trainer
from tfun.util import create_one_hot, create_var, create_conv_layer, create_linear_layer
import tfun.global_config as global_cfg
import tensorflow as tf
import numpy as np
import time
from datetime import datetime
from tfcheck import DNNChecker
import interfaces
import data 

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
        if self.step % global_cfg.log_freq == 0:
            t1 = time.time()
            duration = t1 - self.t0
            self.t0 = t1
            loss_value = run_values.results
            examples_per_sec = global_cfg.log_freq * global_cfg.batch_size / duration
            sec_per_batch = float(duration / global_cfg.log_freq)
            format_str = ('%s: step %d, loss = %.3f (%.1f examples/sec; %.3f sec/batch')
            print(format_str % (datetime.now(), self.step, loss_value, examples_per_sec, sec_per_batch))
            
if __name__ == "__main__":
    global_cfg.batch_size = 128
    global_cfg.train_dir = './logs'

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    train_X, train_Y = (x_train, y_train)
    val_X, val_Y = (x_test[:len(x_test)//2], y_test[:len(y_test)//2])
    test_X, test_Y = (x_test[len(x_test)//2:], y_test[len(y_test)//2:])
    
    global_cfg.num_train = train_X.shape[0]
    global_cfg.num_val = val_X.shape[0]
    global_cfg.num_test = test_X.shape[0]

    train_X = np.reshape(train_X, (global_cfg.num_train, 28, 28, 1))
    #train_Y = create_one_hot(train_Y, 10)

    val_X = np.reshape(val_X, (global_cfg.num_val, 28, 28, 1))
    #val_Y = create_one_hot(val_Y, 10)

    test_X = np.reshape(test_X, (global_cfg.num_test, 28, 28, 1))
    #test_Y = create_one_hot(test_Y, 10)

    # Create placeholder
    features = tf.placeholder(global_cfg.dtype,  [None, 28, 28, 1], name='input_img')
    labels = tf.placeholder(global_cfg.dtype, [None, 10], name='target')
    
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
    x_shape = tf.shape(x) 
    x = tf.reshape(x, (x_shape[0], 1, 1, x_shape[1])) 
    l = loss()
    data_loss = l.softmax_log_loss(x, labels)
    reg_loss = l.l2_loss('weight', lm=0.0)
    total_loss = l.total_loss()

    t = trainer(total_loss)
    t.create_adam_optimizer(0.001)
    t_ = t.get_trainer()

    data_loader_under_test = data.DataLoaderFromArrays(train_X, train_Y, shuffle=True, one_hot=True, normalization=True)
    test_data_loader = data.DataLoaderFromArrays(val_X, val_Y, shuffle=True, one_hot=True, normalization=True)
    model_under_test =  interfaces.build_model_interface(problem_type='classification', 
                          test_mode=False,
                          features=features, 
                          targets=labels, 
                          activations=['conv1/relu:0', 'conv2/relu:0', 'conv3/relu:0', 'conv4/relu:0', 'conv5/relu:0', 'conv6/relu:0'],
                          train_op=t_, 
                          loss=data_loss, 
                          reg_loss=reg_loss,
                          perf=total_loss, 
                          perf_dir='Down', 
                          outs_pre_act=x, 
                          outs_post_act=softmax_out)
    data_under_test = interfaces.build_data_interface(data_loader_under_test, test_data_loader, homogeneous=True)
    checker = DNNChecker(name='b170a9b', data=data_under_test, model=model_under_test, buffer_scale=10)
    checker.run_full_checks()
