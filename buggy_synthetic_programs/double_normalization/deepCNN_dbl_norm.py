'''
Some API functions including tf.image.convert_image_dtype and all tf.image.decode_* would normalize implicitly 
the data to [0, 1] so if the users divide the data, additionally, by 255, the magnitude of data would be critically low.
To illustrate this fault in a general way, we create a custom data loader that doubly normalize the data by dividing twice by the range=(max-min)
'''
import numpy as np
import pandas as pd
import tensorflow as tf
from deep_checker.checkers import DeepChecker
import deep_checker.interfaces as interfaces
import deep_checker.data as data
from deep_checker.settings import EPSILON

class CustomDataLoader(data.DataLoaderFromArrays):

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
            divider[divider==0.0] = EPSILON
        return np.float32(((data-mi)/divider)/divider)
        
class Model:
    # VGG-like
    def __init__(self):
        self.features = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.n_classes = 10
        self.labels = tf.placeholder(tf.float32, [None, self.n_classes])
        self.logits = self.build(self.features, self.n_classes, reuse=False, is_training=True)
        self.probabilities = tf.nn.softmax(self.logits)
        self.test_logits = self.build(self.features, self.n_classes, reuse=True, is_training=False)
        self.test_probabilities = tf.nn.softmax(self.test_logits)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.probabilities, 1)), tf.float32))
        self.test_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.test_probabilities, 1)), tf.float32))
        self.loss = tf.losses.softmax_cross_entropy(self.labels, self.logits)
        self.test_loss = tf.losses.softmax_cross_entropy(self.labels, self.test_logits)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.MomentumOptimizer(learning_rate=0.001,momentum=0.9).minimize(self.loss)

    def build(self, features, n_classes, reuse=False, is_training=False):
        with tf.variable_scope('lenet', reuse=reuse):
            # Convolution Layer with 32 filters and a kernel size of 3
            conv1 = tf.layers.conv2d(inputs=features,
                                    filters=32,
                                    kernel_size=[3, 3], 
                                    padding='same',
                                    activation=tf.nn.relu,
                                    use_bias=False,
                                    kernel_initializer=tf.initializers.he_uniform())
            conv1_norm = tf.layers.batch_normalization(conv1, training=is_training)
            conv2 = tf.layers.conv2d(inputs=conv1_norm,
                                    filters=32,
                                    kernel_size=[3, 3], 
                                    padding='same',
                                    activation=tf.nn.relu,
                                    use_bias=False,
                                    kernel_initializer=tf.initializers.he_uniform())
            conv2_norm = tf.layers.batch_normalization(conv2, training=is_training)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            pool2 = tf.layers.max_pooling2d(inputs=conv2_norm, pool_size=[2, 2], strides=2)
            reg_pool2 = tf.layers.dropout(inputs=pool2, rate=0.2, training=is_training)
            # Convolution Layer with 64 filters and a kernel size of 3
            conv3 = tf.layers.conv2d(inputs=reg_pool2, 
                                    filters=64, 
                                    kernel_size=[3, 3],
                                    padding='same', 
                                    activation=tf.nn.relu,
                                    use_bias=False,
                                    kernel_initializer=tf.initializers.he_uniform())
            conv3_norm = tf.layers.batch_normalization(conv3, training=is_training)
            conv4 = tf.layers.conv2d(inputs=conv3_norm, 
                                    filters=64, 
                                    kernel_size=[3, 3],
                                    padding='same', 
                                    activation=tf.nn.relu,
                                    use_bias=False,
                                    kernel_initializer=tf.initializers.he_uniform())
            conv4_norm = tf.layers.batch_normalization(conv4, training=is_training)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            pool4 = tf.layers.max_pooling2d(inputs=conv4_norm, pool_size=[2, 2], strides=2)
            reg_pool4 = tf.layers.dropout(inputs=pool4, rate=0.3, training=is_training)
            # Convolution Layer with 128 filters and a kernel size of 3
            conv5 = tf.layers.conv2d(inputs=reg_pool4, 
                                    filters=64, 
                                    kernel_size=[3, 3],
                                    padding='same', 
                                    activation=tf.nn.relu,
                                    use_bias=False,
                                    kernel_initializer=tf.initializers.he_uniform())
            conv5_norm = tf.layers.batch_normalization(conv5, training=is_training)
            conv6 = tf.layers.conv2d(inputs=conv5_norm, 
                                    filters=64, 
                                    kernel_size=[3, 3],
                                    padding='same', 
                                    activation=tf.nn.relu,
                                    use_bias=False,
                                    kernel_initializer=tf.initializers.he_uniform())
            conv6_norm = tf.layers.batch_normalization(conv6, training=is_training)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            pool6 = tf.layers.max_pooling2d(inputs=conv6_norm, pool_size=[2, 2], strides=2)
            reg_pool6 = tf.layers.dropout(inputs=pool6, rate=0.4, training=is_training)
            # Flatten the data to a 1-D vector for the fully connected layer
            reg_pool6_flat = tf.layers.flatten(reg_pool6)
            # Fully connected layer 
            dense = tf.layers.dense(inputs=reg_pool6_flat, 
                                    units=128, 
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.initializers.he_uniform())
            dense_norm = tf.layers.batch_normalization(dense, training=is_training)
            # Apply Dropout (if is_training is False, dropout is not applied)
            reg_dense_norm = tf.layers.dropout(inputs=dense_norm, rate=0.5, training=is_training)
            outputs = tf.layers.dense(inputs=reg_dense_norm,
                                      units=n_classes, 
                                      activation=None,
                                      kernel_initializer=tf.initializers.he_uniform())
        return outputs

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    data_loader_under_test = CustomDataLoader(x_train, y_train.squeeze(), shuffle=True, one_hot=True, normalization=True)
    test_data_loader = CustomDataLoader(x_test, y_test.squeeze(), shuffle=True, one_hot=True, normalization=True)
    model = Model()
    model_under_test =  interfaces.build_model_interface(problem_type='classification', 
                          test_mode=True,
                          features=model.features, 
                          targets=model.labels, 
                          train_op=model.train_op, 
                          loss=model.loss, 
                          test_loss=model.test_loss, 
                          perf=model.acc, 
                          test_perf=model.test_acc,
                          perf_dir='Up', 
                          outs_pre_act=model.logits, 
                          outs_post_act=model.probabilities, 
                          test_outs_pre_act=model.test_logits, 
                          test_outs_post_act=model.test_probabilities)
    data_under_test = interfaces.build_data_interface(data_loader_under_test, test_data_loader, homogeneous=True)
    checker = DeepChecker(name='deepCNN_dbl_norm', data=data_under_test, model=model_under_test, buffer_scale=10)
    checker.run_full_checks()