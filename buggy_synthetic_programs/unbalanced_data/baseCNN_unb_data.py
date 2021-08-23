'''
We create a custom data loader with unbalance function that preseve decreasingly ratio of instances
belonging to some classes aiming at creating unbalanced training dataset.
'''
import numpy as np
import pandas as pd
import tensorflow as tf
from deep_checker.checkers import DeepChecker
import deep_checker.interfaces as interfaces
import deep_checker.data as data
from deep_checker.settings import CLASSIFICATION_KEY, REGRESSION_KEY, EPSILON

class CustomDataLoader:

    def __init__(self, features, targets, homogeneous=True, 
                problem_type=CLASSIFICATION_KEY, shuffle=True, 
                one_hot=True, normalization=True):
        features, targets = self.unbalance(features, targets)
        self.curr_pos = 0
        self.rows_count = features.shape[0]
        self.homogeneous = homogeneous
        self.to_shuffle = shuffle
        self.problem_type = problem_type
        self.normalization = normalization
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
        indices = np.arange(0, self.rows_count)  # get all possible indexes
        np.random.shuffle(indices)  # shuffle indexes
        self.features = self.features[indices]
        self.targets = self.targets[indices]

    def preprocess(self, targets):
        if self.problem_type == REGRESSION_KEY:
            targets = self.normalize(targets)
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
            divider[divider==0.0] = EPSILON
        return np.float32((data-mi)/divider)

    def get_epochs(self, batch_size):
        if self.rows_count % batch_size != 0:
            return self.rows_count // batch_size + 1  
        else:
            return self.rows_count // batch_size

    def unbalance(self, features, labels):
        p = 0.01
        total = int(features.shape[0]/10)
        indices = np.argwhere(labels==0).squeeze()
        new_labels = labels[indices]
        new_labels = new_labels[:int(total*p)]
        new_features = features[indices]
        new_features = new_features[:int(total*p)]
        for lbl in range(1,10):
            indices = np.argwhere(labels==lbl).squeeze()
            new_feat = features[indices]
            new_feat = new_feat[:int(total*(p*lbl))] if lbl <= 5 else new_feat
            new_lbl = labels[indices]
            new_lbl =  new_lbl[:int(total*(p*lbl))] if lbl <= 5 else new_lbl
            new_labels = np.concatenate((new_labels, new_lbl), axis=0)
            new_features = np.concatenate((new_features, new_feat), axis=0)
        return new_features, new_labels

    def reset_cursor(self):
        self.curr_pos = 0

class Model:

    def __init__(self):
        self.features = tf.placeholder(tf.float32, [None, 28, 28])
        self.images = tf.reshape(self.features, [-1, 28, 28, 1])
        self.n_classes = 10
        self.labels = tf.placeholder(tf.float32, [None, self.n_classes])
        self.logits = self.build(self.images, self.n_classes)
        self.probabilities = tf.nn.softmax(self.logits)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.labels, 1), tf.argmax(self.logits, 1)), tf.float32))
        self.reg_loss = tf.losses.get_regularization_loss()
        self.loss = tf.losses.softmax_cross_entropy(self.labels, self.logits)
        self.train_op = tf.train.GradientDescentOptimizer(0.1).minimize((self.loss+self.reg_loss))

    def build(self, features, n_classes, l1=0.0, l2=1e-5):
        with tf.variable_scope('lenet'):
            # Convolution Layer with 32 filters and a kernel size of 5
            conv1 = tf.layers.conv2d(inputs=features,
                                    filters=32,
                                    kernel_size=[5, 5], 
                                    padding='same',
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.initializers.he_normal(),
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1,l2))
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            # Convolution Layer with 64 filters and a kernel size of 3
            conv2 = tf.layers.conv2d(inputs=pool1, 
                                    filters=64, 
                                    kernel_size=[3, 3],
                                    padding='same', 
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.initializers.he_normal(),
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1,l2))
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            # Flatten the data to a 1-D vector for the fully connected layer
            pool2_flat = tf.layers.flatten(pool2)

            # Fully connected layer 
            dense = tf.layers.dense(inputs=pool2_flat, 
                                    units=1024, 
                                    activation=tf.nn.relu,
                                    kernel_initializer=tf.initializers.he_normal(),
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1,l2))

            outputs = tf.layers.dense(inputs=dense,
                                      units=n_classes, 
                                      activation=None,
                                      kernel_regularizer=tf.keras.regularizers.l1_l2(l1,l2))
        return outputs

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    data_loader_under_test = CustomDataLoader(x_train, y_train, shuffle=True, one_hot=True, normalization=True)
    test_data_loader = data.DataLoaderFromArrays(x_test, y_test, shuffle=True, one_hot=True, normalization=True)
    model = Model()
    model_under_test =  interfaces.build_model_interface(problem_type='classification', 
                          test_mode=False,
                          features=model.features, 
                          targets=model.labels, 
                          train_op=model.train_op, 
                          loss=model.loss, 
                          reg_loss=model.reg_loss,
                          perf=model.acc, 
                          perf_dir='Up', 
                          outs_pre_act=model.logits, 
                          outs_post_act=model.probabilities)
    data_under_test = interfaces.build_data_interface(data_loader_under_test, test_data_loader, homogeneous=True)
    checker = DeepChecker(name='base_CNN_unbalanced_data', data=data_under_test, model=model_under_test, buffer_scale=10)
    checker.run_full_checks()
    