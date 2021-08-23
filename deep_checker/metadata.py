import collections
import numpy as np
import tensorflow as tf
import scipy.stats as stats
from tensorflow.python.training.basic_session_run_hooks import _as_graph_element
from deep_checker.settings import CLASSIFICATION_KEY, REGRESSION_KEY
import deep_checker.utils as utils

class InputData:

    def __init__(self, data, problem_type):
        self.homogeneous = data.homogeneous
        self.n_repeats = 300
        self.sample_size = 128
        self.sampled_data = self.sampling_data(data.train_loader)
        self.problem_type = problem_type
        self.extract_meta_data()
        
    def sampling_data(self, data_loader):
        sampled_data = {}
        data_features, data_targets = data_loader.next_batch(self.n_repeats*self.sample_size)
        sampled_data['features'] = data_features
        sampled_data['targets'] = data_targets
        return sampled_data

    def get_sample(self, sample_size):
        features, targets = self.sampled_data['features'], self.sampled_data['targets']
        if self.problem_type == CLASSIFICATION_KEY:
            N_labels = self.targets_metadata['labels']
            if self.targets_metadata['onehot']:
                labels = np.argmax(targets, axis=1)  
            else:
                labels = np.squeeze(targets)
            N_per_class = int(sample_size / N_labels)
            for label_idx in range(N_labels):
                lbl_indices = np.argwhere(labels==label_idx).flatten()
                selected_indices = np.random.choice(lbl_indices, N_per_class)
                if label_idx == 0:
                    batch_x, batch_y = features[selected_indices], targets[selected_indices]
                else:
                    batch_x = np.concatenate([batch_x, features[selected_indices]],axis=0)
                    batch_y = np.concatenate([batch_y, targets[selected_indices]],axis=0)
            if len(batch_y.shape) == 1:
                batch_y = batch_y.reshape(batch_y.shape[0], 1)
        elif self.problem_type == REGRESSION_KEY:
            selected_indices = np.random.choice(np.arange(features.shape[0]), sample_size)
            batch_x, batch_y = features[selected_indices], targets[selected_indices]
        return batch_x, batch_y
        
    def extract_meta_data(self):
        if self.homogeneous:
            self.features_metadata = utils.reduce_data(self.sampled_data['features'], 
                                                        reductions=['mean', 'std', 'max', 'min'])
        else:
            self.features_metadata = utils.reduce_data(self.sampled_data['features'], 
                                                        reductions=['mean', 'std', 'max', 'min'], 
                                                        axis=0)
        targets = self.sampled_data['targets']
        if self.problem_type == CLASSIFICATION_KEY:
            self.targets_metadata = {}
            if targets.shape[1] == 1:
                self.targets_metadata['count'] = 1
                self.targets_metadata['labels'] = 2
                self.targets_metadata['onehot'] = False
                labels_probas = np.zeros(2)
                labels_probas[0] = np.mean(1.0 - targets) 
                labels_probas[1] = np.mean(targets) 
            else:
                self.targets_metadata['count'] = targets.shape[1]
                self.targets_metadata['labels'] = targets.shape[1]
                self.targets_metadata['onehot'] = True
                labels_probas = np.zeros(targets.shape[1])
                labels_probas = np.mean(targets, axis=0)
            perplexity = np.exp(stats.entropy(labels_probas))
            self.targets_metadata['probas'] = labels_probas
            self.targets_metadata['balance'] = (perplexity-1)/(self.targets_metadata['labels']-1)
        elif self.problem_type == REGRESSION_KEY:
            self.targets_metadata = utils.reduce_data(targets, 
                                                        reductions=['mean', 'std', 'max', 'min'], 
                                                        axis=0)
            self.targets_metadata['count'] = targets.shape[1] if len(targets.shape) == 2 else 1
            self.targets_metadata['labels'] = None
            self.targets_metadata['max_abs_greater_than_one'] = np.abs(self.targets_metadata['max']) > 1.0
            self.targets_metadata['can_be_negative'] = self.targets_metadata['min'] < 0

class DNNState:

    def __init__(self, model, buff_scale):
        self.model = model
        self.buff_scale = buff_scale
        self.loss_data = []
        self.perf_data = []

    def init_or_reset(self, batch_size):
        self.acts_data = self.init_acts_tensors_and_data(batch_size)
        self.weights_gradients = self.init_gradients_tensors()
        self.weights_reductions = {weight_name:collections.deque(maxlen=self.buff_scale) \
                                   for weight_name in self.model.weights.keys()}
        self.gradients_reductions = {weight_name:collections.deque(maxlen=self.buff_scale) \
                                     for weight_name in self.model.weights.keys()}
        self.biases_reductions = {}
        if self.model.biases: #In case, there is no bias 
            self.biases_reductions = {bias_name:collections.deque(maxlen=self.buff_scale) \
                                    for bias_name in self.model.biases.keys()}
        

    def init_acts_tensors_and_data(self, batch_size):
        acts_data = {}
        for act_name, act_tensor in self.model.activations.items():
            dims = [int(dim) for dim in act_tensor.get_shape()[1:]]
            buffer_size = self.buff_scale * batch_size
            acts_data[act_name] = np.zeros(shape=(buffer_size,*dims)) 
        return acts_data
    
    def init_gradients_tensors(self):
        names = list(self.model.weights.keys())
        tensors = list(self.model.weights.values())
        gradients = tf.gradients(self.model.loss, tensors)
        return {wn:wg for wn,wg in list(zip(names,gradients))}
