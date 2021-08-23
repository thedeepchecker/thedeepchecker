import numpy as np
import tensorflow as tf
from tensorflow.python.training.basic_session_run_hooks import _as_graph_element
from deep_checker.settings import CLASSIFICATION_KEY, REGRESSION_KEY

def build_model_interface(problem_type, features, targets, train_op, \
                          loss, perf, perf_dir, outs_pre_act, test_mode=False, \
                          test_loss=None, test_perf=None, test_activations=None, \
                          outs_post_act=None, test_outs_post_act=None, test_outs_pre_act=None, \
                          reg_loss=None, weights=None, biases=None, activations=None, \
                          test_extra_feed_dict={}, train_extra_feed_dict={}):
    def _is_activation_func(operation):
        act_functions = ['relu', 'elu', 'sigmoid', 'tanh', 'selu', 'relu6', 'leaky_relu']
        for act_func in act_functions:
            if(act_func == operation.lower().split('/')[-1]):
                return True
        return False

    model = Model()

    trainable_params = [var.name for var in tf.trainable_variables()]
    
    if weights is None:
        weights = [param for param in trainable_params if ('kernel' in param or 'weight' in param)]

    weights_dict = {}
    for w_name in weights:
        weights_dict[w_name] = _as_graph_element(w_name) 
        model.set_weights(weights_dict)

    if biases is None:
        biases = [param for param in trainable_params if 'bias' in param]

    biases_dict = {}
    for b_name in biases:
        biases_dict[b_name] = _as_graph_element(b_name) 
        model.set_biases(biases_dict)

    if activations is None:
        all_activations = [op.name for op in tf.get_default_graph().get_operations() if _is_activation_func(op.name)]
        activations = [act_name for act_name in all_activations if act_name.split('/')[0].isalpha()]

    if test_mode:
        if test_activations is None:
            all_activations = [op.name for op in tf.get_default_graph().get_operations() if _is_activation_func(op.name)]
            test_activations = [act_name for act_name in all_activations if not act_name.split('/')[0].isalpha()]
    else:
        test_activations = activations
        test_loss = loss
        test_perf = perf
        if outs_post_act == None:
            outs_post_act = outs_pre_act
        test_outs_post_act = outs_post_act 
        test_outs_pre_act = outs_pre_act
    test_mode = test_mode or (train_extra_feed_dict != test_extra_feed_dict)
    acts_dict = {}
    for act_name in activations:
        acts_dict[act_name] = _as_graph_element(act_name) 
    model.set_activations(acts_dict)
    test_acts_dict = {}
    for act_name, test_act_name in zip(activations,test_activations):
        test_acts_dict[act_name] = _as_graph_element(test_act_name) 
    model.set_test_mode(test_mode)
    model.set_test_activations(test_acts_dict)
    model.set_problem_type(problem_type)
    model.set_features(features)
    model.set_targets(targets)
    model.set_train_op(train_op)
    model.set_loss(loss)
    model.set_reg_loss(reg_loss)
    model.set_perf(perf)
    model.set_perf_dir(perf_dir)
    model.set_test_loss(test_loss)
    model.set_test_perf(test_perf)
    model.set_outs_pre_act(outs_pre_act)
    model.set_outs_post_act(outs_post_act)
    model.set_test_outs_pre_act(test_outs_pre_act)
    model.set_test_outs_post_act(test_outs_post_act)
    model.set_train_extra_feed_dict(train_extra_feed_dict)
    model.set_test_extra_feed_dict(test_extra_feed_dict)
    return model

def build_data_interface(train_loader, test_loader, homogeneous):
    return Data(train_loader, test_loader, homogeneous)

class Model:

    def __init__(self):
        self.problem_type = None
        self.features = None
        self.targets = None
        self.train_op = None
        self.loss = None
        self.reg_loss = None
        self.perf = None
        self.perf_dir = None
        self.test_loss = None
        self.test_perf = None
        self.outs_pre_act = None
        self.outs_post_act = None
        self.test_outs_pre_act = None
        self.test_outs_post_act = None
        self.weights = None
        self.biases = None
        self.activations = None
        self.test_activations = None
        self.train_op = None
        self.train_extra_feed_dict = None
        self.test_extra_feed_dict = None
        self.act_fn_name = None
        self.acts_max_bound = self.acts_min_bound = None
        self.test_mode = None

    def set_activation_func(self):
        activations = [name for name in self.activations.keys()]
        if activations:
            operation = activations[0]
        else:
            operation = 'identity'
        act_functions = ['relu', 'elu', 'sigmoid', 'tanh', 'selu', 'relu6', 'leaky_relu']
        for act_func in act_functions:
            if(act_func == operation.lower().split('/')[-1]):
                break
        self.act_fn_name = act_func

    def set_max_min_bound(self, activation_max_bound=None, activation_min_bound=None):
        if activation_max_bound==None or activation_min_bound==None:
            name = self.act_fn_name
            if name == 'elu':
                activation_max_bound = +np.inf
                activation_min_bound = -1.0
            elif name == 'leaky_relu':
                activation_max_bound = +np.inf
                activation_min_bound = -np.inf
            elif name == 'relu6':
                activation_max_bound = 6.0
                activation_min_bound = 0.0 
            elif name == 'selu':
                activation_max_bound = +np.inf
                activation_min_bound = -np.inf
            elif name == 'tanh':
                activation_max_bound = 1.0
                activation_min_bound = -1.0 
            elif name == 'sigmoid':
                activation_max_bound = 1.0
                activation_min_bound = 0.0 
            elif name == 'relu':
                activation_max_bound = +np.inf
                activation_min_bound = 0.0 
            else:
                activation_max_bound = +np.inf
                activation_min_bound = -np.inf
        self.acts_max_bound, self.acts_min_bound = activation_max_bound, activation_min_bound

    def set_test_mode(self, test_mode):
        self.test_mode = test_mode

    def set_problem_type(self, problem_type):
        self.problem_type = problem_type

    def set_features(self, features):
        self.features = features
    
    def set_targets(self, targets):
        self.targets = targets
    
    def set_train_op(self, train_op):
        self.train_op = train_op
    
    def set_loss(self, loss):
        self.loss = loss
    
    def set_reg_loss(self, reg_loss):
        self.reg_loss = reg_loss
    
    def set_perf(self, perf):
        self.perf = perf
    
    def set_perf_dir(self, perf_dir):
        self.perf_dir = perf_dir
    
    def set_test_loss(self, test_loss):
        self.test_loss = test_loss

    def set_test_perf(self, test_perf):
        self.test_perf = test_perf
    
    def set_outs_pre_act(self, outs_pre_act):
        self.outs_pre_act = outs_pre_act
    
    def set_outs_post_act(self, outs_post_act):
        self.outs_post_act = outs_post_act

    def set_test_outs_pre_act(self, test_outs_pre_act):
        self.test_outs_pre_act = test_outs_pre_act
    
    def set_test_outs_post_act(self, test_outs_post_act):
        self.test_outs_post_act = test_outs_post_act
    
    def set_weights(self, weights):
        self.weights = weights
    
    def set_biases(self, biases):
        self.biases = biases
    
    def set_activations(self, activations):
        self.activations = activations
        self.set_activation_func()
    
    def set_test_activations(self, test_activations):
        self.test_activations = test_activations
    
    def set_train_op(self, train_op):
        self.train_op = train_op
    
    def set_train_extra_feed_dict(self, train_extra_feed_dict):
        self.train_extra_feed_dict = train_extra_feed_dict
    
    def set_test_extra_feed_dict(self, test_extra_feed_dict):
        self.test_extra_feed_dict = test_extra_feed_dict

class Data:

    def __init__(self, train_loader, test_loader, homogeneous):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.homogeneous = homogeneous