import os
import sys
import random
from collections.abc import Iterable
from pathlib import Path 
import numpy as np
import tensorflow as tf
import deep_checker.hooks as hooks
import deep_checker.settings as settings
import deep_checker.metrics as metrics
import deep_checker.utils as utils
from deep_checker.utils import readable
from deep_checker.metadata import DNNState, InputData
from deep_checker.settings import CLASSIFICATION_KEY, REGRESSION_KEY

class DeepChecker:

    def __init__(self, name, data, model, app_path=None, buffer_scale=10):
        app_path = Path.cwd() if app_path == None else app_path
        log_fpath = settings.build_log_file_path(app_path, name)
        self.logger = settings.file_logger(log_fpath, name)
        config_fpath = settings.load_user_config_if_exists(app_path)
        self.config = settings.Config(config_fpath)
        inputs_data = InputData(data, model.problem_type)
        dnn_state = DNNState(model, buffer_scale)
        self.pre_check = PreCheck(dnn_state, inputs_data, self.logger, self.config.pre_check)
        self.post_check = PostCheck(dnn_state, data, self.logger, self.config.post_check)
        self.overfit_check = OverfitCheck(dnn_state, inputs_data, self.logger, self.config.overfit_check)

    def setup(self, fixed_seed=True, tf_seed=42, np_seed=42, python_seed=42, use_multi_cores=False, use_GPU=False):
        if fixed_seed:
            if os.environ.get("PYTHONHASHSEED") != "0":
                self.logger.warning('You must set PYTHONHASHSEED=0 when running the python script If you wanna get reproducible results.')
            tf.set_random_seed(tf_seed)
            np.random.seed(np_seed)
            random.seed(python_seed)
        if not use_multi_cores:
            config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
            with tf.Session(config=config) as sess:
                pass
        if not use_GPU:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

    def run_full_checks(self, 
                        overfit_batch=50,
                        overfit_iters=200,
                        post_fitness_batch=64,
                        post_fitness_epochs=301,
                        fixed_seed=True,
                        use_multi_cores=False, 
                        use_GPU=True,
                        implemented_ops=[]):
        print('Setup...')
        self.setup(fixed_seed=fixed_seed, use_multi_cores=use_multi_cores, use_GPU=use_GPU)
        self.run_pre_checks(overfit_batch, implemented_ops)
        input('Done! Press enter to continue running single-batch overfit checks...')
        self.run_overfit_checks(overfit_batch, overfit_iters)
        input('Done! Press enter to continue running post-checks...')
        self.run_post_checks(post_fitness_batch, post_fitness_epochs)

    def run_pre_checks(self, batch_size, implemented_ops):
        self.pre_check.run(batch_size, implemented_ops)
    
    def run_post_checks(self, post_fitness_batch, post_fitness_epochs):
        self.post_check.run(post_fitness_batch, post_fitness_epochs)
    
    def run_overfit_checks(self, overfit_batch, overfit_iters):
        self.overfit_check.run(overfit_batch, overfit_iters)

class PreCheck:
    
    def __init__(self, nn_data, inputs_data, main_logger, config):
        self.nn_data = nn_data
        self.inputs_data = inputs_data
        self.main_logger = main_logger
        self.config = config
        self.main_msgs = settings.load_messages()
 
    def react(self, message):
        if self.config.fail_on:
            self.main_logger.error(message)
            raise Exception(message)
        else:
            self.main_logger.warning(message)

    def _pre_check_features(self):
        if self.config.data.disabled: return
        if self.inputs_data.homogeneous:
            mas = [self.inputs_data.features_metadata['max']]
            mis = [self.inputs_data.features_metadata['min']]  
            avgs = [self.inputs_data.features_metadata['mean']]
            stds = [self.inputs_data.features_metadata['std']]
        elif isinstance(self.inputs_data.features_metadata['max'], Iterable):
            mas = list(self.inputs_data.features_metadata['max'])
            mis = list(self.inputs_data.features_metadata['min'])  
            avgs = list(self.inputs_data.features_metadata['mean'])
            stds = list(self.inputs_data.features_metadata['std'])
        else:
            mas = [self.inputs_data.features_metadata['max']]
            mis = [self.inputs_data.features_metadata['min']]  
            avgs = [self.inputs_data.features_metadata['mean']]
            stds = [self.inputs_data.features_metadata['std']]
        for idx in range(len(mas)):
            if stds[idx] == 0.0:  
                msg = self.main_msgs['features_constant'] if len(mas) == 1 else self.main_msgs['feature_constant'].format(idx)
                self.react(msg)
            elif any([utils.almost_equal(mas[idx], data_max) for data_max in self.config.data.normalized_data_maxs]) and \
                any([utils.almost_equal(mis[idx], data_min) for data_min in self.config.data.normalized_data_mins]):
                return 
            elif not(utils.almost_equal(stds[idx], 1.0) and utils.almost_equal(avgs[idx], 0.0)):
                msg = self.main_msgs['features_unnormalized'] if len(mas) == 1 else self.main_msgs['feature_unnormalized'].format(idx)
                self.react(msg)

    def _pre_check_targets(self):
        if self.config.data.disabled: return
        if self.inputs_data.problem_type == CLASSIFICATION_KEY:
            if self.inputs_data.targets_metadata['balance'] < self.config.data.labels_perp_min_thresh:
                self.react(self.main_msgs['unbalanced_labels'])
        elif self.inputs_data.problem_type == REGRESSION_KEY:
            if self.inputs_data.targets_metadata['count'] == 1:
                mas = [self.inputs_data.targets_metadata['max']]
                mis = [self.inputs_data.targets_metadata['min']]  
                avgs = [self.inputs_data.targets_metadata['mean']]
                stds = [self.inputs_data.targets_metadata['std']]
            else:
                mas = list(self.inputs_data.targets_metadata['max'])
                mis = list(self.inputs_data.targets_metadata['min'])  
                avgs = list(self.inputs_data.targets_metadata['mean'])
                stds = list(self.inputs_data.targets_metadata['std'])
            for idx in range(len(mas)):
                if utils.almost_equal(stds[idx], 0.0):
                    msg = self.main_msgs['targets_unnormalized'] if len(mas) == 1 else self.main_msgs['target_unnormalized'].format(idx)
                    self.react(msg)
                elif any([utils.almost_equal(mas[idx], data_max) for data_max in self.config.data.normalized_data_maxs]) and \
                    any([utils.almost_equal(mis[idx], data_min) for data_min in self.config.data.normalized_data_mins]):
                    return
                elif not(utils.almost_equal(stds[idx], 1.0) and utils.almost_equal(avgs[idx], 0.0)):
                    msg = self.main_msgs['targets_unnormalized'] if len(mas) == 1 else self.main_msgs['target_unnormalized'].format(idx)
                    self.react(msg)

    def _pre_check_weights(self, session):
        if self.config.init_w.disabled: return
        weights_tensors = self.nn_data.model.weights
        initial_weights = session.run(weights_tensors)
        for weight_name, weight_array in initial_weights.items():
            shape = weight_array.shape
            if len(shape) == 1 and shape[0] == 1: continue
            if utils.almost_equal(np.var(weight_array), 0.0, rtol=1e-8):
                self.react(self.main_msgs['poor_init'].format(weight_name))  
            else:
                if len(shape) == 2:
                    fan_in = shape[0]
                    fan_out = shape[1]
                else:
                    receptive_field_size = np.prod(shape[:-2])
                    fan_in = shape[-2] * receptive_field_size
                    fan_out = shape[-1] * receptive_field_size
                lecun_F, lecun_test = metrics.pure_f_test(weight_array, np.sqrt(1.0/fan_in), self.config.init_w.f_test_alpha)
                he_F, he_test = metrics.pure_f_test(weight_array, np.sqrt(2.0/fan_in), self.config.init_w.f_test_alpha)
                glorot_F, glorot_test = metrics.pure_f_test(weight_array, np.sqrt(2.0/(fan_in + fan_out)), self.config.init_w.f_test_alpha)
                if self.nn_data.model.act_fn_name == 'relu' and not he_test:
                    abs_std_err = np.abs(np.std(weight_array)-np.sqrt(1.0/fan_in))
                    self.react(self.main_msgs['need_he'].format(weight_name, abs_std_err))
                elif self.nn_data.model.act_fn_name == 'tanh' and not glorot_test:
                    abs_std_err = np.abs(np.std(weight_array)-np.sqrt(2.0/fan_in))
                    self.react(self.main_msgs['need_glorot'].format(weight_name, abs_std_err))
                elif self.nn_data.model.act_fn_name == 'sigmoid' and not lecun_test:
                    abs_std_err = np.abs(np.std(weight_array)-np.sqrt(2.0/(fan_in + fan_out)))
                    self.react(self.main_msgs['need_lecun'].format(weight_name, abs_std_err))
                elif not(lecun_test or he_test or glorot_test):
                    self.react(self.main_msgs['need_init_well'].format(weight_name))         

    def _pre_check_biases(self, session):
        if self.config.init_b.disabled: return
        biases_tensors = self.nn_data.model.biases
        if not(biases_tensors):
            self.react(self.main_msgs['need_bias'])
        else:
            initial_biases = session.run(biases_tensors)
            checks = []
            for b_name, b_array in initial_biases.items():
                checks.append(np.sum(b_array)==0.0)
            if self.inputs_data.problem_type == CLASSIFICATION_KEY and \
                self.inputs_data.targets_metadata['balance'] < self.config.data.labels_perp_min_thresh:
                if checks[-1]:
                    self.react(self.main_msgs['last_bias'])
                elif not checks[-1]:
                    bias_indices = np.argsort(b_array) 
                    probas_indices = np.argsort(self.inputs_data.targets_metadata['probas']) 
                    if not np.equal(bias_indices, probas_indices):
                        self.react(self.main_msgs['ineff_bias_cls']) 
            elif self.inputs_data.problem_type == REGRESSION_KEY:
                if self.inputs_data.targets_metadata['count'] == 1:
                    avgs = [self.inputs_data.targets_metadata['mean']]
                    stds = [self.inputs_data.targets_metadata['std']]
                else:
                    avgs = list(self.inputs_data.targets_metadata['mean'])
                    stds = list(self.inputs_data.targets_metadata['std'])
                var_coefs = [std/avg for avg, std in zip(avgs, stds)]
                low_var_coefs_indices = [i for i, var_coef in enumerate(var_coefs) if var_coef <= 1e-3]
                for idx in low_var_coefs_indices:
                    b_value = float(b_array[idx])
                    if not(utils.almost_equal(b_value, avgs[idx])):
                        self.react(self.main_msgs['ineff_bias_regr'].format(idx))       
            elif not np.all(checks):
                self.react(self.main_msgs['zero_bias'])
   
    def _pre_check_loss(self, session):
        if self.config.init_loss.disabled: return
        batch_x, batch_y = self.inputs_data.get_sample(self.config.init_loss.sample_size)
        losses = []
        n = self.config.init_loss.size_growth_rate 
        while n <= (self.config.init_loss.size_growth_rate * self.config.init_loss.size_growth_iters):
            derived_batch_x  = np.concatenate(n*[batch_x], axis=0)
            derived_batch_y = np.concatenate(n*[batch_y], axis=0)
            feed_dict = utils.add_extra_feeds({self.nn_data.model.features: derived_batch_x, 
                                           self.nn_data.model.targets: derived_batch_y},
                                           self.nn_data.model.train_extra_feed_dict)
            losses.append(session.run(self.nn_data.model.loss, feed_dict=feed_dict))
            n *= self.config.init_loss.size_growth_rate
        rounded_loss_rates = [round(losses[i+1]/losses[i]) for i in range(len(losses)-1)]
        equality_checks = sum([(loss_rate==self.config.init_loss.size_growth_rate) for loss_rate in rounded_loss_rates])
        if equality_checks == len(rounded_loss_rates):  
            self.react(self.main_msgs['poor_reduction_loss'])
        if self.nn_data.model.problem_type == CLASSIFICATION_KEY:
            feed_dict = utils.add_extra_feeds({self.nn_data.model.features: batch_x, 
                                            self.nn_data.model.targets: batch_y},
                                            self.nn_data.model.train_extra_feed_dict)
            initial_loss = session.run(self.nn_data.model.loss, feed_dict=feed_dict)
            expected_loss = -np.log(1/self.inputs_data.targets_metadata['labels'])
            err = np.abs(initial_loss-expected_loss)
            if err >= self.config.init_loss.dev_ratio * expected_loss:  
                self.react(self.main_msgs['poor_init_loss'].format(readable(err/expected_loss)))

    def _pre_check_gradients(self):
        if self.config.grad.disabled: return
        with tf.Session() as session:
            session.run(tf.initialize_all_variables())
            for _ in range(self.config.grad.warm_up_steps):
                batch_x, batch_y = self.inputs_data.get_sample(self.config.grad.warm_up_batch)
                feed_dict = {self.nn_data.model.features: batch_x, self.nn_data.model.targets: batch_y}
                session.run(self.nn_data.model.train_op, feed_dict=feed_dict)
            all_weights = list(self.nn_data.model.weights.values())
            weights_shapes = [[int(s) for s in list(weight.shape)] for weight in all_weights]
            init_weights = [session.run(weight) for weight in all_weights]
            few_x, few_y = self.inputs_data.get_sample(self.config.grad.sample_size)
            feed_dict = {self.nn_data.model.features: few_x, self.nn_data.model.targets: few_y}
            for i in range(len(all_weights)):                
                theoretical, numerical = tf.test.compute_gradient(
                    all_weights[i], 
                    weights_shapes[i], 
                    self.nn_data.model.loss,
                    [1],
                    delta=self.config.grad.delta,
                    x_init_value=init_weights[i],
                    extra_feed_dict=feed_dict
                )
                theoretical, numerical = theoretical.flatten(), numerical.flatten()
                total_dims, sample_dims = len(theoretical), int(self.config.grad.ratio_of_dimensions*len(theoretical))
                indices = np.random.choice(np.arange(total_dims), sample_dims, replace=False)
                theoretical_sample = theoretical[indices]
                numerical_sample = numerical[indices]
                numerator = np.linalg.norm(theoretical_sample - numerical_sample)
                denominator = np.linalg.norm(theoretical_sample) + np.linalg.norm(numerical_sample)
                relerr = numerator / denominator
                if relerr > self.config.grad.relative_err_max_thresh:
                    self.react(self.main_msgs['grad_err'].format(all_weights[i], readable(relerr), self.config.grad.relative_err_max_thresh))

    def _pre_check_fitting_data_capability(self, session, batch_size):
        if self.config.prop_fit.disabled: return
        def _loss_is_stable(loss_value):
            if np.isnan(loss_value):
                self.react(self.main_msgs['nan_loss'])
                return False
            if np.isinf(loss_value):
                self.react(self.main_msgs['inf_loss'])
                return False
            return True
        batch_x, batch_y = self.inputs_data.get_sample(self.config.prop_fit.single_batch_size)
        feed_dict = utils.add_extra_feeds({self.nn_data.model.features: batch_x, 
                                           self.nn_data.model.targets: batch_y}, 
                                           self.nn_data.model.train_extra_feed_dict)
        session.run(tf.initialize_all_variables())
        real_losses = []
        real_accs = []
        variables = [self.nn_data.model.train_op, self.nn_data.model.loss, self.nn_data.model.perf]
        variables = variables + [self.nn_data.model.reg_loss] if self.nn_data.model.reg_loss!=None else variables
        for _ in range(self.config.prop_fit.total_iters): 
            results = session.run(variables, feed_dict=feed_dict)
            real_loss, real_acc = results[1], results[2]
            real_accs.append(real_acc)
            real_losses.append(results[1])
            if not(_loss_is_stable(results[1])): 
                self.react(self.main_msgs['underfitting_single_batch'])
                return
        loss, acc = (results[1] + results[3], results[2]) if len(results) == 4 else (results[1], results[2])
        underfitting_prob = False
        if self.inputs_data.problem_type == CLASSIFICATION_KEY:
            if  1.0 - max(real_accs) > self.config.prop_fit.mislabeled_rate_max_thresh: 
                self.react(self.main_msgs['underfitting_single_batch'])
                underfitting_prob = True
        elif self.inputs_data.problem_type == REGRESSION_KEY:
            if min(real_accs) > self.config.prop_fit.mean_error_max_thresh:
                self.react(self.main_msgs['underfitting_single_batch'])
                underfitting_prob = True
        loss_smoothness = metrics.smoothness(np.array(real_losses))
        min_loss = np.min(np.array(real_losses))
        if min_loss <= self.config.prop_fit.abs_loss_min_thresh or (min_loss <= self.config.prop_fit.loss_min_thresh and loss_smoothness > self.config.prop_fit.smoothness_max_thresh):
            self.react(self.main_msgs['zero_loss']); return
        if not(underfitting_prob): return
        zeroed_batch_x = np.zeros_like(batch_x)
        feed_dict = {self.nn_data.model.features: zeroed_batch_x, self.nn_data.model.targets: batch_y}
        session.run(tf.initialize_all_variables())
        fake_losses = []
        for _ in range(self.config.prop_fit.total_iters): 
            _, fake_loss = session.run([self.nn_data.model.train_op,self.nn_data.model.loss], feed_dict=utils.add_extra_feeds(feed_dict, self.nn_data.model.train_extra_feed_dict))
            fake_losses.append(fake_loss)
            if not(_loss_is_stable(fake_loss)): return
        stability_test = np.array([_loss_is_stable(loss_value) for loss_value in (real_losses+fake_losses)])
        if (stability_test == False).any():
            last_real_losses = real_losses[-self.config.prop_fit.sample_size_of_losses:]
            last_fake_losses = fake_losses[-self.config.prop_fit.sample_size_of_losses:]
            if not(metrics.are_significantly_different(last_real_losses, last_fake_losses)):
                self.react(self.main_msgs['data_dep'])

    def _pre_check_operation_dependancy(self, op_name, operation):
        if self.config.ins_wise_op.disabled: return
        op_grad_wrt_inp_tensor = tf.gradients(operation[0], self.nn_data.model.features)
        batch_x, batch_y = self.inputs_data.get_sample(self.config.ins_wise_op.sample_size)
        feed_dict = utils.add_extra_feeds({self.nn_data.model.features: batch_x, 
                                           self.nn_data.model.targets: batch_y},
                                           self.nn_data.model.test_extra_feed_dict)
        with tf.Session() as session:
            session.run(tf.initialize_all_variables())
            grads = []
            for _ in range(self.config.ins_wise_op.trials): 
                op_grad_wrt_inp, _ = session.run([op_grad_wrt_inp_tensor, self.nn_data.model.train_op], feed_dict=feed_dict)
                grads.append(np.sum(op_grad_wrt_inp[0][1:]))
            if sum(grads) != 0.0:
                self.react(self.main_msgs['op_dep'].format(op_name))

    def run(self, batch_size, implemented_ops):
        if self.config.disabled: return
        self._pre_check_features()
        self._pre_check_targets()
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            self._pre_check_weights(session=sess)
            self._pre_check_biases(session=sess)
            self._pre_check_loss(session=sess)
            self._pre_check_fitting_data_capability(session=sess, batch_size=batch_size)
        for op_name, op in implemented_ops:
            self._pre_check_operation_dependancy(op_name, op)
        self._pre_check_gradients()
                  
class PostCheck:
    
    def __init__(self, nn_data, data, logger, config):
        self.nn_data = nn_data
        self.main_logger = logger
        self.config = config
        self.train_data = data.train_loader
        self.test_data = data.test_loader
        self.hooks = self.build_hooks(nn_data, logger) 
        self.main_msgs = settings.load_messages()
    
    def react(self, message):
        if self.config.fail_on:
            self.main_logger.error(message)
            raise Exception(message)
        else:
            self.main_logger.warning(message)
    
    def build_hooks(self, nn_data, logger):
        post_hooks = [
            hooks.PostActivationHook(nn_data=nn_data, main_logger=logger, config=self.config.switch_mode_consist, fail_on=self.config.fail_on),
            hooks.PostLossHook(nn_data=nn_data, main_logger=logger, config=self.config.switch_mode_consist, fail_on=self.config.fail_on)
            ]
        return post_hooks

    def _post_check_labels(self):
        if self.config.corrup_lbls.disabled: return
        model = self.nn_data.model
        self.train_data._pos = 0
        data_size = self.train_data.rows_count
        epoch_iters = self.train_data.get_epochs(self.config.corrup_lbls.batch_size)
        valid_x, valid_y = self.test_data.next_batch(self.config.corrup_lbls.batch_size)
        losses = []
        perfs = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            patience = self.config.corrup_lbls.patience
            for epoch in range(self.config.corrup_lbls.total_epochs):
                for i in range(epoch_iters):
                    batch_x, batch_y = self.train_data.next_batch(self.config.corrup_lbls.batch_size)
                    feed_dict = {model.features: batch_x, model.targets: batch_y}
                    _, loss_value, perf_value = sess.run([model.train_op, model.loss, model.perf], feed_dict=utils.add_extra_feeds(feed_dict, model.train_extra_feed_dict))
                    if i == 0:
                        if loss_value == 0.0: return
                        losses.append(loss_value)
                        perfs.append(perf_value)
                        if len(losses) < 3: continue
                        perf_improv_ratio = round((losses[-2]-losses[-1])/losses[-1],2)
                        if perf_improv_ratio >= self.config.corrup_lbls.perf_improv_ratio_min_thresh:
                            patience -= int(perf_improv_ratio/self.config.corrup_lbls.perf_improv_ratio_min_thresh)
                            if patience <= 0:return
                        else:
                            patience = self.config.corrup_lbls.patience
        if utils.almost_equal(np.var(losses[self.config.corrup_lbls.warmup_epochs:]), 0.):
            self.react(self.main_msgs['corrupted_data'])
    
    def _post_check_augmentation(self):
        if self.config.data_augm.disabled: return
        model = self.nn_data.model
        data_size = self.train_data.rows_count
        epoch_iters = self.train_data.get_epochs(self.config.data_augm.batch_size)
        self.train_data.deactivate_augmentation()
        valid_x, valid_y = self.train_data.next_batch(self.config.data_augm.valid_sample_size)
        self.train_data.reset_cursor()
        self.train_data.activate_augmentation()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epoch_iters*self.config.data_augm.total_epochs):
                batch_x, batch_y = self.train_data.next_batch(self.config.data_augm.batch_size)
                feed_dict = {model.features: batch_x, model.targets: batch_y}
                sess.run(model.train_op, feed_dict=utils.add_extra_feeds(feed_dict, model.train_extra_feed_dict))
            feed_dict = {model.features: valid_x, model.targets: valid_y}
            loss_with_aug, acts_with_aug = sess.run([model.loss, model.test_activations], feed_dict=utils.add_extra_feeds(feed_dict, model.test_extra_feed_dict))
        self.train_data.reset_cursor()
        self.train_data.deactivate_augmentation()
        with tf.Session() as sess: 
            sess.run(tf.global_variables_initializer())
            for i in range(epoch_iters*self.config.data_augm.total_epochs):
                batch_x, batch_y = self.train_data.next_batch(self.config.data_augm.batch_size)
                feed_dict = {model.features: batch_x, model.targets: batch_y}
                sess.run(model.train_op, feed_dict=utils.add_extra_feeds(feed_dict, model.train_extra_feed_dict))
            feed_dict = {model.features: valid_x, model.targets: valid_y}
            loss_without_aug, acts_without_aug = sess.run([model.loss, model.test_activations], feed_dict=utils.add_extra_feeds(feed_dict, model.test_extra_feed_dict))    
        sims = []
        for act_name, act_arr in acts_with_aug.items():
            cka_sim = metrics.feature_space_linear_cka(utils.transform_2d(act_arr, keep='first'),\
                                                     utils.transform_2d(acts_without_aug[act_name], keep='first'),\
                                                     debiased=False)
            sims.append(cka_sim)
        if (np.array(sims) < self.config.data_augm.sim_with_augm_min_thresh).any() and loss_without_aug < loss_with_aug:
            self.react(self.main_msgs['wrong_augm'])
            
    def _post_check_regularization(self):
        if self.config.switch_mode_consist.disabled: return
        self.test_data.reset_cursor()
        self.train_data.reset_cursor()
        model = self.nn_data.model
        data_size = self.train_data.rows_count
        epoch_iters = self.train_data.get_epochs(self.config.switch_mode_consist.batch_size)
        with tf.train.MonitoredTrainingSession(hooks=self.hooks) as mon_sess:
            for i in range(epoch_iters*self.config.switch_mode_consist.total_epochs):
                if i > self.config.start and i % self.config.switch_mode_consist.period == 0:
                    valid_x, valid_y = self.test_data.next_batch(self.config.switch_mode_consist.valid_sample_size)
                    feed_dict = {model.features: valid_x, model.targets: valid_y}
                    loss_on_train_mode, loss_on_test_mode = mon_sess.run([model.loss, model.test_loss], feed_dict=utils.add_extra_feeds(feed_dict, model.test_extra_feed_dict))
                    if len(model.test_extra_feed_dict) > 0 and model.test_extra_feed_dict.values()!=model.train_extra_feed_dict.values():
                        loss_on_train_mode = mon_sess.run(model.loss, feed_dict=utils.add_extra_feeds(feed_dict, model.train_extra_feed_dict))
                else:
                    batch_x, batch_y = self.train_data.next_batch(self.config.switch_mode_consist.batch_size)
                    feed_dict = {model.features: batch_x, model.targets: batch_y}
                    mon_sess.run(model.train_op, feed_dict=utils.add_extra_feeds(feed_dict, model.train_extra_feed_dict))
                    
    def run(self, post_fitness_batch, post_fitness_epochs):
        if self.config.disabled: return
        self._post_check_labels()
        if self.nn_data.model.test_mode:
            self._post_check_regularization()
        if hasattr(self.train_data, '_augmentation'):
            self._post_check_augmentation()

class OverfitCheck:

    def __init__(self, nn_data, inputs_data, logger, config):
        self.nn_data = nn_data
        self.model = nn_data.model 
        self.logger = logger
        self.config = config
        self.inputs_data = inputs_data

    def build_hooks(self):
        overfit_hooks = [
            hooks.OverfitWeightHook(nn_data=self.nn_data, main_logger=self.logger, config=self.config.weight, fail_on=self.config.fail_on),
            hooks.OverfitBiasHook(nn_data=self.nn_data, main_logger=self.logger, config=self.config.bias, fail_on=self.config.fail_on),
            hooks.OverfitActivationHook(targets_metadata=self.inputs_data.targets_metadata, nn_data=self.nn_data, main_logger=self.logger, config=self.config.act, fail_on=self.config.fail_on),
            hooks.OverfitGradientHook(nn_data=self.nn_data, main_logger=self.logger, config=self.config.grad, fail_on=self.config.fail_on),
            hooks.OverfitLossHook(nn_data=self.nn_data, main_logger=self.logger, config=self.config.loss, fail_on=self.config.fail_on)
            ]
        return overfit_hooks
    
    def build_feed_dict(self, overfit_batch):
        batch_x, batch_y = self.inputs_data.get_sample(overfit_batch)
        feed_dict = utils.add_extra_feeds({self.model.features: batch_x, self.model.targets: batch_y}, 
                                           self.model.train_extra_feed_dict)
        return feed_dict

    def run(self, overfit_batch, overfit_iters):
        if self.config.disabled: return
        self.nn_data.init_or_reset(overfit_batch)
        self.hooks = self.build_hooks()  
        self.feed_dict = self.build_feed_dict(overfit_batch)
        patience = self.config.patience
        with tf.train.MonitoredTrainingSession(hooks=self.hooks) as mon_sess:
            for i in range(overfit_iters):
                _, perf_metric = mon_sess.run([self.model.train_op,self.model.perf], feed_dict=self.feed_dict)
                if self.model.problem_type == CLASSIFICATION_KEY and perf_metric  == self.config.classif_perf_thresh:
                    patience -= 1
                elif self.model.problem_type == REGRESSION_KEY and perf_metric < self.config.regr_perf_thresh:
                    patience -= 1
                else:
                    patience = self.config.patience
                if patience == 0: break