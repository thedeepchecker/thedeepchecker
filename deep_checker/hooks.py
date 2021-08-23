import numpy as np
import tensorflow as tf
from tensorflow.python.training.session_run_hook import SessionRunHook
from tensorflow.python.training.session_run_hook import SessionRunArgs
import deep_checker.metrics as metrics
import deep_checker.settings as settings
from deep_checker.settings import CLASSIFICATION_KEY, REGRESSION_KEY
from deep_checker.utils import transform_2d, is_non_2d
from deep_checker.utils import readable

class TFCheckHook(SessionRunHook):

  def __init__(self, nn_data, main_logger, config, fail_on=False):
    self.nn_data = nn_data
    self._fail_on = fail_on
    self.main_logger = main_logger
    self.config = config
    self.main_msgs = settings.load_messages()
  
  def react(self, message):
    if self._fail_on:
      self.main_logger.error(message)
      raise Exception(message)
    else:
      self.main_logger.warning(message)

class OverfitActivationHook(TFCheckHook):

  def __init__(self, targets_metadata, *args, **kwargs):
    self.targets_metadata = targets_metadata
    super(OverfitActivationHook, self).__init__(*args, **kwargs)

  def begin(self):
    self.acts_tensors = self.nn_data.model.activations
    self.outs_tensor = self.nn_data.model.outs_post_act
    self.outputs_metadata = {
                             'non_zero_variance':{'patience':self.config.out.patience,
                                                  'status':np.array([False for _ in range(self.targets_metadata['count'])])},
                             'max_abs_greater_than_one':{'patience':self.config.out.patience,
                                                         'status':np.array([False for _ in range(self.targets_metadata['count'])])}, 
                             'can_be_negative':{'patience':self.config.out.patience,
                                                'status':np.array([False for _ in range(self.targets_metadata['count'])])}
                            }
    self.iter_count = -1
    
  def before_run(self, run_context): 
    return SessionRunArgs([self.acts_tensors, self.outs_tensor])

  def check_activations_range(self, acts_name, acts_array):
    if self.config.range.disabled: return
    if (acts_array < self.nn_data.model.acts_min_bound).any():
      main_msg = self.main_msgs['act_ltn'].format(acts_name,self.nn_data.model.acts_min_bound)
      self.react(main_msg)
    if (acts_array > self.nn_data.model.acts_max_bound).any():
      main_msg = self.main_msgs['act_gtn'].format(acts_name,self.nn_data.model.acts_max_bound)
      self.react(main_msg)

  def check_numerical_instabilities(self, acts_name, acts_array):
    if self.config.numeric_ins.disabled: return
    if np.isinf(acts_array).any():
      self.react(self.main_msgs['act_inf'].format(acts_name))
      return True
    if np.isnan(acts_array).any():
      self.react(self.main_msgs['act_nan'].format(acts_name))
      return True
    return False

  def check_outputs(self, outs_array):
    if self.config.out.disabled: return
    if np.isinf(outs_array).any():
      self.react(self.main_msgs['out_inf']);return
    elif np.isnan(outs_array).any():
      self.react(self.main_msgs['out_nan']);return
    if (self.outputs_metadata['non_zero_variance']['status'] == False).any():
      self.outputs_metadata['non_zero_variance']['patience'] -= 1
      if self.outputs_metadata['non_zero_variance']['patience'] <= 0:
        self.react(self.main_msgs['out_cons'])
    else:
      self.outputs_metadata['non_zero_variance']['patience'] = self.config.out.patience
    if self.nn_data.model.problem_type == CLASSIFICATION_KEY:
      if outs_array.shape[1] == 1:
        positive = (outs_array >= 0.).all() and (outs_array <= 1.).all()
        if not(positive):
          self.react(self.main_msgs['output_invalid'])
      else:
        #cannot check sum to 1.0 because of https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
        sum_to_one = (np.sum(outs_array, axis=1) > 0.95).all() and (np.sum(outs_array, axis=1) < 1.05).all()
        positive = (outs_array >= 0.).all()
        valid_n_outs = outs_array.shape[1] == self.targets_metadata['count']
        if not(positive and sum_to_one and valid_n_outs):
          self.react(self.main_msgs['output_invalid'])
    elif self.nn_data.model.problem_type == REGRESSION_KEY:
      if len(outs_array.shape) > 1:
        valid_n_outs = outs_array.shape[1] == self.targets_metadata['count']
        if not(valid_n_outs):
          self.react(self.main_msgs['output_invalid'])
      if (self.outputs_metadata['max_abs_greater_than_one']['status'] < self.targets_metadata['max_abs_greater_than_one']).any():
        self.outputs_metadata['max_abs_greater_than_one']['patience'] -= 1
        if self.outputs_metadata['max_abs_greater_than_one']['patience'] <= 0:
          self.react(self.main_msgs['lack_of_magnitude_express']) 
      else:
        self.outputs_metadata['max_abs_greater_than_one']['patience'] = self.config.out.patience
      if (self.outputs_metadata['can_be_negative']['status'] < self.targets_metadata['can_be_negative']).any():
        self.outputs_metadata['can_be_negative']['patience'] -= 1
        if self.outputs_metadata['can_be_negative']['patience'] <= 0:
          self.react(self.main_msgs['lack_of_negative_express'])
      else:
        self.outputs_metadata['can_be_negative']['patience'] = self.config.out.patience

  def check_dead_layers(self, acts_name, acts_array, is_conv):
    if self.config.dead.disabled: return
    acts_array = transform_2d(acts_array, keep='last')
    major_values = np.percentile(np.abs(acts_array), q=self.config.dead.act_maj_percentile, axis=0)
    dead_count = np.count_nonzero(major_values < self.config.dead.act_min_thresh)
    dead_ratio = dead_count / major_values.shape[0]
    if dead_ratio > self.config.dead.neurons_ratio_max_thresh:
      main_msg = self.main_msgs['conv_act_dead'] if is_conv else self.main_msgs['fc_act_dead']
      self.react(main_msg.format(dead_count, major_values.size, acts_name))

  def check_acts_distribution(self, acts_name, acts_array, is_conv):
    if self.config.dist.disabled: return
    acts_array = transform_2d(acts_array, keep='last')
    act_std = np.std(acts_array)
    if act_std < self.config.dist.std_acts_min_thresh or act_std > self.config.dist.std_acts_max_thresh:
      if act_std < self.config.dist.std_acts_min_thresh:
        f_test_result = metrics.pure_f_test(acts_array, self.config.dist.std_acts_min_thresh, self.config.dist.f_test_alpha) 
      else:
        f_test_result = metrics.pure_f_test(acts_array, self.config.dist.std_acts_max_thresh, self.config.dist.f_test_alpha) 
      if not(f_test_result[1]):
        main_msg = self.main_msgs['conv_act_unstable'] if is_conv else self.main_msgs['fc_act_unstable']
        self.react(main_msg.format(acts_name, act_std, self.config.dist.std_acts_min_thresh, self.config.dist.std_acts_max_thresh))

  def check_saturated_layers(self, acts_name, acts_array, is_conv):
    if self.config.sat.disabled: return
    acts_array = transform_2d(acts_array, keep='last')
    ro_Bs = np.apply_along_axis(metrics.compute_ro_B, 0, acts_array, min_out=self.config.sat.ro_histo_min, max_out=self.config.sat.ro_histo_max, bins_count=self.config.sat.ro_histo_bins_count)
    saturated_count = np.count_nonzero(ro_Bs > self.config.sat.ro_max_thresh) 
    saturated_ratio = saturated_count / ro_Bs.shape[0]
    if saturated_ratio > self.config.sat.neurons_ratio_max_thresh:
      main_msg = self.main_msgs['conv_act_sat'] if is_conv else self.main_msgs['fc_act_sat']
      self.react(main_msg.format(saturated_count,ro_Bs.size,acts_name))

  def update_buffer(self, acts_name, acts_array):
    n = acts_array.shape[0]
    self.nn_data.acts_data[acts_name][0:-n] = self.nn_data.acts_data[acts_name][-(self.nn_data.buff_scale - 1)*n:]
    self.nn_data.acts_data[acts_name][-n:] = acts_array
    return self.nn_data.acts_data[acts_name]

  def update_outs_conds(self, outs_array):
    self.outputs_metadata['non_zero_variance']['status'] |= (np.var(outs_array, axis=0) > 0)
    self.outputs_metadata['max_abs_greater_than_one']['status'] |= (np.abs(outs_array) > 1).any(axis=0)
    self.outputs_metadata['can_be_negative']['status'] |= (outs_array < 0).any(axis=0)

  def after_run(self, run_context, run_values):
    self.iter_count += 1
    acts, outputs = run_values.results
    self.update_outs_conds(outputs)
    if self.iter_count % self.config.period == 0:
      self.check_outputs(outputs)
    acts = {k:(v,is_non_2d(v)) for k,v in acts.items()} 
    for acts_name, (acts_array, is_conv) in acts.items():
      acts_buffer = self.update_buffer(acts_name, acts_array)
      if self.iter_count < self.config.start or self.iter_count % self.config.period != 0: continue
      self.check_activations_range(acts_name, acts_buffer)
      if self.check_numerical_instabilities(acts_name, acts_array): continue
      if self.nn_data.model.act_fn_name in ['sigmoid', 'tanh']:
        self.check_saturated_layers(acts_name, acts_buffer, is_conv)
      else:
        self.check_dead_layers(acts_name, acts_buffer, is_conv)
      self.check_acts_distribution(acts_name, acts_buffer, is_conv)
      
class OverfitWeightHook(TFCheckHook):

  def __init__(self, *args, **kwargs):
    super(OverfitWeightHook, self).__init__(*args, **kwargs)

  def begin(self):
    self.weights_tensors = self.nn_data.model.weights
    self.iter_count = -1

  def before_run(self, run_context):
    return SessionRunArgs(self.weights_tensors)
  
  def check_numerical_instabilities(self, weight_name, weight_array):
    if self.config.numeric_ins.disabled: return
    if np.isinf(weight_array).any():
      self.react(self.main_msgs['w_inf'].format(weight_name))
      return True
    if np.isnan(weight_array).any():
      self.react(self.main_msgs['w_nan'].format(weight_name))
      return True
    return False
 
  def check_sign(self, weight_name, weight_array, is_conv):
    if self.config.neg.disabled: return
    neg_ratio = np.count_nonzero(weight_array < 0.) / weight_array.size
    if neg_ratio > self.config.neg.ratio_max_thresh:
      main_msg = self.main_msgs['conv_w_sign'] if is_conv else self.main_msgs['fc_w_sign']
      self.react(main_msg.format(weight_name, readable(neg_ratio), self.config.neg.ratio_max_thresh))

  def check_dead(self, weight_name, weight_array, is_conv):
    if self.config.dead.disabled: return
    dead_ratio = np.count_nonzero(np.abs(weight_array) < self.config.dead.value_min_thresh) / weight_array.size
    if dead_ratio > self.config.dead.ratio_max_thresh: 
      main_msg = self.main_msgs['conv_w_dead'] if is_conv else self.main_msgs['fc_w_dead']
      self.react(main_msg.format(weight_name, readable(dead_ratio), self.config.dead.value_min_thresh))

  def check_divergence(self, weight_name, weight_reductions, is_conv):
    if self.config.div.disabled: return
    if weight_reductions[-1] > self.config.div.mav_max_thresh:
      main_msg = self.main_msgs['conv_w_div_1'] if is_conv else self.main_msgs['fc_w_div_1']
      self.react(main_msg.format(weight_name, readable(weight_reductions[-1]), self.config.div.mav_max_thresh))
    elif len(weight_reductions) >= self.config.div.window_size:
      inc_rates = np.array([weight_reductions[-i]/weight_reductions[-i-1] for i in range(1, self.config.div.window_size)])
      if (inc_rates >= self.config.div.inc_rate_max_thresh).all():
        main_msg = self.main_msgs['conv_w_div_2'] if is_conv else self.main_msgs['fc_w_div_2']
        self.react(main_msg.format(weight_name, readable(max(inc_rates)), self.config.div.inc_rate_max_thresh))

  def update_buffer(self, weight_name, weight_array):
    self.nn_data.weights_reductions[weight_name].append(np.mean(np.abs(weight_array)))
    return self.nn_data.weights_reductions[weight_name]

  def after_run(self, run_context, run_values):
    self.iter_count += 1  
    weights = run_values.results
    self.weights = {k:(v,is_non_2d(v)) for k,v in weights.items()}  
    for w_name, (w_array, is_conv) in self.weights.items():
      if self.check_numerical_instabilities(w_name,w_array): continue
      buffer_values = self.update_buffer(w_name, w_array)
      if self.iter_count < self.config.start or self.iter_count % self.config.period != 0: continue
      self.check_sign(w_name, w_array, is_conv)
      self.check_dead(w_name, w_array, is_conv)
      self.check_divergence(w_name, buffer_values, is_conv)
      
class OverfitBiasHook(TFCheckHook):

  def __init__(self, *args, **kwargs):
    super(OverfitBiasHook, self).__init__(*args, **kwargs)

  def begin(self):
    self.bias_tensors = self.nn_data.model.biases
    self.iter_count = -1
    
  def before_run(self, run_context): 
    return SessionRunArgs(self.bias_tensors)
  
  def check_numerical_instabilities(self, bias_name, bias_array):
    if self.config.numeric_ins.disabled: return
    if np.isinf(bias_array).any():
      self.react(self.main_msgs['b_inf'].format(bias_name))
      return True
    if np.isnan(bias_array).any():
      self.react(self.main_msgs['b_nan'].format(bias_name))
      return True
    return False

  def check_divergence(self, bias_name, bias_reductions):
    if self.config.div.disabled: return
    if bias_reductions[-1] > self.config.div.mav_max_thresh:
      self.react(self.main_msgs['b_div_1'].format(bias_name,readable(bias_reductions[-1]),self.config.div.mav_max_thresh))
    elif len(bias_reductions) >= self.config.div.window_size:
      inc_rates = np.array([bias_reductions[-i]/bias_reductions[-i-1] for i in range(1, self.config.div.window_size)])
      if (inc_rates >= self.config.div.inc_rate_max_thresh).all():
        self.react(self.main_msgs['b_div_2'].format(bias_name,readable(max(inc_rates)), self.config.div.inc_rate_max_thresh))

  def update_buffer(self, bias_name, bias_array):
    self.nn_data.biases_reductions[bias_name].append(np.mean(np.abs(bias_array)))
    return self.nn_data.biases_reductions[bias_name]

  def after_run(self, run_context, run_values):
    self.iter_count += 1
    self.biases = run_values.results
    if not(self.biases):return # no bias 
    for b_name, b_array in self.biases.items():
      if self.check_numerical_instabilities(b_name, b_array): continue
      buffer_values = self.update_buffer(b_name, b_array)
      if self.iter_count < self.config.start or self.iter_count % self.config.period != 0: continue
      self.check_divergence(b_name, buffer_values)

class OverfitGradientHook(TFCheckHook):
  
  def __init__(self, *args, **kwargs):
    super(OverfitGradientHook, self).__init__(*args, **kwargs)

  def begin(self):
    self.weights_tensors = self.nn_data.model.weights
    self.gradients_tensors = self.nn_data.weights_gradients
    self.magnitude_update_ratios = {w_name:[] for w_name in self.weights_tensors.keys()}
    self.iter_count = -1

  def after_create_session(self, session, coord):
    self.before_weights = session.run(self.weights_tensors)

  def before_run(self, run_context):
    return SessionRunArgs([self.weights_tensors, self.gradients_tensors])

  def check_numerical_instabilities(self, weight_name, gradient_weight_array):
    if self.config.numeric_ins.disabled: return
    if np.isinf(gradient_weight_array).any():
      self.react(self.main_msgs['gw_inf'].format(weight_name))
      return True
    if np.isnan(gradient_weight_array).any():
      self.react(self.main_msgs['gw_nan'].format(weight_name))
      return True
    return False

  def check_vanishing_gradient(self, weight_name, gradient_reductions, is_conv):
    if self.config.vanish.disabled: return
    if gradient_reductions[-1] < self.config.vanish.mav_min_thresh:
      main_msg = self.main_msgs['conv_gw_van_1'] if is_conv else self.main_msgs['fc_gw_van_1']
      self.react(main_msg.format(weight_name, readable(gradient_reductions[-1]), self.config.vanish.mav_min_thresh))
    elif len(gradient_reductions) >= self.config.vanish.window_size:
      dec_rates = np.array([gradient_reductions[-i]/gradient_reductions[-i-1] for i in range(1, self.config.vanish.window_size)])
      if (dec_rates <= self.config.vanish.dec_rate_min_thresh).all():
        main_msg = self.main_msgs['conv_gw_van_2'] if is_conv else self.main_msgs['fc_gw_van_2']
        self.react(main_msg.format(weight_name, readable(min(dec_rates[0])), self.config.vanish.dec_rate_min_thresh))
      
  def check_exploding_gradient(self, weight_name, gradient_reductions, is_conv):
    if self.config.explod.disabled: return
    if gradient_reductions[-1] > self.config.explod.mav_max_thresh:
      main_msg = self.main_msgs['conv_gw_exp_1'] if is_conv else self.main_msgs['fc_gw_exp_1']
      self.react(main_msg.format(weight_name, gradient_reductions[-1], self.config.explod.mav_max_thresh))
    elif len(gradient_reductions) >= self.config.explod.window_size:
      inc_rates = np.array([gradient_reductions[-i]/gradient_reductions[-i-1] for i in range(1, self.config.explod.window_size)])
      if (inc_rates >= self.config.explod.inc_rate_max_thresh).all():
        main_msg = self.main_msgs['conv_gw_exp_2'] if is_conv else self.main_msgs['fc_gw_exp_2']
        self.react(main_msg.format(weight_name, readable(max(inc_rates)), self.config.explod.inc_rate_max_thresh))
      
  def check_unstable_training(self, weight_name, bw_array, aw_array, is_conv):
    if self.config.unstab.disabled: return
    abs_diff_array = np.abs(self.after_weights[weight_name] - self.before_weights[weight_name])  
    divider = np.mean(np.abs(bw_array))
    if divider == 0: return
    update_ratio = np.divide(np.mean(abs_diff_array), divider)
    log_update_ratio = np.log10(update_ratio) if update_ratio > 0 else 0
    self.magnitude_update_ratios[weight_name].append(log_update_ratio) 
    if self.iter_count > 0 and self.iter_count % self.config.period == 0:
      magnitude_update_ratio = np.mean(self.magnitude_update_ratios[weight_name])
      self.magnitude_update_ratios[weight_name] = []
      if magnitude_update_ratio == 0:
        self.react(self.main_msgs['w_untrained'].format(weight_name))
      elif magnitude_update_ratio > self.config.unstab.high_updates_max_thresh:
        main_msg = self.main_msgs['conv_w_fast'] if is_conv else self.main_msgs['fc_w_fast']
        self.react(main_msg.format(weight_name, readable(magnitude_update_ratio), self.config.unstab.high_updates_max_thresh))
      elif magnitude_update_ratio < self.config.unstab.low_updates_min_thresh:
        main_msg = self.main_msgs['conv_w_slow'] if is_conv else self.main_msgs['fc_w_slow']
        self.react(main_msg.format(weight_name, readable(magnitude_update_ratio), self.config.unstab.low_updates_min_thresh))

  def update_buffer(self, weight_name, gradient_weight_array):
    self.nn_data.gradients_reductions[weight_name].append(np.mean(np.abs(gradient_weight_array)))
    return self.nn_data.gradients_reductions[weight_name]

  def after_run(self, run_context, run_values):
    self.iter_count += 1
    self.after_weights, self.weights_gradients = run_values.results
    self.weights_gradients = {k:(v,is_non_2d(v)) for k,v in self.weights_gradients.items()} 
    for w_name, (gw_array, is_conv) in self.weights_gradients.items():
      if self.check_numerical_instabilities(w_name, gw_array): continue
      buffer_values = self.update_buffer(w_name, gw_array)
      self.check_unstable_training(w_name, self.before_weights[w_name], self.after_weights[w_name], is_conv)
      if self.iter_count < self.config.start or self.iter_count % self.config.period != 0: continue
      self.check_vanishing_gradient(w_name, buffer_values, is_conv)
      self.check_exploding_gradient(w_name, buffer_values, is_conv)
    self.before_weights = self.after_weights
 
class OverfitLossHook(TFCheckHook):

  def __init__(self, *args, **kwargs):
    super(OverfitLossHook, self).__init__(*args, **kwargs)

  def begin(self):
    self.loss = self.nn_data.model.loss 
    self.step_losses = []
    self.perf = self.nn_data.model.perf
    self.step_perfs = []
    if self.nn_data.model.reg_loss is not None:
      weights = list(self.nn_data.model.weights.values())
      self.grad_reg_losses = [tf.gradients(self.nn_data.model.reg_loss, w)[0] for w in weights]
      self.grad_losses = [tf.gradients(self.nn_data.model.loss, w)[0] for w in weights]
      self.grad_losses_norm_ratios = {w_name:[] for w_name in self.nn_data.model.weights.keys()}
    self.min_loss = np.inf
    self.iter_count = -1
    
  def before_run(self, run_context): 
    if self.nn_data.model.reg_loss is None:
      return SessionRunArgs([self.loss, self.perf])
    else:
      return SessionRunArgs([self.loss, self.perf, self.grad_losses, self.grad_reg_losses])
  
  def check_numerical_instabilities(self, loss_value):
    if self.config.numeric_ins.disabled: return
    if np.isnan(loss_value):
      self.react(self.main_msgs['nan_loss'])
      return True
    if np.isinf(loss_value):
      self.react(self.main_msgs['inf_loss'])
      return True
    return False
                  
  def check_loss_curve(self, losses):
    n_losses = len(losses)
    if n_losses >= self.config.non_dec.window_size:
      dec_pers = np.array([(losses[-i-1] - losses[-i])/losses[-i-1] for i in range(1, self.config.non_dec.window_size)])
      if (dec_pers < self.config.non_dec.decr_percentage).all() and not(self.config.non_dec.disabled):
        self.react(self.main_msgs['stagnated_loss'])
    if n_losses >= self.config.div.window_size:
      abs_loss_incrs = [losses[n_losses-i]/self.min_loss for i in range(self.config.div.window_size, 0, -1)]
      inc_rates = np.array([abs_loss_incrs[-i]/abs_loss_incrs[-i-1] for i in range(1, self.config.div.window_size)])
      if (inc_rates >= self.config.div.incr_abs_rate_max_thresh).all() and not(self.config.div.disabled):
        self.react(self.main_msgs['div_loss'].format(readable(max(inc_rates))))
    smoothness = metrics.smoothness(losses[-self.config.fluct.window_size:])
    if smoothness < self.config.fluct.smoothness_ratio_min_thresh and not(self.config.fluct.disabled):
      self.react(self.main_msgs['fluctuated_loss'].format(readable(smoothness), self.config.fluct.smoothness_ratio_min_thresh))

  def check_representativeness(self, losses, perfs):
    if self.config.rep.disabled: return
    if len(losses) > 2 and len(perfs) > 2:
      measure_abs_corr = np.abs(np.corrcoef(losses,perfs))[1,0]
      if measure_abs_corr < self.config.rep.abs_corr_min_thresh:
        self.react(self.main_msgs['non_representative_loss'].format(readable(measure_abs_corr), self.config.rep.abs_corr_min_thresh))

  def check_overwhelming_regularization(self, grad_losses_arr, grad_reg_losses_arr):
    if self.config.over_reg.disabled: return
    def _norm_ratio(array_1, array_2):
      return np.linalg.norm(array_1)/(np.linalg.norm(array_2) + settings.EPSILON)
    for w_name, g_l_w, g_rl_w in zip(self.grad_losses_norm_ratios.keys(), grad_losses_arr, grad_reg_losses_arr):
      self.grad_losses_norm_ratios[w_name].append(_norm_ratio(g_rl_w, g_l_w))
    for w_name, grad_norm_ratios in self.grad_losses_norm_ratios.items():
      if len(grad_norm_ratios) >= self.config.over_reg.window_size: 
        rates = np.array([grad_norm_ratios[-i] for i in range(1, self.config.over_reg.window_size+1)])
        if (rates >= self.config.over_reg.growth_rate_max_thresh).all():
          self.react(self.main_msgs['overwhelm_reg_loss'].format(readable(rates[-1]), self.config.over_reg.growth_rate_max_thresh))

  def update_internals(self, curr_loss, curr_perf):
    self.min_loss = min(curr_loss, self.min_loss)
    self.nn_data.loss_data += [curr_loss]
    self.nn_data.perf_data += [curr_perf]
    return np.array(self.nn_data.loss_data), np.array(self.nn_data.perf_data)

  def after_run(self, run_context, run_values):
    self.iter_count += 1
    if self.nn_data.model.reg_loss is None:
      loss_val, perf_val = run_values.results
    else:
      loss_val, perf_val, grad_losses_arr, grad_reg_losses_arr = run_values.results
      self.check_overwhelming_regularization(grad_losses_arr, grad_reg_losses_arr)
    self.step_losses += [loss_val]
    self.step_perfs += [perf_val]
    if self.check_numerical_instabilities(loss_val): return
    if self.iter_count % self.config.period == 0:
      losses, perfs = self.update_internals(np.mean(self.step_losses), np.mean(self.step_perfs))
      self.check_loss_curve(losses)
      self.check_representativeness(losses, perfs)
      self.step_losses = []
      self.step_perfs = []
      
class PostActivationHook(TFCheckHook):

  def __init__(self, *args, **kwargs):
    super(PostActivationHook, self).__init__(*args, **kwargs)

  def begin(self):
    self.acts_tensors = self.nn_data.model.activations
    self.test_acts_tensors = self.nn_data.model.test_activations
    self.pre_acts_outs = self.nn_data.model.outs_pre_act
    self.test_pre_acts_outs = self.nn_data.model.test_outs_pre_act
    self.iter_count = -1
    self.perform_test = False
    
  def before_run(self, run_context): 
    return SessionRunArgs([self.acts_tensors, self.test_acts_tensors, 
                           self.pre_acts_outs, self.test_pre_acts_outs])

  def check_activations_similarity(self, acts_name, acts_array, test_acts_array):
    acts_array = transform_2d(acts_array, keep='first')
    test_acts_array = transform_2d(test_acts_array, keep='first')
    cka_sim = metrics.feature_space_linear_cka(acts_array, test_acts_array, debiased=True)
    if cka_sim <= self.config.sim_after_switch_mode_min_thresh:
      self.react(self.main_msgs['switch_mode_act_consistency'].format(cka_sim, self.config.sim_after_switch_mode_min_thresh))

  def check_outputs_similarity(self, outs_array, test_outs_array):
    cka_sim = metrics.feature_space_linear_cka(outs_array, test_outs_array, debiased=True)
    if cka_sim <= self.config.sim_after_switch_mode_min_thresh:
      self.react(self.main_msgs['switch_mode_out_consistency'].format(cka_sim, self.config.sim_after_switch_mode_min_thresh))

  def after_run(self, run_context, run_values):
    self.iter_count += 1
    if self.iter_count > self.config.start:
      if len(self.nn_data.model.test_extra_feed_dict) > 0 and \
        self.nn_data.model.test_extra_feed_dict.values()!=self.nn_data.model.train_extra_feed_dict.values():
        if self.iter_count % self.config.period == 0:
          _, self.test_acts, _, self.test_outs = run_values.results
          self.perform_test = True
        elif self.perform_test:
          self.iter_count -= 1
          self.perform_test = False
          acts, _, outs, _ = run_values.results
          for acts_name, acts_array in acts.items():
            test_acts_array = self.test_acts[acts_name]
            self.check_activations_similarity(acts_name, acts_array, test_acts_array)
          self.check_outputs_similarity(outs, self.test_outs)  
      else:
        if self.iter_count % self.config.period == 0:
          acts, test_acts, outs, test_outs = run_values.results
          for acts_name, acts_array in acts.items():
            test_acts_array = test_acts[acts_name]
            self.check_activations_similarity(acts_name, acts_array, test_acts_array)
          self.check_outputs_similarity(outs, test_outs)

class PostLossHook(TFCheckHook):

  def __init__(self, *args, **kwargs):
    super(PostLossHook, self).__init__(*args, **kwargs)

  def begin(self):
    self.previous_loss_value = None
    self.loss = self.nn_data.model.loss
    self.test_loss = self.nn_data.model.test_loss
    self.iter_count = -1
    self.perform_test = False
    
  def before_run(self, run_context): 
    return SessionRunArgs([self.loss,self.test_loss])

  def after_run(self, run_context, run_values):
    self.iter_count += 1
    if self.iter_count > self.config.start:
      if len(self.nn_data.model.test_extra_feed_dict) > 0 and \
        self.nn_data.model.test_extra_feed_dict.values()!=self.nn_data.model.train_extra_feed_dict.values():
        if self.iter_count % self.config.period == 0:
          self.previous_loss_value, _ = run_values.results
          self.perform_test = True
        elif self.perform_test:
          _, test_loss_value = run_values.results
          if abs(self.previous_loss_value - test_loss_value)/max(self.previous_loss_value, test_loss_value) > self.config.relative_loss_diff_max_thresh:
            self.react(self.main_msgs['switch_mode_loss_consistency'].format(readable(loss_value), readable(test_loss_value)))
          self.perform_test = False
      else:
        if self.iter_count % self.config.period == 0:
          loss_value, test_loss_value = run_values.results
          if abs(loss_value - test_loss_value)/max(loss_value, test_loss_value) > self.config.relative_loss_diff_max_thresh:
            self.react(self.main_msgs['switch_mode_loss_consistency'].format(readable(loss_value), readable(test_loss_value)))
