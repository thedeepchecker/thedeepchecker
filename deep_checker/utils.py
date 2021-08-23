import scipy
import numpy as np

def add_extra_feeds(feeds, extra_feed_dict):
  if extra_feed_dict == {}:
    return feeds
  d = {}
  d.update(feeds)
  d.update(extra_feed_dict)
  return d

def almost_equal(value1, value2, rtol=1e-2):
  rerr = np.abs(value1-value2)
  if isinstance(value1, np.ndarray):
    return (rerr <= rtol).all()
  else:
    return rerr <= rtol

def reduce_data(np_data, reductions, axis=None):
  data_reductions = {}
  for reduction_name in reductions:
    data_reductions[reduction_name] = getattr(np, reduction_name)(np_data, axis=axis)
  return data_reductions

def trim_data(data, trim_prop=0.1):
  data.sort()
  trimmed_data = scipy.stats.trimboth(data.flatten(), trim_prop)
  return trimmed_data

def transform_2d(array, keep='first'):
  if keep == 'first':
    return array.reshape(array.shape[0], -1)
  elif keep == 'last':
    return array.reshape(-1, array.shape[-1])

def is_non_2d(array):
  return len(array.shape) > 2

def readable(float_num):
  return round(float_num, 3)