import torch
import numpy as np
import tensorflow as tf

def _is_list(x):
    return isinstance(x, list)

def _is_numpy(x):
    return isinstance(x, np.ndarray)

def _is_torch(x):
    return isinstance(x, torch.Tensor)

def _is_tensorflow(x):
    return isinstance(x, tf.Tensor)