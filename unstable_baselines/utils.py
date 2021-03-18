# --- built in ---
import os
import re
import abc
import cv2
import csv
import sys
import glob
import json
import time
import random
import inspect
import datetime
import tempfile
import multiprocessing

# --- 3rd party ---
import gym 
import cloudpickle

import numpy as np
import tensorflow as tf

# --- my module ---


__all__ = [
    'set_global_seeds',
    'normalize_action',
    'unnormalize_action',
    'tf_soft_update_params'
]

# === utils ===
def set_global_seeds(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def normalize_action(action, high, low):
    assert high is not None and low is not None, 'High and low must be specified'

    return 2.0 * ((action-low)/(high-low)) - 1.0

def unnormalize_action(action, high, low):
    assert high is not None and low is not None, 'High and low must be specified'

    return low + (0.5 * (action + 1.0) * (high - low))


def flatten_obs(obs, space):
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple((np.stack([o[i] for o in obs]) for i in range(obs_len)))
    else:
        return np.stack(obs)

# === tf utils ===
@tf.function
def tf_soft_update_params(target_vars, current_vars, polyak=0.005):
    '''
    Soft update (polyak update)
    '''

    for (tar_var, cur_var) in zip(target_vars, current_vars):
        tar_var.assign((1.-polyak) * tar_var + polyak * cur_var)