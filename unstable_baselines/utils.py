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
import base64
import pickle
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
    'NormalActionNoise',
    'set_global_seeds',
    'normalize',
    'unnormalize',
    'to_json_serializable',
    'from_json_serializable',
    'tf_soft_update_params'
]


# === Action noises ===
class NormalActionNoise():
    def __init__(self, mean, scale):
        self.mean = mean
        self.scale = scale

    def __call__(self, shape=None):
        return np.random.normal(self.mean, self.scale, size=shape)

    def __repr__(self):
        return 'NormalActionNoise(mean={}, scale={})'.format(self.mean, self.scale)

    def reset(self):
        pass

# === utils ===
def set_global_seeds(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def normalize(x, high, low):
    return 2.0 * ((x-low)/(high-low)) - 1.0

def unnormalize(x, high, low):
    return low + (0.5 * (x + 1.0) * (high - low))

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

def is_json_serializable(obj):
    '''
    Check if the object is json serializable
    '''
    try:
        json.dumps(obj, ensure_ascii=False)
        return True
    except:
        return False

SERIALIZED_KEY='#serialized'


def to_json_serializable(d: dict):
    '''
    Serialize dict object to json string
    '''
    serializable_data = {}
    for k, v in d.items():
        if is_json_serializable(v):
            serializable_data[k] = v
        else:
            base64_encoded = base64.b64encode(
                cloudpickle.dumps(v)
            ).decode()

            serializable_data[k] = {SERIALIZED_KEY:base64_encoded}

    return serializable_data

def from_json_serializable(json_dict):
    '''
    Deserialize from json string
    '''

    deserialized_data = {}
    for k, v in json_dict.items():
        if isinstance(v, dict) and SERIALIZED_KEY in v.keys():
            serialized_obj = v[SERIALIZED_KEY]

            try:
                deserialized_obj = cloudpickle.loads(
                    base64.b64decode(serialized_obj.encode())
                )
            except pickle.UnpicklingError:
                raise RuntimeError("Failed to deserialize object {}".format(k))

            deserialized_data[k] = deserialized_obj
        else:
            deserialized_data[k] = v

    return deserialized_data


# === tf utils ===
@tf.function
def tf_soft_update_params(target_vars, current_vars, polyak=0.005):
    '''
    Soft update (polyak update)
    '''

    for (tar_var, cur_var) in zip(target_vars, current_vars):
        tar_var.assign((1.-polyak) * tar_var + polyak * cur_var)