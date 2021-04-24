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

# --- 3rd party ---
import gym 
import cloudpickle

import numpy as np
import tensorflow as tf

from tensorflow.python.lib.io import file_io

# --- my module ---
from unstable_baselines import logger

LOG = logger.getLogger()


__all__ = [
    'NormalActionNoise',
    'StateObject',
    'set_global_seeds',
    'normalize',
    'denormalize',
    'flatten_obs',
    'soft_update',
    'is_json_serializable',
    'to_json_serializable',
    'from_json_serializable',
    'safe_json_dumps',
    'safe_json_loads',
    'safe_json_dump',
    'safe_json_load'
]



# === Const params ===

SERIALIZATION_KEY='#SERIALIZED'


# === Action noises ===
#TODO: Serializable Action noise 
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


# === StateObject ===
class StateObject(dict):
    '''StateObject can store model states, and it can
    be serialized by JSON.
    '''
    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls, *args, **kwargs)
        self.__dict__ = self
        return self
    
    def tostring(self):
        return safe_json_dumps(self, indent=None)

    @classmethod
    def fromstring(cls, string):
        self = StateObject()
        self.update(safe_json_loads(string))
        return self


# === Utils ===

def set_global_seeds(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def normalize(x, low, high, nlow=0.0, nhigh=1.0):
    '''
    Normalize x from [low, high] to [nlow, nhigh]
    '''
    return ((nhigh-nlow)/(high-low)) * (x-low) + nlow

def denormalize(x, low, high, nlow=0.0, nhigh=1.0):
    '''
    Denormalize x from [nlow, nhigh] to [low, high]
    '''
    return ((high-low)/(nhigh-nlow)) * (x-nlow) + low


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


def soft_update(target_vars, source_vars, polyak=0.005):
    '''Perform soft updates

    target_var = (1-polyak) * target_var + polyak * source_var

    Args:
        target_vars (list): A list of tf.Variable update to.
        source_vars (list): A list of tf.Variable update from. 
            The length must be equal to target_vars.
        polyak (float, optional): Smooth rate. Defaults to 0.005.
    '''


    if len(target_vars) != len(source_vars):
        raise ValueError('Length does not match, got {} and {}'.format(
                        len(target_vars), len(source_vars)))

    for (tar_var, src_var) in zip(target_vars, source_vars):
        tar_var.assign((1.-polyak) * tar_var + polyak * src_var)


# === JSON utils ===


def is_json_serializable(obj):
    '''
    Check if the object is json serializable
    '''
    try:
        json.dumps(obj, ensure_ascii=False)
        return True
    except:
        return False

def to_json_serializable(obj):
    '''Convert any object into json serializable object

    Args:
        obj (Any): Any object.
    '''
    if not is_json_serializable(obj):
        encoded_obj = base64.b64encode(
            cloudpickle.dumps(obj)).decode()
        encoded_obj = {SERIALIZATION_KEY: encoded_obj}
    else:
        encoded_obj = obj

    return encoded_obj

def from_json_serializable(encoded_obj):
    '''Convert any serializable object into original object

    Args:
        encoded_obj (Any): Serializable object
    '''
    if (isinstance(encoded_obj, dict) 
            and SERIALIZATION_KEY in encoded_obj.keys()):
        obj = encoded_obj[SERIALIZATION_KEY]
        obj = cloudpickle.loads(
            base64.b64decode(obj.encode()))
    else:
        obj = encoded_obj
    return obj



def safe_json_dumps(obj, 
                    indent=4, 
                    ensure_ascii=False, 
                    default=to_json_serializable, 
                    **kwargs):
    string = json.dumps(obj, indent=indent, 
                        ensure_ascii=ensure_ascii, 
                        default=default,
                        **kwargs)
    return string

def safe_json_loads(string, 
                    object_hook=from_json_serializable, 
                    **kwargs):

    obj = json.loads(string, 
                    object_hook=object_hook, 
                    **kwargs)
    return obj


def safe_json_dump(filepath,
                    obj,
                    **kwargs):
    string = safe_json_dumps(obj, **kwargs)
    file_io.atomic_write_string_to_file(filepath, string)


def safe_json_load(filepath,
                    **kwargs):    
    obj = None
    if file_io.file_exists(filepath):
        string = file_io.read_file_to_string(filepath)
        obj = safe_json_loads(string, **kwargs)

    return obj
