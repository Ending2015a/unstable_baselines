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
    'stack_obs',
    'soft_update',
    'flatten_dicts',
    'is_json_serializable',
    'to_json_serializable',
    'from_json_serializable',
    'get_tensor_ndims',
    'flatten',
    'safe_json_dumps',
    'safe_json_loads',
    'safe_json_dump',
    'safe_json_load'.
    'nested_iter',
    'nested_iter_tuple',
    'nested_to_numpy'
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
    
    def tostring(self, indent=None):
        return safe_json_dumps(self, indent=indent)

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


def stack_obs(obs, space):
    if not isinstance(obs, (list, tuple)):
        raise ValueError('Expecting a list or tuple of observations')
    if len(obs) <= 0:
        raise ValueError('Need observations from at least one environment')

    if isinstance(space, gym.spaces.Dict):
        if not isinstance(space.spaces, OrderedDict):
            raise ValueError('Dict space must have ordered subspaces')
        if not isinstance(obs[0], dict):
            raise ValueError('non-dict observation for environment with '
                    'Dict observation space')
        return OrderedDict([
            (k, stack_obs([o[k] for o in obs], space.spaces[k]))
            for k in space.spaces.keys()
        ])
    elif isinstance(space, gym.spaces.Tuple):
        if not isinstance(obs[0], tuple):
            raise ValueError('non-tuple observation for environment '
                    'with Tuple observation space')
        return tuple([
            stack_obs([o[i] for o in obs], space.spaces[i])
            for i in range(len(space.spaces))
        ])
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


def is_image_obs(obs_space):
    return (isinstance(obs_space, gym.spaces.Box) and
                len(obs_space.shape) == 3)

def flatten_dicts(dicts: list):
    '''Flatten a list of dicts

    Args:
        dicts (list): list of dicts

    Returns:
        dict: flattened dict
    '''
    agg_dict = {}
    for d in dicts:
        for k, v in d.itmes():
            agg_dict.setdefault(k, []).append(v)
    return agg_dict

# === TensorFlow utils ===

def is_bounded(space):
    if isinstance(space, gym.spaces.Box):
        return (not np.any(np.isinf(space.low))
                and not np.any(np.isinf(space.high))
                and np.all(space.high-space.low > 0.))
    return True

def preprocess_observation(inputs, obs_space, dtype=tf.float32):
    if isinstance(obs_space, gym.spaces.Box):
        inputs = tf.cast(inputs, dtype=tf.float32)
        if is_bounded(obs_space):
            low    = tf.constant(obs_space.low, dtype=tf.float32)
            high   = tf.constant(obs_space.high, dtype=tf.float32)
            inputs = normalize(inputs, low, high, 0., 1.)
    elif isinstance(obs_space, gym.spaces.Discrete):
        depth  = tf.constant(obs_space.n, dtype=tf.int32)
        inputs = tf.one_hot(inputs, depth=depth)
    elif isinstance(obs_space, gym.spaces.MultiDiscrete):
        # inputs = [3, 5] obs_space.nvec = [4, 7]
        # [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
        nvec   = tf.constant(obs_space.nvec, dtype=tf.int32)
        inputs = tf.concat([
                tf.one_hot(inputs[:, idx], depth=nvec[idx])
                for idx in range(inputs.shape[-1])
            ], axis=-1)
    elif isinstance(obs_space, gym.spaces.MultiBinary):
        pass
    else:
        raise NotImplementedError("Preprocessing not implemented for "
                                "{}".format(obs_space))

    return tf.cast(inputs, dtype=dtype)

def get_tensor_ndims(tensor: tf.Tensor):
    tensor = tf.convert_to_tensor(tensor)
    ndims = tensor.get_shape().ndims or tf.rank(tensor)
    return ndims


def flatten(tensor: tf.Tensor, begin, end=None):
    '''Collapse dims

    Flatten tensor from dim `begin` to dim `end`

    For example:
        >>> tensor = tf.zeros((2, 3, 4))
        >>> print(collapse_dims(tensor, 1).shape)
        (2, 12)

    Args:
        tensor (tf.Tensor): [description]
        begin ([type]): [description]
        end ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    '''    
    tensor = tf.convert_to_tensor(tensor)
    if end is None:
        flat_shape = tf.concat((
            tf.shape(tensor)[:begin],
            [tf.math.reduce_prod(tf.shape(tensor)[begin:])],
        ), axis=0)
    else:
        flat_shape = tf.concat((
            tf.shape(tensor)[:begin],
            [tf.math.reduce_prod(tf.shape(tensor)[begin:end])],
            tf.shape(tensor)[end:]
        ), axis=0)

    return tf.reshape(tensor, flat_shape)

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


# === nested ops ===

def nested_iter(data, op, *args, first=False, **kwargs):
    '''Iterate over nested data
    NOTE: Use `tuple` instead of `list`. A list type 
    object is treated as an item.

    Args:
        data (tuple or dict): A nested data
        op (function): A function operate on each data
        first (bool): Only iterate on the first item
    '''
    def _inner_nested_iter(data):
        if isinstance(data, dict):
            if first:
                return _inner_nested_iter(
                            next(iter(data.values())))
            else:
                return {k: _inner_nested_iter(v)
                            for k, v in data.items()}
        elif isinstance(data, tuple):
            if first:
                return _inner_nested_iter(
                            next(iter(data)))
            else:
                return tuple(_inner_nested_iter(v)
                                for v in data)
        else:
            return op(data, *args, **kwargs)
    return _inner_nested_iter(data)

def nested_iter_tuple(data_tuple, op, *args, **kwargs):
    '''Iterate over a tuple of nested data. Each nested
    data must have a same nested structure.
    NOTE: Use `tuple` instead of `list`. A list type
    object is treated as an item.

    Args:
        data_tuple (tuple): [description]
        op ([type]): [description]
    '''
    if not isinstance(data_tuple, tuple):
        raise ValueError('`data_tuple` only accepts tuple, '
                'got {}'.format(type(tensor_tuple)))
    
    def _inner_nested_iter_tuple(data_tuple):
        if isinstance(data_tuple[0], dict):
            return {k: _inner_nested_iter_tuple(
                            tuple(data[k]
                            for data in data_tuple))
                        for k in data_tuple[0].keys()}
        elif isinstance(data_tuple[0], tuple):
            return tuple(_inner_nested_iter_tuple(
                            tuple(data[idx] 
                                 for data in data_tuple))
                        for idx in range(len(data_tuple[0])))
        else:
            return op(data_tuple, *args, **kwargs)
    return _inner_nested_iter_tuple(data_tuple)

def nested_to_numpy(data):
    '''Convert all items in a nested data into 
    numpy arrays

    Args:
        data (dict, tuple): A nested data

    Returns:
        dict, tuple: A nested data same as `data`
    '''    
    op = lambda arr: np.asarray(arr)
    return nested_iter(data, op)

