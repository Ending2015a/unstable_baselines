# --- built in ---
import os
import re
import abc
import sys
import json
import time
import base64
import random
import datetime
import itertools
import contextlib

from collections import OrderedDict

# --- 3rd party ---
import gym 
import cloudpickle

import numpy as np
import tensorflow as tf

from tensorflow.python.lib.io import file_io

# --- my module ---

__all__ = [
    'spaces',
    'NormalActionNoise',
    'StateObject',
    'RunningMeanStd',
    'set_seed',
    'normalize',
    'denormalize',
    'stack_obs',
    'soft_update',
    'is_image_space',
    'flatten_dicts',
    'is_json_serializable',
    'to_json_serializable',
    'from_json_serializable',
    'input_tensor',
    'set_optimizer_params',
    'get_tensor_ndims',
    'flatten_dims',
    'run_eagerly',
    'safe_json_dumps',
    'safe_json_loads',
    'safe_json_dump',
    'safe_json_load',
    'iter_nested',
    'map_nested',
    'iter_nested_space',
    'map_nested_space',
    'iter_nested_tuple',
    'map_nested_tuple',
    'nested_to_numpy',
    'unpack_structure',
    'pack_sequence',
    'flatten_space'
]



# === Const params ===

SERIALIZATION_KEY='#SERIALIZED'

# Extra space profiles
class spaces:
    All = [gym.spaces.Box, gym.spaces.Discrete,
           gym.spaces.MultiBinary, gym.spaces.MultiDiscrete,
           gym.spaces.Tuple, gym.spaces.Dict]
    NonNested = [gym.spaces.Box, gym.spaces.Discrete,
                 gym.spaces.MultiBinary, gym.spaces.MultiDiscrete]
    Nested = [gym.spaces.Tuple, gym.spaces.Dict]

# === Action noises ===
#TODO: Move action noises to environment wrapper
class NormalActionNoise():
    def __init__(self, mean, scale):
        self.mean = mean
        self.scale = scale

    def __call__(self, shape=None):
        return np.random.normal(self.mean, self.scale, size=shape)

    def __repr__(self):
        return f'NormalActionNoise(mean={self.mean}, scale={self.scale})'

    def reset(self):
        pass

# === StateObject ===
class StateObject(dict):
    '''An object-like dictionary that you can get/set
    items/attributes by either __getitem__/__setitem__
    or __getattr__/__setattr__.
    For example:
    >>> d = StateObject()
    >>> d['abc'] = 10
    >>> print(d['abc'])
    This is equivalent to
    >>> d.abc = 10
    >>> print(d.abc)
    Also, StateObject can be serialized by JSON by calling
    `tostring` and `fromstring`.
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

# === RunningMeanStd ===
class RunningMeanStd(StateObject):
    def __init__(self, mean: float = 0.0, 
                       std:  float = 1.0):
        self.mean  = mean
        self.var   = std ** 2.0
        self.count = 0

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = len(x)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean, self.var = new_mean, new_var
        self.count = total_count

# === RMS normalizer ===
class RMSNormalizer():
    def __init__(self,
        space:    gym.Space = None,
        enable:        bool = None,
        fixed:         bool = None,
        rms: RunningMeanStd = None,
    ):
        '''A RunningMeanStd normalizer for non-nested spaces.
        This class provides an interface for `VecEnv`s to perform
        running mean std observation normalization. Note that this
        normalizer only supports non-nested spaces. 
        (e.g. gym.spaces.Box)

        Args:
            space (gym.Space): Gym space, expecting a gym.spaces.Boxl, float32.
            enable (bool, optional): Enable normalization. Default to True if
                `space` can be normalized. (Box, float32, not image)
            fixed (bool, optional): Whether to fix RMS value.
                Defaults to False.
            rms (RunningMeanStd, optional): RMS init value. Defaults to None.
        '''
        self.space    = space
        self._enable  = enable
        self._fixed   = fixed
        self._rms     = rms
        self.is_setup = False
        if space is not None:
            self.setup(space)

    def setup(self, space: gym.Space=None):
        '''Setup normalizer'''
        assert not self.is_setup, 'This RMSNormalizer is already been setup'
        self.space    = space
        self._enable  = self._get_enable_opt(self._enable)
        self._fixed   = self._get_fixed_opt(self._fixed)
        self._rms     = self._get_rms_opt(self._rms)
        self.is_setup = True
        return self

    @property
    def rms(self):
        return self._rms

    @property
    def enabled(self):
        return self._enable

    @enabled.setter
    def enabled(self, value: bool):
        self._enable = bool(value)

    @property
    def fixed(self):
        return self._fixed

    @fixed.setter
    def fixed(self, value: bool):
        self._fixed = bool(value)

    def normalize(self, x: np.ndarray, clip_max: float=10.0):
        '''Normalize x with mean and var'''
        # this magic number is from openai baselines
        # see baselines/common/vec_env/vec_normalize.py#L10
        if not self.enabled:
            return x
        rms = self.rms
        eps = np.finfo(np.float32).eps.item()
        x = (x-rms.mean)/np.sqrt(rms.var+eps)
        x = np.clip(x, -clip_max, clip_max)
        return x

    def __call__(self, x:np.ndarray, clip_max: float=10.0):
        '''Shortcut for `normalize()`'''
        return self.normalize(x, clip_max)

    def update(self, x:np.ndarray):
        '''Update RMS value'''
        if self.fixed:
            return
        if len(x.shape) == len(self.space.shape):
            # one sample, expand batch dim
            x = np.expand_dims(x, axis=0)
        self.rms.update(x)

    def load(self, path: str):
        '''Load RMS value from `path`
        Usage:
            >>> rms = RMSNormalizer(
            ...     env.observation_space,
            ...     enable = True,
            ...     fixed = True
            ... ).load('/saved_rms.json')
        '''
        d = safe_json_load(path)
        assert set(['mean','var','count']) == set(d.keys())
        #NOTE: RunningMeanStd is a dict type
        dict.update(self.rms, d)
        return self

    def save(self, path: str):
        '''Save RMS value to `path`'''
        safe_json_dump(path, self.rms)

    def _get_enable_opt(self, opt):
        if opt is None:
            if self.space is None:
                # return True for unknown space
                return True
            # Enabled if space is a Box but not image.
            return (not is_image_space(self.space)
                and isinstance(self.space, gym.spaces.Box)
                and np.dtype(self.space.dtype) == np.float32)
        return bool(opt)
    
    def _get_fixed_opt(self, opt):
        return (not self._enable) or (opt is True)

    def _get_rms_opt(self, opt):
        if isinstance(opt, RunningMeanStd):
            return opt
        return RunningMeanStd()


class NestedRMSNormalizer(RMSNormalizer):
    '''A Universal RunningMeanStd normalizer for both nested and 
    non-nested spaces.
    Since this normalizer may cause performance issue, we don't
    encourage you to use this normalizer. Instead, you can implement
    a custom RMSNormalizer for your custom space.
    '''
    def __init__(self, 
        space:    gym.Space,
        enable:        bool = None,
        fixed:         bool = None,
        rms: RunningMeanStd = None,
    ):
        print('WARN: We don\'t encourage you to use this RMS '
                'normalizer for nested spaces (Dict/Tuple). '
                'See ub.lib.utils.NestedRMSNormalizer')
        #TODO
        raise NotImplementedError
    #     assert isinstance(space, gym.Space)
    #     self.space = space

    #     pyspace = nested_iter_space(self.space, lambda sp:sp)
    #     self._pyspace = pyspace
    #     self._enable_norm = self._get_enable_norm_opt(enable_norm)
    #     self._update_norm = self._get_update_norm_opt(update_norm)
    #     self._norm_rms    = self._get_norm_rms_opt(norm_rms)

    # def _get_enable_norm_opt(self, opt):
    #     nest_opt = self._form_nested_opt(opt, arg_name='enable_norm')
    #     return nested_iter_tuple((nest_opt, self._pyspace),
    #                 lambda t: (t[0] is not False) and is_image_obs(t[1]))

    # def _get_update_norm_opt(self, opt):
    #     nest_opt = self._form_nested_opt(opt, arg_name='update_norm')
    #     return nested_iter_tuple((nest_opt, self._enable_norm),
    #             lambda t: (t[0] is not False) and t[1])

    # def _get_norm_rms_opt(self, opt):


    # def _form_nested_opt(self, opt, arg_name):
    #     if isinstance(opt, (list, tuple, dict)):
    #         try:
    #             nested_iter_tuple((self._pyspace, opt), 
    #                     lambda _: 0)
    #         except:
    #             raise ValueError(f'If `{arg_name}` is a nested data, '
    #                 'it must have a same structure as `space`')
    #         nest_opt = opt
    #     else:
    #         nest_opt = nested_iter(self._pyspace, lambda _: opt)
    #     return nest_opt


# === Utils ===

def set_seed(seed):
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

def is_image_space(space):
    return (isinstance(space, gym.spaces.Box)
                and np.dtype(space.dtype) == np.uint8
                and len(space.shape) == 3)

def flatten_dicts(dicts: list):
    '''Flatten a list of dicts

    Args:
        dicts (list): list of dicts

    Returns:
        dict: flattened dict
    '''
    agg_dict = {}
    for d in dicts:
        for k, v in d.items():
            agg_dict.setdefault(k, []).append(v)
    return agg_dict

def is_bounded(space):
    if isinstance(space, gym.spaces.Box):
        return (not np.any(np.isinf(space.low))
                and not np.any(np.isinf(space.high))
                and np.all(space.high-space.low > 0.))
    return True

# === TensorFlow utils ===

def input_tensor(space, batch_size=1):
    _tensor_op = lambda sp: (tf.zeros((batch_size, *sp.shape), 
                                       dtype=sp.dtype))
    return map_nested_space(space, _tensor_op)

def set_optimizer_params(optimizer, param_list):
    # Set optimizer tracking parameters
    if len(param_list) > 0:
        with tf.name_scope(optimizer._name):
            with tf.init_scope():
                optimizer._create_all_weights(param_list)
    return optimizer

def preprocess_observation(inputs, obs_space, dtype=tf.float32):
    '''Preprocess non-float observations
    If the input's dtype is not float32 or float64 we normalize it.
    '''
    inputs = tf.convert_to_tensor(inputs)
    # Do nothing if input is a float
    if tf.as_dtype(inputs.dtype) in [tf.float32, tf.float64]:
        return tf.cast(inputs, dtype=dtype)
    # Normalize
    if isinstance(obs_space, gym.spaces.Box):
        inputs = tf.cast(inputs, dtype=dtype)
        if is_bounded(obs_space):
            low    = tf.constant(obs_space.low, dtype=dtype)
            high   = tf.constant(obs_space.high, dtype=dtype)
            inputs = normalize(inputs, low, high, 0., 1.)
    elif isinstance(obs_space, gym.spaces.Discrete):
        depth  = tf.constant(obs_space.n, dtype=tf.int32)
        inputs = tf.one_hot(inputs, depth=depth)
    elif isinstance(obs_space, gym.spaces.MultiDiscrete):
        # inputs = [3, 5] obs_space.nvec = [4, 7]
        # [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
        nvec   = tf.constant(obs_space.nvec, dtype=tf.int32)
        inputs = tf.concat([
                tf.one_hot(inputs[..., idx], depth=nvec[idx])
                for idx in range(inputs.shape[-1])
            ], axis=-1)
    elif isinstance(obs_space, gym.spaces.MultiBinary):
        pass
    else:
        raise NotImplementedError("Preprocessing not implemented for "
                                f"{type(obs_space)}")

    return tf.cast(inputs, dtype=dtype)

def get_tensor_ndims(tensor: tf.Tensor):
    tensor = tf.convert_to_tensor(tensor)
    ndims = tensor.get_shape().ndims or tf.rank(tensor)
    return ndims


def flatten_dims(tensor: tf.Tensor, begin, end=None):
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
        raise ValueError('Length does not match, '
            f'got {len(target_vars)} and {len(source_vars)}')

    for (tar_var, src_var) in zip(target_vars, source_vars):
        tar_var.assign((1.-polyak) * tar_var + polyak * src_var)

def broadcast_shape(x_shape, y_shape):
    x_shape_static = tf.get_static_value(x_shape)
    y_shape_static = tf.get_static_value(y_shape)
    if (x_shape_static is None) or (y_shape_static is None):
        return tf.broadcast_dynamic_shape(x_shape, y_shape)
    return tf.broadcast_static_shape(
        tf.TensorShape(x_shape_static), tf.TensorShape(y_shape_static))

@contextlib.contextmanager
def run_eagerly(enabled=True):
    orig_conf = tf.executing_eagerly()
    tf.config.run_functions_eagerly(enabled)
    try:
        yield
    finally:
        tf.config.run_functions_eagerly(orig_conf)

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
                    indent=2, 
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

def iter_nested(data, sortkey=False):
    '''Iterate over nested data structure
    Note: Use `tuple` instead of `list`. A list type
    object is treated as an item.

    For example:
    >>> data = {'a': (1, 2), 'b': 3}
    >>> list(v for v in iter_nested(data))
    [1, 2, 3]

    Args:
        data (tuple or dict): A nested data.
        sortkey (bool): Whether to sort dict's key. Defaults
            to False.
    '''
    def _inner_iter_nested(data):
        if isinstance(data, dict):
            keys = sorted(data.keys()) if sortkey else data.keys()
            for k in keys:
                yield from _inner_iter_nested(data[k])
        elif isinstance(data, tuple):
            for v in data:
                yield from _inner_iter_nested(v)
        else:
            yield data
    return _inner_iter_nested(data)

def map_nested(data, op, *args, sortkey=False, **kwargs):
    '''A nested version of map function
    NOTE: Use `tuple` instead of `list`. A list type 
    object is treated as an item.

    Args:
        data (tuple or dict): A nested data
        op (function): A function operate on each data
        sortkey (bool): Whether to sort dict's key. Defaults
            to False.
    '''
    if not callable(op):
        raise ValueError('`op` must be a callable')

    def _inner_map_nested(data):
        if isinstance(data, dict):
            keys = sorted(data.keys()) if sortkey else data.keys()
            return {k: _inner_map_nested(data[k])
                        for k in keys}
        elif isinstance(data, tuple):
            return tuple(_inner_map_nested(v)
                            for v in data)
        else:
            return op(data, *args, **kwargs)
    return _inner_map_nested(data)

def iter_nested_space(space, sortkey=False):
    '''Iterate over nested gym space. Similar to iter_nested
    but it's for gym spaces.

    Args:
        space (gym.Space): Nested or non-nested gym space
        sortkey (bool): Whether to sort dict's key. Defaults
            to False.
    '''
    def _inner_iter_nested(space):
        if isinstance(space, gym.spaces.Dict):
            spaces = space.spaces
            keys = sorted(spaces.keys()) if sortkey else spaces.keys()
            for k in keys:
                yield from _inner_iter_nested(spaces[k])
        elif isinstance(space, gym.spaces.Tuple):
            for v in space.spaces:
                yield from _inner_iter_nested(v)
        else:
            yield space
    return _inner_iter_nested(space)

def map_nested_space(space, op, *args, sortkey=False, **kwargs):
    '''A nested version of map function. Similar to map_nested
    but it's for gym spaces.

    Args:
        space (gym.Space): Nested or non-nested gym space
        op (callable): A function operate on each data
        sortkey (bool): Whether to sort dict's key. Defaults
            to False.
    '''
    if not callable(op):
        raise ValueError('`op` must be a callable')

    def _inner_map_nested(space):
        if isinstance(space, gym.spaces.Dict):
            spaces = space.spaces
            keys = sorted(spaces.keys()) if sortkey else spaces.keys()
            return {k: _inner_map_nested(spaces[k])
                        for k in keys}
        elif isinstance(space, gym.spaces.Tuple):
            return tuple(_inner_map_nested(v)
                            for v in space.spaces)
        else:
            return op(space, *args, **kwargs)
    return _inner_map_nested(space)

def iter_nested_tuple(data_tuple, sortkey=False):
    '''Iterate over a tuple of nested structures. Similar to iter_nested
    but it iterates each of each nested data in the input tuple.
    For example:
    >>> a = {'x': 1, 'y': (2, 3)}
    >>> b = {'u': 4, 'v': (5, 6)}
    >>> list(iter_nested_tuple((a, b)))
    [(1, 4), (2, 5), (3, 6)]

    Args:
        data_tuple (tuple): A tuple of nested data.
        sortkey (bool): Whether to sort dict's key. Defaults
            to False.
    '''
    if not isinstance(data_tuple, tuple):
        raise TypeError('`data_tuple` only accepts tuple, '
                f'got {type(data_tuple)}')
    def _inner_iter_nested(data_tuple):
        if isinstance(data_tuple[0], dict):
            keys = data_tuple[0].keys()
            keys = sorted(keys) if sortkey else keys
            for k in keys:
                yield from _inner_iter_nested(
                        tuple(data[k] for data in data_tuple))
        elif isinstance(data_tuple[0], tuple):
            for k in range(len(data_tuple[0])):
                yield from _inner_iter_nested(
                    tuple(data[k] for data in data_tuple))
        else:
            yield data_tuple
    return _inner_iter_nested(data_tuple)

def map_nested_tuple(data_tuple, op, *args, sortkey=False, **kwargs):
    '''A nested version of map function. Similar to map_nested
    but it iterates each of each nested data in the input tuple.

    Args:
        data_tuple (tuple): A tuple of nested data.
        op (callable): A function operate on each data
        sortkey (bool): Whether to sort dict's key. Defaults
            to False.
    '''
    if not callable(op):
        raise ValueError('`op` must be a callable')
    if not isinstance(data_tuple, tuple):
        raise TypeError('`data_tuple` only accepts tuple, '
                f'got {type(data_tuple)}')
    def _inner_map_nested(data_tuple):
        if isinstance(data_tuple[0], dict):
            keys = data_tuple[0].keys()
            keys = sorted(keys) if sortkey else keys
            return {k: _inner_map_nested(
                        tuple(data[k] for data in data_tuple))
                    for k in keys}
        elif isinstance(data_tuple[0], tuple):
            return tuple(_inner_map_nested(
                        tuple(data[idx] for data in data_tuple))
                    for idx in range(len(data_tuple[0])))
        else:
            return op(data_tuple, *args, **kwargs)
    return _inner_map_nested(data_tuple)

def nested_to_numpy(data, sortkey=False):
    '''Convert all items in a nested data into 
    numpy arrays

    Args:
        data (dict, tuple): A nested data
        sortkey (bool): Whether to sort dict's key. Defaults
            to False.

    Returns:
        dict, tuple: A nested data same as `data`
    '''    
    op = lambda arr: np.asarray(arr)
    return map_nested(data, op, sortkey=sortkey)

def unpack_structure(data, sortkey=False):
    '''Extract structure and flattened data from a nested data
    For example:
        >>> data = {'a': 'abc', 'b': (2.0, [3, 4, 5])}
        >>> struct, flat_data = extract_struct(data)
        >>> flat_data
        ['abc', 2.0, [3, 4, 5]]
        >>> struct
        {'a': 0, 'b': (1, 2)}
    
    Args:
        data (dict, tuple): A nested data
        sortkey (bool): Whether to sort dict's key. Defaults
            to False.

    '''
    _count_op = lambda v, c: next(c)
    counter = itertools.count(0)
    struct = map_nested(data, _count_op, counter, sortkey=sortkey)
    size = next(counter)
    flat_data = [None] * size
    def _flat_op(ind_and_data, flat_data):
        ind, data = ind_and_data
        flat_data[ind] = data
    map_nested_tuple((struct, data), _flat_op, flat_data, sortkey=sortkey)
    return struct, flat_data

def pack_sequence(struct, flat_data, sortkey=False):
    '''An inverse operation of `extract_structure`

    Args:
        struct: A nested structure each data field contains
            an index of elements in `flat_data`
        flat_data (list): flattened data.
        sortkey (bool): Whether to sort dict's key. Defaults
            to False.
    '''
    _struct_op = lambda ind, flat: flat[ind]
    data = map_nested(struct, _struct_op, flat_data, sortkey=sortkey)
    return data

def flatten_space(space: gym.Space, sortkey=False):
    '''Flatten gym's nested space, like gym.Dict or gym.Tuple,
    to a tuple of spaces.

    Args:
        space (gym.Space): Nested or non-nested space.
        sortkey (bool): Whether to sort dict's key. Defaults
            to False.

    Returns:
        tuple: Flattened gym spaces.
    '''
    return tuple(v for v in iter_nested_space(space, sortkey=sortkey))