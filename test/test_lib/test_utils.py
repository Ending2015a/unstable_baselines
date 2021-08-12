# --- built in ---
import os
import sys
import time
import json
import logging
import unittest

# --- 3rd party ---
import gym
import numpy as np
import tensorflow as tf

# --- my module ---
from unstable_baselines.lib import utils
from test.utils import TestCase

def to_json_from_json(obj):
    serialized_obj = utils.to_json_serializable(obj)
    obj = utils.from_json_serializable(serialized_obj)
    return obj

def safe_json_dumps_loads(obj):
    string = utils.safe_json_dumps(obj)
    obj = utils.safe_json_loads(string)
    return obj

class TestUtilsModule(TestCase):
    '''Test unstable_baselines.utils module

    Class list:
    [x] NormalActionNoise
    [x] StateObject
    [ ] RunningMeanStd
    [ ] RMSNormalizer
    [ ] NestedRMSNormalizer (NotImplemented)

    Function list:
    [ ] set_seed
    [ ] normalize
    [ ] denormalize
    [ ] stack_obs
    [ ] soft_update
    [ ] is_image_space
    [ ] flatten_dicts
    [ ] is_bounded
    [ ] preprocess_observation
    [ ] get_tensor_ndims
    [ ] flatten_tensor
    [x] is_json_serializable
    [x] to_json_serializable
    [x] from_json_serializable
    [x] safe_json_dumps
    [x] safe_json_loads
    [x] safe_json_dump
    [x] safe_json_load
    [x] nested_iter
    [x] nested_iter_tuple
    [x] nested_to_numpy
    [ ] extract_structure
    [ ] pack_sequence
    '''

    def test_normal_action_noise(self):
        mean = 0.
        scale = 1.
        shape = (2, 3)
        noise = utils.NormalActionNoise(mean, scale)
        self.assertArrayEqual(noise(shape).shape, shape)
        self.assertEqual(
            repr(noise),
            f'NormalActionNoise(mean={mean}, scale={scale})'
        )
        noise.reset()

    def test_running_mean_std(self):
        mean = 0.0
        std = 1.0
        arr = np.arange(20, dtype=np.float32).reshape((10, 2))*0.1
        a = arr[:3]
        b = arr[3:]
        rms = utils.RunningMeanStd(mean, std)
        rms.update(a)
        self.assertArrayClose(rms.mean, np.mean(a, axis=0), decimal=6)
        self.assertArrayClose(rms.var, np.var(a, axis=0), decimal=6)
        rms.update(b)
        self.assertArrayClose(rms.mean, np.mean(arr, axis=0), decimal=6)
        self.assertArrayClose(rms.var, np.var(arr, axis=0), decimal=6)

    def test_rms_normalizer_box(self):
        pass

    def test_rms_normalizer_discrete(self):
        pass

    def test_rms_normalizer_image(self):
        pass

    def test_nested_rms_normalizer(self):
        pass

    def test_set_seed(self):
        utils.set_seed(1)
        a = np.random.normal(size=(3,))
        utils.set_seed(1)
        b = np.random.normal(size=(3,))
        self.assertArrayEqual(a, b)
    
    def test_normalize_denormalize(self):
        x = np.arange(21, dtype=np.float32)*0.1
        y = np.arange(21, dtype=np.float32)*0.05
        x_ = utils.normalize(x, x.min(), x.max())
        self.assertArrayClose(x_, y, decimal=6)
        y_ = utils.denormalize(y, x.min(), x.max())
        self.assertArrayClose(x, y_, decimal=6)

    def test_normalize_denormalize_tf(self):
        x = tf.range(21, dtype=tf.float32)*0.1
        y = tf.range(21, dtype=tf.float32)*0.05
        x_ = utils.normalize(x, tf.reduce_min(x), tf.reduce_max(x))
        self.assertArrayClose(x_, y, decimal=6)
        y_ = utils.denormalize(y, tf.reduce_min(x), tf.reduce_max(x))
        self.assertArrayClose(x, y_, decimal=6)

    def test_stack_obs_box(self):
        space = gym.spaces.Box(low=np.zeros((64,64,3)),
                               high=np.ones((64,64,3))*255, 
                               dtype=np.uint8)
        obs1 = space.sample()
        obs2 = space.sample()
        stacked = utils.stack_obs((obs1, obs2), space)
        self.assertArrayEqual(stacked, np.stack((obs1, obs2)))

    def test_stack_obs_dict(self):
        space1 = gym.spaces.Box(low=np.zeros((64,64,3)),
                                high=np.ones((64,64,3))*255, 
                                dtype=np.uint8)
        space2 = gym.spaces.MultiBinary(5)
        space = gym.spaces.Dict({'pov': space1, 'vec': space2})
        obs1 = space.sample()
        obs2 = space.sample()
        stacked = utils.stack_obs((obs1, obs2), space)
        self.assertArrayEqual(stacked['pov'], np.stack((obs1['pov'], obs2['pov'])))
        self.assertArrayEqual(stacked['vec'], np.stack((obs1['vec'], obs2['vec'])))
    
    def test_stack_obs_tuple(self):
        space1 = gym.spaces.Box(low=np.zeros((64,64,3)),
                                high=np.ones((64,64,3))*255, 
                                dtype=np.uint8)
        space2 = gym.spaces.MultiBinary(5)
        space = gym.spaces.Tuple((space1, space2))
        obs1 = space.sample()
        obs2 = space.sample()
        stacked = utils.stack_obs((obs1, obs2), space)
        self.assertArrayEqual(stacked[0], np.stack((obs1[0], obs2[0])))
        self.assertArrayEqual(stacked[1], np.stack((obs1[1], obs2[1])))
    
    def test_soft_update(self):
        a = tf.Variable(0.0, dtype=tf.float32)
        b = tf.Variable(1.0, dtype=tf.float32)
        utils.soft_update([a], [b], polyak=0.1)
        self.assertArrayClose(a.numpy(), np.asarray(0.1), decimal=6)

    def test_soft_update_exception(self):
        batch_size = 5
        feat_dim = 64
        model1 = tf.keras.Sequential([
            tf.keras.layers.Dense(64)
        ])
        model2 = tf.keras.Sequential([
            tf.keras.layers.Dense(64),
            tf.keras.layers.Dense(64)
        ])

        model1(tf.zeros((batch_size, feat_dim)))
        model2(tf.zeros((batch_size, feat_dim)))
        with self.assertRaises(ValueError):
            utils.soft_update(model1.trainable_variables, 
                              model2.trainable_variables)

    def test_is_image_space(self):
        # Box 3D + uint8: True
        space = gym.spaces.Box(low=np.zeros((64,64,3)),
                               high=np.ones((64,64,3))*255, 
                               dtype=np.uint8)
        self.assertTrue(utils.is_image_space(space))
        # Box 2D + uint8: False
        space = gym.spaces.Box(low=np.zeros((64,64)),
                               high=np.ones((64,64))*255, 
                               dtype=np.uint8)
        self.assertFalse(utils.is_image_space(space))
        # Box 3D + float32: False
        space = gym.spaces.Box(low=np.zeros((64,64,3)),
                               high=np.ones((64,64,3)),
                               dtype=np.float32)
        self.assertFalse(utils.is_image_space(space))

    def test_flatten_dicts(self):
        d1 = {'a': 1, 'b': [2, 3]}
        d2 = {'b': [4], 'a': 5}
        d3 = {'c': 6}
        d = utils.flatten_dicts([d1, d2, d3])
        self.assertEqual(list(d.keys()), ['a', 'b', 'c'])
        self.assertEqual(d['a'], [1, 5])
        self.assertEqual(d['b'], [[2, 3], [4]])
        self.assertEqual(d['c'], [6])
        
    def test_is_bounded(self):
        # bounded non-box
        space = gym.spaces.Discrete(5)
        self.assertTrue(utils.is_bounded(space))
        # bounded box
        low = np.zeros((64,64,3))
        high = np.ones((64,64,3))
        space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.assertTrue(utils.is_bounded(space))
        # unbounded box: low >= high
        low = np.zeros((64,64,3))
        high = np.ones((64,64,3))
        high[0,0,0] = 0
        space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.assertFalse(utils.is_bounded(space))
        # unbounded box: inf
        low = np.zeros((64,64,3))
        high = np.ones((64,64,3))
        high[0,0,0] = np.inf
        space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.assertFalse(utils.is_bounded(space))

    def test_preprocess_observation(self):
        pass

    def test_get_tensor_ndims(self):
        a = tf.zeros((1, 2, 3, 4), dtype=tf.float32)
        a = utils.get_tensor_ndims(a)
        self.assertEqual(a, 4)
        a = tf.keras.Input((3, 3), dtype=tf.float32)
        a = utils.get_tensor_ndims(a)
        self.assertEqual(a, 3)

    def test_flatten_tensor(self):
        a, b, c, d = 3, 4, 5, 6
        tensor = tf.zeros((a, b, c, d), dtype=np.float32)
        f = utils.flatten_tensor(tensor, 0)
        self.assertArrayEqual(f.shape, (a*b*c*d,))
        f = utils.flatten_tensor(tensor, 2)
        self.assertArrayEqual(f.shape, (a, b, c*d))
        f = utils.flatten_tensor(tensor, 1, 3)
        self.assertArrayEqual(f.shape, (a, b*c, d))

    def test_json_serializable_simple(self):
        '''Test
            utils.to_json_serializable
            utils.from_json_serializable

            Test object:
                int, float, dict, tuple
        '''
        # json serializable
        a = dict(a=10, b=1.0, c=[4.0, 5.0],
                d=('df', 'as'))

        self.assertEqual(utils.to_json_serializable(a), a)
        self.assertEqual(to_json_from_json(a), a)

        # not json serializable
        a[(8, 9, 10)] = 'asdf'
        self.assertEqual(to_json_from_json(a), a)

    def test_json_serializable_complex(self):
        '''Test
            utils.to_json_serializable
            utils.from_json_serializab.e

            Test object:
                class, numpy
        '''

        class A:
            c = 20
            def __init__(self, a, b):
                self.a = a
                self.b = b

        # class
        a = A(10, 11)
        a2 = to_json_from_json(a)
        self.assertEqual(type(a), type(a2))
        self.assertEqual(a.a, a2.a)
        self.assertEqual(a.b, a2.b)
        self.assertEqual(A.c, a2.c)

        # numpy
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        a2 = to_json_from_json(a)
        self.assertTrue(np.array_equal(a, a2))
        self.assertEqual(a.dtype, a2.dtype)

    def test_safe_json_dumps_loads_simple(self):
        # json serializable
        a = dict(a=10, b=1.0, c=[4.0, 5.0],
                d=('df', 'as'))

        self.assertEqual(utils.safe_json_dumps(a), 
                        json.dumps(a,indent=4,ensure_ascii=False))
        self.assertEqual((a), a)

    def test_safe_json_dumps_loads_complex(self):

        class A:
            c = 20
            def __init__(self, a, b):
                self.a = a
                self.b = b

        # class
        a = A(10, 11)
        a2 = safe_json_dumps_loads(a)
        self.assertEqual(type(a), type(a2))
        self.assertEqual(a.a, a2.a)
        self.assertEqual(a.b, a2.b)
        self.assertEqual(A.c, a2.c)

        # class in dict
        a = {'a': 10, 'b': A(10, 11)}
        a2 = json.loads(utils.safe_json_dumps(a))
        self.assertEqual(a['a'], a2['a'])
        self.assertTrue(utils.is_json_serializable(a2))
        self.assertTrue(isinstance(a2['b'], dict))
        self.assertTrue(utils.SERIALIZATION_KEY in a2['b'].keys())

        # numpy
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        a2 = safe_json_dumps_loads(a)
        self.assertTrue(np.array_equal(a, a2))
        self.assertEqual(a.dtype, a2.dtype)

    def test_state_object(self):

        # test state object
        s = utils.StateObject(a=10, b=20)
        s2 = utils.StateObject.fromstring(s.tostring())
        self.assertEqual(s2, s)

        # test assign values to state object
        s = utils.StateObject(a=np.array([1, 2, 3], dtype=np.uint8))
        s.b = 20
        s.c = None
        s['d'] = 30
        s2 = utils.StateObject.fromstring(s.tostring())

        self.assertEqual(s2['b'], 20)
        self.assertEqual(s2['c'], None)
        self.assertEqual(s2.d, 30)
        self.assertEqual(s.keys(), s2.keys())
        for k, v in s.items():
            if isinstance(v, np.ndarray):
                self.assertTrue(np.array_equal(s[k], s2[k]))
            else:
                self.assertEqual(s[k], s2[k])

    def test_nested_iter(self):
        op = lambda v: len(v)
        data = {'a': (np.arange(3), np.arange(4)), 
                'b': np.arange(5)}
        
        res = utils.nested_iter(data, op, first=False)
        self.assertEqual(res['a'][0], 3)
        self.assertEqual(res['a'][1], 4)
        self.assertEqual(res['b'], 5)

        res = utils.nested_iter(data, op, first=True)
        self.assertEqual(res, 3)

    def test_nested_iter_space(self):
        space1 = gym.spaces.Box(low=np.zeros((64,64,3)),
                                high=np.ones((64,64,3))*255, 
                                dtype=np.uint8)
        space2 = gym.spaces.MultiBinary(5)
        spacet = gym.spaces.Tuple((space1, space2))
        space  = gym.spaces.Dict({'tuple': spacet})
        op = lambda s:s
        space = utils.nested_iter_space(space, op)
        self.assertTrue(isinstance(space, dict))
        self.assertEqual(list(space.keys()), ['tuple'])
        self.assertTrue(isinstance(space['tuple'], tuple))
        self.assertEqual(space['tuple'], (space1, space2))

    def test_nested_iter_tuple(self):
        op = lambda data_tuple: np.asarray(data_tuple).shape
        data1 = {'a': (np.arange(3), np.arange(4)), 
                'b': np.arange(5)}
        data2 = {'a': (np.arange(3), np.arange(4)), 
                'b': np.arange(5)}

        res = utils.nested_iter_tuple((data1, data2), op)
        self.assertEqual(res['a'][0], (2, 3))
        self.assertEqual(res['a'][1], (2, 4))
        self.assertEqual(res['b'], (2, 5))

    def test_nested_iter_tuple_exception(self):
        op = lambda data_tuple: np.asarray(data_tuple).shape
        data1 = {'a': (np.arange(3), np.arange(4)), 
                'b': np.arange(5)}
        data2 = {'a': (np.arange(3), np.arange(4)), 
                'b': np.arange(5)}
        with self.assertRaises(TypeError):
            res = utils.nested_iter_tuple([data1, data2], op)

        with self.assertRaises(ValueError):
            op = None
            res = utils.nested_iter_tuple((data1, data2), op)


    def test_nested_to_numpy(self):
        data = {'a': (list(range(3)), list(range(4))),
                'b': list(range(5))}
        
        res = utils.nested_to_numpy(data)
        self.assertArrayEqual(res['a'][0], np.arange(3))
        self.assertArrayEqual(res['a'][1], np.arange(4))
        self.assertArrayEqual(res['b'], np.arange(5))