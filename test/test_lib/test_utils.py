# --- built in ---
import os
import sys
import time
import json
import logging
import tempfile
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
    '''Test unstable_baselines.lib.utils module
    '''

    def test_normal_action_noise(self):
        mean = 0.0
        scale = 1.0
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

    def test_running_mean_std_3d(self):
        space = gym.spaces.Box(low=np.ones((64,64,3))*0.0,
                               high=np.ones((64,64,3))*1.0,
                               dtype=np.float32)
        batch_size = 3
        obs = [space.sample() for _ in range(batch_size)]
        obs = utils.stack_obs(obs, space)
        mean = 0.0
        std = 1.0
        rms = utils.RunningMeanStd(mean, std)
        rms.update(obs)
        self.assertArrayClose(rms.mean, np.mean(obs, axis=0), decimal=6)
        self.assertArrayClose(rms.var, np.var(obs, axis=0), decimal=6)

    def test_rms_normalizer_box(self):
        # image box
        space = gym.spaces.Box(low=np.zeros((64,64,3)),
                               high=np.ones((64,64,3))*255, 
                               dtype=np.uint8)
        rms_norm = utils.RMSNormalizer(space)
        self.assertFalse(rms_norm.enabled)
        self.assertTrue(rms_norm.fixed)
        batch_size = 3
        obs = [space.sample() for _ in range(batch_size)]
        obs = utils.stack_obs(obs, space)
        res_obs = rms_norm.normalize(obs)
        self.assertArrayEqual(obs, res_obs)
        # non-image box
        space = gym.spaces.Box(low=np.ones((64,64,3))*0.0,
                               high=np.ones((64,64,3))*1.0,
                               dtype=np.float32)
        rms_norm = utils.RMSNormalizer(space)
        self.assertTrue(rms_norm.enabled)
        self.assertFalse(rms_norm.fixed)
        batch_size = 3
        obs = [space.sample() for _ in range(batch_size)]
        obs = utils.stack_obs(obs, space)
        rms_norm.update(obs)
        res_obs = rms_norm.normalize(obs)
        obs_mean = np.mean(obs, axis=0)
        obs_var = np.var(obs, axis=0)
        self.assertArrayClose(rms_norm.rms.mean, obs_mean)
        self.assertArrayClose(rms_norm.rms.var, obs_var)
        eps = np.finfo(np.float32).eps.item()
        obs_norm = (obs-obs_mean)/np.sqrt(obs_var+eps)
        self.assertArrayClose(obs_norm, res_obs, decimal=3)
        # sampling
        obs2 = [space.sample() for _ in range(batch_size)]
        obs2 = utils.stack_obs(obs2, space)
        rms_norm.update(obs2)
        res_obs = rms_norm.normalize(obs2)
        concat_obs = np.concatenate((obs, obs2), axis=0)
        obs_mean = np.mean(concat_obs, axis=0)
        obs_var = np.var(concat_obs, axis=0)
        self.assertArrayClose(rms_norm.rms.mean, obs_mean)
        self.assertArrayClose(rms_norm.rms.var, obs_var)
        obs_norm = (obs2-obs_mean) / np.sqrt(obs_var+eps)
        self.assertArrayClose(obs_norm, res_obs, decimal=3)
        # test save/load
        with tempfile.NamedTemporaryFile() as f:
            rms_norm.save(f.name)
            new_rms_norm = utils.RMSNormalizer(space).load(f.name)
        res_obs = new_rms_norm.normalize(obs2)
        self.assertArrayClose(obs_norm, res_obs, decimal=3)

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
        # test exceptions
        with self.assertRaises(ValueError):
            utils.stack_obs(obs1, space)
        with self.assertRaises(ValueError):
            utils.stack_obs([], space)

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
        # test exceptions
        with self.assertRaises(ValueError):
            tuple_space = gym.spaces.Tuple((space1, space2))
            utils.stack_obs((obs1, obs2), tuple_space)


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
        # test exceptions
        with self.assertRaises(ValueError):
            dict_space = gym.spaces.Dict({'pov': space1, 'vec': space2})
            utils.stack_obs((obs1, obs2), dict_space)

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

    def test_preprocess_observation_box_bounded(self):
        space = gym.spaces.Box(low=np.zeros((64,64,3)),
                               high=np.ones((64,64,3))*255, 
                               dtype=np.uint8)
        obs = space.sample()
        norm_obs = utils.normalize(obs, space.low, space.high, 0., 1.)
        res_obs = utils.preprocess_observation(obs, space)
        self.assertArrayClose(norm_obs, res_obs)
        # batch
        batch_size = 8
        obses = [space.sample() for _ in range(batch_size)]
        obs = utils.stack_obs(obses, space)
        norm_obs = utils.normalize(obs, space.low, space.high, 0., 1.)
        res_obs = utils.preprocess_observation(obs, space)
        self.assertArrayClose(norm_obs, res_obs)

    def test_preprocess_observation_box_unbounded(self):
        space = gym.spaces.Box(low=np.full((64,), -np.inf, dtype=np.float32),
                               high=np.full((64,), np.inf, dtype=np.float32),
                               dtype=np.float32)
        obs = space.sample()
        res_obs = utils.preprocess_observation(obs, space)
        self.assertArrayClose(obs, res_obs)

    def test_preprocess_observation_discrete(self):
        space_dim = 5
        space = gym.spaces.Discrete(space_dim)
        obs = space.sample()
        # one hot
        norm_obs = np.zeros((space_dim,), dtype=np.float32)
        norm_obs[obs] = 1.0
        res_obs = utils.preprocess_observation(obs, space)
        self.assertArrayClose(norm_obs, res_obs)
        # batch
        batch_size = 8
        obses = [space.sample() for _ in range(batch_size)]
        obs = utils.stack_obs(obses, space)
        # one hot
        norm_obs = np.zeros((obs.size, space_dim), dtype=np.float32)
        norm_obs[np.arange(obs.size), obs] = 1.0
        res_obs = utils.preprocess_observation(obs, space)
        self.assertArrayClose(norm_obs, res_obs)

    def test_preprocess_observation_multidiscrete(self):
        space_dims = [4, 7]
        space = gym.spaces.MultiDiscrete(space_dims)
        obs = space.sample()
        # one hot
        offset = np.cumsum([0] + space_dims)[:len(space_dims)]
        norm_obs = np.zeros((np.sum(space_dims),), dtype=np.float32)
        norm_obs[obs+offset] = 1.0
        res_obs = utils.preprocess_observation(obs, space)
        self.assertArrayClose(norm_obs, res_obs)
        # batch
        batch_size = 8
        obses = [space.sample() for _ in range(batch_size)]
        obs = utils.stack_obs(obses, space)
        # one hot
        norm_obs = np.zeros((batch_size, np.sum(space_dims)), dtype=np.float32)
        for batch, item in zip(np.arange(batch_size), obs+offset):
            norm_obs[batch, item] = 1.0
        res_obs = utils.preprocess_observation(obs, space)
        self.assertArrayClose(norm_obs, res_obs)

    def test_preprocess_observation_multibinary(self):
        space_dim = 6
        space = gym.spaces.MultiBinary(space_dim)
        obs = space.sample()
        norm_obs = utils.preprocess_observation(obs, space)
        self.assertArrayClose(obs, norm_obs)

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
        d1 = np.arange(3)
        d2 = np.arange(4)
        d3 = np.arange(5)
        data = {'a': (d1, d2), 'b': d3}
        res = utils.nested_iter(data, op, first=False)
        self.assertEqual(res['a'][0], 3)
        self.assertEqual(res['a'][1], 4)
        self.assertEqual(res['b'], 5)
        res = utils.nested_iter(data, op, first=True)
        self.assertEqual(res, 3)

    def test_nested_iter_exception(self):
        # ValueError when op is not a callable type
        op = True
        d1 = np.arange(3)
        d2 = np.arange(4)
        d3 = np.arange(5)
        data = {'a': (d1, d2), 'b': d3}
        with self.assertRaises(ValueError):
            utils.nested_iter(data, op)

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

    def test_nested_iter_space_exception(self):
        # ValueError when op is not a callable type
        space1 = gym.spaces.Box(low=np.zeros((64,64,3)),
                                high=np.ones((64,64,3))*255, 
                                dtype=np.uint8)
        space2 = gym.spaces.MultiBinary(5)
        spacet = gym.spaces.Tuple((space1, space2))
        space  = gym.spaces.Dict({'tuple': spacet})
        op = True
        with self.assertRaises(ValueError):
            utils.nested_iter_space(space, op)

    def test_nested_iter_tuple(self):
        op = lambda data_tuple: np.asarray(data_tuple).shape
        d1 = np.arange(3)
        d2 = np.arange(4)
        d3 = np.arange(5)
        data1 = {'a': (d1, d2), 'b': d3}
        data2 = {'a': (d1, d2), 'b': d3}
        res = utils.nested_iter_tuple((data1, data2), op)
        self.assertEqual(res['a'][0], (2, 3))
        self.assertEqual(res['a'][1], (2, 4))
        self.assertEqual(res['b'], (2, 5))

    def test_nested_iter_tuple_exception(self):
        op = lambda data_tuple: np.asarray(data_tuple).shape
        d1 = np.arange(3)
        d2 = np.arange(4)
        d3 = np.arange(5)
        data1 = {'a': (d1, d2), 'b': d3}
        data2 = {'a': (d1, d2), 'b': d3}
        with self.assertRaises(TypeError):
            res = utils.nested_iter_tuple([data1, data2], op)
        with self.assertRaises(ValueError):
            op = None
            res = utils.nested_iter_tuple((data1, data2), op)

    def test_nested_to_numpy(self):
        d1 = list(range(3))
        d2 = list(range(4))
        d3 = list(range(5))
        data = {'a': (d1, d2), 'b': d3}
        
        res = utils.nested_to_numpy(data)
        self.assertArrayEqual(res['a'][0], np.arange(3))
        self.assertArrayEqual(res['a'][1], np.arange(4))
        self.assertArrayEqual(res['b'], np.arange(5))

    def test_extract_structure(self):
        d1 = np.arange(3)
        d2 = np.arange(4)
        d3 = np.arange(5)
        data = {'a': (d1, d2), 'b': d3}
        struct, flat_data = utils.extract_structure(data)

        self.assertEqual(len(flat_data), 3)
        self.assertArrayEqual(flat_data[0], d1)
        self.assertArrayEqual(flat_data[1], d2)
        self.assertArrayEqual(flat_data[2], d3)
        self.assertEqual(list(struct.keys()), list(data.keys()))
        self.assertTrue(isinstance(struct['a'], tuple))
        self.assertEqual(struct['a'][0], 0)
        self.assertEqual(struct['a'][1], 1)
        self.assertEqual(struct['b'], 2)

    def test_pack_sequence(self):
        d1 = np.arange(3)
        d2 = np.arange(4)
        d3 = np.arange(5)
        data = {'a': (d1, d2), 'b': d3}
        struct, flat_data = utils.extract_structure(data)
        res_data = utils.pack_sequence(struct, flat_data)
        self.assertEqual(list(data.keys()), list(res_data.keys()))
        self.assertTrue(isinstance(res_data['a'], tuple))
        self.assertArrayEqual(res_data['a'][0], d1)
        self.assertArrayEqual(res_data['a'][1], d2)
        self.assertArrayEqual(res_data['b'], d3)