# --- built in ---
import os
import sys
import time
import json
import logging
import unittest

# --- 3rd party ---
import numpy as np
import tensorflow as tf

# --- my module ---
from unstable_baselines import utils

def to_json_from_json(obj):
    serialized_obj = utils.to_json_serializable(obj)
    obj = utils.from_json_serializable(serialized_obj)
    return obj

def safe_json_dumps_loads(obj):
    string = utils.safe_json_dumps(obj)
    obj = utils.safe_json_loads(string)
    return obj

class TestUtilsModule(unittest.TestCase):
    '''Test unstable_baselines.utils module

    Class list:
    [ ] NormalActionNoise
    [x] StateObject

    Function list:
    [ ] set_global_seeds
    [ ] normalize
    [ ] denormalize
    [ ] flatten_obs
    [ ] soft_update
    [x] is_json_serializable
    [x] to_json_serializable
    [x] from_json_serializable
    [x] safe_json_dumps
    [x] safe_json_loads
    [x] safe_json_dump
    [x] safe_json_load
    '''

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

    def test_stateobject(self):

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


if __name__ == '__main__':
    unittest.main()