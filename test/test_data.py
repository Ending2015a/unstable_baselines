# --- built in ---
import os
import sys
import time
import logging
import unittest

# --- 3rd party ---
import numpy as np

# --- my module ---
from unstable_baselines import data as ub_data
from test.utils import TestCase

class TestDataModule(TestCase):
    '''Test unstable_baselines.data module

    Class list:
    [x] NestedReplayBuffer
    [x] DictReplayBuffer
    [x] TrajReplayBuffer

    Function list:
    [ ] compute_advantages
    '''

    def test_nested_replay_buffer(self):
        buf = ub_data.NestedReplayBuffer(10)
        for i in range(20):
            buf.add({'a': ([i], [i+1])})
        self.assertTrue(buf.full)
        exp_0 = np.arange(10, 20)
        exp_1 = np.arange(11, 21)
        self.assertArrayEqual(buf.data['a'][0], exp_0)
        self.assertArrayEqual(buf.data['a'][1], exp_1)
        # test sample (batch)
        batch = buf(batch_size=3)
        self.assertEqual(tuple(batch['a'][0].shape), (3,))
        self.assertEqual(tuple(batch['a'][1].shape), (3,))
        batch_a = batch['a']
        batch['a'] = (batch_a[0]+1, batch_a[1]+2)
        exp_0[buf._cached_inds] += 1
        exp_1[buf._cached_inds] += 2
        # test update
        buf.update(batch)
        self.assertArrayEqual(buf.data['a'][0], exp_0)
        self.assertArrayEqual(buf.data['a'][1], exp_1)

        # test sample (batch & seq)
        batch = buf(batch_size=3, seq_len=2)
        self.assertEqual(tuple(batch['a'][0].shape), (3, 2))
        self.assertEqual(tuple(batch['a'][0].shape), (3, 2))
        batch_a = batch['a']
        batch['a'] = (batch_a[0]+1, batch_a[1]+2)
        exp_0[buf._cached_inds] += 1
        exp_1[buf._cached_inds] += 2
        # test update
        buf.update(batch)
        self.assertArrayEqual(buf.data['a'][0], exp_0)
        self.assertArrayEqual(buf.data['a'][1], exp_1)

    def test_dict_replay_buffer(self):
        buf = ub_data.DictReplayBuffer(10)
        for i in range(20):
            buf.add(a=[i], b=[i+1])
        self.assertTrue(buf.full)
        exp_0 = np.arange(10, 20)
        exp_1 = np.arange(11, 21)
        self.assertArrayEqual(buf.data['a'], exp_0)
        self.assertArrayEqual(buf.data['b'], exp_1)
        # test sample (batch)
        batch = buf(batch_size=3)
        self.assertEqual(tuple(batch['a'].shape), (3,))
        self.assertEqual(tuple(batch['b'].shape), (3,))
        batch['a'] += 1
        batch['b'] += 2
        exp_0[buf._cached_inds] += 1
        exp_1[buf._cached_inds] += 2
        # test update
        buf.update(**batch)
        self.assertArrayEqual(buf.data['a'], exp_0)
        self.assertArrayEqual(buf.data['b'], exp_1)

        # test sample (batch & seq)
        batch = buf(batch_size=3, seq_len=2)
        self.assertEqual(tuple(batch['a'].shape), (3, 2))
        self.assertEqual(tuple(batch['b'].shape), (3, 2))
        batch['a'] += 1
        batch['b'] += 2
        exp_0[buf._cached_inds] += 1
        exp_1[buf._cached_inds] += 2
        # test update
        buf.update(**batch)
        self.assertArrayEqual(buf.data['a'], exp_0)
        self.assertArrayEqual(buf.data['b'], exp_1)

    def test_traj_replay_buffer(self):
        buf = ub_data.TrajReplayBuffer()
        for i in range(20):
            buf.add(a=[i], b=[(i+1)%5==0])
        self.assertEqual(buf.pos, 20)
        self.assertFalse(buf.ready_for_sampling)
        with self.assertRaises(RuntimeError):
            buf(3)
        buf.make()
        self.assertEqual(buf.n_samples, 20)
        exp_a = np.arange(20)
        exp_b = (np.arange(20)+1)%5==0
        self.assertArrayEqual(buf.data['a'], exp_a)
        self.assertArrayEqual(buf.data['b'], exp_b)
        
        batches = []
        for batch in buf(batch_size=3):
            self.assertEqual(batch['a'].shape, (3,))
            self.assertEqual(batch['b'].shape, (3,))
            batches.append(batch)
        self.assertEqual(len(batches), 7)


if __name__ == '__main__':
    unittest.main()