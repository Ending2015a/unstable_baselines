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
    '''

    def test_nested_replay_buffer(self):
        capacity = 10
        n_samples = 20
        buf = ub_data.NestedReplayBuffer(capacity)
        for i in range(n_samples):
            buf.add({'a': ([i], [i+1])})
        self.assertTrue(buf.isfull())
        exp_0 = np.arange(capacity, n_samples)
        exp_1 = np.arange(capacity+1, n_samples+1)
        self.assertArrayEqual(buf.data['a'][0], exp_0)
        self.assertArrayEqual(buf.data['a'][1], exp_1)
        # test sample (batch=None)
        batch = buf()
        self.assertArrayEqual((capacity,), batch['a'][0].shape)
        self.assertArrayEqual((capacity,), batch['a'][1].shape)
        self.assertArrayEqual(batch['a'][0], buf[buf._cached_inds]['a'][0])
        self.assertArrayEqual(batch['a'][1], buf[buf._cached_inds]['a'][1])
        # test sample (batch=3)
        batch_size = 3
        batch = buf(batch_size=batch_size)
        self.assertArrayEqual((batch_size,), batch['a'][0].shape)
        self.assertArrayEqual((batch_size,), batch['a'][1].shape)
        self.assertArrayEqual(batch['a'][0], buf[buf._cached_inds]['a'][0])
        self.assertArrayEqual(batch['a'][1], buf[buf._cached_inds]['a'][1])
        # test update
        batch = buf(batch_size=batch_size)
        batch_a = batch['a']
        batch['a'] = (batch_a[0]+1, batch_a[1]+2)
        exp_0[buf._cached_inds] += 1
        exp_1[buf._cached_inds] += 2
        buf.update(batch)
        self.assertArrayEqual(buf.data['a'][0], exp_0)
        self.assertArrayEqual(buf.data['a'][1], exp_1)
        buf[buf._cached_inds] = batch
        self.assertArrayEqual(buf.data['a'][0], exp_0)
        self.assertArrayEqual(buf.data['a'][1], exp_1)
        # test sample (batch & seq)
        batch_size = 3
        seq_len = 2
        batch = buf(batch_size=batch_size, seq_len=seq_len)
        self.assertArrayEqual((batch_size, seq_len), batch['a'][0].shape)
        self.assertArrayEqual((batch_size, seq_len), batch['a'][1].shape)
        # test update
        batch_a = batch['a']
        batch['a'] = (batch_a[0]+1, batch_a[1]+2)
        exp_0[buf._cached_inds] += 1
        exp_1[buf._cached_inds] += 2
        buf.update(batch)
        self.assertArrayEqual(buf.data['a'][0], exp_0)
        self.assertArrayEqual(buf.data['a'][1], exp_1)
        buf[buf._cached_inds] = batch
        self.assertArrayEqual(buf.data['a'][0], exp_0)
        self.assertArrayEqual(buf.data['a'][1], exp_1)

    def test_nested_replay_buffer_multidim(self):
        capacity = 10
        n_samples = 20
        dim = 3
        buf = ub_data.NestedReplayBuffer(capacity)
        exp = np.arange(n_samples*dim).reshape((n_samples, 1, dim))
        for i in range(n_samples):
            buf.add({'a': exp[i]})
        exp = exp[capacity:, 0]
        self.assertArrayEqual((capacity, dim), buf.data['a'].shape)
        self.assertArrayEqual(exp, buf.data['a'])
        # test sample
        batch = buf()
        self.assertArrayEqual((capacity, dim), batch['a'].shape)
        self.assertArrayEqual(batch['a'], buf[buf._cached_inds]['a'])
        # test sample (batch=3)
        batch_size = 3
        batch = buf(batch_size=batch_size)
        self.assertArrayEqual((batch_size, dim), batch['a'].shape)
        self.assertArrayEqual(batch['a'], buf[buf._cached_inds]['a'])
        # test update
        batch = buf(batch_size=batch_size)
        batch['a'] = batch['a'] + 1
        exp[buf._cached_inds] += 1
        buf.update(batch)
        self.assertArrayEqual(exp, buf.data['a'])
        # update by assignment
        buf[buf._cached_inds] = batch
        self.assertArrayEqual(exp, buf.data['a'])
        # test sample (batch & seq)
        batch_size = 3
        seq_len = 2
        batch = buf(batch_size=batch_size, seq_len=seq_len)
        self.assertArrayEqual((batch_size, seq_len, dim), batch['a'].shape)
        self.assertArrayEqual(batch['a'], buf[buf._cached_inds]['a'])
        # test update
        batch['a'] = batch['a'] + 1
        exp[buf._cached_inds] += 1
        buf.update(batch)
        self.assertArrayEqual(exp, buf.data['a'])
        buf[buf._cached_inds] = batch
        self.assertArrayEqual(exp, buf.data['a'])

    def test_nested_replay_buffer_exception(self):
        with self.assertRaises(ValueError):
            buf = ub_data.NestedReplayBuffer(0)
        with self.assertRaises(ValueError):
            buf = ub_data.NestedReplayBuffer(None)
        buf = ub_data.NestedReplayBuffer(1)
        with self.assertRaises(RuntimeError):
            buf[0]
        with self.assertRaises(RuntimeError):
            buf._set_data(1, indices=0, _auto_create_space=False)

    def test_dict_replay_buffer(self):
        capacity = 10
        n_samples = 20
        buf = ub_data.DictReplayBuffer(capacity)
        for i in range(n_samples):
            buf.add(a=[i], b=[i+1])
        self.assertEqual(set(buf.keys()), set(['a', 'b']))
        self.assertTrue(buf.isfull())
        exp_0 = np.arange(capacity, n_samples)
        exp_1 = np.arange(capacity+1, n_samples+1)
        self.assertArrayEqual(buf.data['a'], exp_0)
        self.assertArrayEqual(buf.data['b'], exp_1)
        # test sample (batch=None)
        batch = buf()
        self.assertArrayEqual((capacity,), batch['a'].shape)
        self.assertArrayEqual((capacity,), batch['b'].shape)
        self.assertArrayEqual(batch['a'], buf[buf._cached_inds]['a'])
        self.assertArrayEqual(batch['b'], buf[buf._cached_inds]['b'])
        # test sample (batch=3)
        batch_size = 3
        batch = buf(batch_size=batch_size)
        self.assertArrayEqual((batch_size,), batch['a'].shape)
        self.assertArrayEqual((batch_size,), batch['b'].shape)
        self.assertArrayEqual(batch['a'], buf[buf._cached_inds]['a'])
        self.assertArrayEqual(batch['b'], buf[buf._cached_inds]['b'])
        # test update
        batch = buf(batch_size=batch_size)
        batch['a'] += 1
        batch['b'] += 2
        exp_0[buf._cached_inds] += 1
        exp_1[buf._cached_inds] += 2
        buf.update(a=batch['a'], b=batch['b'])
        self.assertArrayEqual(buf.data['a'], exp_0)
        self.assertArrayEqual(buf.data['b'], exp_1)
        buf[buf._cached_inds] = batch
        self.assertArrayEqual(buf.data['a'], exp_0)
        self.assertArrayEqual(buf.data['b'], exp_1)
        # test sample (batch & seq)
        batch_size = 3
        seq_len = 2
        batch = buf(batch_size=batch_size, seq_len=seq_len)
        self.assertArrayEqual((batch_size, seq_len), batch['a'].shape)
        self.assertArrayEqual((batch_size, seq_len), batch['b'].shape)
        # test update
        batch['a'] += 1
        batch['b'] += 2
        exp_0[buf._cached_inds] += 1
        exp_1[buf._cached_inds] += 2
        buf.update(a=batch['a'], b=batch['b'])
        self.assertArrayEqual(buf.data['a'], exp_0)
        self.assertArrayEqual(buf.data['b'], exp_1)
        buf[buf._cached_inds] = batch
        self.assertArrayEqual(buf.data['a'], exp_0)
        self.assertArrayEqual(buf.data['b'], exp_1)

    def test_dict_replay_buffer_multidim(self):
        capacity = 10
        n_samples = 20
        dim = 3
        buf = ub_data.DictReplayBuffer(capacity)
        exp = np.arange(n_samples*dim).reshape((n_samples, 1, dim))
        for i in range(n_samples):
            buf.add(a=exp[i])
        exp = exp[capacity:, 0]
        self.assertEqual(set(buf.keys()), set(['a']))
        self.assertArrayEqual((capacity, dim), buf.data['a'].shape)
        self.assertArrayEqual(exp, buf.data['a'])
        # test sample
        batch = buf()
        self.assertArrayEqual((capacity, dim), batch['a'].shape)
        self.assertArrayEqual(batch['a'], buf[buf._cached_inds]['a'])
        # test sample (batch=3)
        batch_size = 3
        batch = buf(batch_size=batch_size)
        self.assertArrayEqual((batch_size, dim), batch['a'].shape)
        self.assertArrayEqual(batch['a'], buf[buf._cached_inds]['a'])
        # test update
        batch = buf(batch_size=batch_size)
        batch['a'] = batch['a'] + 1
        exp[buf._cached_inds] += 1
        buf.update(a=batch['a'])
        self.assertArrayEqual(exp, buf.data['a'])
        # update by assignment
        buf[buf._cached_inds] = batch
        self.assertArrayEqual(exp, buf.data['a'])
        # test sample (batch & seq)
        batch_size = 3
        seq_len = 2
        batch = buf(batch_size=batch_size, seq_len=seq_len)
        self.assertArrayEqual((batch_size, seq_len, dim), batch['a'].shape)
        self.assertArrayEqual(batch['a'], buf[buf._cached_inds]['a'])
        # test update
        batch['a'] = batch['a'] + 1
        exp[buf._cached_inds] += 1
        buf.update(a=batch['a'])
        self.assertArrayEqual(exp, buf.data['a'])
        buf[buf._cached_inds] = batch
        self.assertArrayEqual(exp, buf.data['a'])

    def test_dict_replay_buffer_exception(self):
        with self.assertRaises(RuntimeError):
            buf = ub_data.DictReplayBuffer(1)
            buf.keys()

    def test_sequential_replay_buffer(self):
        n_samples = 20
        buf = ub_data.SequentialReplayBuffer()
        for i in range(n_samples):
            buf.add(a=[i], b=[(i+1)%5==0])
        self.assertEqual(n_samples, buf.pos)
        self.assertEqual(n_samples, len(buf))
        self.assertFalse(buf.ready_for_sampling)
        self.assertFalse(buf.isfull())
        with self.assertRaises(RuntimeError):
            buf(3)
        buf.make()
        with self.assertRaises(RuntimeError):
            buf.make()
        self.assertEqual(n_samples, buf.n_samples)
        self.assertEqual(n_samples, len(buf))
        self.assertFalse(buf.isfull())
        exp_a = np.arange(n_samples)
        exp_b = (np.arange(n_samples)+1)%5==0
        self.assertArrayEqual(exp_a, buf.data['a'])
        self.assertArrayEqual(exp_b, buf.data['b'])
        # test sample (batch=3)
        batch_size = 3
        batches = []
        for batch in buf(batch_size=batch_size):
            self.assertEqual((batch_size,), batch['a'].shape)
            self.assertEqual((batch_size,), batch['b'].shape)
            batches.append(batch)
        self.assertEqual(7, len(batches))
        # test sample
        batches = []
        for batch in buf():
            self.assertEqual((n_samples,), batch['a'].shape)
            self.assertEqual((n_samples,), batch['b'].shape)
            batches.append(batch)
        self.assertEqual(1, len(batches))
    
    def test_sequential_replay_buffer_multidim(self):
        n_samples = 20
        buf = ub_data.SequentialReplayBuffer()
        for i in range(n_samples):
            buf.add(a=np.array([[i]]), b=np.array([[(i+1)%5==0]]))
        self.assertEqual(n_samples, buf.pos)
        self.assertEqual(n_samples, len(buf))
        self.assertFalse(buf.ready_for_sampling)
        self.assertFalse(buf.isfull())
        with self.assertRaises(RuntimeError):
            buf(3)
        buf.make()
        self.assertEqual(n_samples, buf.n_samples)
        self.assertEqual(n_samples, len(buf))
        self.assertFalse(buf.isfull())
        exp_a = np.arange(n_samples).reshape((-1, 1))
        exp_b = ((np.arange(n_samples)+1)%5==0).reshape((-1, 1))
        self.assertArrayEqual(exp_a, buf.data['a'])
        self.assertArrayEqual(exp_b, buf.data['b'])
        # test sample (batch=3)
        batch_size = 3
        batches = []
        for batch in buf(batch_size=batch_size):
            self.assertEqual((batch_size, 1), batch['a'].shape)
            self.assertEqual((batch_size, 1), batch['b'].shape)
            batches.append(batch)
        self.assertEqual(7, len(batches))
        # test sample
        batches = []
        for batch in buf():
            self.assertEqual((n_samples, 1), batch['a'].shape)
            self.assertEqual((n_samples, 1), batch['b'].shape)
            batches.append(batch)
        self.assertEqual(1, len(batches))

    def test_sequential_replay_buffer_exception(self):
        buf = ub_data.SequentialReplayBuffer()
        with self.assertRaises(RuntimeError):
            buf._append_data({'a':0}, _auto_create_space=False)

    def test_compute_advantage(self):
        rew = np.asarray([0, 0, 1, 0, 1], dtype=np.float32)
        val = np.asarray([.5, .5, .5, .5, .5], dtype=np.float32)
        done = np.asarray([0, 0, 1, 0, 0], dtype=np.bool_)
        adv = ub_data.compute_advantage(rew=rew, val=val, done=done, 
                                    gamma=0.99, gae_lambda=0.95)
        exp = np.asarray([-0.47525, -0.5, 1.8704151, 0.9307975, 0.995],
                        dtype=np.float32)
        self.assertAllClose(exp, adv)