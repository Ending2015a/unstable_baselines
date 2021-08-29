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
        n_samples = 15 # test circular
        buf = ub_data.NestedReplayBuffer(capacity)
        self.assertEqual(buf.capacity, capacity)
        self.assertTrue(buf.isnull)
        for i in range(n_samples):
            buf.add({'a': ([i], [i+1])})
            if i < capacity-1:
                self.assertFalse(buf.isfull)
                self.assertEqual(i+1, len(buf))
            else:
                self.assertTrue(buf.isfull)
                self.assertEqual(capacity, len(buf))
        exp = np.arange(n_samples-capacity, n_samples)
        exp_a0 = np.roll(exp, n_samples % capacity)
        exp_a1 = exp_a0 + 1
        self.assertArrayEqual(buf.data['a'][0], exp_a0)
        self.assertArrayEqual(buf.data['a'][1], exp_a1)
        # test getitem
        data = buf[np.arange(n_samples % capacity)]
        exp_a0 = np.arange(n_samples - n_samples % capacity, n_samples)
        exp_a1 = exp_a0 + 1
        self.assertArrayEqual(data['a'][0], exp_a0)
        self.assertArrayEqual(data['a'][1], exp_a1)
        # test setitem
        n = n_samples - capacity
        new_data = np.arange(n - n_samples % capacity, n)
        new_data = {'a': (new_data, new_data+1)}
        buf[np.arange(n_samples % capacity)] = new_data
        n = n_samples - capacity - n_samples % capacity
        exp_a0 = np.arange(n, n + capacity)
        exp_a1 = exp_a0 + 1
        self.assertArrayEqual(buf.data['a'][0], exp_a0)
        self.assertArrayEqual(buf.data['a'][1], exp_a1)
        # test update (should have the same results as setitem)
        buf.update(new_data, indices=np.arange(n_samples % capacity))
        self.assertArrayEqual(buf.data['a'][0], exp_a0)
        self.assertArrayEqual(buf.data['a'][1], exp_a1)
    
    def test_uniform_sampler_with_nested_replay_buffer(self):
        capacity = 10
        n_samples = 15 # test circular
        buf = ub_data.NestedReplayBuffer(capacity)
        samp = ub_data.UniformSampler(buf)
        for i in range(n_samples):
            buf.add({'a': ([i], [i+1])})
            if i < capacity-1:
                self.assertFalse(buf.isfull)
                self.assertEqual(i+1, len(buf))
            else:
                self.assertTrue(buf.isfull)
                self.assertEqual(capacity, len(buf))
        exp = np.arange(n_samples-capacity, n_samples)
        exp_a0 = np.roll(exp, n_samples % capacity)
        exp_a1 = exp_a0 + 1
        self.assertArrayEqual(buf.data['a'][0], exp_a0)
        self.assertArrayEqual(buf.data['a'][1], exp_a1)
        # test sample (batch=None)
        batch = samp()
        self.assertArrayEqual((capacity,), batch['a'][0].shape)
        self.assertArrayEqual((capacity,), batch['a'][1].shape)
        self.assertArrayEqual(batch['a'][0], buf[samp._cached_inds]['a'][0])
        self.assertArrayEqual(batch['a'][1], buf[samp._cached_inds]['a'][1])
        # test sample (batch=3)
        batch_size = 3
        batch = samp(batch_size=batch_size)
        self.assertArrayEqual((batch_size,), batch['a'][0].shape)
        self.assertArrayEqual((batch_size,), batch['a'][1].shape)
        self.assertArrayEqual(batch['a'][0], buf[samp._cached_inds]['a'][0])
        self.assertArrayEqual(batch['a'][1], buf[samp._cached_inds]['a'][1])
        # test sample (batch=3, seq=2)
        batch_size = 3
        seq_len = 2
        batch = samp(batch_size=batch_size, seq_len=seq_len)
        self.assertArrayEqual((batch_size, seq_len), batch['a'][0].shape)
        self.assertArrayEqual((batch_size, seq_len), batch['a'][1].shape)
        self.assertArrayEqual(batch['a'][0], buf[samp._cached_inds]['a'][0])
        self.assertArrayEqual(batch['a'][1], buf[samp._cached_inds]['a'][1])

    def test_uniform_sampler_with_nested_replay_buffer_multidim(self):
        capacity = 10
        n_samples = 15 # test circular
        batch = 2
        dim = 2
        buf = ub_data.NestedReplayBuffer(capacity)
        samp = ub_data.UniformSampler(buf)
        data = np.arange(n_samples*batch*dim).reshape((n_samples, batch, dim))
        for i in range(n_samples):
            buf.add({'a': data[i]})
        data = data.reshape((n_samples*batch, dim))
        exp = data[-capacity:]
        self.assertArrayEqual((capacity, dim), buf.data['a'].shape)
        self.assertArrayEqual(exp, buf.data['a'])
        # test sample
        batch = samp()
        self.assertArrayEqual((capacity, dim), batch['a'].shape)
        self.assertArrayEqual(batch['a'], buf[samp._cached_inds]['a'])
        # test sample (batch=3)
        batch_size = 3
        batch = samp(batch_size=batch_size)
        self.assertArrayEqual((batch_size, dim), batch['a'].shape)
        self.assertArrayEqual(batch['a'], buf[samp._cached_inds]['a'])
        # test sample (batch=3, seq=2)
        batch_size = 3
        seq_len = 2
        batch = samp(batch_size=batch_size, seq_len=seq_len)
        self.assertArrayEqual((batch_size, seq_len, dim), batch['a'].shape)
        self.assertArrayEqual(batch['a'], buf[samp._cached_inds]['a'])

    def test_permute_sampler_with_nested_replay_buffer(self):
        capacity = 10
        n_samples = 15 # test circular
        buf = ub_data.NestedReplayBuffer(capacity)
        samp = ub_data.PermuteSampler(buf)
        self.assertEqual(buf.capacity, capacity)
        self.assertTrue(buf.isnull)
        for i in range(n_samples):
            buf.add({'a': ([i], [i+1])})
            if i < capacity-1:
                self.assertFalse(buf.isfull)
                self.assertEqual(i+1, len(buf))
            else:
                self.assertTrue(buf.isfull)
                self.assertEqual(capacity, len(buf))
        exp = np.arange(n_samples-capacity, n_samples)
        exp_a0 = np.roll(exp, n_samples % capacity)
        exp_a1 = exp_a0 + 1
        self.assertArrayEqual(buf.data['a'][0], exp_a0)
        self.assertArrayEqual(buf.data['a'][1], exp_a1)
        # test sample (batch=None)
        batch_size = len(buf)
        batches = []
        indices = []
        for batch in samp():
            self.assertArrayEqual((batch_size,), batch['a'][0].shape)
            self.assertArrayEqual((batch_size,), batch['a'][1].shape)
            self.assertArrayEqual(buf[samp._cached_inds]['a'][0], batch['a'][0])
            self.assertArrayEqual(buf[samp._cached_inds]['a'][1], batch['a'][1])
            batches.append(batch)
            indices.append(samp._cached_inds)
        self.assertEqual(1, len(batches))
        unique, counts = np.unique(indices, return_counts=True)
        # check if contains all elements
        self.assertTrue(len(buf), len(unique))
        # check if all elements are sampled at least once
        self.assertTrue(np.all(counts == 1))
        # test sample (batch=3)
        batch_size = 3
        batches = []
        indices = []
        for batch in samp(batch_size=batch_size):
            self.assertArrayEqual((batch_size,), batch['a'][0].shape)
            self.assertArrayEqual((batch_size,), batch['a'][1].shape)
            self.assertArrayEqual(buf[samp._cached_inds]['a'][0], batch['a'][0])
            self.assertArrayEqual(buf[samp._cached_inds]['a'][1], batch['a'][1])
            batches.append(batch)
            indices.append(samp._cached_inds)
        self.assertEqual(4, len(batches)) # total samples == capacity
        unique, counts = np.unique(indices, return_counts=True)
        # check if contains all elements
        self.assertTrue(len(buf), len(unique))
        # check if all elements are sampled at least once but less than 2
        self.assertTrue(np.all(counts >= 1) and np.all(counts <= 2))

    def test_nested_replay_buffer_exception(self):
        with self.assertRaises(ValueError):
            # buffer_size <= 0
            buf = ub_data.NestedReplayBuffer(0)
        with self.assertRaises(ValueError):
            # not isinstance(buffer_size, int)
            buf = ub_data.NestedReplayBuffer(None)
        buf = ub_data.NestedReplayBuffer(1)
        self.assertTrue(buf.isnull)
        with self.assertRaises(RuntimeError):
            # raise RuntimeError('Buffer space not created')
            buf[0]
        # auto create space
        buf._set_data(1, indices=0)
        buf[0]

    def test_dict_replay_buffer(self):
        capacity = 10
        n_samples = 15 # test circular
        buf = ub_data.DictReplayBuffer(capacity)
        self.assertEqual(buf.capacity, capacity)
        self.assertTrue(buf.isnull)
        for i in range(n_samples):
            buf.add(a=[i], b=[i+1])
            if i < capacity-1:
                self.assertFalse(buf.isfull)
                self.assertEqual(i+1, len(buf))
            else:
                self.assertTrue(buf.isfull)
                self.assertEqual(capacity, len(buf))
        self.assertEqual(set(buf.keys()), set(['a', 'b']))
        self.assertFalse('c' in buf)
        exp = np.arange(n_samples-capacity, n_samples)
        exp_a = np.roll(exp, n_samples % capacity)
        exp_b = exp_a + 1
        self.assertArrayEqual(exp_a, buf.data['a'])
        self.assertArrayEqual(exp_b, buf.data['b'])
        # test update
        n = n_samples - capacity
        new_data = np.arange(n - n_samples % capacity, n)
        new_data = {'a': new_data, 'b': new_data+1}
        indices = np.arange(n_samples % capacity)
        buf.update(a=new_data['a'], indices=indices) # partial update
        n = n_samples - capacity - n_samples % capacity
        exp_a = np.arange(n, n + capacity)
        self.assertArrayEqual(exp_a, buf.data['a'])
        self.assertArrayEqual(exp_b, buf.data['b'])
        buf.update(b=new_data['b'], indices=indices) # partial update
        exp_b = exp_a + 1
        self.assertArrayEqual(exp_a, buf.data['a'])
        self.assertArrayEqual(exp_b, buf.data['b'])

    def test_uniform_sampler_with_dict_replay_buffer(self):
        capacity = 10
        n_samples = 15 # test circular
        buf = ub_data.DictReplayBuffer(capacity)
        samp = ub_data.UniformSampler(buf)
        self.assertEqual(buf.capacity, capacity)
        self.assertTrue(buf.isnull)
        for i in range(n_samples):
            buf.add(a=[i], b=[i+1])
            if i < capacity-1:
                self.assertFalse(buf.isfull)
                self.assertEqual(i+1, len(buf))
            else:
                self.assertTrue(buf.isfull)
                self.assertEqual(capacity, len(buf))
        self.assertEqual(set(buf.keys()), set(['a', 'b']))
        exp = np.arange(n_samples-capacity, n_samples)
        exp_a = np.roll(exp, n_samples % capacity)
        exp_b = exp_a + 1
        self.assertArrayEqual(buf.data['a'], exp_a)
        self.assertArrayEqual(buf.data['b'], exp_b)
        # test sample (batch=None)
        batch = samp()
        self.assertArrayEqual((capacity,), batch['a'].shape)
        self.assertArrayEqual((capacity,), batch['b'].shape)
        self.assertArrayEqual(batch['a'], buf[samp._cached_inds]['a'])
        self.assertArrayEqual(batch['b'], buf[samp._cached_inds]['b'])
        # test sample (batch=3)
        batch_size = 3
        batch = samp(batch_size=batch_size)
        self.assertArrayEqual((batch_size,), batch['a'].shape)
        self.assertArrayEqual((batch_size,), batch['b'].shape)
        self.assertArrayEqual(batch['a'], buf[samp._cached_inds]['a'])
        self.assertArrayEqual(batch['b'], buf[samp._cached_inds]['b'])
        # test update
        batch = samp(batch_size=batch_size)
        batch['a'] += 1
        batch['b'] += 2
        exp_a[samp._cached_inds] += 1
        exp_b[samp._cached_inds] += 2
        samp.update(a=batch['a'], b=batch['b'])
        self.assertArrayEqual(buf.data['a'], exp_a)
        self.assertArrayEqual(buf.data['b'], exp_b)
        # test sample (batch=3, seq=2)
        batch_size = 3
        seq_len = 2
        batch = samp(batch_size=batch_size, seq_len=seq_len)
        self.assertArrayEqual((batch_size, seq_len), batch['a'].shape)
        self.assertArrayEqual((batch_size, seq_len), batch['b'].shape)
        self.assertArrayEqual(batch['a'], buf[samp._cached_inds]['a'])
        self.assertArrayEqual(batch['b'], buf[samp._cached_inds]['b'])
        # test update
        batch['a'] += 1
        batch['b'] += 2
        exp_a[samp._cached_inds] += 1
        exp_b[samp._cached_inds] += 2
        samp.update(a=batch['a'], b=batch['b'])
        self.assertArrayEqual(buf.data['a'], exp_a)
        self.assertArrayEqual(buf.data['b'], exp_b)

    def test_dict_replay_buffer_exception(self):
        buf = ub_data.DictReplayBuffer(1)
        self.assertEqual([], buf.keys())
        self.assertFalse('k' in buf)
        with self.assertRaises(AssertionError):
            buf.add(a=[1])
            buf.add(b=[2]) # key not exists

    def test_sequential_replay_buffer(self):
        n_samples = 10
        buf = ub_data.SequentialReplayBuffer()
        for i in range(n_samples):
            buf.add(a=[i, i+n_samples])
        self.assertEqual(n_samples*2, len(buf))
        self.assertFalse(buf.ready_for_sample)
        self.assertFalse(buf.isfull)
        self.assertEqual(np.inf, buf.capacity)
        buf.make() 
        self.assertEqual(n_samples*2, len(buf))
        self.assertTrue(buf.ready_for_sample)
        self.assertFalse(buf.isfull)
        exp = np.arange(n_samples*2)
        self.assertArrayEqual(exp, buf.data['a'])
        # test getitem
        data = buf[np.arange(n_samples//2)]
        exp = np.arange(n_samples//2)
        self.assertArrayEqual(exp, data['a'])
        # test setitem
        new_data = np.arange(n_samples//2)+1
        buf[np.arange(n_samples//2)] = {'a':new_data}
        exp = np.arange(n_samples*2)
        exp[:n_samples//2] = np.arange(n_samples//2)+1
        self.assertArrayEqual(exp, buf.data['a'])
        # test update (should have the same results as setitem)
        buf.update(a=np.arange(n_samples//2)+1, indices=np.arange(n_samples//2))
        self.assertArrayEqual(exp, buf.data['a'])

    def test_sequential_replay_buffer_multidim(self):
        n_samples = 10
        buf = ub_data.SequentialReplayBuffer()
        for i in range(n_samples):
            buf.add(a=[[i, i+n_samples]])
        self.assertEqual(n_samples, len(buf))
        self.assertFalse(buf.ready_for_sample)
        self.assertFalse(buf.isfull)
        buf.make()
        self.assertEqual(n_samples, len(buf))
        self.assertTrue(buf.ready_for_sample)
        self.assertFalse(buf.isfull)
        exp = np.arange(n_samples*2).reshape(2, n_samples).T
        self.assertArrayEqual(exp, buf.data['a'])
        # test getitem
        data = buf[np.arange(n_samples//2)]
        exp1 = np.arange(n_samples//2)
        exp2 = exp1+n_samples
        exp = np.asarray([exp1, exp2]).T
        self.assertArrayEqual(exp, data['a'])
        # test setitem
        new_data = exp + 1
        buf[np.arange(n_samples//2)] = {'a':new_data}
        exp1 = np.arange(n_samples*2).reshape(2, n_samples).T
        exp1[:n_samples//2] = exp + 1
        exp = exp1
        self.assertArrayEqual(exp, buf.data['a'])
        # test update (should have the same results as setitem)
        buf.update(a=new_data, indices=np.arange(n_samples//2))
        self.assertArrayEqual(exp, buf.data['a'])

    def test_sequential_replay_buffer_exception(self):
        buf = ub_data.SequentialReplayBuffer()
        buf.add(a=[1])
        buf.make() # ready for sample
        with self.assertRaises(RuntimeError):
            # make twice: "The buffer has already made."
            buf.make()
        with self.assertRaises(RuntimeError):
            # "The buffer can not add data after calling `buffer.make()`"
            buf.add(a=[2])
        buf = ub_data.SequentialReplayBuffer()
        with self.assertRaises(AssertionError):
            # rank must > 0
            buf.add(a=1)
        buf = ub_data.SequentialReplayBuffer()
        buf.add(a=[1])
        with self.assertRaises(RuntimeError):
            buf.update(a=1, indices=0)

    def test_permute_sampler_with_sequential_replay_buffer(self):
        n_samples = 10
        buf = ub_data.SequentialReplayBuffer()
        samp = ub_data.PermuteSampler(buf)
        for i in range(n_samples):
            buf.add(a=[i, i+n_samples])
        self.assertEqual(n_samples*2, len(buf))
        self.assertFalse(buf.ready_for_sample)
        self.assertFalse(buf.isfull)
        buf.make() 
        self.assertEqual(n_samples*2, len(buf))
        self.assertTrue(buf.ready_for_sample)
        self.assertFalse(buf.isfull)
        exp = np.arange(n_samples*2)
        self.assertArrayEqual(exp, buf.data['a'])
        # test sample (batch=None)
        batch_size = len(buf)
        batches = []
        indices = []
        for batch in samp():
            self.assertArrayEqual((batch_size,), batch['a'].shape)
            self.assertArrayEqual(buf[samp._cached_inds]['a'], batch['a'])
            batches.append(batch)
            indices.append(samp._cached_inds)
        self.assertEqual(1, len(batches))
        unique, counts = np.unique(indices, return_counts=True)
        # check if contains all elements
        self.assertTrue(len(buf), len(unique))
        # check if all elements are sampled at least once
        self.assertTrue(np.all(counts == 1))
        # test sample (batch=3)
        batch_size = 3
        batches = []
        indices = []
        for batch in samp(batch_size=batch_size):
            self.assertArrayEqual((batch_size,), batch['a'].shape)
            self.assertArrayEqual(buf[samp._cached_inds]['a'], batch['a'])
            batches.append(batch)
            indices.append(samp._cached_inds)
        self.assertEqual(7, len(batches)) # total samples == 20
        unique, counts = np.unique(indices, return_counts=True)
        # check if contains all elements
        self.assertTrue(len(buf), len(unique))
        # check if all elements are sampled at least once but less than 2
        self.assertTrue(np.all(counts >= 1) and np.all(counts <= 2))
        # test update
        batch_size = 3
        batch = next(iter(samp(batch_size=batch_size)))
        exp = np.arange(n_samples*2)
        batch['a'] += 1
        exp[samp._cached_inds] += 1
        samp.update(a=batch['a'])
        self.assertArrayEqual(exp, buf.data['a'])


    def test_sampler_exception(self):
        with self.assertRaises(ValueError):
            # `buffer` must be an instance of BaseBuffer
            ub_data.UniformSampler(None)

    def test_compute_advantage(self):
        rew = np.asarray([0, 0, 1, 0, 1], dtype=np.float32)
        val = np.asarray([.0, .1, .5, .4, .5], dtype=np.float32)
        done = np.asarray([0, 0, 1, 0, 0], dtype=np.bool_)
        adv = ub_data.compute_advantage(rew=rew, val=val, done=done, 
                                    gamma=0.99, gae_lambda=0.95)
        # openai baselines style GAE
        exp = np.asarray([0.00495, -0.1, 1.865465, 1.0307975, 0.995],
                        dtype=np.float32)
        # tianshou style GAE:
        # exp = np.asarray([1.26304564, 1.23768806, 0.89600003, 0.56525001, 0.5],
        #                 dtype=np.float32)
        self.assertAllClose(exp, adv)