# --- built in ---
import os
import sys
import math
import time
import logging
import unittest

# --- 3rd party ---
import numpy as np

# --- my module ---
from unstable_baselines.lib import utils as ub_utils
from unstable_baselines.lib import data as ub_data
from test.utils import TestCase

class TestDataModule(TestCase):
    '''Test unstable_baselines.data module
    '''
    def test_segment_tree(self):
        tree = ub_data.SegmentTree(10)
        # test getitem/setitem
        tree[:] = np.arange(1, 11)
        self.assertArrayEqual(np.arange(1, 11), tree[:])
        tree[np.arange(3)] = [3, 2, 1]
        self.assertArrayEqual([3, 2, 1, 4], tree[np.arange(4)])
        # test sum
        tree[:] = np.arange(1, 11)
        self.assertEqual(55, tree.sum())
        self.assertEqual(5, tree.sum(start=1, end=3))
        self.assertEqual(7, tree.sum(start=2, end=4))
        self.assertEqual(44, tree.sum(start=1, end=-1))
        # test update
        tree.update(1, 3)
        self.assertEqual(6, tree.sum(start=1, end=3))
        # test index
        tree[:] = np.arange(1, 11)
        self.assertEqual(0, tree.index(1))
        self.assertEqual(1, tree.index(2))
        self.assertEqual(1, tree.index(3))
        self.assertEqual(2, tree.index(4))
        self.assertEqual(3, tree.index(7))
        self.assertEqual(8, tree.index(45))
        self.assertEqual(9, tree.index(54))
        # test index vectorized
        self.assertArrayEqual([[0], [9]], tree.index([[1], [46]]))

    def test_base_buffer(self):
        capacity = 10
        batch = 1
        n_samples = 15 # test circular
        buf = ub_data.BaseBuffer(capacity, batch=batch)
        self.assertEqual(capacity, buf.capacity)
        self.assertEqual(capacity, buf.slots)
        self.assertEqual(batch, buf.batch)
        self.assertEqual(0, buf.head)
        self.assertEqual(0, buf.tail)
        self.assertTrue(buf.isnull)
        self.assertFalse(buf.isfull)
        self.assertTrue(buf.ready_for_sample)
        for i in range(n_samples):
            buf.add({'a': ([i], [i+1])})
            if i < capacity-1:
                self.assertFalse(buf.isfull)
                self.assertEqual(i+1, len(buf))
                self.assertEqual(i+1, buf.len_slots())
                self.assertEqual(0, buf.head)
            else:
                self.assertTrue(buf.isfull)
                self.assertEqual(capacity, len(buf))
                self.assertEqual(capacity, buf.len_slots())
            self.assertEqual((i+1)%capacity, buf.tail)
        exp = np.arange(n_samples-capacity, n_samples)
        exp_a0 = np.roll(exp, n_samples % capacity)
        exp_a1 = exp_a0 + 1
        exp_a0 = np.expand_dims(exp_a0, axis=-1)
        exp_a1 = np.expand_dims(exp_a1, axis=-1)
        self.assertArrayEqual(exp_a0, buf.data['a'][0])
        self.assertArrayEqual(exp_a1, buf.data['a'][1])
        # test getitem
        data = buf[np.arange(n_samples % capacity)]
        exp_a0 = np.arange(n_samples - n_samples % capacity, n_samples)
        exp_a1 = exp_a0 + 1
        exp_a0 = np.expand_dims(exp_a0, axis=-1)
        exp_a1 = np.expand_dims(exp_a1, axis=-1)
        self.assertArrayEqual(exp_a0, data['a'][0])
        self.assertArrayEqual(exp_a1, data['a'][1])
        # test setitem
        n = n_samples - capacity
        new_data = np.arange(n - n_samples % capacity, n)
        new_data = np.expand_dims(new_data, axis=-1)
        new_data = {'a': (new_data, new_data+1)}
        buf[np.arange(n_samples % capacity)] = new_data
        n = n_samples - capacity - n_samples % capacity
        exp_a0 = np.arange(n, n + capacity)
        exp_a1 = exp_a0 + 1
        exp_a0 = np.expand_dims(exp_a0, axis=-1)
        exp_a1 = np.expand_dims(exp_a1, axis=-1)
        self.assertArrayEqual(exp_a0, buf.data['a'][0])
        self.assertArrayEqual(exp_a1, buf.data['a'][1])
        # test update (should have the same results as setitem)
        buf.update(new_data, indices=np.arange(n_samples % capacity))
        self.assertArrayEqual(exp_a0, buf.data['a'][0])
        self.assertArrayEqual(exp_a1, buf.data['a'][1])
        # test ravel/unravel index
        def test_ravel(indices):
            self.assertArrayEqual(
                np.ravel_multi_index(indices, (buf.slots, buf.batch)),
                buf.ravel_index(indices))
        test_ravel(([1, 2, 3], 0))
        test_ravel(([1, 2, 3], [0]))
        def test_unravel(indices):
            self.assertArrayEqual(
                np.unravel_index(indices, (buf.slots, buf.batch)),
                buf.unravel_index(indices))
        test_unravel([4, 5, 6])
        test_unravel(7)


    def test_base_buffer_multidim(self):
        capacity = 20
        batch = 2
        dim = 2
        n_samples = 15 # test circular
        buf = ub_data.BaseBuffer(capacity, batch=batch)
        data = np.arange(n_samples*batch*dim).reshape((n_samples, batch, dim))
        for i in range(n_samples):
            buf.add({'a': data[i]})
            if (i+1)*batch < capacity:
                self.assertFalse(buf.isfull)
                self.assertEqual((i+1)*batch, len(buf))
                self.assertEqual(i+1, buf.len_slots())
                self.assertEqual(0, buf.head)
            else:
                self.assertTrue(buf.isfull)
                self.assertEqual(capacity, len(buf))
                self.assertEqual(capacity//batch, buf.len_slots())
            self.assertEqual((i+1)%(capacity//batch), buf.tail)
        exp = np.arange(n_samples*batch*dim-capacity*dim, n_samples*batch*dim)
        exp = exp.reshape(-1, 2, 2)
        exp = np.roll(exp, n_samples % (capacity//batch), axis=0)
        self.assertArrayEqual(exp, buf.data['a'])
        # test ravel/unravel index
        def test_ravel(indices):
            self.assertArrayEqual(
                np.ravel_multi_index(indices, (buf.slots, buf.batch)),
                buf.ravel_index(indices))
        test_ravel(([1, 2, 3], 0))
        test_ravel(([[1], [2], [3]], [0, 1]))
        def test_unravel(indices):
            self.assertArrayEqual(
                np.unravel_index(indices, (buf.slots, buf.batch)),
                buf.unravel_index(indices))
        test_unravel([4, 5, 6])
        test_unravel(7)

    def test_base_buffer_auto_calc_space(self):
        capacity = 10
        batch = 1
        buf = ub_data.BaseBuffer(capacity, batch=batch)
        self.assertEqual(0, len(buf))
        self.assertEqual(0, buf.len_slots())
        self.assertEqual(capacity, buf.capacity)
        self.assertEqual(capacity, buf.slots)
        self.assertEqual(batch, buf.batch)
        self.assertEqual(0, buf.head)
        self.assertEqual(0, buf.tail)
        self.assertTrue(buf.isnull)
        self.assertFalse(buf.isfull)
        self.assertTrue(buf.ready_for_sample)
        capacity = 10
        n_samples = 15 # test circular
        buf = ub_data.BaseBuffer(capacity, batch=None)
        self.assertEqual(0, len(buf))
        self.assertEqual(0, buf.len_slots())
        self.assertEqual(None, buf.capacity)
        self.assertEqual(None, buf.slots)
        self.assertEqual(None, buf.batch)
        self.assertEqual(0, buf.head)
        self.assertEqual(0, buf.tail)
        self.assertTrue(buf.isnull)
        self.assertFalse(buf.isfull)
        self.assertTrue(buf.ready_for_sample)
        buf.add({'a': [0, 1]})
        self.assertEqual(2, len(buf))
        self.assertEqual(1, buf.len_slots())
        self.assertEqual(capacity, buf.capacity)
        self.assertEqual(math.ceil(capacity/2), buf.slots)
        self.assertEqual(2, buf.batch)
        self.assertEqual(0, buf.head)
        self.assertEqual(1, buf.tail)
        self.assertFalse(buf.isnull)
        self.assertFalse(buf.isfull)
        self.assertTrue(buf.ready_for_sample)

    def test_base_buffer_relative_index(self):
        capacity = 10
        batch = 1
        n_samples = 15 # test circular
        buf = ub_data.BaseBuffer(capacity, batch=batch)
        for i in range(n_samples):
            buf.add({'a': ([i], [i+1])})
        head = n_samples%capacity
        self.assertEqual(head, buf.head)
        self.assertEqual(head, buf.tail)
        # test int, slice key
        data = buf.rel[1]
        self.assertArrayEqual([head+1], data['a'][0])
        self.assertArrayEqual([head+2], data['a'][1])
        data = buf.rel[-1]
        self.assertArrayEqual([n_samples-1], data['a'][0])
        self.assertArrayEqual([n_samples], data['a'][1])
        data = buf.rel[1:3]
        exp = np.arange(2).reshape(-1, 1)
        self.assertArrayEqual(exp+head+1, data['a'][0])
        self.assertArrayEqual(exp+head+2, data['a'][1])
        data = buf.rel[-3:-1]
        exp = np.arange(2, 0, -1).reshape(-1, 1)
        self.assertArrayEqual(n_samples-1-exp, data['a'][0])
        self.assertArrayEqual(n_samples-exp, data['a'][1])
        data = buf.rel[-1:1]
        self.assertEqual((0, 1), data['a'][0].shape)
        self.assertEqual((0, 1), data['a'][1].shape)
        # test tuple key
        data = buf.rel[1, 0]
        self.assertArrayEqual(head+1, data['a'][0])
        self.assertArrayEqual(head+2, data['a'][1])
        data = buf.rel[-1, 0]
        self.assertArrayEqual(n_samples-1, data['a'][0])
        self.assertArrayEqual(n_samples, data['a'][1])
        data = buf.rel[1:3, 0]
        exp = np.arange(2)
        self.assertArrayEqual(exp+head+1, data['a'][0])
        self.assertArrayEqual(exp+head+2, data['a'][1])
        data = buf.rel[-3:-1, 0]
        exp = np.arange(2, 0, -1)
        self.assertArrayEqual(n_samples-1-exp, data['a'][0])
        self.assertArrayEqual(n_samples-exp, data['a'][1])
        data = buf.rel[-1:1, 0]
        self.assertEqual((0,), data['a'][0].shape)
        self.assertEqual((0,), data['a'][1].shape)
        # test list key
        data = buf.rel[[1,3]]
        self.assertArrayEqual([[head+1], [head+3]], data['a'][0])
        self.assertArrayEqual([[head+2], [head+4]], data['a'][1])
        # test np key
        data = buf.rel[np.asarray([1,3])]
        self.assertArrayEqual([[head+1], [head+3]], data['a'][0])
        self.assertArrayEqual([[head+2], [head+4]], data['a'][1])
        # test index out of range
        with self.assertRaises(IndexError):
            buf.rel[capacity+1]

    def test_base_buffer_shape(self):
        capacity = 10
        batch = 3
        n_samples = 15
        buf = ub_data.BaseBuffer(capacity, batch=batch)
        for i in range(n_samples):
            buf.add({'a': np.arange(batch)})
        self.assertArrayEqual((3,), buf[1]['a'].shape)
        self.assertArrayEqual((2,3), buf[1:3]['a'].shape)
        self.assertArrayEqual((2,), buf[1:3, 0]['a'].shape)
        self.assertArrayEqual((2,2), buf[1:3, :2]['a'].shape)
        self.assertArrayEqual((3,), buf.rel[1]['a'].shape)
        self.assertArrayEqual((2,3), buf.rel[1:3]['a'].shape)
        self.assertArrayEqual((2,), buf.rel[1:3, 0]['a'].shape)
        self.assertArrayEqual((2,2), buf.rel[1:3, :2]['a'].shape)

    def test_base_buffer_exception(self):
        with self.assertRaises(ValueError):
            # size <= 0
            ub_data.BaseBuffer(0, 1)
        with self.assertRaises(AssertionError):
            # batch <= 0
            ub_data.BaseBuffer(1, 0)
        buf = ub_data.BaseBuffer(1, 1)
        self.assertTrue(buf.isnull)
        with self.assertRaises(AssertionError):
            # AssertionError: Buffer space not created
            buf[0]
        # auto create space
        buf._set_data(1, indices=0)
        buf[0]

    def test_replay_buffer(self):
        capacity = 10
        batch = 1
        n_samples = 15 # test circular
        buf = ub_data.ReplayBuffer(capacity, batch=batch)
        self.assertEqual(capacity, buf.capacity)
        self.assertEqual(capacity, buf.slots)
        self.assertEqual(batch, buf.batch)
        self.assertEqual(0, buf.head)
        self.assertEqual(0, buf.tail)
        self.assertTrue(buf.isnull)
        self.assertFalse(buf.isfull)
        self.assertTrue(buf.ready_for_sample)
        for i in range(n_samples):
            buf.add(a=[i], b=[i+1])
            if i < capacity-1:
                self.assertFalse(buf.isfull)
                self.assertEqual(i+1, len(buf))
                self.assertEqual(i+1, buf.len_slots())
                self.assertEqual(0, buf.head)
            else:
                self.assertTrue(buf.isfull)
                self.assertEqual(capacity, len(buf))
                self.assertEqual(capacity, buf.len_slots())
            self.assertEqual((i+1)%capacity, buf.tail)
        self.assertEqual(set(buf.keys()), set(['a', 'b']))
        self.assertFalse('c' in buf)
        exp = np.arange(n_samples-capacity, n_samples)
        exp_a0 = np.roll(exp, n_samples % capacity)
        exp_a1 = exp_a0 + 1
        exp_a0 = np.expand_dims(exp_a0, axis=-1)
        exp_a1 = np.expand_dims(exp_a1, axis=-1)
        self.assertArrayEqual(exp_a0, buf.data['a'])
        self.assertArrayEqual(exp_a1, buf.data['b'])
        # test getitem
        data = buf[np.arange(n_samples % capacity)]
        exp_a0 = np.arange(n_samples - n_samples % capacity, n_samples)
        exp_a1 = exp_a0 + 1
        exp_a0 = np.expand_dims(exp_a0, axis=-1)
        exp_a1 = np.expand_dims(exp_a1, axis=-1)
        self.assertArrayEqual(exp_a0, data['a'])
        self.assertArrayEqual(exp_a1, data['b'])
        # test setitem
        n = n_samples - capacity
        new_data = np.arange(n - n_samples % capacity, n)
        new_data = np.expand_dims(new_data, axis=-1)
        new_data = {'a': new_data, 'b': new_data+1}
        buf[np.arange(n_samples % capacity)] = new_data
        n = n_samples - capacity - n_samples % capacity
        exp_a0 = np.arange(n, n + capacity)
        exp_a1 = exp_a0 + 1
        exp_a0 = np.expand_dims(exp_a0, axis=-1)
        exp_a1 = np.expand_dims(exp_a1, axis=-1)
        self.assertArrayEqual(exp_a0, buf.data['a'])
        self.assertArrayEqual(exp_a1, buf.data['b'])
        # test update (should have the same results as setitem)
        buf.update(a=new_data['a'], b=new_data['b'], 
                  indices=np.arange(n_samples % capacity))
        self.assertArrayEqual(exp_a0, buf.data['a'])
        self.assertArrayEqual(exp_a1, buf.data['b'])

    def test_replay_buffer_exception(self):
        buf = ub_data.ReplayBuffer(1, 1)
        self.assertEqual([], buf.keys())
        self.assertFalse('k' in buf)
        with self.assertRaises(AssertionError):
            buf.add(a=[1])
            buf.add(b=[2]) # key not exists

    def test_dynamic_buffer(self):
        n_samples = 10
        buf = ub_data.DynamicBuffer(batch=2)
        self.assertEqual(np.inf, buf.capacity)
        self.assertEqual(0, buf.head)
        for i in range(n_samples):
            buf.add(a=[i, i+n_samples])
        self.assertEqual(n_samples*2, len(buf))
        self.assertEqual(n_samples, buf.len_slots())
        self.assertEqual(n_samples, buf.slots)
        self.assertFalse(buf.ready_for_sample)
        self.assertFalse(buf.isfull)
        buf.make()
        self.assertEqual(n_samples*2, len(buf))
        self.assertTrue(buf.ready_for_sample)
        self.assertFalse(buf.isfull)
        exp = np.arange(n_samples*2).reshape(2, -1).T
        self.assertArrayEqual(exp, buf.data['a'])
        # test getitem
        data = buf[np.arange(n_samples//2)]
        exp = np.c_[np.arange(n_samples//2), np.arange(n_samples//2)+n_samples]
        self.assertArrayEqual(exp, data['a'])
        # test setitem
        new_data = exp+1
        buf[np.arange(n_samples//2)] = {'a':new_data}
        exp = np.c_[np.arange(n_samples), np.arange(n_samples)+n_samples]
        exp[:n_samples//2] += 1
        self.assertArrayEqual(exp, buf.data['a'])
        # test update (should have the same results as setitem)
        buf.update(a=new_data, indices=np.arange(n_samples//2))
        self.assertArrayEqual(exp, buf.data['a'])

    def test_dynamic_buffer_auto_calc_space(self):
        batch = 2
        buf = ub_data.DynamicBuffer(batch=batch)
        self.assertEqual(0, len(buf))
        self.assertEqual(0, buf.len_slots())
        self.assertEqual(np.inf, buf.capacity)
        self.assertEqual(0, buf.slots)
        self.assertEqual(batch, buf.batch)
        self.assertEqual(0, buf.head)
        self.assertEqual(0, buf.tail)
        self.assertTrue(buf.isnull)
        self.assertFalse(buf.isfull)
        self.assertFalse(buf.ready_for_sample)
        # auto calc space
        buf = ub_data.DynamicBuffer(batch=None)
        self.assertEqual(0, len(buf))
        self.assertEqual(0, buf.len_slots())
        self.assertEqual(np.inf, buf.capacity)
        self.assertEqual(0, buf.slots)
        self.assertEqual(None, buf.batch)
        self.assertEqual(0, buf.head)
        self.assertEqual(0, buf.tail)
        self.assertTrue(buf.isnull)
        self.assertFalse(buf.isfull)
        self.assertFalse(buf.ready_for_sample)
        buf.add(a=[0, 1])
        self.assertEqual(2, len(buf))
        self.assertEqual(1, buf.len_slots())
        self.assertEqual(np.inf, buf.capacity)
        self.assertEqual(1, buf.slots)
        self.assertEqual(2, buf.batch)
        self.assertEqual(0, buf.head)
        self.assertEqual(1, buf.tail)
        self.assertFalse(buf.isnull)
        self.assertFalse(buf.isfull)
        self.assertFalse(buf.ready_for_sample)

    def test_dynamic_buffer_exception(self):
        buf = ub_data.DynamicBuffer(batch=1)
        buf.add(a=[1])
        self.assertFalse(buf.ready_for_sample)
        with self.assertRaises(RuntimeError):
            # not ready for sample
            buf.update(indices=0, a=[2])
        buf.make()
        with self.assertRaises(RuntimeError):
            # make twice: "The buffer has already made."
            buf.make()
        with self.assertRaises(RuntimeError):
            # "The buffer can not add data after calling `buffer.make()`"
            buf.add(a=[2])
        with self.assertRaises(RuntimeError):
            buf._append_data({'a': [1]})

    def test_uniform_sampler_with_base_buffer(self):
        capacity = 10
        batch = 1
        n_samples = 15
        buf = ub_data.BaseBuffer(capacity, batch=batch)
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
        exp_a0 = np.expand_dims(exp_a0, axis=-1)
        exp_a1 = np.expand_dims(exp_a1, axis=-1)
        self.assertArrayEqual(buf.data['a'][0], exp_a0)
        self.assertArrayEqual(buf.data['a'][1], exp_a1)
        # test sample (batch=None)
        batch = samp()
        self.assertArrayEqual((capacity,), batch['a'][0].shape)
        self.assertArrayEqual((capacity,), batch['a'][1].shape)
        self.assertArrayEqual(batch['a'][0], buf[samp.indices]['a'][0])
        self.assertArrayEqual(batch['a'][1], buf[samp.indices]['a'][1])
        # test sample (batch=3)
        batch_size = 3
        batch = samp(batch_size=batch_size)
        self.assertArrayEqual((batch_size,), batch['a'][0].shape)
        self.assertArrayEqual((batch_size,), batch['a'][1].shape)
        self.assertArrayEqual(batch['a'][0], buf[samp.indices]['a'][0])
        self.assertArrayEqual(batch['a'][1], buf[samp.indices]['a'][1])
        # test sample (batch=3, seq=2)
        batch_size = 3
        seq_len = 2
        batch = samp(batch_size=batch_size, seq_len=seq_len)
        self.assertArrayEqual((batch_size, seq_len), batch['a'][0].shape)
        self.assertArrayEqual((batch_size, seq_len), batch['a'][1].shape)
        self.assertArrayEqual(batch['a'][0], buf[samp.indices]['a'][0])
        self.assertArrayEqual(batch['a'][1], buf[samp.indices]['a'][1])
        # test update
        batch['a'] = (np.zeros_like(batch['a'][0]), 
                      np.zeros_like(batch['a'][1]))
        samp.update(batch)
        self.assertArrayEqual(buf[samp.indices]['a'][0], batch['a'][0])
        self.assertArrayEqual(buf[samp.indices]['a'][1], batch['a'][1])


    def test_uniform_sampler_with_base_buffer_rel(self):
        capacity = 10
        batch = 1
        n_samples = 15
        buf = ub_data.BaseBuffer(capacity, batch=batch)
        samp = ub_data.UniformSampler(buf)
        for i in range(n_samples):
            buf.add({'a': ([i], [i+1])})
        # test relative indexing
        inds1 = np.arange(3)
        inds2 = np.zeros(3, dtype=np.int64)
        samp._cached_inds = (inds1, inds2)
        self.assertArrayEqual(buf[samp.indices]['a'][0], samp.rel[0]['a'][0])
        self.assertArrayEqual(buf[samp.indices]['a'][1], samp.rel[0]['a'][1])
        self.assertArrayEqual(buf[(inds1-3, inds2)]['a'][0], samp.rel[-3]['a'][0])
        self.assertArrayEqual(buf[(inds1-3, inds2)]['a'][1], samp.rel[-3]['a'][1])
        self.assertArrayEqual(buf[(inds1+3, inds2)]['a'][0], samp.rel[3]['a'][0])
        self.assertArrayEqual(buf[(inds1+3, inds2)]['a'][1], samp.rel[3]['a'][1])
        add = np.array([[1, 2]], dtype=np.int64).T
        self.assertArrayEqual(buf[(inds1+add, inds2)]['a'][0], samp.rel[1:3]['a'][0])
        self.assertArrayEqual(buf[(inds1+add, inds2)]['a'][1], samp.rel[1:3]['a'][1])
        add = np.array([[-3, -2]], dtype=np.int64).T
        self.assertArrayEqual(buf[(inds1+add, inds2)]['a'][0], samp.rel[-3:-1]['a'][0])
        self.assertArrayEqual(buf[(inds1+add, inds2)]['a'][1], samp.rel[-3:-1]['a'][1])
        # test shape
        self.assertArrayEqual((0, 3), samp.rel[-1:1]['a'][0].shape)
        # test setitem
        add = np.array([[1, 2]], dtype=np.int64).T
        batch = samp.rel[1:3]
        batch['a'] = (np.zeros_like(batch['a'][0]),
                        np.zeros_like(batch['a'][1]))
        samp.rel[1:3] = batch
        self.assertTrue(np.all(samp.rel[1:3]['a'][0] == 0))
        self.assertTrue(np.all(samp.rel[1:3]['a'][1] == 0))

    def test_permute_sampler_with_base_buffer(self):
        capacity = 10
        batch = 1
        n_samples = 15
        buf = ub_data.BaseBuffer(capacity, batch=batch)
        samp = ub_data.PermuteSampler(buf)
        for i in range(n_samples):
            buf.add({'a': ([i], [i+1])})
        # test sample (batch=None)
        batch_size = len(buf)
        batches = []
        indices = []
        for batch in samp():
            self.assertArrayEqual((batch_size,), batch['a'][0].shape)
            self.assertArrayEqual((batch_size,), batch['a'][1].shape)
            self.assertArrayEqual(buf[samp.indices]['a'][0], batch['a'][0])
            self.assertArrayEqual(buf[samp.indices]['a'][1], batch['a'][1])
            batches.append(batch)
            indices.append(
                np.ravel_multi_index(samp.indices, 
                            (buf.len_slots(), buf.batch))
            )
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
            self.assertArrayEqual(buf[samp.indices]['a'][0], batch['a'][0])
            self.assertArrayEqual(buf[samp.indices]['a'][1], batch['a'][1])
            batches.append(batch)
            indices.append(
                np.ravel_multi_index(samp.indices, 
                            (buf.len_slots(), buf.batch))
            )
        self.assertEqual(4, len(batches)) # total samples == capacity
        unique, counts = np.unique(indices, return_counts=True)
        # check if contains all elements
        self.assertTrue(len(buf), len(unique))
        # check if all elements are sampled at least once but less than 2
        self.assertTrue(np.all(counts >= 1))
        self.assertTrue(np.all(counts <= 2))
        # test sample (batch=3, seq_len=2)
        batch_size = 3
        seq_len = 2
        batches = []
        indices = []
        for batch in samp(batch_size=batch_size, seq_len=seq_len):
            self.assertArrayEqual((batch_size, seq_len), batch['a'][0].shape)
            self.assertArrayEqual((batch_size, seq_len), batch['a'][1].shape)
            self.assertArrayEqual(buf[samp.indices]['a'][0], batch['a'][0])
            self.assertArrayEqual(buf[samp.indices]['a'][1], batch['a'][1])
            batches.append(batch)
            indices.append(
                np.ravel_multi_index(samp.indices, 
                            (buf.len_slots(), buf.batch))
            )
        self.assertEqual(3, len(batches)) # total samples == capacity
        unique, counts = np.unique(indices, return_counts=True)
        # check if contains all elements
        self.assertTrue(len(buf), len(unique))
        # check if all elements are sampled at least once but less than 3
        self.assertTrue(np.all(counts >= 1))
        self.assertTrue(np.all(counts <= 4))
        
    def test_uniform_sampler_with_dynamic_buffer(self):
        n_samples = 10
        batch_ = 2
        buf = ub_data.DynamicBuffer(batch_)
        samp = ub_data.UniformSampler(buf)
        for i in range(n_samples):
            samp.add(a=[i, i+n_samples])
        self.assertEqual(n_samples*2, len(buf))
        self.assertFalse(buf.ready_for_sample)
        self.assertFalse(buf.isfull)
        # test sample (batch=None)
        with self.assertRaises(RuntimeError):
            # buffer is not ready for sampling
            samp()
        buf.make()
        # test sample (batch=None)
        batch = samp()
        self.assertArrayEqual((n_samples*batch_,), batch['a'].shape)
        self.assertArrayEqual(batch['a'], buf[samp.indices]['a'])
        # test sample (batch=3)
        batch_size = 3
        batch = samp(batch_size=batch_size)
        self.assertArrayEqual((batch_size,), batch['a'].shape)
        self.assertArrayEqual(batch['a'], buf[samp.indices]['a'])
        # test sample (batch=3, seq=2)
        batch_size = 3
        seq_len = 2
        batch = samp(batch_size=batch_size, seq_len=seq_len)
        self.assertArrayEqual((batch_size, seq_len), batch['a'].shape)
        self.assertArrayEqual(batch['a'], buf[samp.indices]['a'])
        # test update
        batch['a'] = np.zeros_like(batch['a'])
        samp.update(a=batch['a'])
        self.assertArrayEqual(buf[samp.indices]['a'], batch['a'])

    def test_permute_sampler_with_dynamic_buffer(self):
        n_samples = 10
        batch_ = 2
        buf = ub_data.DynamicBuffer(batch_)
        samp = ub_data.PermuteSampler(buf)
        for i in range(n_samples):
            buf.add(a=[i, i+n_samples])
        self.assertEqual(n_samples*2, len(buf))
        self.assertFalse(buf.ready_for_sample)
        self.assertFalse(buf.isfull)
        # test sample (batch=None)
        with self.assertRaises(RuntimeError):
            # buffer is not ready for sampling
            samp()
        buf.make()
        # test sample (batch=None)
        batch_size = len(buf)
        batches = []
        indices = []
        ub_utils.set_seed(2)
        for batch in samp():
            self.assertArrayEqual((batch_size,), batch['a'].shape)
            self.assertArrayEqual(buf[samp.indices]['a'], batch['a'])
            batches.append(batch)
            indices.append(
                np.ravel_multi_index(samp.indices,
                            (buf.len_slots(), buf.batch))
            )
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
            self.assertArrayEqual(buf[samp.indices]['a'], batch['a'])
            batches.append(batch)
            indices.append(
                np.ravel_multi_index(samp.indices, 
                            (buf.len_slots(), buf.batch))
            )
        self.assertEqual(7, len(batches)) # total samples == capacity
        unique, counts = np.unique(indices, return_counts=True)
        # check if contains all elements
        self.assertTrue(len(buf), len(unique))
        # check if all elements are sampled at least once but less than 2
        self.assertTrue(np.all(counts >= 1))
        self.assertTrue(np.all(counts <= 2))
        # test sample (batch=3, seq_len=2)
        batch_size = 3
        seq_len = 2
        batches = []
        indices = []
        ub_utils.set_seed(2)
        for batch in samp(batch_size=batch_size, seq_len=seq_len):
            self.assertArrayEqual((batch_size, seq_len), batch['a'].shape)
            self.assertArrayEqual(buf[samp.indices]['a'], batch['a'])
            batches.append(batch)
            indices.append(
                np.ravel_multi_index(samp.indices, 
                            (buf.len_slots(), buf.batch))
            )
        self.assertEqual(6, len(batches)) # total samples == capacity
        unique, counts = np.unique(indices, return_counts=True)
        # check if contains all elements
        self.assertTrue(len(buf), len(unique))
        # check if all elements are sampled at least once but less than 3
        self.assertTrue(np.all(counts >= 1))
        self.assertTrue(np.all(counts <= 3))

    def test_prioritized_sampler_with_replay_buffer(self):
        capacity = 10
        batch = 1
        alpha = 1.0
        # test create space
        buf = ub_data.ReplayBuffer(capacity)
        self.assertTrue(buf.isnull)
        samp = ub_data.PrioritizedSampler(buf, alpha)
        self.assertTrue(samp._weight_tree is None)
        # add one sample to create space
        samp.add(i=[1])
        self.assertTrue(samp._weight_tree is not None)
        self.assertEqual(buf.capacity, samp._weight_tree._size)
        self.assertTrue(samp._weight_tree._base > buf.capacity)
        self.assertEqual(1, samp._weight_tree.sum())
        samp.add(i=[2])
        samp.add(i=[3])
        self.assertEqual(3, samp._weight_tree.sum())
        # test sample (batch=None)
        batch = samp.sample(beta=-1.0)
        self.assertEqual((3,), batch['i'].shape)
        self.assertArrayEqual(np.ones((3,), dtype=np.float32), batch['w'])
        # test sample (batch=2)
        ub_utils.set_seed(1) # i=[2, 3]
        batch_size = 2
        batch = samp.sample(batch_size=batch_size, beta=-1.0)
        self.assertArrayEqual([1, 2], samp.indices[0])
        self.assertEqual((2,), batch['i'].shape)
        self.assertArrayEqual(np.ones((2,), dtype=np.float32), batch['w'])
        samp.update(w=[0.5, 0.5])
        self.assertAllClose(0.5, samp._min_w) # exponent
        self.assertAllClose(1.0, samp._max_w) # exponent
        self.assertAllClose(2, samp._weight_tree.sum())
        batches = []
        for n in range(10000):
            batch = samp.sample(batch_size=batch_size, beta=-1.0) # i=[1, 1]
            batches.append(batch['i'])
        samples = np.asarray(batches).flatten()
        self.assertAllClose(0.5, np.sum(samples==1)/(n*2), atol=1e-2)
        self.assertAllClose(0.25, np.sum(samples==2)/(n*2), atol=1e-2)
        self.assertAllClose(0.25, np.sum(samples==3)/(n*2), atol=1e-2)
        # test sample (batch=3, seq=2)
        ub_utils.set_seed(2) # i=[[1, 2], [1, 2], [2, 3]]
        batch_size = 3
        seq_len = 2
        batch = samp.sample(batch_size=batch_size, seq_len=seq_len, beta=-1.0)
        self.assertArrayEqual([[0, 1], [0, 1], [1, 2]], samp.indices[0])
        self.assertEqual((3, 2), batch['i'].shape)
        self.assertAllClose([[1, 0.5], [1, 0.5], [0.5, 0.5]], batch['w'], atol=1e-6)
        samp.update(w=[[0.1, 0.2], [1, 0.5], [0.4, 0.5]])
        self.assertAllClose(0.1, samp._min_w)
        self.assertAllClose(1.0, samp._max_w)
        self.assertAllClose(1.9, samp._weight_tree.sum())

    def test_sampler_exception(self):
        with self.assertRaises(ValueError):
            # `buffer` must be an instance of BaseBuffer
            ub_data.UniformSampler(None)
        with self.assertRaises(ValueError):
            # `buffer` must be an instance of BaseBuffer
            ub_data.PermuteSampler(None)
        with self.assertRaises(ValueError):
            # `buffer` must be ReplayBuffer
            ub_data.PrioritizedSampler(ub_data.BaseBuffer(10), 1.0)
        with self.assertRaises(ValueError):
            # `buffer` must be ReplayBuffer
            ub_data.PrioritizedSampler(ub_data.DynamicBuffer(), 1.0)

    def test_compute_nstep_rew(self):
        rew = np.asarray([[1., 0., 1.], [1., 1., 1.]], dtype=np.float32).T
        done = np.asarray([[0., 0., 0.], [0., 1., 0.]], dtype=np.float32).T
        gamma = 0.95
        res = ub_data.compute_nstep_rew(rew, done, gamma=gamma)
        exp = np.asarray([1.9025, 1.95], dtype=np.float32)
        self.assertAllClose(exp, res)

    def test_compute_nstep_rew_empty(self):
        rew = np.asarray([], dtype=np.float32)
        done = np.asarray([], dtype=np.float32)
        res = ub_data.compute_nstep_rew(rew, done, 0.95)
        self.assertArrayEqual(rew, res)

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