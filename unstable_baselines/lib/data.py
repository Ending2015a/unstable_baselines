# --- built in ---
import os
import sys
import time
import logging

# --- 3rd party ---
import numpy as np
import numba as nb

# --- my module ---
from unstable_baselines.lib import utils as ub_utils

__all__ = [
    'NestedReplayBuffer',
    'ReplayBuffer',
    'SequentialBuffer'
]

class NestedReplayBuffer():
    def __init__(self, buffer_size):
        if not isinstance(buffer_size, int) or buffer_size <= 0:
            raise ValueError('`buffer_size` must be greater than 0')
        
        self.buffer_size = buffer_size
        self.reset()

    def reset(self):
        self.pos = 0
        self.full = False

        self.data = None
        self._cached_inds = None

    def add(self, data):
        # count the first dimension
        _len_op = lambda v: len(v)
        n_samples = ub_utils.nested_iter(data, _len_op, first=True)
        # prepare indices
        inds = np.arange(self.pos, self.pos+n_samples) % self.buffer_size
        # copy data into the buffer
        self._set_data(data, indices=inds, _auto_create_space=True)
        # increase pointer
        self.pos += n_samples
        # circular buffer
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = self.pos % self.buffer_size
    
    def update(self, data, indices=None):
        if indices is None:
            indices = self._cached_inds
        self._set_data(data, indices=indices, _auto_create_space=True)
    
    def melloc_by_batch_samples(self, data):
        def _create_space(v):
            v = np.asarray(v)
            if len(v.shape) > 1:
                shape = (self.buffer_size,) + v.shape[1:]
            else:
                shape = (self.buffer_size,)
            return np.zeros(shape, dtype=v.dtype)
        self.data = ub_utils.nested_iter(data, op=_create_space)

    def isnull(self):
        return self.data is None

    def isfull(self):
        return self.full

    def __call__(self, batch_size: int=None, seq_len: int=None):
        '''Randomly sample a batch from replay buffer

        Args:
            batch_size (int, optional): Batch size to sample. Defaults to None.
            seq_len (int, optional): Length of data sequences. Defaults to None.

        Returns:
            dict, tuple: A nested structure of np.ndarray, each with shape 
                (batch, *data.shape) if `seq_len` is None, otherwise, 
                (batch, seq_len, *data.shape)
        '''        
        if batch_size is None:
            batch_size = len(self)
        if seq_len is None:
            batch_inds = np.random.randint(0, len(self), size=batch_size)
        else:
            batch_inds = np.random.randint(0, len(self)-seq_len+1, size=batch_size)
            batch_inds = np.expand_dims(batch_inds, axis=-1)
            batch_inds = batch_inds + np.arange(seq_len)
        self._cached_inds = batch_inds
        return self._get_data(batch_inds)

    def __len__(self):
        return self.buffer_size if self.full else self.pos

    def __getitem__(self, key):
        '''Get slice

        Args:
            key: Indices slices
        '''
        return self._get_data(key)
    
    def __setitem__(self, key, val):
        return self._set_data(val, indices=key)

    # --- private functions ---
    def _get_data(self, indices):
        if self.isnull():
            raise RuntimeError('Buffer space not created, please `add` data, '
                    ' or calling `melloc_by_batch_samples` first.')
        _slice_op = lambda v: v[indices]
        return ub_utils.nested_iter(self.data, _slice_op)

    def _set_data(self, data, indices, _auto_create_space=True):
        # Create space or raise empty error, if the spae is not created
        if self.isnull():
            if _auto_create_space:
                self.melloc_by_batch_samples(data)
            else:
                raise RuntimeError('Buffer space not created, please `add` data, '
                        'or calling `melloc_by_batch_samples` first.')
        def _assign_op(data_tuple, idx):
            v, data = data_tuple
            data[idx, ...] = np.asarray(v).astype(data.dtype).copy()
        ub_utils.nested_iter_tuple((data, self.data), _assign_op, idx=indices)
        

class DictReplayBuffer(NestedReplayBuffer):
    def keys(self):
        if self.isnull():
            raise RuntimeError('Data is empty')
        return self.data.keys()

    def add(self, **kwargs):
        return super().add(kwargs)

    def update(self, _indices=None, **kwargs):
        return super().update(kwargs, indices=_indices)

class SequentialReplayBuffer(DictReplayBuffer):
    def __init__(self):
        self.reset()

    def reset(self):
        self.data      = None
        self.pos       = 0
        self.n_samples = 0
        self.ready_for_sampling = False

        self._cached_inds = None

    def add(self, **kwargs):
        # copy data into the buffer
        self._append_data(kwargs, _auto_create_space=True)
        # increase number of batch samples
        self.pos += 1

    def make(self):
        if self.ready_for_sampling:
            raise RuntimeError('The buffer has already made.')
        data = ub_utils.nested_to_numpy(self.data)
        # flatten data (steps, n_samples, ...) -> (n_samples*steps, ...)
        def _swap_flatten(v):
            shape = v.shape
            if len(shape) < 2:
                raise RuntimeError('The data must have rank > 2, '
                        'got {}'.format(shape))
            if len(shape) < 3:
                v = v.swapaxes(0, 1).reshape(shape[0]*shape[1])
            else:
                v = v.swapaxes(0, 1).reshape(shape[0]*shape[1], *shape[2:])
            return v
        _len_op = lambda v: len(v)
        data = ub_utils.nested_iter(data, _swap_flatten)
        n_samples = ub_utils.nested_iter(data, _len_op, first=True)
        self.data = data
        self.n_samples = n_samples
        self.ready_for_sampling = True

    def melloc_by_batch_samples(self, data):
        def _create_space(v):
            return []
        self.data = ub_utils.nested_iter(data, op=_create_space)
    
    def isnull(self):
        return self.data is None

    def isfull(self):
        return False

    def __call__(self, batch_size: int=None):
        if not self.ready_for_sampling:
            raise RuntimeError('The buffer is not ready for sampling, '
                    'call `make` before sampling')
        return self._sample_data(batch_size)

    def __len__(self):
        return self.n_samples if self.ready_for_sampling else self.pos

    def _append_data(self, data, _auto_create_space=True):
        # Create space or raise empty error, if the spae is not created
        if self.isnull():
            if _auto_create_space:
                self.melloc_by_batch_samples(data)
            else:
                raise RuntimeError('Buffer space not created, please `add` data, '
                        'or calling `melloc_by_batch_samples` first.')
        def _append_op(data_tuple):
            v, data = data_tuple
            data.append(v)
        ub_utils.nested_iter_tuple((data, self.data), _append_op)

    def _sample_data(self, batch_size):
        n_samples = len(self)
        if batch_size is None:
            batch_size = n_samples
        rand_inds = np.random.permutation(n_samples)
        start_ind = 0
        while start_ind < n_samples:
            _slice = range(start_ind, start_ind+batch_size)
            indices = np.take(rand_inds, _slice, mode='wrap')
            self._cached_inds = indices
            # return itertor
            yield self._get_data(indices)
            start_ind += batch_size


# alias
ReplayBuffer = DictReplayBuffer
SequentialBuffer = SequentialReplayBuffer

# === data utils ===
def compute_advantage(rew:        np.ndarray,
                      val:        np.ndarray,
                      done:       np.ndarray,
                      gamma:      float = 0.99,
                      gae_lambda: float = 1.0):
    '''Compute GAE

    Args:
        rewards (np.ndarray): Rewards (steps, ...)
        values (np.ndarray): Predicted values (steps, ...)
        dones (np.ndarray): Done flags (steps, ...)
        gamma (float, optional): Discount factor. Defaults to 0.99.
        gae_lambda (float, optional): GAE lambda. Defaults to 1.0.

    Returns:
        np.ndarray: GAE
    '''
    # ensure the inputs are np.ndarray
    rew  = np.asarray(rew).astype(np.float32)
    val  = np.asarray(val).astype(np.float32)
    done = np.asarray(done).astype(np.float32)
    return _compute_advantage(rew=rew, val=val, done=done,
                    gamma=gamma, gae_lambda=gae_lambda)
@nb.njit
def _compute_advantage(rew:        np.ndarray,
                       val:        np.ndarray,
                       done:       np.ndarray,
                       gamma:      float = 0.99,
                       gae_lambda: float = 1.0):
    adv  = np.zeros_like(rew)
    gae = 0.
    next_ndone = 1.-done[-1]
    next_val = val[-1]
    for t in range(len(done)-1, -1, -1):
        delta = rew[t] + gamma * next_val * next_ndone - val[t]
        gae = delta + gamma * gae_lambda * next_ndone * gae
        adv[t] = gae
        next_ndone = 1.-done[t]
        next_val = val[t]
    return adv
