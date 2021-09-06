# --- built in ---
import os
import abc
import sys
import time
import logging

# --- 3rd party ---
import numpy as np

# --- my module ---
from unstable_baselines.lib import utils as ub_utils

__all__ = [
    'NestedReplayBuffer',
    'ReplayBuffer',
    'SequentialBuffer'
]


# === Segment Tree ===

class SegmentTree():
    pass


# === Replay Buffers ===
class BaseBuffer(metaclass=abc.ABCMeta):
    def __init__(self):
        raise NotImplementedError
    
    @property
    @abc.abstractmethod
    def isnull(self):
        '''Return True if the buffer space is not created'''
        raise NotImplementedError
    
    @property
    @abc.abstractmethod
    def isfull(self):
        '''Return True if the buffer space is full'''
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def capacity(self):
        '''Return the capacity of the buffer'''
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def ready_for_sample(self):
        '''Return True, if the buffer is ready for sampling'''
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        '''Reset the buffer'''
        raise NotImplementedError

    @abc.abstractmethod
    def add(self, data):
        '''Add new data into the buffer'''
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, data):
        '''Update existed data in the buffer'''
        raise NotImplementedError

    @abc.abstractmethod
    def make(self):
        '''Prepare for sampling'''
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, path):
        '''Save buffer data to path'''
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, path):
        '''Load buffer data from path'''
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        '''Return buffer length'''
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, key):
        '''Get items from the buffer'''
        raise NotImplementedError

    @abc.abstractmethod
    def __setitem__(self, key):
        '''Set items in the buffer'''
        raise NotImplementedError


class NestedReplayBuffer(BaseBuffer):
    def __init__(self, buffer_size):
        '''A replay buffer to store nested type of data
        Note that the nested structure can only be performed
        by python Tuple and python Dict. A python list object
        is treated as a single element.

        Example:
        >>> buffer = NestedReplayBuffer(3)
        >>> buffer.add({'a': [0.1], 'b': ([1], [True])})
        >>> buffer.add({'a': [0.2], 'b': ([2], [True])})
        >>> buffer.data
        {'a': array([0.1, 0.2, 0.0]),
         'b': (array([1, 2, 0]), array([ True, True, False]))}

        Args:
            buffer_size (int): Buffer size, capacity.

        Raises:
            ValueError: `buffer_size` must be an int and greater than 0.
        '''
        if not isinstance(buffer_size, int) or buffer_size <= 0:
            raise ValueError('`buffer_size` must be greater than 0')
        
        self._buffer_size = buffer_size
        self._pos         = 0
        self._full        = False
        self._data        = None
        self.reset()

    @property
    def isnull(self):
        '''Return True, if the buffer space is not created'''
        return self._data is None

    @property
    def isfull(self):
        '''Return True, if the buffer is full'''
        return self._full
    
    @property
    def capacity(self):
        '''Return the size/capacity of the buffer'''
        return self._buffer_size

    @property
    def ready_for_sample(self):
        return True

    @property
    def data(self):
        '''Retrieve data by key'''
        return self._data

    def reset(self):
        '''Reset buffer'''
        self._pos = 0
        self._full = False
        self._data = None

    def add(self, data):
        '''Add new BATCH data into the buffer

        Args:
            data (Any): Any nested type of data. Note that the python list
                is treated as single element.
        '''
        # count the first/batch dimension
        first = next(iter(ub_utils.iter_nested(data)))
        assert not np.isscalar(first), f'rank must be > 0'
        n_samples = len(first)
        # n_samples = ub_utils.nested_iter(data, _len_op, first=True)
        # prepare indices
        inds = np.arange(self._pos, self._pos+n_samples) % self._buffer_size
        # copy data into the buffer
        self._set_data(data, indices=inds)
        # update pointer position (circular)
        self._pos += n_samples
        if self._pos >= self._buffer_size:
            self._full = True
            self._pos = self._pos % self._buffer_size

    def update(self, data, indices):
        '''Update buffer contants'''
        self._set_data(data, indices=indices)

    def make(self):
        '''Prepare for sampling'''
        pass

    def __len__(self):
        return (self._buffer_size if self.isfull
                    else self._pos)

    def __getitem__(self, key):
        '''Get slice of data

        Args:
            key (int, slice): Indices slices

        Returns:
            nested: A nested data slice
        '''
        return self._get_data(key)

    def __setitem__(self, key, val):
        '''Set slice of data

        Args:
            key (int, slice): Indices slices
            val (nested): New data
        '''
        self._set_data(val, indices=key)

    # --- private methods ---
    def _melloc_by_batch_data(self, data):
        def _create_space(v):
            v = np.asarray(v)
            if len(v.shape) <= 1:
                shape = (self._buffer_size,)
            else:
                shape = (self._buffer_size,) + v.shape[1:]
            return np.zeros(shape, dtype=v.dtype)
        self._data = ub_utils.map_nested(data, op=_create_space)
    
    def _get_data(self, indices):
        '''Retrieve data'''
        if self.isnull:
            raise RuntimeError('Buffer space not created')
        _slice_op = lambda v: v[indices]
        return ub_utils.map_nested(self._data, _slice_op)

    def _set_data(self, data, indices):
        '''Set data'''
        assert indices is not None, '`indices` is not set'
        if self.isnull:
            # create space if the space is empty
            self._melloc_by_batch_data(data)
        def _assign_op(data_tuple, idx):
            new_data, data = data_tuple
            data[idx, ...] = np.asarray(new_data).astype(data.dtype).copy()
        ub_utils.map_nested_tuple((data, self._data), _assign_op, idx=indices)

    def load(self, path):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

class DictReplayBuffer(NestedReplayBuffer):
    '''A replay buffer to store nested type of data
    Note that the nested structure can only be performed
    by python Tuple and python Dict. A python list object
    is treated as a single element.

    Example:
        >>> buffer = DictReplayBuffer(3)
        >>> buffer.add(a=[0.1], b=([1], [True]))
        >>> buffer.add(a=[0.2], b=([2], [True]))
        >>> buffer.data
        {'a': array([0.1, 0.2, 0.0]),
         'b': (array([1, 2, 0]), array([ True, True, False]))}
    '''
    def keys(self):
        '''Return keys'''
        return [] if self.isnull else self._data.keys()

    def __contains__(self, key):
        '''Return True, if key is in the dict'''
        return not self.isnull and key in self._data.keys()

    def add(self, **data):
        if not self.isnull:
            self._assert_keys_exist(data.keys())
        return super().add(data)

    def update(self, indices, **data):
        self._assert_keys_exist(data.keys())
        return super().update(data, indices=indices)

    def _assert_keys_exist(self, keys):
        for key in keys:
            assert key in self, f'Key "{key}" does not exist'

class SequentialReplayBuffer(DictReplayBuffer):
    def __init__(self):
        '''An unlimited replay buffer which stores each sample 
        sequentially. Note that `buffer.make()` must be called 
        before sampling data.

        Using SequentialReplayBuffer must follow the procedure:
        Create -> add -> make -> update -> reset
        
        '''
        self._ready_for_sample = False
        self.reset()

    @property
    def capacity(self):
        '''Return the size/capacity of the buffer'''
        return float('inf')

    @property
    def ready_for_sample(self):
        '''Return True if buffer is ready for sampling'''
        return self._ready_for_sample

    def reset(self):
        '''Reset buffer'''
        super().reset()
        self._ready_for_sample = False

    def add(self, **data):
        '''Add new batch data into the buffer'''
        # count the first/batch dimension
        first = next(iter(ub_utils.iter_nested(data)))
        assert not np.isscalar(first), f'rank must be > 0'
        n_samples = len(first)
        #n_samples = ub_utils.nested_iter(data, _len_op, first=True)
        # copy data into the buffer
        self._append_data(data)
        # increase number of batch samples
        self._pos += n_samples

    def update(self, indices, **data):
        '''Update buffer contents
        You should call `make()` before calling `update()`
        '''
        if not self.ready_for_sample:
            raise RuntimeError('Call `buffer.make()` before calling '
                '`buffer.update()`')
        super().update(indices=indices, **data)

    def make(self):
        '''Prepare for sampling
        We need to flatten buffer data before sampling.
        We first swap the first and second axes
            (steps, batch, ...) -> (batch, steps, ...)
        then flatten the first and second dims
            (batch, steps, ...) -> (batch*steps, ...)
        The data becomes sequential ordered.
        '''
        if self.ready_for_sample:
            raise RuntimeError('The buffer has already made.')
        data = ub_utils.nested_to_numpy(self._data)
        # flatten data (steps, n_samples, ...) -> (n_samples*steps, ...)
        def _swap_flatten(v):
            shape = v.shape
            assert len(shape) > 1, f'The data must have rank > 1, got {shape}'
            if len(shape) <= 2:
                v = v.swapaxes(0, 1).reshape(shape[0]*shape[1])
            else:
                v = v.swapaxes(0, 1).reshape(shape[0]*shape[1], *shape[2:])
            return v
        data = ub_utils.map_nested(data, _swap_flatten)
        self._data = data
        self._ready_for_sample = True

    def __len__(self):
        return self._pos

    # --- private method ---
    def _melloc_by_batch_data(self, data):
        '''Create buffer space'''
        def _create_space(v):
            return []
        self._data = ub_utils.map_nested(data, op=_create_space)
    
    def _append_data(self, data):
        if self.isnull:
            self._melloc_by_batch_data(data)
        if self.ready_for_sample:
            raise RuntimeError('The buffer can not append data after '
                'calling `buffer.make()`.')
        def _append_op(data_tuple):
            new_data, data = data_tuple
            data.append(np.asarray(new_data))
        ub_utils.map_nested_tuple((data, self.data), _append_op)

# alias
ReplayBuffer = DictReplayBuffer
SequentialBuffer = SequentialReplayBuffer

# === Sampler ===

class BaseSampler(metaclass=abc.ABCMeta):
    def __init__(self, buffer: BaseBuffer):
        if not isinstance(buffer, BaseBuffer):
            raise ValueError('`buffer` must be an instance of BaseBuffer, '
                f'got {type(buffer)}')
        self._buffer = buffer
    
    @property
    def buffer(self):
        return self._buffer

    def __call__(self, *args, **kwargs):
        '''A shortcut to self.sample()'''
        return self.sample(*args, **kwargs)

    @abc.abstractmethod
    def sample(self, batch_size, seq_len):
        '''Random sample replay buffer'''
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, *args, **kwargs):
        '''Update sampled data to the buffer'''
        raise NotImplementedError


class UniformSampler(BaseSampler):
    def __init__(self, buffer: BaseBuffer):
        '''Uniform sampler samples buffer uniformly.
        This sampler is mainly for off-policy algorithms.

        Args:
            buffer (BaseBuffer): Replay buffer
        '''
        super().__init__(buffer)
        self._cached_inds = None

    def sample(self, batch_size: int=None, seq_len: int=None):
        '''Randomly sample a batch or a batch sequence of data from the 
        replay buffer

        Args:
            batch_size (int, optional): Batch size to sample. Defaults to None.
            seq_len (int, optional): Length of sequences. If it's None, the sampled
                data has shape (b, *data.shape), otherwise, (b, seq, *data.shape).
                Defaults to None.
        
        Returns:
            Any: Returns a slice of data from the replay buffer. Each element has 
                shape (b, *data.shape) if seq_len is None, otherwise, 
                (b, seq, *data.shape).
        '''
        buf = self.buffer
        assert buf.ready_for_sample, 'Buffer is not ready for sampling, ' \
            'call `buffer.make()` before sampling'
        if batch_size is None:
            batch_size = len(buf)
        if seq_len is None:
            batch_inds = np.random.randint(0, len(buf), size=batch_size)
        else:
            batch_inds = np.random.randint(0, len(buf)-seq_len+1, size=batch_size)
            batch_inds = np.expand_dims(batch_inds, axis=-1)
            batch_inds = batch_inds + np.arange(seq_len)
        self._cached_inds = batch_inds
        return self.buffer[batch_inds]

    def update(self, *args, **kwargs):
        '''Update sampled data to the buffer'''
        self.buffer.update(*args, **kwargs, indices=self._cached_inds)


class PermuteSampler(BaseSampler):
    def __init__(self, buffer: BaseBuffer):
        '''Permute sampler samples buffer by random permutation. It
        returns an iterator iterates through all rollout data.

        Args:
            buffer (BaseBuffer): Replay buffer.
        '''
        super().__init__(buffer)
        self._cached_inds = None
    
    def sample(self, batch_size: int=None, seq_len: int=None):
        '''Create an iterator which randomly iterate through all data 
        in the replay buffer with a given batch size and sequence length.

        Args:
            batch_size (int, optional): Batch size to sample. Defaults to None.
            seq_len (int, optional): Length of sequences. If it's None, the sampled
                data has shape (b, *data.shape), otherwise, (b, seq, *data.shape).
                Defaults to None.
        
        Returns:
            Any: Returns an iterator which produces slices of data from the replay 
                buffer. Each element has shape (b, *data.shape) if seq_len is None, 
                otherwise, (b, seq, *data.shape).
        '''
        buf = self.buffer
        assert buf.ready_for_sample, 'Buffer is not ready for sampling, ' \
            'call `buffer.make()` before sampling'
        if seq_len is not None:
            # TODO: implement sampling sequences
            raise NotImplementedError
        return self._iter(batch_size=batch_size, seq_len=seq_len)

    def _iter(self, batch_size: int=None, seq_len: int=None):
        buf = self.buffer
        buf_len = len(buf)
        if batch_size is None:
            batch_size = buf_len
        permute = np.random.permutation(buf_len)
        start_ind = 0
        while start_ind < buf_len:
            _slice = range(start_ind, start_ind+batch_size)
            indices = np.take(permute, _slice, mode='wrap')
            self._cached_inds = indices
            # return iterator
            yield self.buffer[indices]
            start_ind += batch_size

    def update(self, *args, **kwargs):
        '''Update sampled data to the buffer'''
        self.buffer.update(*args, **kwargs, indices=self._cached_inds)

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
    mask = 1.-done
    return _compute_advantage(rew=rew, val=val, mask=mask,
        gamma=gamma, gae_lambda=gae_lambda)

def _compute_advantage(rew:        np.ndarray,
                       val:        np.ndarray,
                       mask:       np.ndarray,
                       gamma:      float,
                       gae_lambda: float):
    '''Perform GAE computation'''
    adv = np.zeros_like(rew)
    gae = 0.
    next_mask = mask[-1]
    next_val  = val[-1]
    for t in reversed(range(len(mask))):
        delta = rew[t] + gamma * next_val * next_mask - val[t]
        gae = delta + gamma * gae_lambda * next_mask * gae
        adv[t] = gae
        next_mask = mask[t]
        next_val  = val[t]
    return adv
