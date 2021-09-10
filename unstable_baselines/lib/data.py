# --- built in ---
import os
import abc
import sys
import math
import time
import logging

# --- 3rd party ---
import numpy as np

# --- my module ---
from unstable_baselines.lib import utils as ub_utils

__all__ = [

]

# === Segment Tree ===

class SegmentTree(metaclass=abc.ABCMeta):
    def __init__(self, size: int):
        '''An implementation of segment tree used to efficiently O(logN)
        compute the sum of a query range [start, end)

        Args:
            size (int): Number of elements.
        '''
        assert isinstance(size, int) and size > 0
        base = 1<<(size-1).bit_length()
        self._size = size
        self._base = base
        self._value = np.zeros([base * 2], dtype=np.float64)
    
    def __getitem__(self, key: np.ndarray):
        # formalize indices
        if isinstance(key, (int, slice)):
            key = np.asarray(range(self._size)[key], dtype=np.int64)
        else:
            key = np.asarray(key, dtype=np.int64)
        key = key % self._size + self._base
        return self._value[key]

    def __setitem__(self, key: np.ndarray, value: np.ndarray):
        self.update(key, value)

    def update(self, key: np.ndarray, value: np.ndarray):
        '''Update elements' values'''
        # formalize indices
        if isinstance(key, (int, slice)):
            key = np.asarray(range(self._size)[key], dtype=np.int64)
        else:
            key = np.asarray(key, dtype=np.int64)
        key = key % self._size + self._base
        key = key.flatten()
        value = np.asarray(value, dtype=np.float64).flatten()
        # set values
        self._value[key] = value
        # update tree (all keys have the same depth)
        while key[0] > 1:
            self._value[key>>1] = self._value[key] + self._value[key^1]
            key >>= 1

    def sum(self, start: int=None, end: int=None):
        '''Compute the sum of the given range [start, end)'''
        if (start == None) and (end == None):
            # shortcut
            return self._value[1]
        start, end, _ = slice(start, end).indices(self._size)
        start += self._base
        end += self._base
        res = 0.0
        while start < end:
            if start & 1:
                res += self._value[start]
            if end & 1:
                res += self._value[end-1]
            start = (start+1) >> 1
            end = end >> 1
        return res

    def find(self, value: np.ndarray):
        '''Find the largest ind such that
        self.sum(0, ind+1) < value
        '''
        assert np.min(value) >= 0.0
        assert np.max(value) < self._value[1]
        # if input is a scalar, return should be a scalar too.
        one_value = np.isscalar(value)
        # convert to 1D array
        value = np.asarray(value, dtype=np.float64)
        orig_shape = value.shape
        value = value.flatten()
        inds = np.ones_like(value, dtype=np.int64)
        # find inds (all inds have the same depth)
        while inds[0] < self._base:
            inds <<= 1
            lsum = self._value[inds]
            d = lsum < value
            value -= lsum * d
            inds += d
        inds -= self._base
        inds = inds.reshape(orig_shape)
        return inds.item() if one_value else inds

# === Replay buffer ===

class RelativeIndex():
    def __init__(self, buffer:  'BaseBuffer',
                       offsets: tuple = None):
        '''This class is used to relative indexing

        For example, if the buffer has 10 slots and the last element
        (pos) is at 6th slot, you can get the last element by absolute 
        indexing
        >>> buffer[6]
        by relative indexing (relative to `buffer.head`)
        >>> buffer.rel[-1]
        `-1` is the last element in the buffer.
        This is useful when you want to calculate N-step returns

        This also support vectorized relative indexing, for example:
        >>> rel = RelativeIndex(buffer, [1, 3, 5])
        >>> rel[-1]
        this is equivelent to
        >>> buffer[[0, 2, 4]]

        Note that relative indexing only supports int, np.ndarray and 
        slice. It does not support Ellipsis.

        Args:
            buffer (BaseBuffer): Buffer
            offsets (int, tuple, np.ndarray): The indices relative to.
                If None, it is set to `buffer.head`. Defaults to None.
        '''
        assert isinstance(buffer, BaseBuffer)
        self.buffer = buffer
        if offsets is None:
            offsets = self.buffer.head
        self.offsets = np.index_exp[offsets]

    def toabs(self, key, offset, max_size):
        '''Convert to absolute indexing'''
        if key is Ellipsis:
            raise NotImplementedError("Relative indexing does not"
                "support Ellipsis")
        # formalize indices (slice to indices)
        if isinstance(key, (int, slice)):
            key = np.asarray(range(max_size)[key], dtype=np.int64)
        else:
            key = np.asarray(key, dtype=np.int64)
        if not np.isscalar(offset):
            # vectorized indexing
            key = key.reshape(key.shape + (1,)*offset.ndim)
        key = (key + offset) % max_size
        return key
    
    def cvtkeys(self, keys):
        keys = np.index_exp[keys]
        # Currently only support the first dim
        max_size = self.buffer.len_slots()
        key = self.toabs(keys[0], self.offsets[0], max_size)
        if len(self.offsets) > 1:
            # TODO Support multi-dims
            assert len(keys) == 1, 'Relative indexing does not support multi-dims'
            keys = tuple([key, *self.offsets[1:]])
        else:
            keys = tuple([key, *keys[1:]])
        return keys

    def __getitem__(self, keys):
        return self.buffer[self.cvtkeys(keys)]

    def __setitem__(self, keys, vals):
        self.buffer[self.cvtkeys(keys)] = vals

class BaseBuffer(metaclass=abc.ABCMeta):
    '''The base class of replay buffers.
    BaseBuffer stores sample in a 2D manner (slots, batch), where `slots`
    can be seen as the timesteps of each samples, and `batch` is the 
    number of samples each time user `add()` into the buffer.
    '''
    def __init__(self, size: int, batch: int=None):
        '''Create a buffer
        
        Args:
            size (int): Buffer size, capacity
            batch (int, optional): batch size of replay samples, commonly 
                referred to as number of envs in a vectorized env. This 
                value is automatically set when user first adds a batch 
                of data. Defaults to None.
        '''
        if not isinstance(size, int) or size <= 0:
            raise ValueError('`size` must be greater than 0, '
                f'got {size}')
        self._min_size = size
        self._batch = batch
        self._slots = None
        self._size  = None
        self._pos   = 0
        self._full  = False
        self._data  = None
        self.reset()
        # calculate buffer spaces
        if batch is not None:
            self._calc_space(batch)

    @property
    def capacity(self):
        '''Return the capacity of the buffer'''
        return self._size
    
    @property
    def slots(self):
        '''Return number of slots'''
        return self._slots
    
    @property
    def batch(self):
        '''Return batch size'''
        return self._batch
    
    @property
    def head(self):
        '''Return the index of the first slot'''
        return self._pos if self.isfull else 0

    @property
    def tail(self):
        '''Return the index of the last slot'''
        return self._pos
    
    @property
    def isnull(self):
        '''Return True, if the buffer space is not created'''
        return self._data is None

    @property
    def isfull(self):
        '''Return True, if the buffer is full'''
        return self._full

    @property
    def ready_for_sample(self):
        '''Return True, if it is ready for sampling'''
        return True

    @property
    def data(self):
        '''Retrieve data'''
        return self._data

    @property
    def rel(self):
        '''Relative indexing'''
        return RelativeIndex(self)

    def reset(self):
        '''Reset buffer'''
        self._pos    = 0
        self._full   = False
        self._data   = None

    def add(self, data):
        '''Add new batch data into the buffer

        Args:
            data (Any): Any nested type of data. Note that the python list
                is treated as a single element. Each data must have shape 
                (b, *)
        '''
        # count the batch dimension
        arr = next(iter(ub_utils.iter_nested(data)))
        assert not np.isscalar(arr), f'rank must be > 0'
        # prepare indices and copy data into the buffer
        self._set_data(data, indices=self._pos)
        # update cursor position (circular buffer)
        self._pos += 1
        if self._pos >= self._slots:
            self._full = True
            self._pos = self._pos % self._slots

    def update(self, data, indices):
        '''Update buffer data'''
        self._set_data(data, indices=indices)

    def __len__(self):
        '''Return the number of total samples'''
        if self.isnull:
            return 0
        return (self._size if self.isfull else self._pos*self._batch)

    def len_slots(self):
        '''Return the number of filled slots'''
        return (self._slots if self.isfull else self._pos)

    def __getitem__(self, key):
        '''Get a slice of data
        
        Args:
            key(int, np.ndarray): Indices.

        Returns:
            Any: A slice of nested data
        '''
        return self._get_data(key)

    def __setitem__(self, key, val):
        '''Set slice of data
        
        Args:
            key (int, slice): Indices slices
            val (Any): New data
        '''
        self._set_data(val, indices=key)

    # --- private methods ---
    def _calc_space(self, batch: int):
        '''Calculate buffer spaces'''
        assert isinstance(batch, int)
        assert batch > 0
        self._batch = batch
        self._slots = math.ceil(self._min_size/batch)
        self._size  = self._slots * self._batch # true size

    def _melloc(self, data):
        '''Create spaces from the given data example'''
        def _create_space(v):
            v = np.asarray(v)
            shape = (self._slots, self._batch)
            if len(v.shape) > 1:
                shape += v.shape[1:]
            return np.zeros(shape, dtype=v.dtype)
        # calculate buffer spaces
        if self._batch is None:
            arr = next(iter(ub_utils.iter_nested(data)))
            assert not np.isscalar(arr), f'rank must be > 0'
            self._calc_space(len(arr))
        self._data = ub_utils.map_nested(data, op=_create_space)

    def _get_data(self, indices):
        assert not self.isnull, 'Buffer space not created'
        _slice_op = lambda v: v[indices]
        return ub_utils.map_nested(self._data, _slice_op)

    def _set_data(self, data, indices):
        assert indices is not None, '`indices` not set'
        # create space if the space is empty
        if self.isnull:
            self._melloc(data)
        # assign to buffer
        def _assign_op(data_tuple):
            new_data, data = data_tuple
            data[indices] = np.asarray(new_data).astype(data.dtype)
        ub_utils.map_nested_tuple((data, self._data), _assign_op)

    def load(self, path):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError


class ReplayBuffer(BaseBuffer):
    '''A replay buffer to store nested type of data
    Note that you can only use python tuple and python
    dict to represent the nested data. A python list
    is treated as a single element.

    Example:
    >>> buffer = DictReplayBuffer(3, 1)
    >>> buffer.add(a=[0.1], b=([1], [True]))
    >>> buffer.add(a=[0.2], b=([2], [True]))
    >>> buffer.data
    {'a': array([[0.1, 0.2, 0.0]]),
        'b': (array([[1, 2, 0]]), array([[ True, True, False]]))}
    '''
    def keys(self):
        '''Return keys'''
        return [] if self.isnull else self._data.keys()
    
    def __contains__(self, key):
        '''Return True, if key is in the dict'''
        return not self.isnull and key in self._data.keys()

    def add(self, **data):
        return super().add(data)

    def update(self, indices, **data):
        return super().update(data, indices=indices)

    def _assert_keys_exist(self, keys):
        for key in keys:
            assert key in self, f'Key "{key}" does not exist'

    def _set_data(self, data, indices):
        if not self.isnull:
            self._assert_keys_exist(data.keys())
        return super()._set_data(data, indices)

class SequentialBuffer(ReplayBuffer):
    def __init__(self, batch: int=None):
        '''An unlimited replay buffer which stores each sample sequentially. 
        Note that `buffer.make()` must be called before sampling data.

        Using SequentialReplayBuffer must follow the procedure:
            Create -> add -> make -> update -> reset

        Args:
            batch (int, optional): batch size of replay samples, commonly 
                referred to as number of envs in a vectorized env. This 
                value is automatically set when user first adds a batch 
                of data. Defaults to None.
        '''
        self._batch = batch
        self._ready_for_sample = False
        self.reset()

    @property
    def capacity(self):
        '''Return the capacity of the buffer'''
        return float('inf')

    @property
    def slots(self):
        '''Return number of slots'''
        return self._pos

    @property
    def head(self):
        '''Return the index of the first slot'''
        return 0

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
        if self.ready_for_sample:
            raise RuntimeError('Buffer can\'t add data when it '
                'is ready for sampling')
        arr = next(iter(ub_utils.iter_nested(data)))
        assert not np.isscalar(arr), f'rank must be > 0'
        # copy data into the buffer
        self._append_data(data)
        # increase number of batch samples
        self._pos += 1

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
        Convert list to np.ndarray
        '''
        if self.ready_for_sample:
            raise RuntimeError('The buffer has already made.')
        self._data = ub_utils.nested_to_numpy(self._data)
        self._ready_for_sample = True

    def __len__(self):
        return 0 if self.isnull else self._pos*self._batch

    def len_slots(self):
        return self._pos

    # --- private method ---
    def _calc_space(self, batch: int):
        '''Calculate buffer spaces'''
        assert isinstance(batch, int)
        assert batch > 0
        self._batch = batch

    def _melloc(self, data):
        '''Create buffer space'''
        _create_space = lambda v: []
        if self._batch is None:
            arr = next(iter(ub_utils.iter_nested(data)))
            assert not np.isscalar(arr), f'rank must be > 0'
            self._calc_space(len(arr))
        self._data = ub_utils.map_nested(data, op=_create_space)
    
    def _append_data(self, data):
        if self.isnull:
            self._melloc(data)
        if self.ready_for_sample:
            raise RuntimeError('The buffer can not append data after '
                'calling `buffer.make()`.')
        self._assert_keys_exist(data.keys())
        def _append_op(data_tuple):
            new_data, data = data_tuple
            data.append(np.asarray(new_data))
        ub_utils.map_nested_tuple((data, self.data), _append_op)

# === Sampler ===

class BaseSampler(metaclass=abc.ABCMeta):
    def __init__(self, buffer: BaseBuffer):
        if not isinstance(buffer, BaseBuffer):
            raise ValueError('`buffer` must be an instance of BaseBuffer, '
                f'got {type(buffer)}')
        self._buffer = buffer
        self._cached_inds = None
    
    @property
    def buffer(self):
        return self._buffer

    @property
    def indices(self):
        '''Cached indices'''
        return self._cached_inds

    @property
    def rel(self):
        '''Relative indexing, relative to sampled indices'''
        return RelativeIndex(self._buffer, self._cached_inds)

    def __call__(self, *args, **kwargs):
        '''A shortcut to self.sample()'''
        return self.sample(*args, **kwargs)

    @abc.abstractmethod
    def sample(self, batch_size, seq_len):
        '''Random sample replay buffer'''
        raise NotImplementedError
        
    def update(self, *args, **kwargs):
        '''Update sampled data to the buffer'''
        self.buffer.update(*args, **kwargs, indices=self._cached_inds)

class UniformSampler(BaseSampler):
    def __init__(self, buffer: BaseBuffer):
        '''Uniform sampler samples buffer uniformly.
        This sampler is mainly for off-policy algorithms.

        Args:
            buffer (BaseBuffer): Replay buffer
        '''
        super().__init__(buffer)

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
        if not buf.ready_for_sample:
            raise RuntimeError('Buffer is not ready for sampling, '
                'call `buffer.make()` before sampling')
        if batch_size is None:
            batch_size = len(buf)
        if seq_len is None:
            inds = np.random.randint(len(buf), size=batch_size)
        else:
            inds = np.random.randint(len(buf)-seq_len*buf.batch, 
                                     size=(batch_size, 1))
        inds1, inds2 = np.unravel_index(inds, (buf.slots, buf.batch))
        if seq_len is not None:
            inds1 = inds1 + np.arange(seq_len)
            inds2 = inds2 + np.zeros_like(inds1) # broadcast
        # add offset
        inds1 = (inds1 + buf.head) % buf.len_slots()
        self._cached_inds = (inds1, inds2)
        return self.buffer[self._cached_inds]

class PermuteSampler(BaseSampler):
    def __init__(self, buffer: BaseBuffer):
        '''Permute sampler samples buffer by random permutation. It
        returns an iterator iterates through all rollout data.

        Args:
            buffer (BaseBuffer): Replay buffer.
        '''
        super().__init__(buffer)
    
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
        if not buf.ready_for_sample:
            raise RuntimeError('Buffer is not ready for sampling, '
                'call `buffer.make()` before sampling')
        return self._iter(batch_size=batch_size, seq_len=seq_len)

    def _iter(self, batch_size: int=None, seq_len: int=None):
        buf = self.buffer
        if batch_size is None:
            batch_size = len(buf)
        if seq_len is None:
            inds = np.arange(len(buf))
        else:
            #TODO: skip seq_len steps
            inds = np.arange(len(buf)-seq_len*buf.batch).reshape(-1, 1)
        inds1, inds2 = np.unravel_index(inds, (buf.slots, buf.batch))
        if seq_len is not None:
            inds1 = inds1 + np.arange(seq_len)
            inds2 = inds2 + np.zeros_like(inds1) # broadcast
        # shuffle the indices of the indices of the samples
        permute = np.arange(len(inds1))
        np.random.shuffle(permute)
        start_ind = 0
        while start_ind < len(inds):
            slice_ = range(start_ind, start_ind+batch_size)
            indices = np.take(permute, slice_, mode='wrap')
            self._cached_inds = (inds1[indices], inds2[indices])
            # return iterator
            yield self.buffer[self._cached_inds]
            start_ind += batch_size

# === data utils ===
def compute_nstep_rew(rew: np.ndarray,
                      done: np.ndarray,
                      gamma: float = 0.99):
    '''Compute N step rewards
    aggregate the folloing N-1 steps discounted rewards to the 
    first reward:
    rew = rew[0] + gamma**1 * rew[1] + gamma**2 * rew[2] + ...

    Args:
        rew (np.ndarray): Reward sequences, (step, ...)
        done (np.ndarray): Done sequences, (step, ...)
        gamma (float, optional): Discount factor. Defaults to 0.99

    Returns:
        np.ndarray: N step rewards (...)
    '''
    rew = np.asarray(rew).astype(np.float32)
    done = np.asarray(done).astype(np.float32)
    mask = 1.-done
    if len(done) == 0:
        return rew
    return _compute_nstep_rew(rew=rew, mask=mask, gamma=gamma)

def _compute_nstep_rew(rew: np.ndarray,
                       mask: np.ndarray,
                       gamma: float = 0.99):
    gams = gamma ** np.arange(len(mask))
    res = np.zeros_like(rew[0])
    prev_mask = np.ones_like(mask[0])
    for t in range(len(mask)):
        res += gams[t] * rew[t] * prev_mask
        prev_mask *= mask[t]
    return res

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