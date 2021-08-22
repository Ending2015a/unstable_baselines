# --- built in ---
import os
import abc
import sys
import time

# --- 3rd party ---
import numpy as np
import tensorflow as tf

# --- my module ---
from unstable_baselines.lib import utils as ub_utils


__all__ = [
    'Categorical',
    'Normal',
    'MultiNormal',
    'Tanh'
]


# === tf probability ===

class Distribution(tf.Module, 
                    metaclass=abc.ABCMeta):
    def __init__(self, dtype, **kwargs):
        super().__init__(**kwargs)
        self.dtype = dtype

    def prob(self, x):
        '''
        Probability of given outcomes (x)
        '''
        return tf.math.exp(self.log_prob(x))

    @abc.abstractmethod
    def log_prob(self, x):
        '''
        Probability of given outcomes (x)
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def mode(self):
        '''
        Mode
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, n: int=[]):
        '''
        Sample outcomes
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def entropy(self):
        '''
        Entropy
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def kl(self, q):
        '''
        KL divergence

        q: target probability ditribution (Categorical)
        '''
        raise NotImplementedError

class Categorical(Distribution):
    def __init__(self, logits: tf.Tensor, dtype=tf.int32, **kwargs):
        self._logits = tf.convert_to_tensor(logits)
        super().__init__(dtype=dtype, **kwargs)

    @property
    def logits(self):
        return self._logits

    def _p(self):
        '''
        Probability distribution
        '''
        return tf.math.exp(self._log_p())

    def _log_p(self):
        '''
        Log probability distribution
        '''
        x = self.logits - tf.math.reduce_max(self.logits, axis=-1, keepdims=True)
        e = tf.math.exp(x)
        z = tf.math.reduce_sum(e, axis=-1, keepdims=True)
        return x - tf.math.log(z)

    def log_prob(self, x):
        '''
        Log probability of given outcomes (x)
        '''
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(
                                labels=tf.cast(x, dtype=self.dtype), 
                                logits=self.logits)

    def mode(self):
        '''
        Mode
        '''
        return tf.math.argmax(self.logits, axis=-1, output_type=self.dtype)

    def sample(self, n: int=[]):
        '''
        Sample outcomes
        '''
        n = tf.TensorShape(n)
        shape = tf.concat((n, tf.shape(self.logits)), axis=0)
        e  = tf.random.uniform(shape, dtype=self.logits.dtype)
        it = self.logits - tf.math.log(-tf.math.log(e))
        return tf.math.argmax(it, axis=-1, output_type=self.dtype)

    def entropy(self):
        '''
        Entropy
        '''
        m   = tf.math.reduce_max(self.logits, axis=-1, keepdims=True)
        x   = self.logits - m
        z   = tf.math.reduce_sum(tf.math.exp(x), axis=-1)
        xex = tf.math.multiply_no_nan(self.logits, tf.math.exp(x))
        p   = tf.math.reduce_sum(xex, axis=-1) / z
        return m[..., 0] + tf.math.log(z) - p

    def kl(self, q: Distribution):
        '''
        KL divergence

        q: target probability distribution (Categorical)
        '''
        logp = self._log_p()
        logq = q._log_p()
        p    = tf.math.exp(logp)
        return tf.math.reduce_sum(p * (logp-logq), axis=-1)


class Normal(Distribution):
    def __init__(self, mean, scale, dtype=tf.float32, **kwargs):
        self._mean = tf.cast(tf.convert_to_tensor(mean), dtype=dtype)
        self._scale = tf.cast(tf.convert_to_tensor(scale), dtype=dtype)
        super().__init__(dtype=dtype, **kwargs)

    @property
    def mean(self):
        return self._mean

    @property
    def scale(self):
        return self._scale

    def log_prob(self, x):
        '''
        Probability of given outcomes (x)
        '''
        x = tf.cast(tf.convert_to_tensor(x), dtype=self.dtype)
        p = 0.5 * tf.math.squared_difference(x/self.scale, self.mean/self.scale)
        c = tf.constant(0.5 * np.log(2. * np.pi), dtype=self.dtype)
        z = c + tf.math.log(self.scale)
        return -(p+z)

    def mode(self):
        '''
        Mode
        '''
        return self.mean * tf.ones_like(self.scale)

    def sample(self, n: int=[]):
        '''
        Sample outcomes
        '''
        n = tf.TensorShape(n)
        shape = ub_utils.broadcast_shape(self.mean.shape, self.scale.shape)
        shape = tf.concat((n, shape), axis=0)
        x = tf.random.normal(shape=shape, mean=0., stddev=1.,
                            dtype=self.dtype)
        return self.mean + self.scale * x

    def entropy(self):
        '''
        Entropy
        '''
        c = tf.constant(0.5 * np.log(2. * np.pi), dtype=self.dtype) + 0.5
        return c + tf.math.log(self.scale) * tf.ones_like(self.mean)

    def kl(self, q: Distribution):
        '''
        KL divergence

        q: target probability distribution (Normal)
        '''
        log_diff = (tf.math.log(self.scale * tf.ones_like(self.mean)) 
                    - tf.math.log(q.scale * tf.ones_like(q.mean)))
        return (0.5 * tf.math.squared_difference(self.mean/q.scale, q.mean/q.scale) +
                0.5 * tf.math.expm1(2. * log_diff) - log_diff)

class MultiNormal(Normal):
    def __init__(self, mean, scale, dtype=tf.float32, **kwargs):
        self._mean = tf.cast(tf.convert_to_tensor(mean), dtype=dtype)
        self._scale = tf.cast(tf.convert_to_tensor(scale), dtype=dtype)
        shape = ub_utils.broadcast_shape(self._mean.shape, self._scale.shape)
        if len(shape) < 1:
            raise RuntimeError('MultiNormal needs at least 1 dimension')
        Distribution.__init__(self, dtype=dtype, **kwargs)

    def log_prob(self, x):
        '''
        Log probability
        '''
        return tf.math.reduce_sum(super().log_prob(x), axis=-1)

    def entropy(self):
        '''
        Entropy
        '''
        return tf.math.reduce_sum(super().entropy(), axis=-1)

    def kl(self, q: Distribution):
        '''
        KL divergence

        q: target probability distribution (MultiNormal)
        '''
        return tf.math.reduce_sum(super().kl(q), axis=-1)


# === Probability bijection ===

class Bijector(Distribution):
    def __init__(self, distribution: Distribution):
        '''Wrapping the base distribution with a bijector wrapper

        Args:
            distribution (Distribution): Base distribution
        '''        
        if not isinstance(distribution, Distribution):
            raise ValueError('`distribution` must be a type of '
                f'Distribution, got {type(distribution)}')
        self.dist = distribution

    @property
    def distribution(self):
        return self.dist

    @abc.abstractmethod
    def forward(self, x):
        '''Forward bijection

        Args:
            x (tf.Tensor): Outcomes
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def inverse(self, y):
        '''Inverse bijection

        Args:
            y (tf.Tensor): Outcomes
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def log_det_jacob(self, x):
        '''Compute the log-det-jacobian matrix of a given outcome

        Args:
            x (tf.Tensor): Outcomes
        '''
        raise NotImplementedError

    def mode(self):
        return self.forward(self.distribution.mode())

    def log_prob(self, x):
        agg_jacob = tf.math.reduce_sum(self.log_det_jacob(x), axis=-1)
        return self.distribution.log_prob(x) * agg_jacob

    def sample(self):
        '''
        Sample outcomes
        '''
        return self.forward(self.distribution.sample())

    @abc.abstractmethod
    def entropy(self):
        '''
        Entropy
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def kl(self, q):
        '''
        KL divergence
        '''
        raise NotImplementedError
    

class Tanh(Bijector):
    def forward(self, x):
        return tf.math.tanh(x)

    def inverse(self, x):
        return tf.math.atanh(x)

    def log_det_jacob(self, x):
        # log(dy/dx) = log(1 - tanh(x)**2)
        return 2. * (np.log(2.) - x - tf.math.softplus(-2.*x))
    
    def entropy(self):
        '''
        Entropy
        '''
        raise NotImplementedError('Entropy for Tanh squashed does not '
            'exist an analytical solution')
        
    def kl(self, q: Distribution):
        '''
        KL divergence
        '''
        raise NotImplementedError('KL-divergence for Tanh squashed does not '
            'exist an analytical solution')