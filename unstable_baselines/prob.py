# --- built in ---
import os
import abc
import sys
import time

# --- 3rd party ---
import numpy as np
import tensorflow as tf

# --- my module ---


__all__ = [
    'Categorical',
    'Normal'
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
        raise NotImplementedError("Method not implemented")

    @abc.abstractmethod
    def mode(self):
        '''
        Mode
        '''
        raise NotImplementedError("Method not implemented")

    @abc.abstractmethod
    def sample(self):
        '''
        Sample outcomes
        '''
        raise NotImplementedError("Method not implemented")

    @abc.abstractmethod
    def entropy(self):
        '''
        Entropy
        '''
        raise NotImplementedError("Method not implemented")

    @abc.abstractmethod
    def kl(self, q):
        '''
        KL divergence

        q: target probability ditribution (Categorical)
        '''
        raise NotImplementedError("Method not implemented")



class Categorical(Distribution):
    def __init__(self, logits, **kwargs):
        self._logits = tf.convert_to_tensor(logits)

        super().__init__(dtype=self._logits.dtype,
                        **kwargs)
        
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
                            labels=x, logits=self.logits)

    def mode(self):
        '''
        Mode
        '''
        return tf.math.argmax(self.logits, axis=-1)

    def sample(self):
        '''
        Sample outcomes
        '''
        e = tf.random.uniform(tf.shape(self.logits))
        it = self.logits - tf.math.log(-tf.math.log(e))
        return tf.math.argmax(it, axis=-1)

    def entropy(self):
        '''
        Entropy
        '''
        m = tf.math.reduce_max(self.logits, axis=-1, keepdims=True)
        x = self.logits - m
        z = tf.math.reduce_sum(tf.math.exp(x), axis=-1)
        xex = tf.math.multiply_no_nan(self.logits, tf.math.exp(x))
        p = tf.math.reduce_sum(xex, axis=-1) / z
        return m[..., 0] + tf.math.log(z) - p

    def kl(self, q):
        '''
        KL divergence

        q: target probability distribution (Categorical)
        '''
        logp = self._log_p()
        logq = q._log_p()
        p = tf.math.exp(logp)
        return tf.math.reduce_sum(p * (logq-logp), axis=-1)



class Normal(Distribution):
    def __init__(self, mean, scale, dtype=tf.float32, **kwargs):

        self._mean = tf.cast(tf.convert_to_tensor(mean), dtype=dtype)
        self._scale = tf.cast(tf.convert_to_tensor(scale), dtype=dtype)

        super().__init__(dtype=dtype, 
                        **kwargs)

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
        x = tf.cast(x, dtype=self.dtype)
        p = 0.5 * tf.math.squared_difference(x/self.scale, self.mean/self.scale)
        c = tf.constant(0.5 * np.log(2. * np.pi), dtype=self.dtype)
        z = c + tf.math.log(self.scale)
        return -(p+z)

    def mode(self):
        '''
        Mode
        '''
        return self.mean

    def sample(self):
        '''
        Sample outcomes
        '''
        x = tf.random.normal(tf.shape(self.mean), dtype=self.dtype)
        return self.mean + self.scale * x

    def entropy(self):
        '''
        Entropy
        '''
        c = tf.constant(0.5 * np.log(2. * np.pi), dtype=self.dtype) + 0.5
        return c + tf.math.log(self.scale)

    def kl(self, q):
        '''
        KL divergence

        q: target probability distribution (Normal)
        '''
        log_diff = tf.math.log(self.scale) - tf.math.log(q.scale)
        return (0.5 * tf.math.square_difference(self.mean/q.scale, q.mean/q.scale) +
                0.5 * tf.math.expm1(2. * log_diff) - log_diff)

class MultiNormal(Normal):
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

    def kl(self, q):
        '''
        KL divergence

        q: target probability distribution (MultiNormal)
        '''
        return tf.math.reduce_sum(super().kl(q), axis=-1)

