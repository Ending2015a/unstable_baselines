__copyright__ = '''
 The MIT License (MIT)
 Copyright (c) 2021 Joe Hsiao
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
 OR OTHER DEALINGS IN THE SOFTWARE.
'''
__license__ = 'MIT'


# --- built in ---
import os
import sys
import time
import random
import datetime

# --- 3rd party ---
import gym
import cloudpickle

import numpy as np
import tensorflow as tf

# --- my module ---
from unstable_baslines import logger

from unstable_baselines.bugs import ReLU

# create logger 
LOG = logger.getLogger('nets')

# === Common nets ===

class Identity(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs, training=True):
        return inputs

# CNN feature extractor
class NatureCnn(tf.keras.Model):
    def __init__(self, **kwargs):
        '''Nature CNN originated from 
        "Playing Atari with Deep Reinforcement Learning"
        '''        
        super().__init__(**kwargs)

        self._model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 8, 4),
            ReLU(),
            tf.keras.layers.Conv2D(64, 4, 2),
            ReLU(),
            tf.keras.layers.Conv2D(64, 3, 1),
            ReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512),
            ReLU()
        ])

    def call(self, inputs, training=True):
        '''Forward network

        Args:
            inputs (tf.Tensor): Expecing a 4-D batch observations 
                with shape (b, h, w, c)
            training (bool, optional): Training mode. Defaults to True.

        Returns:
            tf.Tensor: Latent vectors, shape (b, 512)
        '''        
        return self._model(inputs, training=training)

# Mlp feature extrator
class MlpNet(tf.keras.Model):
    '''MLP feature extractor'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._model = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64),
            ReLU(),
            tf.keras.layers.Dense(64),
            ReLU()
        ])

    def call(self, inputs, training=True):
        '''Forward network

        Args:
            inputs (tf.Tensor): Expecting a batch observations
                with shape (b, obs_space.shape)
            training (bool, optional): Training mode. Defaults to True.

        Returns:
            tf.Tensor: Latent vectors, shape (b, 64)
        '''        
        return self._model(inputs, training=training)


# === Policies ===

class CategoricalPolicyNet(tf.keras.Model):
    '''Categorical policy for discrete action space
    '''
    support_spaces = [gym.spaces.Discrete]
    def __init__(self, action_space, **kwargs):
        '''Create categorical policy net

        Args:
            action_space (gym.spaces.Spaces): Action space, must be
                one of the types listed in `support_spaces`
        '''
        super().__init__(**kwargs)

        if not isinstance(action_space, self.support_spaces):
            raise ValueError(f'{type(self).__name__} does not '
                f'suprt action space of type `{type(action_space)}`')

        self.action_space = action_space
        self._model = None

    def build(self, input_shape):
        self._model = self.get_model()
        super().build(input_shape)
    
    def call(self, inputs, training=True):
        '''Forward network

        Args:
            inputs (tf.Tensor): Expecting a latent vector in shape
                (b, latent), tf.float32
            training (bool, optional): Training mode. Defaults to True.

        Returns:
            Categorical: A categorical distribution
        '''        
        return ub_prob.Categorical(self._model(inputs, training=training))

    def get_model(self):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(self.action_space.n)
        ])

class DiagGaussianPolicyNet(tf.keras.Model):
    '''Tanh squashed diagonal Gaussian mixture policy net 
    with state-dependent covariance
    '''
    support_spaces = [gym.spaces.Box]
    def __init__(self, action_space, squash=False, **kwargs):
        '''Create diagonal gaussian policy net

        Args:
            action_space (gym.spaces): Action space, must be
                one of the types listed in `support_spaces`
            squash (bool): Tanh squashing. Default to False.
        '''
        super().__init__(**kwargs)

        if not isinstance(action_space, self.support_spaces):
            raise ValueError(f'{type(self).__name__} does not '
                f'suprt action space of type `{type(action_space)}`')

        self.action_space = action_space
        self.action_dims  = int(np.prod(action_space.shape))
        self.output_shape = action_space.shape
        self.squash       = squash

        self._mean_model   = None
        self._logstd_model = None

    def build(self, input_shape):
        '''Build model
        '''
        self._mean_model   = self.get_mean_model()
        self._logstd_model = self.get_logstd_model()
        super().build(input_shape)

    def call(self, inputs, training=True):
        '''Forward network

        Args:
            inputs (tf.Tensor): Expecting a latent vector in shape
                (b, latent), tf.float32
            training (bool, optional): Training mode. Defaults to True.

        Returns:
            MultiNormal: A multi variate gaussian distribution
        '''
        # forward model
        mean   = self._mean_model(inputs, training=training)
        logstd = self._logstd_model(inputs, training=training)
        std    = tf.math.softplus(logstd) + 1e-5
        # reshape as action space space
        mean = tf.reshape(mean, self.output_shape)
        std  = tf.reshape(std, self.output_shape)
        # create multi variate gauss dist with tah squashed
        distrib = ub_prob.MultiNormal(mean, std)
        if self.squash:
            distrib = ub_prob.Tanh(distrib)
        return distrib

    def get_mean_model(self):
        '''Mean/Loc model

        Returns:
            tf.keras.Model: mean/log model
        '''        
        return tf.keras.Sequential([
            tf.keras.layers.Dense(self.action_dims)
        ])

    def get_logstd_model(self):
        '''Log-std model

        Returns:
            tf.keras.Model: log-std model
        '''        
        return tf.keras.Sequential([
            tf.keras.layers.Dense(self.action_dims)
        ])

class PolicyNet(tf.keras.Model):
    support_spaces = [gym.spaces.Box, gym.spaces.Discrete]
    def __init__(self, action_space, squash=False, **kwargs):
        super().__init__(**kwargs)

        if not isinstance(action_space, self.support_spaces):
            raise ValueError(f'{type(self).__name__} does not '
                f'support action space of type `{type(action_space)}`')

        self.squash       = squash
        self.action_space = action_space
        self._model  = None
        self._policy = None

    def build(self, input_shape):
        self._model = self.get_model()

        if isinstance(self.action_space, gym.spaces.Discrete):
            self._policy = self.get_categorical_policy()
        elif isinstance(self.action_space, gym.spaces.Box):
            self._policy = self.get_gaussian_policy()
        else:
            raise ValueError(f'{type(self).__name__} does not '
                f'support action space of type `{type(action_space)}`')
        super().build(input_shape)

    def call(self, inputs, training=True):
        '''Forward network

        Args:
            inputs (tf.Tensor): Expecting a latent vectors
            training (bool, optional): Training mode. Defaults to True.
        '''
        x = inputs
        if self._model is not None:
            x = self._model(x, training=training)
        return self._policy(x, training=training)

    def get_model(self):
        '''Create model

        Returns:
            tf.keras.Model or None: model
        '''
        return None

    def get_categorical_policy(self):
        return CategoricalPolicyNet(self.action_space)

    def get_gaussian_policy(self):
        return DiagGaussianPolicyNet(self.action_space, self.squash)


# === Value networks ===

class ValueNet(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._model       = None

    def build(self, input_shape):
        self._model = self.get_model()
        super().build(input_shape)

    def call(self, inputs, training=True):
        return self._model(inputs, training=training)

    def get_model(self):
        '''Create model

        Returns:
            tf.keras.Model: model
        '''
        return tf.keras.Sequential([
            tf.keras.layers.Dense(1)
        ])

class MultiHeadValueNets(tf.keras.Model):
    def __init__(self, n_critics: int=2, **kwargs):
        super().__init__(**kwargs)

        self.n_critics    = n_critics
        self._models      = None

    def build(self, input_shape):
        self._models = [
            self.get_model()
            for n in range(n_critics)
        ]
        super().build(input_shape)
    
    def __getitem__(self, key):
        if not isinstance(key, int):
            raise KeyError(f'Key must be an `int`, got {type(key)}')
        
        return self._models[key]

    def call(self, inputs, training=True):
        '''Forward all critics

        Args:
            inputs (tf.Tensor): Expecting a batch latents with shape
                (b, latent), tf.float32
            training (bool, optional): Training mode. Defaults to True.

        Returns:
            tf.Tensor: (n_critics, b, 1)
        '''
        return tf.stack([
            self._models[n](inputs, training=training)
            for n in range(self.n_critics)
        ])

    def get_model(self):
        '''Create model

        Returns:
            tf.keras.Model: model
        '''        
        return tf.keras.Sequential([
            tf.keras.layers.Dense(1)
        ])