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
from unstable_baselines.lib import prob as ub_prob
from unstable_baselines.lib import patch as ub_patch

# === Common nets ===

class Identity(tf.keras.Model):
    '''Do nothing'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs, training=True):
        return inputs

class Constant(tf.keras.Model):
    '''Return a constant'''
    def __init__(self, constant, **kwargs):
        super().__init__(**kwargs)
        self.constant = constant
    def call(self, inputs, training=True):
        # broadcast shape to batch shape (b, *constant.shape)
        batch_shape = tf.shape(inputs)[0]
        value = tf.expand_dims(self.constant, axis=0)
        return tf.repeat(value, batch_shape, axis=0)

class MlpNet(tf.keras.Model):
    '''MLP feature extractor'''
    def __init__(self, hiddens=[64, 64], **kwargs):
        super().__init__(**kwargs)

        layers = [tf.keras.layers.Flatten()]
        for h in hiddens:
            layers.extend([
                tf.keras.layers.Dense(h),
                ub_patch.ReLU()
            ])
        
        self._model = tf.keras.Sequential(layers)

    def call(self, inputs, training=True):
        return self._model(inputs, training=training)

class NatureCnn(tf.keras.Model):
    def __init__(self, **kwargs):
        '''Nature CNN originated from 
        "Playing Atari with Deep Reinforcement Learning"
        '''        
        super().__init__(**kwargs)

        self._model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 8, 4),
            ub_patch.ReLU(),
            tf.keras.layers.Conv2D(64, 4, 2),
            ub_patch.ReLU(),
            tf.keras.layers.Conv2D(64, 3, 1),
            ub_patch.ReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512),
            ub_patch.ReLU()
        ])

    def call(self, inputs, training=True):
        return self._model(inputs, training=training)

# === Policies ===

class CategoricalPolicyNet(tf.keras.Model):
    '''Categorical policy for discrete action space'''
    support_spaces = [gym.spaces.Discrete]
    def __init__(self, action_space, **kwargs):
        '''Create categorical policy net

        Args:
            action_space (gym.spaces.Spaces): Action space, must be
                one of the types listed in `support_spaces`
        '''
        super().__init__(**kwargs)

        if not isinstance(action_space, tuple(self.support_spaces)):
            raise ValueError(f'{type(self)} does not support '
                f'action spaces of type `{type(action_space)}`')

        self.action_space = action_space
        self._model = None

    def build(self, input_shape):
        self._model = self.create_model()
    
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

    def create_model(self):
        '''Override this method to customize model arch'''
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

        if not isinstance(action_space, tuple(self.support_spaces)):
            raise ValueError(f'{type(self).__name__} does not '
                f'suprt action space of type `{type(action_space)}`')

        self.action_space = action_space
        self.action_dims  = int(np.prod(action_space.shape))
        self.action_shape = action_space.shape
        self.squash       = squash

        self._mean_model   = None
        self._logstd_model = None

    def build(self, input_shape):
        '''Build model
        '''
        self._mean_model   = self.create_mean_model()
        self._logstd_model = self.create_logstd_model()

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
        output_shape = [-1] + list(self.action_shape)
        mean = tf.reshape(mean, output_shape)
        std  = tf.reshape(std, output_shape)
        # create multi variate gauss dist with tah squashed
        distrib = ub_prob.MultiNormal(mean, std)
        if self.squash:
            distrib = ub_prob.Tanh(distrib)
        return distrib

    def create_mean_model(self):
        '''Mean/Loc model
        Override this method to customize model arch

        Returns:
            tf.keras.Model: mean/log model
        '''        
        return tf.keras.Sequential([
            tf.keras.layers.Dense(self.action_dims)
        ])

    def create_logstd_model(self):
        '''Log-std model
        Override this method to customize model arch

        Returns:
            tf.keras.Model: log-std model
        '''        
        return tf.keras.Sequential([
            tf.keras.layers.Dense(self.action_dims)
        ])

class PolicyNet(tf.keras.Model):
    support_spaces = [gym.spaces.Box, gym.spaces.Discrete]
    #TODO: support other spaces
    def __init__(self, action_space, squash=False, net=None, **kwargs):
        '''Base Policy net
        Override `create_model` or `create_*_policy` method to customize
        model architecture

        Args:
            action_space (gym.Space): The action space.
            squash (bool, optional): Apply Tanh squash for continuous (Box) 
                action space. Defaults to False.
            net (tf.keras.Model, optional): Base network, feature extractor.
                Defaults to None.

        Raises:
            ValueError: if the given action space type does not contain in the 
                `support_spaces` list
        '''
        super().__init__(**kwargs)
        # check if space supported
        if not isinstance(action_space, tuple(self.support_spaces)):
            raise ValueError(f'{type(self)} does not '
                f'support the action space of type {type(action_space)}')
        
        self.action_space = action_space
        self.squash  = squash
        self._model  = net
        self._policy = None
    
    def build(self, input_shape):
        # create feature extraction model
        if self._model is None:
            self._model = Identity()
        # create policy
        if isinstance(self.action_space, gym.spaces.Discrete):
            self._policy = self.create_categorical_policy()
        elif isinstance(self.action_space, gym.spaces.Box):
            self._policy = self.create_gaussian_policy()
        else:
            raise ValueError(f'{type(self)} dose not '
                f'support the action space of type {type(action_space)}')

    def call(self, inputs, training=True):
        '''Forward network

        Args:
            inputs (tf.Tensor): Input tensors, shape (b, *)
            training (bool, optional): Training mode. Defaults to True.

        Returns:
            Distribution: Action distributions.
        '''
        x = self._model(inputs, training=training)
        return self._policy(x, training=training)
    
    def create_categorical_policy(self):
        '''Policy for Discrete action spaces'''
        return CategoricalPolicyNet(self.action_space)

    def create_gaussian_policy(self):
        '''Policy for Box action space'''
        return DiagGaussianPolicyNet(self.action_space, self.squash)


# === Value networks ===

class ValueNet(tf.keras.Model):
    def __init__(self, net=None, **kwargs):
        '''Base value net

        Args:
            net (tf.keras.Model, optional): Base network, feature extractor. 
                Defaults to None.
        '''        
        super().__init__(**kwargs)
        self._model = net

    def build(self, input_shape):
        # create empty net
        if self._model is None:
            self._model = Identity()
        self._model = tf.keras.Sequential([
            self._model,
            self.create_value_model()
        ])

    def call(self, inputs, training=True):
        '''Forward value net

        Args:
            inputs (tf.Tensor): Input tensors, shape (b, *).
            training (bool, optional): Training mode. Defaults to True.

        Returns:
            tf.Tensor: value predictions, (Default) shape (b, 1)
        '''        
        return self._model(inputs, training=training)
    
    def create_value_model(self):
        '''Create value model
        Override this method to customize model arch

        Returns:
            tf.keras.Model: model
        '''
        return tf.keras.Sequential([
            tf.keras.layers.Dense(1)
        ])

class MultiHeadValueNets(tf.keras.Model):
    def __init__(self, n_heads: int=2, nets=None, **kwargs):
        '''Base multi head value net

        Args:
            n_heads (int, optional): Number of heads. Defaults to 2.
            nets (tf.keras.Model, list, optional): Base networks, 
                feature extractors. Can be a tf.keras.Model or a list
                of tf.keras.Model. Note that if only a tf.keras.Model 
                was provided, it is shared across all heads. The count 
                of tf.keras.Model of the list must match to `n_heads`. 
                Defaults to None.
        '''
        super().__init__(**kwargs)
        self.n_heads   = n_heads
        self._models   = nets

    def build(self, input_shape):
        if self._models is None:
            self._models = [
                Identity()
                for n in range(self.n_heads)
            ]
        elif isinstance(self._models, (list, tuple)):
            pass
        else:
            # share net, copy the same net multiple times
            self._models = [
                self._models
                for n in range(self.n_heads)
            ]
        assert len(self._models) == self.n_heads
        # create value nets
        self._models = [
            tf.keras.Sequential([
                model,
                self.create_value_model()
            ])
            for model in self._models
        ]
    
    def __getitem__(self, key):
        '''Get a specified value net

        Args:
            key (int): model index

        Returns:
            tf.keras.Model: a specified value net
        '''
        if self._models is None:
            raise ValueError(f'Models for {self.name} have not yet been'
                'created. Models are created when the Model is first called'
                'on inputs or `build()` is called with and `input_shape`')
        if not isinstance(key, int):
            raise KeyError(f'Key must be an `int`, got {type(key)}')
        
        try:
            return self._models[key]
        except IndexError:
            raise IndexError(f'Index out of range, n_heads={self.n_heads}'
                f'got key={key}')

    def call(self, inputs, axis=0, training=True):
        '''Forward all critics

        Args:
            inputs (tf.Tensor): Expecting a batch latents with shape
                (b, latent), tf.float32
            axis (int): axis to stack values.
            training (bool, optional): Training mode. Defaults to True.

        Returns:
            tf.Tensor: Value predictions. (Default) shape (n_heads, b, 1)
        '''
        return tf.stack([
            self._models[n](inputs, training=training)
            for n in range(self.n_heads)
        ], axis=axis)

    def create_value_model(self):
        '''Create model
        Override this method to customize model arch

        Returns:
            tf.keras.Model: model
        '''        
        return tf.keras.Sequential([
            tf.keras.layers.Dense(1)
        ])