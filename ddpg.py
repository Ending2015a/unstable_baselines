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
import logging
import argparse

# --- 3rd party ---
import gym

import numpy as np
import tensorflow as tf


# --- my module ---
from unstable_baselines import logger

logger.Config.use(level='DEBUG', colored=True, reset=False)
LOG = logger.getLogger('DDPG')


# === Noises ===
class AdaptiveParamNoiseSpec():
    '''
    Implements adaptive parameter noise
    see paper: https://arxiv.org/abs/1706.01905
    '''
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coeeficient=1.01)
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coeeficient = adoption_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev
            self.current_stddev /= self.adoption_coefficient
        else:
            # Increase stddev
            self.current_stddev *= self.adoption_coefficient

    def get_stats(self):
        return {'param_noise_stddev': self.current_stddev}

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adoption_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adoption_coefficient)

class OrnsteinUhlenbeckActionNoise():
    '''
    Implements Ornstein Unlenbeck noise, used by the original DDPG algorithm
    TODO
    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    '''
    def __init__(self, mean, stddev, theta=.15, dt=1e-2):
        self.theta = theta
        self.mean = mean
        self.stddev = stddev
        self.dt = dt
        self.noise_prev = None
        self.reset()

    def __call__(self):
        noise = self.noise_prev

    def reset(self):
        pass

# === Replay buffer ===
class Buffer():

# === Networks ===


class PolicyNet(tf.keras.Model):
    def __init__(self, action_space):
        super().__init__()
        
        assert isinstance(action_space, gym.spaces.Box)
        self.action_space = action_space

        self._layers = [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, name='fc1'),
            tf.keras.layers.LayerNormalization(name='layernorm1'),
            tf.keras.layers.ReLU(name='relu1'),
            tf.keras.layers.Dense(64, name='fc2'),
            tf.keras.layers.LayerNormalization(name='layernorm1'),
            tf.keras.layers.ReLU(name='relu1'),
            tf.keras.layers.Dense(action_space.shape[0], name='fc3'),
            tf.keras.layers.Activation(activation='tanh', name='tanh')
        ]
    
    @tf.function
    def call(self, inputs, training=False):
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x

class ValueNet(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self._obs_layers = [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, name='fc1'),
            tf.keras.layers.LayerNormalization(name='layernorm1'),
            tf.keras.layers.ReLU(name='relu1'),
        ]
        self._concat = tf.keras.layers.Concatenate()
        self._layers = [
            tf.keras.layers.Dense(64, name='fc2'),
            tf.keras.layers.LayerNormalization(name='layernorm1'),
            tf.keras.layers.ReLU(name='relu1'),
            tf.keras.layers.Dense(1, name='fc3')
        ]
    
    @tf.function
    def call(self, obs, action, training=False):
        x = obs
        for layer in self._obs_layers:
            x = layer(x)
        
        x = self._concat([x, action])

        for layer in self._layers:
            x = layer(x)

        return x



class DDPG(tf.keras.Model):
    def __init__(self):
        pass
    
    def call(self):
        pass

if __name__ == '__main__':

    # Reset logger
    logger.Config.use(filename='test.log', level='DEBUG', colored=True, reset=False)
    LOG = logger.getLogger('DDPG')

    # Print welcome message
    LOG.add_row('')
    LOG.add_rows('DDPG', fmt='{:@f:ANSI_Shadow}', align='center')
    LOG.add_line()
    LOG.add_rows('{}'.format(__copyright__))
    LOG.flush('INFO')