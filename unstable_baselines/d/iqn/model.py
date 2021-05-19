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

import numpy as np
import tensorflow as tf

# --- my module ---
from unstable_baselines import logger


from unstable_baselines.base import (SavableModel, 
                                     TrainableModel)
from unstable_baselines.bugs import ReLU
from unstable_baselines.sche import Scheduler
from unstable_baselines.utils import (is_image_observation,
                                      preprocess_observation,
                                      get_input_tensor_from_space)

# create logger
LOG = logger.getLogger('IQN')

# === Buffers ===

class ReplayBuffer():
    '''
    Replay buffer
    '''
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.reset()

    def reset(self):
        self.pos = 0
        self.full = False

        self.obss      = None
        self.acts      = None
        self.next_obss = None
        self.rews      = None
        self.dones     = None

    def add(self, observations, next_observations, actions, rewards, dones):
        '''Add new samples into replay buffer

        Args:
            observations (np.ndarray): numpy array of type np.uint8,
                shape (n_envs, obs_space.shape).
            next_observations (np.ndarray): numpy array of type np.uint8,
                shape (n_envs, obs_space.shape).
            actions (np.ndarray): discrete actions, numpy array of type 
                np.int32 or np.int64, shape (n_envs, act_space.n)
            rewards (np.ndarray): numpy array of type np.float32 or 
                np.float64, shape (n_envs,)
            dones (np.ndarray): numpy array of type np.float32 or
                np.bool, shape (n_envs,)
        '''

        obss      = np.asarray(observations)
        next_obss = np.asarray(next_observations)
        actions   = np.asarray(actions)
        rewards   = np.asarray(rewards)
        dones     = np.asarray(dones)

        n_env = obss.shape[0]

        if self.obss is None:
            # create spaces
            self.obss      = np.zeros((self.buffer_size, ) + obss.shape[1:],    dtype=obss.dtype)
            self.acts      = np.zeros((self.buffer_size, ) + actions.shape[1:], dtype=actions.dtype)
            self.next_obss = np.zeros((self.buffer_size, ) + obss.shape[1:],    dtype=obss.dtype)
            self.rews      = np.zeros((self.buffer_size, ) + rewards.shape[1:], dtype=rewards.dtype)
            self.dones     = np.zeros((self.buffer_size, ) + dones.shape[1:],   dtype=dones.dtype)

        idx = np.arange(self.pos, self.pos+n_env) % self.buffer_size

        self.obss[idx, ...]      = obss.copy()
        self.acts[idx, ...]      = actions.copy()
        self.next_obss[idx, ...] = next_obss.copy()
        self.rews[idx, ...]      = rewards.copy()
        self.dones[idx, ...]     = dones.copy()

        # increase start position
        self.pos += n_env

        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = self.pos % self.buffer_size

    def __len__(self):
        if self.full:
            return self.buffer_size
        else:
            return self.pos

    def __call__(self, batch_size=None):
        '''Randomly sample a batch from replay buffer

        Args:
            batch_size (int, optional): Batch size. Defaults to None.

        Returns:
            np.ndarray: observations, shape (batch_size, obs_space.shape)
            np.ndarray: actions, shape (batch_size, act_space.n)
            np.ndarray: next observations, shape (batch_size, obs_space.shape)
            np.ndarray: dones, shape (batch_size,)
            np.ndarray: rewards, shape (batch_size,)
        '''
        if batch_size is None:
            batch_size = len(self)

        batch_inds = np.random.randint(0, len(self), size=batch_size)

        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds):
        return (self.obss[batch_inds],
                self.acts[batch_inds],
                self.next_obss[batch_inds],
                self.dones[batch_inds],
                self.rews[batch_inds])

# === Networks ===

class PhiNet(tf.keras.Model):
    '''Function Phi in IQN paper'''
    def __init__(self, latent_size, embed_size=64, **kwargs):
        '''Phi net (quantile distortion function)

        Args:
            latent_size (int): The size of latent vectors from
                feature extraction nets.
            embed_size (int, optional): The embedding dimensions 
                (number of cosine samples) Defaults to 64.
        '''
        super().__init__(**kwargs)

        self.embed_size = embed_size

        self._layers = [
            tf.keras.layers.Dense(latent_size),
            ReLU(name='relu1')
        ]

    @tf.function
    def call(self, inputs, training=False):
        '''Forward network

        Args:
            inputs (tf.Tensor): taus in original paper. Expecting a tf.float32 
                vector in shape (batch, n_taus).
            training (bool, optional): Training mode. Defaults to False.

        Returns:
            tf.Tensor: phi(tau) = ReLU(sum(cos(pi*i*tau)w + b))
                shape (batch, n_taus, latent_size), dtype tf.float32.
        '''
        tau     = tf.expand_dims(inputs, axis=-1)     # (batch, n_taus, 1)
        i_pi    = np.pi * tf.range(self.embed_size, dtype=inputs.dtype)
        cos_tau = tf.math.cos(i_pi * tau)    # (batch, n_taus, embed_size)
        # forward mlp
        x = cos_tau
        for layer in self._layers:
            x = layer(x)
        return x

class IqnNatureCnn(tf.keras.Model):
    '''IQN version of Nature CNN, originated from
    "Playing Atari with Deep Reinforcement Learning"
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._h_layers = [
            tf.keras.layers.Conv2D(32, 8, 4, name='conv1'),
            ReLU(name='relu1'),
            tf.keras.layers.Conv2D(64, 4, 2, name='conv2'),
            ReLU(name='relu2'),
            tf.keras.layers.Conv2D(64, 3, 1, name='conv3'),
            ReLU(name='relu3'),
            tf.keras.layers.Flatten()
        ]

        self._t_layers = [
            tf.keras.layers.Dense(512, name='fc'),
            ReLU(name='relu4')
        ]

    def call(self, inputs, training=False, _get_middle=False):
        '''Forward network

        If `_get_middle` is True, inputs accept batch observations.
        If it's False, inputs accept a (latent, taus) tuple. The arg 
        `_get_middle` is only used to retrieve the shape of the hidden 
        layer in the middle part of the feature extraction networks, 
        which is needed when constructing the networks (PhiNet's 
        output_size). So, leave it as False.

        Args:
            inputs (tf.Tensor or Tuple): A tf.Tensor if `_get_middle` is 
                False, or a tuple if it is True.
                [0] (tf.Tensor): Expecting 4-D batch observations, shape 
                    (batch, height, width, channel) for `_get_middle=False`.
                    Otherwise, a latent vector in shape (batch, latent), 
                    tf.float32.
                [1] (tf.Tensor): Distorted quantile samples phi(tau), shape
                    (batch, n_taus, latent), tf.float32.
            _get_middle (bool, optional): This arg is used to get the output
                shape of the middle part during the networks construction
                phase. Defaults to False.
            training (bool, optional): Training mode. Defaults to False.

        Returns:
            tf.Tensor: Latent vectors. If `extract_features` is True,
                return a tensor in shape (batch, latent). Otherwise,
                return shape (batch, n_taus, latent). tf.float32.
        '''
        if _get_middle:
            x = inputs
        else:
            if not isinstance(inputs, (tuple, list)):
                raise RuntimeError('`inputs` must be of type tuple, got '
                                '{}'.format(type(inputs)))
            x = inputs[0]
        
        # forward head layers
        for layer in self._h_layers:
            x = layer(x)

        if _get_middle:
            return x

        x = tf.expand_dims(x, axis=1) * inputs[1]
        # forward remaining layers
        for layer in self._t_layers:
            x = layer(x)
        
        return x


# Mlp feature extractor
class IqnMlpNet(tf.keras.Model):
    '''IQN version of MLP network'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._h_layers = [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, name='fc1'),
            ReLU(name='relu1')
        ]

        self._t_layers = [
            tf.keras.layers.Dense(64, name='fc2'),
            ReLU(name='relu2')
        ]
    
    def call(self, inputs, training=False, _get_middle=False):
        '''Forward network

        If `_get_middle` is True, inputs accept batch observations.
        If it's False, inputs accept a (latent, taus) tuple. The arg 
        `_get_middle` is only used to retrieve the shape of the hidden 
        layer in the middle part of the feature extraction networks, 
        which is needed when constructing the networks (PhiNet's 
        output_size). So, leave it as False.

        Args:
            inputs (tf.Tensor or Tuple): A tf.Tensor if `_get_middle` is 
                False, or a tuple if it is True.
                [0] (tf.Tensor): Expecting batch observations, shape 
                    (batch, obs_space.shape) for `_get_middle=False`.
                    Otherwise, a latent vector in shape (batch, latent), 
                    tf.float32.
                [1] (tf.Tensor): Distorted quantile samples phi(tau), shape
                    (batch, n_taus, latent), tf.float32.
            _get_middle (bool, optional): This arg is used to get the output
                shape of the middle part during the networks construction
                phase. Defaults to False.
            training (bool, optional): Training mode. Defaults to False.

        Returns:
            tf.Tensor: Latent vectors. If `extract_features` is True,
                return a tensor in shape (batch, latent). Otherwise,
                return shape (batch, n_taus, latent). tf.float32.
        '''
        if _get_middle:
            x = inputs
        else:
            if not isinstance(inputs, tuple):
                raise RuntimeError('`inputs` must be of type tuple, got '
                                '{}'.format(type(inputs)))
            x = inputs[0]
        
        # forward head layers
        for layer in self._h_layers:
            x = layer(x)

        if _get_middle:
            return x
        
        x = tf.expand_dims(x, axis=1) * inputs[1]
        # forward remaining layers
        for layer in self._t_layers:
            x = layer(x)
        
        return x

# Q-value network
class QNet(tf.keras.Model):
    def __init__(self, action_space, **kwargs):
        super().__init__(**kwargs)

        self._layers = [
            tf.keras.layers.Dense(action_space.n)
        ]

    @tf.function
    def call(self, inputs, training=False):
        '''Forward network

        Args:
            inputs (tf.Tensor): Expecting a latent vector in shape 
                (batch, n_taus, latent), tf.float32.
            training (bool, optional): Training mode. Defaults to False.

        Returns:
            tf.Tensor: Predicted Q values in shape (batch, n_taus, act_space.n)
        '''
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x

# === Agent, Model ===

class Agent(SavableModel):
    def __init__(self, observation_space, action_space, 
                       embed_size=64, force_mlp=False, **kwargs):
        '''IQN Agent

        Args:
            observation_space (gym.Spaces): The observation space of the 
                environment. Set None for delayed setup.
            action_space (gym.Spaces): The action space of the environment. 
                Set None for delayed setup.
            embed_size (int, optional): The embedding dimensions (number 
                of cosine samples) used in Phi net. Default to 64.
            force_mlp (bool, optional): Force to use MLP feature extractor.
                Defaults to False.
        '''
        super().__init__(**kwargs)

        self.embed_size  = embed_size
        self.force_mlp   = force_mlp

        # --- Initialize ---
        self.observation_space = None
        self.action_space      = None
        self.net               = None
        self.phi_net           = None
        self.q_net             = None

        if (observation_space is not None) and (action_space is not None):
            self.setup_model(observation_space, action_space)

    def setup_model(self, observation_space, action_space):
        '''Setup model and networks

        Args:
            observation_space (gym.Spaces): The observation space of the
                environment.
            action_space (gym.Spaces): The action space of the environment.
        '''
        
        self.observation_space = observation_space
        self.action_space      = action_space

        # --- setup model ---
        if (is_image_observation(observation_space)
                and (not self.force_mlp)):
            self.net = IqnNatureCnn()
        else:
            self.net = IqnMlpNet()

        # construct networks
        inputs = get_input_tensor_from_space(observation_space) # (batch, obs_space.shape)
        latents = self.net(inputs, _get_middle=True) # (batch, latent)

        latent_size = latents.shape[-1]
        self.phi_net = PhiNet(latent_size=latent_size, 
                              embed_size=self.embed_size)
        self.q_net = QNet(action_space)

        taus = tf.keras.Input(shape=(None,), dtype=tf.float32) # (batch, n_taus)
        phitaus = self.phi_net(taus) # (batch, n_taus, latent_size)
        outputs = self.net([inputs, phitaus]) # (batch, n_taus, latent)
        self.q_net(outputs) # (batch, n_taus, act_space.n)

    @tf.function
    def _forward(self, inputs, training=True):
        '''Forward actor

        Args:
            inputs (tf.Tensor): A (observation, tau) tuple
                [0] (tf.Tensor): batch observations in shape (batch, obs_space.shape).
                    tf.uint8 for image observations and tf.float32 for non-image 
                    observations.
                [1] (tf.Tensor): taus in shape (batch, n_taus)
            training (bool, optional): Training mode. Default to True.

        Return:
            tf.Tensor: predicted quentile Q values in shape (batch, act_space.n, n_taus),
                tf.float32
        '''
        inputs, taus = inputs

        # cast and normalize non-float32 inputs (e.g. image with uint8)
        inputs = preprocess_observation(inputs, self.observation_space)
        # forward network
        phi_t  = self.phi_net(taus, training=training)
        latent = self.net([inputs, phi_t], training=training)
        # forward value net
        values = self.q_net(latent, training=training) # (batch, n_taus, act_space.n)
        values = tf.transpose(values, (0, 2, 1))       # (batch, act_space.n, n_taus)

        return values

    @tf.function
    def call(self, inputs, training=True):
        '''Batch predict actions

        Args:
            inputs (tf.Tensor): A (observation, tau) tuple
                [0] (tf.Tensor): batch observations in shape (batch, obs_space.shape).
                    tf.uint8 for image observations and tf.float32 for non-image 
                    observations.
                [1] (tf.Tensor): taus in shape (batch, n_taus)
            training (bool, optional): Training mode. Default to True.

        Returns:
            tf.Tensor: Predicted actions in shape (batch, ), tf.int64
            tf.Tensor: Predicted state-action values in shape (batch, act_space.n),
                tf.float32
            tf.Tensor: Predicted quantile Q values shape (batch, act_space.n, n_taus)
        '''

        # forward
        quan_vals = self._forward(inputs, training=training)  # (batch, act_space.n, n_taus)
        values    = tf.math.reduce_mean(quan_vals, axis=-1)   # (batch, act_space.n)
        actions   = tf.math.argmax(values, axis=-1)           # (batch,)
        
        return actions, values, quan_vals
    
    def predict(self, inputs, n_ks: int=32):
        '''Predict actions

        Args:
            inputs (np.ndarray): batch observations in shape (batch, obs_space.shape)
                or a single observation in shape (obs_space.shape). np.uint8 for image
                observations and np.float32 for non-image observations.
            n_ks (int, optional): Number of quantile samples. Defaults to 32.

        Returns:
            np.ndarray: predicted actions in shape (batch, ) or (), np.int64
        '''

        one_sample = (len(inputs.shape) == len(self.observation_space.shape))

        if one_sample:
            inputs = np.expand_dims(inputs, axis=0)

        # random generate taus
        taus = tf.random.uniform(shape=(inputs.shape[0], n_ks), 
                                maxval=1.0, dtype=tf.float32)
        # predict
        outputs, *_ = self([inputs, taus], training=False)
        outputs     = np.asarray(outputs)

        if one_sample:
            outputs = np.squeeze(outputs, axis=0)

        # predict
        return outputs

    def get_config(self):
        
        config = {'observation_space': self.observation_space,
                  'action_space':      self.action_space,
                  'embed_size':        self.embed_size,
                  'force_mlp':         self.force_mlp}

        return config


class IQN(TrainableModel):
    def __init__(self, env, learning_rate:        float = 3e-4,
                            buffer_size:            int = int(1e6),
                            min_buffer:             int = 50000,
                            n_taus:                 int = 64,
                            n_target_taus:          int = 64,
                            n_ks:                   int = 32,
                            n_steps:                int = 4,
                            n_gradsteps:            int = 1,
                            batch_size:             int = 64,
                            embed_size:             int = 64,
                            gamma:                float = 0.99,
                            tau:                  float = 1.0,
                            kappa:                float = 1.0,
                            max_grad_norm:        float = 0.5,
                            force_mlp:             bool = False,
                            explore_schedule: Scheduler = 0.3,
                            verbose:                int = 0,
                            **kwargs):
        '''Implicit Quantile Networks (IQN)

        The implementation mainly follows its originated paper
        `Implicit Quantile Networks for Distributional Reinforcement Learning` by Dabney et al.

        The first argument `env` can be `None` for delayed model setup. You 
        should call `set_env()` then call `setup_model()` to manually setup 
        the model.

        Args:
            env (gym.Env): Training environment. Can be `None`.
            learning_rate (float, optional): Learning rate. Defaults to 3e-4.
            buffer_size (int, optional): Maximum size of the replay buffer. Defaults to 1000000.
            min_buffer (int, optional): Minimum size of the replay buffer before training. 
                Defaults to 50000.
            n_taus (int, optional): Number of quantile samples. Default to 64.
            n_target_taus (int, optional): Number of target quantile samples. Default to 64.
            n_ks (int, optional): Number of quantile samples for action selections. Default to 32.
            n_steps (int, optional): Number of steps of rollouts to collect for every epoch. 
                Defaults to 100.
            n_gradsteps (int, optional): Number of gradient steps in one epoch. 
                Defaults to 200.
            batch_size (int, optional): Training batch size. Defaults to 128.
            gamma (float, optional): Decay rate. Defaults to 0.99.
            tau (float, optional): Polyak update parameter. Defaults to 1.0.
            kappa (float, optional): Kappa for quantile huber loss. Defaults to 1.0.
            max_grad_norm (float, optional): Gradient clip range. Defaults to 0.5.
            force_mlp (bool, optional): Force to use MLP feature extractor. Defaults to False.
            explore_schedule (Sheduler, optional): Epsilon greedy scheduler. Defaults to 0.3.
            verbose (int, optional): More training log. Defaults to 0.
        '''
        super().__init__(**kwargs)

        self.env = env

        self.learning_rate    = learning_rate
        self.buffer_size      = buffer_size
        self.min_buffer       = min_buffer
        self.n_taus           = n_taus
        self.n_target_taus    = n_target_taus
        self.n_ks             = n_ks
        self.n_steps          = n_steps
        self.n_gradsteps      = n_gradsteps
        self.batch_size       = batch_size
        self.embed_size       = embed_size
        self.gamma            = gamma
        self.tau              = tau
        self.kappa            = kappa
        self.max_grad_norm    = max_grad_norm
        self.force_mlp        = force_mlp
        self.explore_schedule = explore_schedule
        self.verbose          = verbose

        # initialize states
        self.buffer            = None
        self.tb_writer         = None
        self.observation_space = None
        self.action_space      = None
        self.n_envs            = 0

        if env is not None:
            self.set_env(env)
            self.setup_model(env.observation_space, env.action_space)


    def set_env(self, env):
        '''Set environment

        If the environment is already set, you can call this function
        to change the environment. But the observation space and action
        space must be consistent with the original one.

        Args:
            env (gym.Env): Training environment.
        '''

        if self.observation_space is not None:
            assert env.observation_space == self.observation_space, 'Observation space mismatch, expect {}, got {}'.format(
                                                                        self.observation_space, env.observation_space)

        if self.action_space is not None:
            assert env.action_space == self.action_space, 'Action space mismatch, expect {}, got {}'.format(
                                                                self.action_space, env.action_space)
        
        self.env    = env
        self.n_envs = env.n_envs

    def setup_model(self, observation_space, action_space):
        '''Setup model, optimizer and scheduler for training

        Args:
            observation_space (gym.Spaces): The observation space of the
                environment.
            action_space (gym.Spaces): The action space of the environment.
        '''
        
        self.observation_space = observation_space
        self.action_space      = action_space

        # --- setup model ---
        self.buffer       = ReplayBuffer(buffer_size=self.buffer_size)
        self.agent        = Agent(self.observation_space, self.action_space, 
                                embed_size=self.embed_size, force_mlp=self.force_mlp)
        self.agent_target = Agent(self.observation_space, self.action_space, 
                                embed_size=self.embed_size, force_mlp=self.force_mlp)

        self.optimizer    = tf.keras.optimizers.Adam(learning_rate=self.learning_rate,
                                                     clipnorm=self.max_grad_norm)

        # initialize target
        self.agent_target.update(self.agent)

        # setup scheduler
        self.explore_schedule = Scheduler.get_scheduler(self.explore_schedule, 
                                                        state_object=self.state)

    @tf.function
    def _forward(self, inputs, training=True):
        '''Forward actor

        Args:
            inputs (tf.Tensor): A (observation, tau) tuple
                [0] (tf.Tensor): batch observations in shape (batch, obs_space.shape).
                    tf.uint8 for image observations and tf.float32 for non-image 
                    observations.
                [1] (tf.Tensor): taus in shape (batch, n_taus)
            training (bool, optional): Training mode. Default to True.

        Return:
            tf.Tensor: predicted quentile Q values in shape (batch, act_space.n, n_taus),
                tf.float32
        '''
        return self.agent._forward(inputs, training=training)

    @tf.function
    def call(self, inputs, training=True):
        '''Batch predict actions

        Args:
            inputs (tf.Tensor): A (observation, tau) tuple
                [0] (tf.Tensor): batch observations in shape (batch, obs_space.shape).
                    tf.uint8 for image observations and tf.float32 for non-image 
                    observations.
                [1] (tf.Tensor): taus in shape (batch, n_taus)
            training (bool, optional): Training mode. Default to True.

        Returns:
            tf.Tensor: Predicted actions in shape (batch, ), tf.int64
            tf.Tensor: Predicted state-action values in shape (batch, act_space.n),
                tf.float32
            tf.Tensor: Predicted quantile Q values shape (batch, act_space.n, n_taus)
        '''
        return self.agent(inputs, training=training)

    def predict(self, inputs, n_ks: int=None):
        '''Predict actions

        Args:
            inputs (np.ndarray): batch observations in shape (batch, obs_space.shape)
                or a single observation in shape (obs_space.shape). np.uint8 for image
                observations and np.float32 for non-image observations.
            n_ks (int, optional): Number of quantile samples. Defaults to 32.

        Returns:
            np.ndarray: predicted actions in shape (batch, ) or (), np.int64
        '''
        return self.agent.predict(inputs, n_ks=n_ks or self.n_ks)

    @tf.function
    def value_loss(self, obs, action, next_obs, done, reward):
        '''Compute loss

        Args:
            obs (tf.Tensor): batch observations, shape (batch, obs_space.shape),
                tf.uint8 for image observations, tf.float32 for non-image observations
            action (tf.Tensor): batch actions, shape (batch, ),
                tf.int32 or tf.int64 for discrete action space
            next_obs (tf.Tensor): batch next observations, shape (batch, obs_space.shape),
                tf.uint8 for image observations, tf.float32 for non-image observations
            done (tf.Tensor): batch done, shape (batch, ), tf.bool or tf.float32
            reward (tf.Tensor): batch reward, shape (batch, ), tf.float32

        Returns:
            tf.Tensor: loss, tf.float32
        '''

        action = tf.cast(action, dtype=tf.int64)
        reward = tf.cast(reward, dtype=tf.float32)
        done   = tf.cast(done, dtype=tf.float32)

        reward = tf.expand_dims(reward, axis=-1) # (batch, 1)
        done   = tf.expand_dims(done, axis=-1)   # (batch, 1)

        # sample target quantiles
        target_taus = tf.random.uniform(shape=(obs.shape[0], self.n_target_taus), 
                                        maxval=1.0, dtype=tf.float32) # (batch, n_target_taus)
        # compute target quantile q values
        next_act, _, next_qs = self.agent_target([next_obs, target_taus])
        target_q = tf.gather(next_qs, indices=next_act, batch_dims=1) # (batch, n_target_taus)

        y     = reward + (1.-done) * self.gamma * target_q # (batch, n_target_taus)
        y     = tf.stop_gradient(y)

        # sample quantiles
        taus  = tf.random.uniform(shape=(obs.shape[0], self.n_taus),
                                maxval=1.0, dtype=tf.float32) # (batch, n_taus)
        # compute current quantile q values
        qs    = self.agent._forward([obs, taus]) # (batch, act_space.n, n_taus)
        q     = tf.gather(qs, indices=action, batch_dims=1) # (batch, n_taus)

        # compute huber loss
        y     = tf.expand_dims(y, axis=-2) # (batch, 1, n_target_taus)
        q     = tf.expand_dims(q, axis=-1) # (batch, n_taus, 1)
        u     = y - q                     # (batch, n_taus, n_target_taus) td error
        abs_u = tf.math.abs(u)
        huber = tf.where(abs_u > self.kappa, self.kappa * (abs_u - 0.5*self.kappa), 
                                             0.5 * tf.math.square(u))      # (batch, n_taus, n_target_taus)
        tau_i = tf.expand_dims(taus, axis=-1) # (batch, n_taus, 1)
        loss  = tf.abs(tau_i - tf.cast(u < 0.0, dtype=tf.float32)) * huber # (batch, n_taus, n_target_taus)
        loss  = tf.math.reduce_mean(tf.math.reduce_sum(loss, axis=-2))

        return loss

    @tf.function
    def _train_step(self, obs, action, next_obs, done, reward):
        '''Perform one gradient update

        Args:
            obs (tf.Tensor): batch observations, shape (batch, obs_space.shape),
                tf.uint8 for image observations, tf.float32 for non-image observations
            action (tf.Tensor): batch actions, shape (batch, ),
                tf.int32 or tf.int64 for discrete action space
            next_obs (tf.Tensor): batch next observations, shape (batch, obs_space.shape),
                tf.uint8 for image observations, tf.float32 for non-image observations
            done (tf.Tensor): batch done, shape (batch, ), tf.bool or tf.float32
            reward (tf.Tensor): batch reward, shape (batch, ), tf.float32

        Returns:
            tf.Tensor: loss, tf.float32
        '''

        variables = self.agent.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(variables)

            loss = self.value_loss(obs, action, next_obs, done, reward)

        # perform gradients
        grads = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))

        return loss

    def _run(self, steps, obs=None):
        '''Run environments, collect rollouts

        Args:
            steps (int): number of timesteps
            obs (np.ndarray, optional): the last observations. If `None`, 
                reset the environment.

        Returns:
            np.ndarray: the last observations.
        '''

        if obs is None:
            obs = self.env.reset()

        for _ in range(steps):

            if (len(self.buffer) < self.min_buffer or
                    np.random.rand() < self.explore_schedule()):
                
                # random action
                action = np.asarray([self.action_space.sample() 
                                        for n in range(self.n_envs)])

            else:
                # predict action
                action = self.predict(obs)

            # step environment
            new_obs, reward, done, infos = self.env.step(action)

            # add to buffer
            self.buffer.add(obs, new_obs, action, reward, done)
            obs = new_obs

            # update state
            self.num_timesteps += self.n_envs

        return new_obs

    def train(self, steps, batch_size, target_update):
        '''Train one epoch

        Args:
            steps (int): gradient steps
            batch_size (int): batch size
            target_update (int): target network update frequency (gradient steps)

        Returns:
            float: mean loss
        '''

        all_loss = []

        for _step in range(steps):
            (obs, action, next_obs, done, reward) = self.buffer(batch_size)

            loss = self._train_step(obs, action, next_obs, done, reward)

            all_loss.append(loss)

            self.num_gradsteps += 1

            # update target networks
            if self.num_gradsteps % target_update == 0:
                self.agent_target.update(self.agent, polyak=self.tau)

        m_loss = np.mean(np.hstack(np.asarray(all_loss)))

        return m_loss

    def eval(self, env, n_episodes=5, max_steps=-1):
        '''Evaluate model (use default evaluation method)

        Args:
            env (gym.Env): the environment for evaluation
            n_episodes (int, optional): number of episodes to evaluate. 
                Defaults to 5.
            max_steps (int, optional): maximum steps in one episode. 
                Defaults to 10000. Set to -1 to run episodes until done.

        Returns:
            list: total rewards for each episode
            list: episode length for each episode
        '''

        return super().eval(env, n_episodes=n_episodes,
                                 max_steps=max_steps)

    def learn(self, total_timesteps:  int,
                    log_interval:     int = 1000,
                    eval_env:     gym.Env = None,
                    eval_interval:    int = 10000,
                    eval_episodes:    int = 5,
                    eval_max_steps:   int = 3000,
                    save_interval:    int = 10000,
                    save_path:        str = None,
                    target_update:    int = 2500,
                    tb_logdir:        str = None,
                    reset_timesteps: bool = False):
        '''Train IQN

        Args:
            total_timesteps (int): Total timesteps to train agent.
            log_interval (int, optional): Print log every ``log_interval`` 
                epochs. Defaults to 1.
            eval_env (gym.Env, optional): Environment for evaluation. 
                Defaults to None.
            eval_interval (int, optional): Evaluate every ``eval_interval``
                epochs. Defaults to 1.
            eval_episodes (int, optional): Evaluate ``eval_episodes`` episodes. 
                for every evaluation. Defaults to 5.
            eval_max_steps (int, optional): maximum steps every evaluation. 
                Defaults to 10000.
            save_interval (int, optional): Save model every ``save_interval``
                epochs. Default to None.
            save_path (str, optional): Model saving path. Default to None.
            target_update (int, optional): Frequency of updating target network.
                update every ``target_update`` gradient steps. Defaults to 10000.
            tb_logdir (str, optional): tensorboard log directory. Defaults to None.
            reset_timesteps (bool, optional): reset timesteps. Defaults to False.

        Returns:
            IQN: self
        '''        

        assert self.env is not None, 'Env not set, call set_env() before training'

        # create tensorboard writer
        if tb_logdir is not None:
            self.tb_writer = tf.summary.create_file_writer(tb_logdir)

        # initialize state
        if reset_timesteps:
            self.num_timesteps = 0
            self.num_gradsteps = 0
            self.num_epochs    = 0
            self.progress      = 0
            # reset buffer
            self.buffer.reset()

        obs        = None
        time_start = time.time()
        time_spent = 0
        timesteps_per_epoch = self.n_steps * self.n_envs
        total_epochs = int(float(total_timesteps-self.num_timesteps) /
                                        float(timesteps_per_epoch) + 0.5)
        

        while self.num_timesteps < total_timesteps:

            # collect rollouts
            obs = self._run(steps=self.n_steps, obs=obs)

            # update state
            self.num_epochs += 1
            self.progress = float(self.num_timesteps) / float(total_timesteps)

            if len(self.buffer) > self.min_buffer:

                # training
                loss = self.train(self.n_gradsteps, 
                                  batch_size=self.batch_size, 
                                  target_update=target_update)

                # write tensorboard
                if self.tb_writer is not None:
                    with self.tb_writer.as_default():
                        tf.summary.scalar('loss', loss, step=self.num_timesteps)
                        tf.summary.scalar('explore_rate', self.explore_schedule(),
                                                        step=self.num_timesteps)

                    self.tb_writer.flush()

            # print training log
            if (log_interval is not None) and (self.num_epochs % log_interval == 0):
                # current time
                time_now       = time.time()
                # execution time (one epoch)
                execution_time = (time_now - time_start) - time_spent
                # total time spent
                time_spent     = (time_now - time_start)
                # remaining time
                remaining_time = (time_spent / self.progress)*(1.0-self.progress)
                # eta
                eta            = (datetime.datetime.now() + datetime.timedelta(seconds=remaining_time)).strftime('%Y-%m-%d %H:%M:%S')
                # average steps per second
                fps            = float(self.num_timesteps) / time_spent

                LOG.set_header('Epoch {}/{}'.format(self.num_epochs, total_epochs))
                LOG.add_line()
                LOG.add_row('Timesteps', self.num_timesteps, total_timesteps, fmt='{}: {}/{}')
                LOG.add_row('Steps/sec', fps,                                 fmt='{}: {:.2f}')
                LOG.add_row('Progress',  self.progress*100.0,                 fmt='{}: {:.2f}%')

                if self.verbose > 0:
                    LOG.add_row('Execution time', datetime.timedelta(seconds=execution_time))
                    LOG.add_row('Elapsed time',   datetime.timedelta(seconds=time_spent))
                    LOG.add_row('Remaining time', datetime.timedelta(seconds=remaining_time))
                    LOG.add_row('ETA',            eta)
                    LOG.add_line()

                    if len(self.buffer) > self.min_buffer:
                        LOG.add_row('Loss',         loss,                    fmt='{}: {:.6f}')
                        LOG.add_row('Explore rate', self.explore_schedule(), fmt='{}: {:.6f}')
                    else:
                        LOG.add_row('Collecting rollouts {}/{}'.format(len(self.buffer), self.min_buffer))

                LOG.add_line()
                LOG.flush('INFO')

            # evaluate model
            eval_metrics = None
            if (eval_env is not None) and (self.num_epochs % eval_interval == 0):

                eps_rews, eps_steps = self.eval(env=eval_env, 
                                                n_episodes=eval_episodes, 
                                                max_steps=eval_max_steps)
                
                max_idx    = np.argmax(eps_rews)
                max_rews   = eps_rews[max_idx]
                max_steps  = eps_steps[max_idx]
                mean_rews  = np.mean(eps_rews)
                std_rews   = np.std(eps_rews)
                mean_steps = np.mean(eps_steps)

                # eval metrics to select the best model
                eval_metrics = mean_rews

                if self.tb_writer is not None:
                    with self.tb_writer.as_default():
                        tf.summary.scalar('max_rewards',  max_rews,   step=self.num_timesteps)
                        tf.summary.scalar('mean_rewards', mean_rews,  step=self.num_timesteps)
                        tf.summary.scalar('std_rewards',  std_rews,   step=self.num_timesteps)
                        tf.summary.scalar('mean_length',  mean_steps, step=self.num_timesteps)

                    self.tb_writer.flush()

                if self.verbose > 1:

                    for ep in range(eval_episodes):
                        LOG.set_header('Eval episode {}/{}'.format(ep+1, eval_episodes))
                        LOG.add_line()
                        LOG.add_row('Rewards', eps_rews[ep])
                        LOG.add_row(' Length', eps_steps[ep])
                        LOG.add_line()
                        LOG.flush('INFO')

                LOG.set_header('Evaluate {}/{}'.format(self.num_epochs, total_epochs))
                LOG.add_line()
                LOG.add_row('Max rewards',  max_rews)
                LOG.add_row('     Length',  max_steps)
                LOG.add_line()
                LOG.add_row('Mean rewards', mean_rews)
                LOG.add_row(' Std rewards', std_rews, fmt='{}: {:.3f}')
                LOG.add_row(' Mean length', mean_steps)
                LOG.add_line()
                LOG.flush('INFO')

            # save model
            if ((save_path is not None) and (save_interval is not None)
                    and (self.num_epochs % save_interval) == 0):
                
                saved_path = self.save(save_path, checkpoint_number=self.num_epochs,
                                    checkpoint_metrics=eval_metrics)

                if self.verbose > 0:
                    LOG.info('Checkpoint saved to: {}'.format(saved_path))

                    # find the best model path
                    best_path = self._preload(save_path, best=True)
                    if best_path == os.path.abspath(saved_path):
                        LOG.debug(' (Current the best)')

        return self

    def get_config(self):
        
        init_config = { 'learning_rate':       self.learning_rate,
                        'buffer_size':         self.buffer_size,
                        'min_buffer':          self.min_buffer,
                        'n_taus':              self.n_taus,
                        'n_target_taus':       self.n_target_taus,
                        'n_ks':                self.n_ks,
                        'n_steps':             self.n_steps,
                        'n_gradsteps':         self.n_gradsteps,
                        'batch_size':          self.batch_size,
                        'embed_size':          self.embed_size,
                        'gamma':               self.gamma,
                        'tau':                 self.tau,
                        'kappa':               self.kappa,
                        'max_grad_norm':       self.max_grad_norm,
                        'force_mlp':           self.force_mlp,
                        'explore_schedule':    self.explore_schedule,
                        'verbose':             self.verbose}

        setup_config = {'observation_space': self.observation_space,
                        'action_space':      self.action_space}

        return {'init_config': init_config,
                'setup_config': setup_config}
    
    @classmethod
    def from_config(cls, config):

        assert 'init_config' in config, 'Failed to load {} config, init_config not found'.format(cls.__name__)
        assert 'setup_config' in config, 'Failed to load {} config, setup_config not found'.format(cls.__name__)

        init_config = config['init_config']
        setup_config = config['setup_config']

        # construct model
        self = cls(env=None, **init_config)
        self.setup_model(**setup_config)

        return self