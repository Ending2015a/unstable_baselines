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
import re
import abc
import cv2
import csv
import sys
import glob
import json
import time
import random
import inspect
import logging
import argparse
import datetime
import multiprocessing

# --- 3rd party ---
import gym
import cloudpickle

import numpy as np
import tensorflow as tf

# --- my module ---
from unstable_baselines import logger

from unstable_baselines.base_v2 import (SavableModel, 
                                        TrainableModel)
from unstable_baselines.bugs import ReLU
from unstable_baselines.utils_v2 import (normalize,
                                         denormalize,
                                         is_image_observation,
                                         preprocess_observation,
                                         get_input_tensor_from_space)

# create logger
LOG = logger.getLogger('TD3')

# === Buffer ===

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
            observations (np.ndarray): numpy array of type np.float32,
                shape (n_envs, obs_space.shape).
            next_observations (np.ndarray): numpy array of type np.float32,
                shape (n_envs, obs_space.shape).
            actions (np.ndarray): continuous actions, numpy array of type 
                np.float32, shape (n_envs, act_space.shape)
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
            np.ndarray: actions, shape (batch_size, act_space.shape)
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

# CNN feature extractor
class NatureCnn(tf.keras.Model):
    def __init__(self, **kwargs):
        '''Nature CNN originated from 
        "Playing Atari with Deep Reinforcement Learning"
        '''
        super().__init__(**kwargs)

        self._layers = [
            tf.keras.layers.Conv2D(32, 8, 4, name='conv1'),
            ReLU(name='relu1'),
            tf.keras.layers.Conv2D(64, 4, 2, name='conv2'),
            ReLU(name='relu2'),
            tf.keras.layers.Conv2D(64, 3, 1, name='conv3'),
            ReLU(name='relu3'),
            tf.keras.layers.Flatten(name='flatten'),
            tf.keras.layers.Dense(512, name='fc'),
            ReLU(name='relu4')
        ]

    @tf.function
    def call(self, inputs, training=False):
        '''Forward network

        Args:
            inputs (tf.Tensor): Expecting 4-D batch observations, shape
                (batch, height, width, channel)
            training (bool, optional): Training mode. Defaults to False.

        Returns:
            tf.Tensor: Latent vectors.
        '''
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x

class Actor(tf.keras.Model):
    '''
    Actor network (continuous state, action)
    from paper: https://arxiv.org/abs/1802.09477 (Appendix C)
    '''
    def __init__(self, action_space, **kwargs):
        super().__init__(**kwargs)

        assert isinstance(action_space, gym.spaces.Box)
        self.action_space = action_space

        self._layers = [
            tf.keras.layers.Dense(400, name='fc1'),
            ReLU(name='relu1'),
            tf.keras.layers.Dense(300, name='fc2'),
            ReLU(name='relu2'),
            tf.keras.layers.Dense(action_space.shape[0], name='fc3'),
            tf.keras.layers.Activation(activation='tanh', name='tanh')
        ]

    @tf.function
    def call(self, inputs, training=False):
        '''Forward network

        Args:
            inputs (tf.Tensor): Expecting batch observations, shape 
                (batch, obs_space.shape)
            training (bool, optional): Training mode. Defaults to False.

        Returns:
            tf.Tensor: Predicted actions, shape (batch, act_space.shape),
                tf.float32
        '''
        x = inputs
        for layer in self._layers:
            x = layer(x)

        return x


class Critic(tf.keras.Model):
    '''
    Critic network (continuous state, action)
    from paper: https://arxiv.org/abs/1802.09477 (Appendix C)
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._concat = tf.keras.layers.Concatenate()

        self._layers = [
            tf.keras.layers.Dense(400, name='fc1'),
            ReLU(name='relu1'),
            tf.keras.layers.Dense(300, name='fc2'),
            ReLU(name='relu2'),
            tf.keras.layers.Dense(1, name='fc3')
        ]

    @tf.function
    def call(self, inputs, training=False):
        '''Forward network

        Args:
            inputs (tuple): Observation, action tuple
                [0] (tf.Tensor): Batch observations, shape 
                    (batch, obs_space.shape)
                [1] (tf.Tensor): Batch actions, shape
                    (batcj, act_space.shape)
            training (bool, optional): Training mode. Defaults to False.

        Returns:
            tf.Tensor: Predicted state-action values, shape (batch, 1), 
                tf.float32
        '''

        x = self._concat(inputs)
        for layer in self._layers:
            x = layer(x)

        return x


# === Agent, Model ===

class Agent(SavableModel):
    def __init__(self, observation_space, action_space, force_mlp=False, **kwargs):
        '''TD3 Agent

        Args:
            observation_space (gym.Spaces): The observation space of the environment.
                None for delayed setup.
            action_space (gym.Spaces): The action space of the environment.
                None for delayed setup.
            force_mlp (bool, optional): Force to use MLP feature extractor. 
                Defaults to False.
        '''
        super().__init__(**kwargs)

        self.force_mlp = force_mlp

        # --- Initialize ---
        self.observation_space = None
        self.action_space = None
        self.actor = None
        self.critic_1 = None
        self.critic_2 = None

        if observation_space is not None and action_space is not None:
            self.setup_model(observation_space, action_space)

    def setup_model(self, observation_space, action_space):
        '''Setup model and networks

        Args:
            observation_space (gym.Spaces): The observation space of the
                environment.
            action_space (gym.Spaces): The action space of the environment.
        '''

        # check observation/action space
        if observation_space is None:
            raise ValueError("Observation space not provided: "
                            "{}".format(observation_space))
        if action_space is None:
            raise ValueError("Action space not provided: {}".format(
                            action_space))
        if not isinstance(action_space, gym.spaces.Box):
            raise ValueError("Action space only support Box type, "
                        "got {}.".format(type(action_space).__name__))


        self.observation_space = observation_space
        self.action_space      = action_space

        # --- setup model ---
        if (is_image_observation(observation_space) and
                (not self.force_mlp)):
            self.net = NatureCnn()
        else:
            # NOOP
            self.net = tf.keras.Sequential(
                layers=[tf.keras.layers.Flatten()])
        
        self.actor = Actor(self.action_space)
        self.critic_1 = Critic()
        self.critic_2 = Critic()

        # construct networks
        obs_inputs = get_input_tensor_from_space(observation_space)
        act_inputs = get_input_tensor_from_space(action_space)
        fea_inputs = self.net(obs_inputs)
        self.actor(features)
        self.critic_1([fea_inputs, act_inputs])
        self.critic_2([fea_inputs, act_inputs])

    @tf.function
    def _forward(self, inputs, training=True):
        '''Forward actor

        Args:
            inputs (tf.Tensor): batch observations in shape 
                (batch, obs_space.shape), tf.float32.
            training (bool, optional): Training mode. Default
                to True.

        Return:
            tf.Tensor: predicted raw actions (batch, act_space.shape), tf.float32
            tf.Tensor: Extracted input features (batch, latent), tf.float32
        '''

        # cast and normalize non-float32 inputs (e.g. image in uint8)
        # NOTICE: image in float32 is considered as having been normalized
        inputs   = preprocess_observation(inputs, self.observation_space)
        features = self.net(inputs, training=training)

        return self.actor(features, training=training), features

    @tf.function
    def call(self, inputs, normalized_act=True, training=True):
        '''Batch predict actions

        Args:
            inputs (tf.Tensor): batch observations in shape 
                (batch, obs_space.shape), tf.float32.
            normalized_act (bool, optional): Return normalized actions 
                or not. Default to True.
            training (bool, optional): Training mode. Default to True.

        Returns:
            tf.Tensor: predicted actions in shape (batch, act_space.shape), 
                tf.float32
            tf.Tensor: Extracted input features (batch, latent), tf.float32
        '''

        action, features = self._forward(inputs, training=training)

        # denormalize action
        if not normalized_act:
            low = tf.cast(self.action_space.low, dtype=tf.float32)
            high = tf.cast(self.action_space.high, dtype=tf.float32)
            action = denormalize(action, low=low, high=high, nlow=-1., nhigh=1.)
        
        return action, features

    def predict(self, inputs, normalized_act=False):
        '''Predict actions

        Args:
            inputs (np.ndarray): batch observations in shape (batch, obs_space.shape)
                or a single observation in shape (obs_space.shape), np.float32.
            normalized_act (bool, optional): Return normalized actions or not. Default
                to True.

        Returns:
            np.ndarray: predicted actions in shape (batch, act_space.shape) or 
                (act_space.shape) for a single sample, np.float32
        '''
        one_sample = (len(inputs.shape) == len(self.observation_space.shape))

        if one_sample:
            inputs = np.expand_dims(inputs, axis=0)

        # predict
        outputs, *_ = self(inputs, normalized_act=normalized_act, training=False)
        outputs = np.asarray(outputs)

        if one_sample:
            outputs = np.squeeze(outputs, axis=0)

        return outputs

    def get_config(self):
        
        config = {'observation_space': self.observation_space, 
                  'action_space': self.action_space,
                  'force_mlp': self.force_mlp}

        return config


class TD3(TrainableModel):
    def __init__(self, env, learning_rate:       float = 1e-3,
                            buffer_size:           int = int(1e6),
                            min_buffer:            int = 10000,
                            n_steps:               int = 100,
                            n_gradsteps:           int = 100,
                            batch_size:            int = 100,
                            policy_update:         int = 2,
                            gamma:               float = 0.99,
                            tau:                 float = 0.005, # polyak
                            max_grad_norm:       float = 0.5,
                            action_noise:        float = 0.2,
                            action_noise_clip:   float = 0.5,
                            explore_noise              = None,
                            force_mlp:            bool = False,
                            verbose:               int = 0,
                            **kwargs):
        '''TD3

        The implementation mainly follows its originated paper
        "Addressing Function Approximation Error in Actor-Critic Methods" 
        by Fujimoto et al.

        The first argument `env` can be `None` for delayed model setup. You 
        should call `set_env()` then call `setup_model()` to manually setup 
        the model.

        Args:
            env (gym.Env): Training environment. Can be `None`.
            learning_rate (float, optional): Learning rate. Defaults to 1e-3.
            buffer_size (int, optional): Maximum size of the replay buffer. Defaults to 1000000.
            min_buffer (int, optional): Minimum size of the replay buffer before training. 
                Defaults to 10000.
            n_steps (int, optional): number of steps of rollouts to collect for every epoch. 
                Defaults to 100.
            n_gradsteps (int, optional): number of gradient steps in one epoch. 
                Defaults to 100.
            batch_size (int, optional): Training batch size. Defaults to 100.
            policy_update (int, optional): Policy update frequency (gradsteps). Defaults to 2.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            tau (float, optional): Polyak update parameter. Defaults to 0.005.
            max_grad_norm (float, optional): Adam gradients clip range. Defaults to 0.5.
            action_noise (float, optional): Noise scale added to target policy. Defaults to 0.2.
            action_noise_clip (float, optional): Noise range added to target policy. Defaults to 0.5.
            explore_noise (ActionNoise, optional): Gaussian noise for exploreation. Defaults to None.
            force_mlp (bool, optional): Force to use MLP net as feature extractor. Defaults to False.
            verbose (int, optional): More training log. Defaults to 0.
        '''
        super().__init__(**kwargs)

        self.env = env

        self.learning_rate       = learning_rate
        self.buffer_size         = buffer_size
        self.min_buffer          = min_buffer
        self.n_steps             = n_steps
        self.n_gradsteps         = n_gradsteps
        self.batch_size          = batch_size
        self.policy_update       = policy_update
        self.gamma               = gamma
        self.tau                 = tau
        self.max_grad_norm       = max_grad_norm
        self.action_noise        = action_noise
        self.action_noise_clip   = action_noise_clip
        self.explore_noise       = explore_noise
        self.force_mlp           = force_mlp
        self.verbose             = verbose

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
            if env.observation_space != self.observation_space:
                raise RuntimeError("Observation space does not match, "
                                "expected {}, got {}".format(
                                    self.observation_space,
                                    env.observation_space))
        if self.action_space is not None:
            if env.action_space != self.action_space:
                raise RuntimeError("Action space does not match, "
                                "expected {}, got {}".format(
                                    self.action_space,
                                    env.action_space))
        self.env = env
        self.n_envs = env.n_envs

    def setup_model(self, observation_space, action_space):
        '''Setup model, optimizer and scheduler for training

        Args:
            observation_space (gym.Spaces): The observation space of the
                environment.
            action_space (gym.Spaces): The action space of the environment.
        '''

        if observation_space is None:
            raise ValueError("Observation space not provided: "
                            "{}".format(observation_space))
        if action_space is None:
            raise ValueError("Action space not provided: {}".format(
                            action_space))
        
        self.observation_space = observation_space
        self.action_space      = action_space

        # --- setup model ---
        self.buffer       = ReplayBuffer(buffer_size=self.buffer_size)
        self.agent        = Agent(observation_space, action_space, force_mlp=self.force_mlp)
        self.agent_target = Agent(observation_space, action_space, force_mlp=self.force_mlp)

        self.actor_optimizer  = tf.keras.optimizers.Adam(learning_rate=self.learning_rate,
                                                         clipnorm=max_grad_norm)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, 
                                                         clipnorm=max_grad_norm)

        # initialize target
        self.agent_target.update(self.agent)

    

    @tf.function
    def _forward(self, inputs, training=True):
        '''Forward actor

        Args:
            inputs (tf.Tensor): batch observations in shape (batch, obs_space.shape).
                tf.float32.
            training (bool, optional): Training mode. Default to True.

        Return:
            tf.Tensor: predicted raw actions (batch, act_space.shape), tf.float32
            tf.Tensor: Extracted input features (batch, latent), tf.float32
        '''
        return self.agent._forward(inputs, training=training)

    @tf.function
    def call(self, inputs, normalized_act=True, training=True):
        '''Batch predict actions

        Args:
            inputs (tf.Tensor): batch observations in shape (batch, obs_space.shape).
                tf.float32.
            normalized_act (bool, optional): Return normalized actions or not. Default
                to True.
            training (bool, optional): Training mode. Default to True.

        Returns:
            tf.Tensor: predicted actions in shape (batch, act_space.shape), tf.float32
            tf.Tensor: Extracted input features (batch, latent), tf.float32
        '''
        return self.agent(inputs, normalized_act=normalized_act, training=training)

    def predict(self, inputs, normalized_act=False):
        '''Predict actions

        Args:
            inputs (np.ndarray): batch observations in shape (batch, obs_space.shape)
                or a single observation in shape (obs_space.shape). np.float32.
            normalized_act (bool, optional): Return normalized actions or not. Default
                to True

        Returns:
            np.ndarray: predicted actions in shape (batch, act_space.shape) or 
                (act_space.shape), np.float32
        '''
        return self.agent.predict(inputs, normalized_act=normalized_act)

    @tf.function
    def actor_loss(self, obs):
        '''Compute actor loss

        Actor loss = -Q(s, mu(s))

        Args:
            obs (tf.Tensor): batch observations, shape (batch, obs_space.shape),
                tf.float32.

        Returns:
            tf.Tensor: Actor loss , tf.float32
        '''
        act, fea = self.agent._forward(obs)
        return -tf.reduce_mean(self.agent.critic_1([fea, act]))

    @tf.function
    def critic_loss(self, obs, action, next_obs, done, reward):
        '''Compute critic loss

        Critic loss:
        y = r + gamma * min( Q1*(s', mu*(s') + noise), 
                             Q2*(s', mu*(s') + noise) )
        
        L = MSE(y, Q1(s, mu(s))) + MSE(y, Q2(s, mu(s)))

        Args:
            obs (tf.Tensor): batch observations, shape (batch, obs_space.shape),
                tf.float32.
            action (tf.Tensor): batch actions, shape (batch, act_space.shape), tf.float32
            next_obs (tf.Tensor): batch next observations, shape (batch, obs_space.shape),
                tf.float32.
            done (tf.Tensor): batch done, shape (batch, ), tf.bool or tf.float32
            reward (tf.Tensor): batch reward, shape (batch, ), tf.float32

        Returns:
            tf.Tensor: Critic loss, tf.float32
        '''

        reward = tf.cast(reward, dtype=tf.float32)
        done   = tf.cast(done,   dtype=tf.float32)

        noise    = tf.random.normal(shape=action.shape) * self.action_noise
        noise    = tf.clip_by_value(noise, -self.action_noise_clip, self.action_noise_clip)
        
        next_act, next_fea = self.agent_target._forward(next_obs)
        next_act           = tf.clip_by_value(next_act + noise, -1., 1.)

        # compute the target Q value
        q1       = tf.squeeze(self.agent_target.critic_1([next_fea, next_act]), axis=-1)
        q2       = tf.squeeze(self.agent_target.critic_2([next_fea, next_act]), axis=-1)
        target_q = tf.minimum(q1, q2) # (batch, )

        y = reward + tf.stop_gradient( (1.-done) * self.gamma * target_q)

        # compute Q estimate
        _, fea = self.agent._forward(obs)
        q1 = tf.squeeze(self.agent.critic_1([fea, action]), axis=-1) # (batch, )
        q2 = tf.squeeze(self.agent.critic_2([fea, action]), axis=-1) # (batch, )
    
        # compute critic loss
        return tf.keras.losses.MSE(q1, y) + tf.keras.losses.MSE(q2, y)

    @tf.function
    def _train_actor(self, obs):
        '''Perform one gradient update to the actor network

        Args:
            obs (tf.Tensor): batch observations, shape (batch, obs_space.shape),
                tf.float32.

        Returns:
            tf.Tensor: Actor loss, tf.float32
        '''
        variables = self.agent.actor.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(variables)

            # compute actor loss
            loss = self.actor_loss(obs)

        # perform gradient
        grads = tape.gradient(loss, variables)
        self.actor_optimizer.apply_gradients(zip(grads, variables))

        return loss

    @tf.function
    def _train_critic(self, obs, action, next_obs, done, reward):
        '''Perform one gradient update to critic networks

        Args:
            obs (tf.Tensor): batch observations, shape (batch, obs_space.shape),
                tf.float32.
            action (tf.Tensor): batch actions, shape (batch, act_space.shape), tf.float32
            next_obs (tf.Tensor): batch next observations, shape (batch, obs_space.shape),
                tf.float32.
            done (tf.Tensor): batch done, shape (batch, ), tf.bool or tf.float32
            reward (tf.Tensor): batch reward, shape (batch, ), tf.float32

        Returns:
            tf.Tensor: Critic loss, tf.float32
        '''
        # combine two variable list
        variables = (self.agent.critic_1.trainable_variables + 
                        self.agent.critic_2.trainable_variables +
                        self.agent.net.trainable_variables)
        with tf.GradientTape() as tape:
            tape.watch(variables)

            loss = self.critic_loss(obs, action, next_obs, done, reward)

        # perform gradients
        grads = tape.gradient(loss, variables)
        self.critic_optimizer.apply_gradients(zip(grads, variables))

        return loss

    def _run(self, steps, obs=None):
        '''Run environments, coolect rollouts

        Args:
            steps (int): number of timesteps
            obs (np.ndarray, optional): the last observations. If `None`, 
                reset the environment.

        Returns:
            np.ndarray: the last observations.
        '''

        if obs is None:
            obs = self.env.reset()

        if self.explore_noise is not None:
            self.explore_noise.reset()

        for _ in range(steps):

            if len(self.buffer) < self.min_buffer:
                # random sample (collecting rollouts)
                action = np.array([self.action_space.sample() for n in range(self.n_envs)])
                action = normalize(action, high=self.action_space.high, low=self.action_space.low, nlow=-1., nhigh=1.)
            else:
                # sample from policy (normalized)
                action = self(obs, normalized_act=True)
            
            # add action noise
            if self.explore_noise is not None:
                action = np.clip(action + self.explore_noise(shape=action.shape), -1, 1)

            # step environment
            raw_action = denormalize(action, high=self.action_space.high, low=self.action_space.low, nlow=-1., nhigh=1.)
            new_obs, reward, done, infos = self.env.step(raw_action)
            
            # add to buffer
            self.buffer.add(obs, new_obs, action, reward, done)
            obs = new_obs

            if self.n_envs == 1 and done[0]:
                if self.explore_noise is not None:
                    self.explore_noise.reset()
            
            # update state
            self.num_timesteps += self.n_envs

        return new_obs

    def train(self, steps, batch_size, policy_update, target_update):
        '''Train one epoch

        Args:
            steps (int): gradient steps
            batch_size (int): batch size
            policy_update (int): policy delayed update frequency (gradsteps)
            target_update (int): target network update frequency (gradsteps)

        Returns:
            float: mean actor loss, None if actor not updated, np.float32
            float: mean critic loss, np.float32
        '''

        m_actor_loss = None
        m_critic_loss = None

        all_actor_loss = []
        all_critic_loss = []

        for gradient_steps in range(steps):
            (obs, action, next_obs, done, reward) = self.buffer(batch_size)

            critic_loss = self._train_critic(obs, action, next_obs, done, reward)

            self.num_gradsteps += 1

            # update policy
            if self.num_gradsteps % policy_update == 0:

                # update critic
                actor_loss = self._train_actor(obs)
                all_actor_loss.append(actor_loss)
            
            # update target agent
            if self.num_gradsteps % target_update == 0:
                self.agent_target.update(self.agent)
            
            all_critic_loss.append(critic_loss)

        if len(all_actor_loss) > 0:
            m_actor_loss = np.mean(np.hstack(np.asarray(all_actor_loss)))
        m_critic_loss = np.mean(np.hstack(np.asarray(all_critic_loss)))
        
        return m_actor_loss, m_critic_loss

    def eval(self, env, n_episodes=5, max_steps=-1):
        '''Evaluate model (use default evaluation method)

        Args:
            env (gym.Env): the environment for evaluation
            n_episodes (int, optional): number of episodes to evaluate. 
                Defaults to 5.
            max_steps (int, optional): maximum steps in one episode. 
                Set to -1 to run episodes until done. Defaults to -1.

        Returns:
            list: total rewards for each episode
            list: episode length for each episode
        '''

        return super().eval(env, n_episodes=n_episodes,
                                 max_steps=max_steps)

    def learn(self, total_timesteps:  int, 
                    log_interval:     int = 1,
                    eval_env:     gym.Env = None, 
                    eval_interval:    int = 1, 
                    eval_episodes:    int = 5, 
                    eval_max_steps:   int = 1000,
                    save_interval:    int = 1,
                    save_path:        str = None,
                    target_update:    int = 2,
                    tb_logdir:        str = None, 
                    reset_timesteps: bool = False):
        '''Train TD3

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
                Defaults to 1000.
            save_interval (int, optional): Save model every ``save_interval``
                epochs. Default to 1.
            save_path (str, optional): Model saving path. Default to None.
            target_update (int, optional): Frequency of updating target network.
                update every ``target_update`` gradient steps. Defaults to 2.
            tb_logdir (str, optional): tensorboard log directory. Defaults to None.
            reset_timesteps (bool, optional): reset timesteps. Defaults to False.

        Returns:
            TD3: self
        '''        

        assert self.env is not None, 'Please set env before training'

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

        # initialize
        obs        = None
        time_start = time.time()
        time_spent = 0
        timesteps_per_epoch = self.n_steps * self.n_envs
        total_epoch = int(float(total_timesteps-self.num_timesteps) / 
                                        float(timesteps_per_epoch) + 0.5)


        while self.num_timesteps < total_timesteps:

            # collect rollouts
            obs = self._run(steps=self.n_steps, obs=obs)

            # update state
            self.num_epochs += 1
            self.progress = float(self.num_timesteps) / float(total_timesteps)

            if len(self.buffer) > self.min_buffer:

                # training
                actor_loss, critic_loss = self.train(self.n_gradsteps, 
                                                     batch_size=self.batch_size, 
                                                     policy_update=self.policy_update,
                                                     target_update=target_update)

                # write tensorboard
                if self.tb_writer is not None:
                    with self.tb_writer.as_default():
                        if actor_loss is not None:
                            tf.summary.scalar('actor_loss', actor_loss,  step=self.num_timesteps)
                        tf.summary.scalar('critic_loss',    critic_loss, step=self.num_timesteps)

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
                LOG.add_row('Progress',  self.progress*100.0,                      fmt='{}: {:.2f}%')

                if self.verbose > 0:
                    LOG.add_row('Execution time', datetime.timedelta(seconds=execution_time))
                    LOG.add_row('Elapsed time',   datetime.timedelta(seconds=time_spent))
                    LOG.add_row('Remaining time', datetime.timedelta(seconds=remaining_time))
                    LOG.add_row('ETA',            eta)
                    LOG.add_line()

                    if len(self.buffer) > self.min_buffer:
                        if actor_loss is not None:
                            LOG.add_row('Actor loss', actor_loss,  fmt='{}: {:.6f}')
                        LOG.add_row('Critic loss',    critic_loss, fmt='{}: {:.6f}')
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
                LOG.add_row('Mean rewards', mean_rews.round(3))
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
                        'n_steps':             self.n_steps,
                        'n_gradsteps':         self.n_gradsteps,
                        'batch_size':          self.batch_size,
                        'policy_update':       self.policy_update,
                        'gamma':               self.gamma,
                        'tau':                 self.tau,
                        'max_grad_norm':       self.max_grad_norm,
                        'action_noise':        self.action_noise,
                        'action_noise_clip':   self.action_noise_clip,
                        'explore_noise':       self.explore_noise,
                        'force_mlp':           self.force_mlp,
                        'verbose':             self.verbose}

        setup_config = {'observation_space': self.observation_space,
                        'action_space': self.action_space}

        return {'init_config': init_config, 
                'setup_config': setup_config}
    
    @classmethod
    def from_config(cls, config):

        assert 'init_config' in config, 'Failed to load {} config, init_config not found'.format(cls.__name__)
        assert 'setup_config' in config, 'Failed to load {} config, setup_config not found'.format(cls.__name__)

        init_config  = config['init_config']
        setup_config = config['setup_config']

        # construct model
        self = cls(env=None, **init_config)
        self.setup_model(**setup_config)

        return self
