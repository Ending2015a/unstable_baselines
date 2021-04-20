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
from unstable_baselines.prob import Categorical
from unstable_baselines.utils import (normalize,
                                      to_json_serializable,
                                      from_json_serializable,
                                      tf_soft_update_params,
                                      StateObject)



# create logger
LOG = logger.getLogger('C51')


def calc_bins(v_min, v_max, n_atoms):
    '''Calculate canonical returns of each category

    Args:
        v_min (float): Minimum of value.
        v_max (float): Maximum of value.
        n_atoms (int): Number of categories.

    Returns:
        list: a list of canonical returns for each category
        float: bin width
    '''
    delta_z = (v_max-v_min) / (n_atoms-1)
    bins = [v_min + i * delta_z for i in range(n_atoms)]
    return bins, delta_z


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

class NatureCnn(tf.keras.Model):

    def __init__(self, **kwargs):
        '''
        Nature CNN originated from 
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

# Mlp feature extractor
class MlpNet(tf.keras.Model):
    '''MLP feature extractor'''
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)

        self._layers = [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, name='fc1'),
            ReLU(name='relu1'),
            tf.keras.layers.Dense(64, name='fc2'),
            ReLU(name='relu2'),
        ]
    
    @tf.function
    def call(self, inputs, training=False):
        '''Forward network

        Args:
            inputs (tf.Tensor): Expecting batch observations, shape 
                (batch, obs_space.shape)
            training (bool, optional): Training mode. Defaults to False.

        Returns:
            tf.Tensor: Latent vectors.
        '''
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x


# Categorical Q-value network
class CategoricalQNet(tf.keras.Model):
    def __init__(self, action_space, n_atoms, **kwargs):
        super().__init__(**kwargs)

        self._layers = [
            tf.keras.layers.Dense(action_space.n * n_atoms)
        ]

        # output shape
        self._o_shape = (-1, action_space.n, n_atoms)

    @tf.function
    def call(self, inputs, training=False):
        '''Forward network

        Args:
            inputs (tf.Tensor): Expecting a latent vector in shape 
                (batch, latent_size), tf.float32.
            training (bool, optional): Training mode. Defaults to False.

        Returns:
            tf.Tensor: Predicted Categorical Q values in shape 
                (batch, act_space.n, n_atoms)
        '''
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return tf.reshape(x, self._o_shape)

# === Agent, Model ===

class Agent(SavableModel):
    def __init__(self, observation_space, action_space, v_min=0.0, v_max=3.0, 
                       n_atoms=51, force_mlp=False, **kwargs):
        '''C51 Agent

        Args:
            observation_space (gym.Spaces): The observation space of the environment.
                Can be None for delayed setup.
            action_space (gym.Spaces): The action space of the environment.
                Can be None for delayed setup.
            v_min (float, optional): Minimum of value. Defaults to 0.0.
            v_max (float, optional): Maximum of value. Defaults to 3.0.
            n_atoms (int, optional): Number of categories of Q net. Defaults to 51.
            force_mlp (bool, optional): Force to use MLP feature extractor. 
                Defaults to False.
        '''
        super().__init__(**kwargs)

        self.v_min     = v_min
        self.v_max     = v_max
        self.n_atoms   = n_atoms
        self.force_mlp = force_mlp

        # --- Initialize ---
        self.observation_space = None
        self.action_space      = None
        self.net               = None
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
        if (len(self.observation_space.shape) == 3) and (not self.force_mlp):
            self.net = NatureCnn()
        else:
            self.net = MlpNet()

        self.q_net = CategoricalQNet(action_space, n_atoms=self.n_atoms)

        # construct networks
        inputs = tf.keras.Input(shape=self.observation_space.shape, dtype=tf.float32)

        outputs = self.net(inputs)
        self.q_net(outputs)

    @tf.function
    def _forward(self, inputs, training=True):
        '''Forward actor

        Args:
            inputs (tf.Tensor): batch observations in shape (batch, obs_space.shape).
                tf.uint8 for image observations and tf.float32 for non-image 
                observations.
            training (bool, optional): Determine whether in training mode. Default
                to True.

        Return:
            tf.Tensor: predicted logits in shape (batch, act_space.n, n_atoms),
                tf.float32
        '''

        # cast and normalize non-float32 inputs (e.g. image with uint8)
        if tf.as_dtype(inputs.dtype) != tf.float32:
            # cast observations to float32
            inputs = tf.cast(inputs, dtype=tf.float32)
            low    = tf.cast(self.observation_space.low, dtype=tf.float32)
            high   = tf.cast(self.observation_space.high, dtype=tf.float32)
            # normalize observations [0, 1]
            inputs = normalize(inputs, low=low, high=high, nlow=0., nhigh=1.)

        # forward network
        latent = self.net(inputs, training=training)
        # forward policy net
        logits = self.q_net(latent, training=training)

        return logits

    @tf.function
    def call(self, inputs, training=True):
        '''Batch predict actions

        Args:
            inputs (tf.Tensor): batch observations in shape (batch, obs_space.shape).
                tf.uint8 for image observations and tf.float32 for non-image 
                observations.
            training (bool, optional): Determine whether in training mode. Default
                to True.

        Returns:
            tf.Tensor: Predicted actions in shape (batch, ), tf.int64
            tf.Tensor: Predicted state-action values in shape (batch, act_space.n),
                tf.float32
            tf.Tensor: Predicted Q value probability distributions
                shape (batch, act_space.n, n_atoms)
        '''

        bins, _ = calc_bins(self.v_min, self.v_max, self.n_atoms)
        zi = tf.constant(bins, dtype=tf.float32)

        # forward
        logits  = self._forward(inputs, training=training) # (batch, act_space.n, n_atoms)
        probs   = tf.nn.softmax(logits, axis=-1)           # (batch, act_space.n, n_atoms)
        values  = tf.math.reduce_sum(zi * probs, axis=-1)  # (batch, act_space.n)
        actions = tf.math.argmax(values, axis=-1)          # (batch,)
        
        return actions, values, probs
    
    def predict(self, inputs):
        '''Predict actions

        Args:
            inputs (np.ndarray): batch observations in shape (batch, obs_space.shape)
                or a single observation in shape (obs_space.shape). np.uint8 for image
                observations and np.float32 for non-image observations.

        Returns:
            np.ndarray: predicted actions in shape (batch, ) or (), np.int64
        '''

        one_sample = (len(inputs.shape) == len(self.observation_space.shape))

        if one_sample:
            inputs = np.expand_dims(inputs, axis=0)

        # predict
        outputs, *_ = self(inputs, training=False)
        outputs     = outputs.numpy()

        if one_sample:
            outputs = np.squeeze(outputs, axis=0)

        # predict
        return outputs

    def get_config(self):
        
        config = {'observation_space': self.observation_space,
                  'action_space':      self.action_space,
                  'v_min':             self.v_min,
                  'v_max':             self.v_max,
                  'n_atoms':           self.n_atoms,
                  'force_mlp':         self.force_mlp}

        return to_json_serializable(config)


class C51(TrainableModel):
    def __init__(self, env, learning_rate:        float = 3e-4,
                            buffer_size:            int = int(1e6),
                            min_buffer:             int = 50000,
                            n_atoms:                int = 51,
                            v_min:                float = 0.0,
                            v_max:                float = 3.0,
                            n_steps:                int = 4,
                            n_gradsteps:            int = 1,
                            batch_size:             int = 64,
                            gamma:                float = 0.99,
                            tau:                  float = 1.0,
                            max_grad_norm:        float = 0.5,
                            force_mlp:             bool = False,
                            explore_schedule: Scheduler = 0.3,
                            verbose:                int = 0,
                            **kwargs):
        '''Categorical DQN (C51)

        The implementation mainly follows its originated paper
        `A Distributional Perspective on Reinforcement Learning` by Bellemare et al.

        The first argument `env` can be `None` for delayed model setup. You 
        should call `set_env()` then call `setup_model()` to manually setup 
        the model.

        Args:
            env (gym.Env): Training environment. Can be `None`.
            learning_rate (float, optional): Learning rate. Defaults to 3e-4.
            buffer_size (int, optional): Maximum size of the replay buffer. Defaults to 1000000.
            min_buffer (int, optional): Minimum size of the replay buffer before training. 
                Defaults to 50000.
            n_atoms (int, optional): Number of categories of Q network.
            v_min (float, optional): Minimum of Q value.
            v_max (float, optional): Maximum of Q value.
            n_steps (int, optional): number of steps of rollouts to collect for every epoch. 
                Defaults to 100.
            n_gradsteps (int, optional): number of gradient steps in one epoch. 
                Defaults to 200.
            batch_size (int, optional): Training batch size. Defaults to 128.
            gamma (float, optional): Decay rate. Defaults to 0.99.
            tau (float, optional): Polyak update parameter. Defaults to 1.0.
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
        self.n_atoms          = n_atoms
        self.v_min            = v_min
        self.v_max            = v_max
        self.n_steps          = n_steps
        self.n_gradsteps      = n_gradsteps
        self.batch_size       = batch_size
        self.gamma            = gamma
        self.tau              = tau
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
        self.agent        = Agent(self.observation_space, self.action_space, v_min=self.v_min, 
                                  v_max=self.v_max, n_atoms=self.n_atoms, force_mlp=self.force_mlp)
        self.agent_target = Agent(self.observation_space, self.action_space, v_min=self.v_min, 
                                  v_max=self.v_max, n_atoms=self.n_atoms, force_mlp=self.force_mlp)

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
            inputs (tf.Tensor): batch observations in shape (batch, obs_space.shape).
                tf.uint8 for image observations and tf.float32 for non-image 
                observations.
            training (bool, optional): Determine whether in training mode. Default
                to True.

        Return:
            tf.Tensor: predicted logits in shape (batch, act_space.n, n_atoms),
                tf.float32
        '''
        return self.agent._forward(inputs, training=training)

    @tf.function
    def call(self, inputs, training=True):
        '''Batch predict actions

        Args:
            inputs (tf.Tensor): batch observations in shape (batch, obs_space.shape).
                tf.uint8 for image observations and tf.float32 for non-image 
                observations.
            training (bool, optional): Determine whether in training mode. Default
                to True.

        Returns:
            tf.Tensor: Predicted actions in shape (batch, ), tf.int64
            tf.Tensor: Predicted state-action values in shape (batch, act_space.n),
                tf.float32
            tf.Tensor: Predicted Q value probability distributions
                shape (batch, act_space.n, n_atoms)
        '''
        return self.agent(inputs, training=training)

    def predict(self, inputs):
        '''Predict actions

        Args:
            inputs (np.ndarray): batch observations in shape (batch, obs_space.shape)
                or a single observation in shape (obs_space.shape). np.uint8 for image
                observations and np.float32 for non-image observations.

        Returns:
            np.ndarray: predicted actions in shape (batch, ) or (), np.int64
        '''
        return self.agent.predict(inputs)

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

        action  = tf.cast(action, dtype=tf.int64)
        reward  = tf.cast(reward, dtype=tf.float32)
        done    = tf.cast(done, dtype=tf.float32)

        reward  = tf.expand_dims(reward, axis=-1) # (batch, 1)
        done    = tf.expand_dims(done, axis=-1)   # (batch, 1)

        # compute bins and delta_z
        bins, delta_z = calc_bins(self.v_min, self.v_max, self.n_atoms)
        bins    = tf.constant(bins, dtype=tf.float32)     # (n_atoms)
        delta_z = tf.constant(delta_z, dtype=tf.float32)  # ()

        # compute TZ
        zi      = tf.repeat(tf.expand_dims(bins, axis=0), tf.shape(done)[0], axis=0) # (batch, n_atoms)
        Tzi     = reward + (1.-done) * self.gamma * zi                               # (batch, n_atoms)
        Tzi     = tf.clip_by_value(Tzi, self.v_min, self.v_max)                      # (batch, n_atoms)

        mat_zi  = tf.repeat(tf.expand_dims(zi,  axis=-1), self.n_atoms, axis=-1) # (batch, n_atoms, n_atoms)
        mat_Tzi = tf.repeat(tf.expand_dims(Tzi, axis=-1), self.n_atoms, axis=-1) # (batch, n_atoms, n_atoms)
        mat_Tzj = tf.transpose(mat_Tzi, (0, 2, 1))

        mat_TZ  = tf.clip_by_value(1. - tf.math.abs(mat_Tzj - mat_zi)/delta_z, 0., 1.)

        # compute pj(x', a')
        next_act, _, next_ps = self.agent_target(next_obs) 
        next_pj = tf.gather(next_ps, indices=next_act, batch_dims=1) # (batch, n_atoms)

        # compute PhiTZ(x', a')
        #PhiTZ  = tf.reduce_sum(mat_TZ * tf.expand_dims(next_pj, axis=1), axis=-1)
        PhiTZ   = tf.einsum('bij,bj->bi', mat_TZ, next_pj)  # (batch, n_atoms)
        PhiTZ   = tf.stop_gradient(PhiTZ)

        # compute ceoss entropy loss
        its     = self.agent._forward(obs)                     # (batch, act_space.n, n_atoms)
        it      = tf.gather(its, indices=action, batch_dims=1) # (batch, n_atoms)
        loss    = tf.nn.softmax_cross_entropy_with_logits(labels=PhiTZ, logits=it)
        loss    = tf.math.reduce_mean(loss)

        # # compute logp(x, a)
        # it      = self.agent._forward(obs)                       # (batch, act_space.n, n_atoms)
        # x       = it - tf.math.reduce_max(it, axis=-1, keepdims=True)
        # e       = tf.math.exp(x)
        # z       = tf.math.reduce_sum(e, axis=-1, keepdims=True)
        # logps   = x - tf.math.log(z)                             # (batch, act_space.n, n_atoms)
        # logp    = tf.gather(logps, indices=action, batch_dims=1) # (batch, n_atoms)

        # # compute cross entropy loss
        # ce      = -tf.stop_gradient(PhiTZ) * logp          # (batch, n_atoms)
        # loss    = tf.math.reduce_mean(tf.math.reduce_sum(ce, axis=-1))

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
                action, *_ = self(obs)

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
        '''Train C51

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
            C51: self
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
                
                saved_path = self.save(save_path, checkpoint_number=self.num_epochs)

                if self.verbose > 0:
                    LOG.info('Checkpoint saved to: {}'.format(saved_path))

        return self

    def get_config(self):
        
        init_config = { 'learning_rate':       self.learning_rate,
                        'buffer_size':         self.buffer_size,
                        'min_buffer':          self.min_buffer,
                        'n_atoms':             self.n_atoms,
                        'v_min':               self.v_min,
                        'v_max':               self.v_max,
                        'n_steps':             self.n_steps,
                        'n_gradsteps':         self.n_gradsteps,
                        'batch_size':          self.batch_size,
                        'gamma':               self.gamma,
                        'tau':                 self.tau,
                        'max_grad_norm':       self.max_grad_norm,
                        'force_mlp':           self.force_mlp,
                        'explore_schedule':    self.explore_schedule,
                        'verbose':             self.verbose}

        setup_config = {'observation_space': self.observation_space,
                        'action_space':      self.action_space}

        init_config  = to_json_serializable(init_config)
        setup_config = to_json_serializable(setup_config)

        return {'init_config': init_config,
                'setup_config': setup_config}
    
    @classmethod
    def from_config(cls, config):

        assert 'init_config' in config, 'Failed to load {} config, init_config not found'.format(cls.__name__)
        assert 'setup_config' in config, 'Failed to load {} config, setup_config not found'.format(cls.__name__)

        init_config = from_json_serializable(config['init_config'])
        setup_config = from_json_serializable(config['setup_config'])

        # construct model
        self = cls(env=None, **init_config)
        self.setup_model(**setup_config)

        return self