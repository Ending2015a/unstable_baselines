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
from unstable_baselines import logger


from unstable_baselines.base import (SavableModel, 
                                     TrainableModel)
from unstable_baselines.bugs import ReLU
from unstable_baselines.prob import (Categorical,
                                     MultiNormal)
from unstable_baselines.utils import (set_global_seeds,
                                      normalize,
                                      unnormalize,
                                      to_json_serializable,
                                      from_json_serializable)



# create logger
LOG = logger.getLogger('PPO')

# === Buffers ===
class GaeBuffer():
    '''A Generalized Advantage Estimation Buffer'''
    def __init__(self, gae_lambda=1.0, gamma=0.99):
        '''GAE Buffer
            refer to: https://arxiv.org/abs/1506.02438

        Args:
            gae_lambda (float, optional): Smoothing parameter. Defaults to 1.0.
            gamma (float, optional): Discount factor. Defaults to 0.99.
        '''        
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.reset()

    def reset(self):
        self.observations = []
        self.actions      = []
        self.rewards      = []
        self.returns      = []
        self.dones        = []
        self.values       = []
        self.log_probs    = []
        self.advantages   = []
        self.ready_for_sampling = False

    def add(self, obs, action, reward, done, value, log_prob):
        '''Add batch samples into buffer

        Args:
            obs (np.ndarray): observations, shape (n_envs, obs_space.shape), 
                np.uint8 for image observations, np.float32 for continuous 
                control observaitons.
            action (np.ndarray): actions, shape (n_envs, act_space.shape),
                np.int32 or np.int64 for categorical actions, np.float32 for
                continuous control actions.
            reward (np.ndarray): rewards, shape (n_envs,), np.float32
            done (np.ndarray): dones, shape (n_envs,), np.bool or np.float32
            value (np.ndarray): predicted values, shape (n_envs,), np.float32
            log_prob (np.ndarray): log probabilities, shape (n_envs), 
                np.float32
        '''
        
        assert self.ready_for_sampling == False, 'The buffer is ready for sampling, please reset the buffer'

        self.observations.append(np.asarray(obs).copy())
        self.actions.append(np.asarray(action).copy())
        self.rewards.append(np.asarray(reward).copy())
        self.dones.append(np.asarray(done).copy())
        self.values.append(np.asarray(value).copy())
        self.log_probs.append(np.asarray(log_prob).copy())

    def __len__(self):
        return len(self.observations)

    def __call__(self, batch_size=None):
        '''Sample batch

        Args:
            batch_size (int, optional): batch size. Defaults to None.

        Yields:
            np.ndarray: observations, shape (batch, obs_space.shape)
            np.ndarray: actions, shape (batch, act_space.shape)
            np.ndarray: values, shape (batch,)
            np.ndarray: log probabilities, shape (batch,)
            np.ndarray: advantages, shape (batch,)
            np.ndarray: returns, shape (batch,)
        '''        
        assert self.ready_for_sampling, 'The buffer is not ready for sampling, please call make() first'

        buffer_size = len(self)

        if batch_size is None:
            batch_size = buffer_size
        
        indices = np.random.permutation(buffer_size)

        start_idx = 0
        while start_idx < buffer_size:
            # return generator
            yield self._get_samples(indices[start_idx:start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, indices):
        return (self.observations[indices], # (batch, obs_space.shape)
                self.actions[indices],      # (batch, act_space.shape)
                self.values[indices],       # (batch, )
                self.log_probs[indices],    # (batch, )
                self.advantages[indices],   # (batch, )
                self.returns[indices])      # (batch, )

    def make(self):
        '''Compute GAE
        
        Call this function before sampling this buffer.
        '''        

        self.observations = np.asarray(self.observations) # (steps, n_envs, obs_space.shape)
        self.actions      = np.asarray(self.actions)      # (steps, n_envs, act_space.shape)
        self.rewards      = np.asarray(self.rewards)      # (steps, n_envs)
        self.dones        = np.asarray(self.dones)        # (steps, n_envs)
        self.values       = np.asarray(self.values)       # (steps, n_envs)
        self.log_probs    = np.asarray(self.log_probs)    # (steps, n_envs)
        self.advantages   = np.zeros(self.values.shape,   dtype=np.float32) # (steps, n_envs)

        # convert type bool to float32
        self.dones        = self.dones.astype(np.float32)

        # compute GAE
        last_gae_lam = 0
        buffer_size  = len(self)
        next_non_terminal = 1.0 - self.dones[-1]
        next_value   = self.values[-1]

        for step in reversed(range(buffer_size)):

            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            
            self.advantages[step] = last_gae_lam
            
            # prepare for next step
            next_non_terminal = 1.0 - self.dones[step]
            next_value = self.values[step]
        
        self.returns = self.advantages + self.values

        # flatten 
        self.observations = self._swap_flatten(self.observations) # (n_envs*steps, obs_space.shape)
        self.actions      = self._swap_flatten(self.actions)      # (n_envs*steps, act_space.shape)
        self.values       = self._swap_flatten(self.values)       # (n_envs*steps,)
        self.log_probs    = self._swap_flatten(self.log_probs)    # (n_envs*steps,)
        self.advantages   = self._swap_flatten(self.advantages)   # (n_envs*steps,)
        self.returns      = self._swap_flatten(self.returns)      # (n_envs*steps,)

        self.ready_for_sampling = True
    
    def _swap_flatten(self, v):
        '''
        Swap and flatten first two dimensions

        (steps, n_envs, ...) -> (n_envs*steps, ...)
        '''
        shape = v.shape
        if len(shape) < 3:
            v = v.swapaxes(0, 1).reshape(shape[0]*shape[1])
        else:
            v = v.swapaxes(0, 1).reshape(shape[0]*shape[1], *shape[2:])

        return v

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


class CategoricalPolicyNet(tf.keras.Model):
    '''Categorical policy for discrete action space'''
    def __init__(self, action_space, **kwargs):
        '''Create categorical policy network

        Args:
            action_space (gym.Spaces): discrete action space.
        '''        
        super().__init__(**kwargs)

        self._layers = [
            tf.keras.layers.Dense(action_space.n)
        ]
    
    @tf.function
    def call(self, inputs, training=False):
        '''Forward network

        Args:
            inputs (tf.Tensor): Expecting a latent vector in shape 
                (batch, latent_size), tf.float32.
            training (bool, optional): Training mode. Defaults to False.

        Returns:
            tf.Tensor: Predicted latent categorical distributions.
                (batch, act_space.n)
        '''
        x = inputs
        for layer in self._layers:
            x = layer(x)
        
        return x

    def get_distribution(self, logits):
        '''Create categorical distributions.

        Args:
            logits (tf.Tensor): Latent categorical distributions.

        Returns:
            Categorical: categorical distribution object.
        '''
        return Categorical(logits)


class DiagGaussianPolicyNet(tf.keras.Model):
    '''Diagonal Gaussian policy with state-independent std
    for coninuous (box) action space'''
    def __init__(self, action_space, **kwargs):
        '''Create Diagonal Gaussian policy network

        Args:
            action_space (gym.Spaces): box action space.
        '''        
        super().__init__(**kwargs)

        self._layers = [
            tf.keras.layers.Dense(action_space.shape[0])
        ]
        
        # State-independent std
        self._logstd = tf.Variable(np.zeros(shape=(action_space.shape[0],),
                            dtype=np.float32))

    @tf.function
    def call(self, inputs, training=False):
        '''Forward network

        Args:
            inputs (tf.Tensor): Expecting a latent vector in shape 
                (batch, latent_size), tf.float32.
            training (bool, optional): Training mode. Defaults to False.

        Returns:
            tuple: Predicted latent multivariate normal distributions.
                [0] (tf.Tensor): means, shape (batch, act_space.shape)
                [1] (tf.Tensor): standard deviations, shape
                    (batch, act_space.shape)
        '''
        x = inputs
        for layer in self._layers:
            x = layer(x)
        
        std = tf.ones_like(x) * tf.math.exp(self._logstd)
        return x, std

    def get_distribution(self, logits):
        '''Create multivariate normal distributions.

        Args:
            logits (tuple): Latent multivariate normal distributions.
                [0] (tf.Tensor): means, shape (batch, act_space.shape)
                [1] (tf.Tensor): standard deviations, shape 
                    (batch, act_space.shape)

        Returns:
            MultiNormal: multivariate normal distribution object.
        '''
        return MultiNormal(logits[0], logits[1])


class PolicyNet(tf.keras.Model):
    '''Policy network'''
    def __init__(self, action_space, **kwargs):
        '''Create policy network

        Args:
            action_space (gym.Spaces): action space, supporting
                Discrete or Box.
        '''
        super().__init__(**kwargs)
        
        if isinstance(action_space, gym.spaces.Discrete):
            self._net = CategoricalPolicyNet(action_space)
        elif isinstance(action_space, gym.spaces.Box):
            self._net = DiagGaussianPolicyNet(action_space)
        else:
            raise NotImplementedError('Action space not supported: {}'.format(type(action_space)))

    @tf.function
    def call(self, inputs, training=False):
        '''Forward network

        Args:
            inputs (tf.Tensor): Expecting a latent vector in shape 
                (batch, latent_size), tf.float32.
            training (bool, optional): Training mode. Defaults to False.

        Returns:
            tf.Tensor or tuple: Predicted latent distributions.
                tf.Tensor for discrete action space (categorical)
                tuple (mean, std) for box action space (multinormal)
        '''
        return self._net(inputs, training=training)

    def get_distribution(self, logits):
        '''Create distribution objects

        Args:
            logits (tf.Tensor or tuple): Predicted latent distributions.
                tf.Tensor for discrete action space (categorical)
                tuple (mean, std) for box action space (multinormal)

        Returns:
            Categorical or MultiNormal: distribution objects corresponds 
                to the type of action space.
        '''
        return self._net.get_distribution(logits)


class ValueNet(tf.keras.Model):
    '''Value network'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._layers = [
            tf.keras.layers.Dense(1)
        ]

    @tf.function
    def call(self, inputs, training=False):
        '''Forward network

        Args:
            inputs (tf.Tensor): Expecting a latent vector in shape 
                (batch, latent_size), tf.float32.
            training (bool, optional): Training mode. Defaults to False.

        Returns:
            tf.Tensor: Predicted state values in shape (batch, 1)
        '''
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x


# ==== Agent, Model ===

class Agent(SavableModel):
    def __init__(self, observation_space, action_space, shared_net=False, force_mlp=False, **kwargs):
        '''PPO Agent

        Args:
            observation_space (gym.Spaces): The observation space of the environment.
                Set to None for delayed setup.
            action_space (gym.Spaces): The action space of the environment.
                Set to None for delayed setup.
            shared_net (bool, optional): determine whether to share the feature 
                extractor between policy and value network. Defaults to False.
            force_mlp (bool, optional): Force to use MLP feature extractor. 
                Defaults to False.
        '''
        super().__init__(**kwargs)

        self.shared_net = shared_net
        self.force_mlp  = force_mlp

        # --- Initialize ---
        self.observation_space = None
        self.action_space      = None
        self.net               = None
        self.net2              = None
        self.policy_net        = None
        self.value_net         = None

        if observation_space is not None and action_space is not None:
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
        
        self.policy_net = PolicyNet(self.action_space)
        
        # not sharing the backbone network
        if not self.shared_net:
            self.net2 = type(self.net)()
        
        self.value_net  = ValueNet()

        # construct networks
        inputs  = tf.keras.Input(shape=self.observation_space.shape, dtype=tf.float32)
        
        outputs = self.net(inputs)
        self.policy_net(outputs)

        if self.net2 is not None:
            outputs = self.net2(inputs)
        self.value_net(outputs)

    @tf.function
    def _forward(self, inputs, training=True):
        '''Forward networks

        Args:
            inputs (tf.Tensor): batch observations in shape (batch, obs_space.shape).
                tf.uint8 for image observations and tf.float32 for non-image 
                observations.
            training (bool, optional): training mode. Default to True.

        Returns:
            tf.Tensor or tuple: Predicted latent distributions.
                tf.Tensor for discrete action space (categorical)
                    shape (batch, act_space.n)
                tuple (mean, std) for box action space (multinormal)
                    shapes are both (batch, act_space.shape)
            tf.Tensor: Predicted state values, shape (batch,)
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
        logits = self.policy_net(latent, training=training)

        # forward value net
        if self.net2 is not None:
            latent = self.net2(inputs, training=training)
        values = self.value_net(latent, training=training) # (batch, 1)
        values = tf.squeeze(values, axis=-1)               # (batch, )

        return logits, values

    @tf.function
    def call(self, inputs, deterministic=False, training=True):
        '''Batch predict actions

        Args:
            inputs (tf.Tensor): batch observations in shape (batch, obs_space.shape).
                tf.uint8 for image observations and tf.float32 for non-image 
                observations.
            deterministic (bool, optional): deterministic actions. Defaults to False.
            training (bool, optional): training mode. Default to True.

        Returns:
            tf.Tensor: predicted actions, shape (batch, act_space.shape)
            tf.Tensor: predicted values, shape (batch,)
            tf.Tensor: action log likelihood, shape (batch,)
        '''

        # forward
        logits, values = self._forward(inputs, training=training)
        distrib = self.policy_net.get_distribution(logits)

        if deterministic:
            actions = distrib.mode()
        else:
            actions = distrib.sample()

        log_probs = distrib.log_prob(actions)

        return actions, values, log_probs

    def predict(self, inputs, clip_action=True, deterministic=True):
        '''Predict actions

        Args:
            inputs (np.ndarray): batch observations in shape (batch, obs_space.shape)
                or a single observation in shape (obs_space.shape). np.uint8 for image
                observations and np.float32 for non-image observations.
            clip_action (bool, optional): perform action clipping for box action spaces.
                ignored in discrete action settings. Defaults to True.
            deterministic (bool, optional): deterministic actions. Defaults to True.

        Returns:
            np.ndarray: predicted actions in shape (batch,) or (), np.int64 for 
                discrete actions, shape (batch, act_space.shape) or (act_space.shape), 
                np.float32 for continuous actions.
        '''
        one_sample  = (len(inputs.shape) == len(self.observation_space.shape))

        if one_sample:
            inputs  = np.expand_dims(inputs, axis=0)

        # predict
        outputs, *_ = self(inputs, deterministic=deterministic, training=False)
        outputs     = outputs.numpy()

        # clip action to a valid range (Continuous action)
        if clip_action and isinstance(self.action_space, gym.spaces.Box):
            outputs = np.clip(outputs, self.action_space.low, self.action_space.high)

        if one_sample:
            outputs = np.squeeze(outputs, axis=0)

        # predict
        return outputs

    def get_config(self):

        config = {'observation_space': self.observation_space,
                  'action_space':      self.action_space,
                  'shared_net':        self.shared_net,
                  'force_mlp':         self.force_mlp}

        return to_json_serializable(config)


class PPO(TrainableModel):
    def __init__(self, env, learning_rate: float = 3e-4, 
                            n_steps:         int = 2048, 
                            batch_size:      int = 64, 
                            n_subepochs:     int = 10,
                            gamma:         float = 0.99, 
                            gae_lambda:    float = 0.95,
                            clip_range:    float = 0.2, 
                            clip_range_vf: float = None,
                            ent_coef:      float = 0.01, 
                            vf_coef:       float = 0.5, 
                            max_grad_norm: float = 0.5,
                            target_kl:     float = None,
                            shared_net:     bool = False,
                            force_mlp:      bool = False,
                            verbose:         int = 0, 
                            **kwargs):
        '''Proximal policy optimization

        The implementeation mainly follows its originated paper
        "Proximal Policy Optimization Algorithms" by Schulman et al.

        The first argument `env` can be `None` for delayed model setup. You
        should call `set_env()` then call `setup_model()` to manually setup
        the model.

        Args:
            env (gym.Env): Training environment. Set to `None` for delayed setup.
            learning_rate (float, optional): Learning rate. Defaults to 3e-4.
            n_steps (int, optional): number of steps of rollouts to collect for
                each epoch. Defaults to 2048.
            batch_size (int, optional): Training batch size. Defaults to 64.
            n_subepochs (int, optional): Number of subepochs for each epoch.
                Defaults to 10.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            gae_lambda (float, optional): GAE smooth parameter. Defaults to 0.95.
            clip_range (float, optional): Policy ratio clip range. Defaults to 0.2.
            clip_range_vf (float, optional): Target value clip range. 
                Defaults to None.
            ent_coef (float, optional): Entropy loss coefficient. Defaults to 0.01.
            vf_coef (float, optional): Value loss coefficient. Defaults to 0.5.
            max_grad_norm (float, optional): Adam clipnorm. Defaults to 0.5.
            target_kl (float, optional): Target KL divergence to early stop. 
                Defaults to None.
            shared_net (bool, optional): Whether to share the feature extractor
                between policy and value networks. Defaults to False.
            force_mlp (bool, optional): Force to use MLP net as feature extractor. Defaults to False.
            verbose (int, optional): More training log. Defaults to 0.
        '''        
        super().__init__(**kwargs)

        self.env = env
        
        self.learning_rate   = learning_rate
        self.batch_size      = batch_size
        self.n_subepochs     = n_subepochs
        self.n_steps         = n_steps
        self.gamma           = gamma
        self.gae_lambda      = gae_lambda
        self.clip_range      = clip_range
        self.clip_range_vf   = clip_range_vf
        self.ent_coef        = ent_coef
        self.vf_coef         = vf_coef
        self.max_grad_norm   = max_grad_norm
        self.target_kl       = target_kl
        self.shared_net      = shared_net
        self.force_mlp       = force_mlp
        self.verbose         = verbose

        self.buffer            = None
        self.tb_writer         = None
        self.observation_space = None
        self.action_space      = None
        self.n_envs            = None

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
        '''Setup model, optimizer for training

        Args:
            observation_space (gym.Spaces): The observation space of the
                environment.
            action_space (gym.Spaces): The action space of the environment.
        '''

        self.observation_space = observation_space
        self.action_space      = action_space

        # --- setup model ---
        self.buffer     = GaeBuffer(gae_lambda=self.gae_lambda, gamma=self.gamma)
        self.agent      = Agent(self.observation_space, self.action_space, shared_net=self.shared_net, force_mlp=self.force_mlp)
        
        self.optimizer  = tf.keras.optimizers.Adam(learning_rate=self.learning_rate,
                                                   epsilon=1e-5,
                                                   clipnorm=self.max_grad_norm)

    @tf.function
    def _forward(self, input, training=True):
        '''Forward agent

        Args:
            inputs (tf.Tensor): batch observations in shape (batch, obs_space.shape).
                tf.uint8 for image observations and tf.float32 for non-image 
                observations.
            training (bool, optional): training mode. Default to True.

        Returns:
            tf.Tensor or tuple: Predicted latent distributions.
                tf.Tensor for discrete action space (categorical)
                    shape (batch, act_space.n)
                tuple (mean, std) for box action space (multinormal)
                    shapes are both (batch, act_space.shape)
            tf.Tensor: Predicted state values, shape (batch,)
        '''
        return self.agent._forward(input, training=training)
    
    @tf.function
    def call(self, inputs, deterministic=False, training=True):
        '''Batch predict actions

        Args:
            inputs (tf.Tensor): batch observations in shape (batch, obs_space.shape).
                tf.uint8 for image observations and tf.float32 for non-image 
                observations.
            deterministic (bool, optional): deterministic actions. Defaults to False.
            training (bool, optional): training mode. Default to True.

        Returns:
            tf.Tensor: predicted actions, shape (batch, act_space.shape)
            tf.Tensor: predicted values, shape (batch,)
            tf.Tensor: action log likelihood, shape (batch,)
        '''
        return self.agent(inputs, deterministic=deterministic, training=training)

    def predict(self, inputs, clip_action=True, deterministic=True):
        '''Predict actions

        Args:
            inputs (np.ndarray): batch observations in shape (batch, obs_space.shape)
                or a single observation in shape (obs_space.shape). np.uint8 for image
                observations and np.float32 for non-image observations.
            clip_action (bool, optional): perform action clipping for box action spaces.
                ignored in discrete action settings. Defaults to True.
            deterministic (bool, optional): deterministic actions. Defaults to True.

        Returns:
            np.ndarray: predicted actions in shape (batch,) or (), np.int64 for 
                discrete actions, shape (batch, act_space.shape) or (act_space.shape), 
                np.float32 for continuous actions.
        '''
        return self.agent.predict(inputs, clip_action=clip_action, deterministic=deterministic)

    @tf.function
    def policy_loss(self, advantage, log_prob, old_log_prob):
        '''Compute policy loss (clipped surrogate loss)

        Args:
            advantage (tf.Tensor): batch advantages, shape (batch,), tf.float32
            log_prob (tf.Tensor): batch action log likelihoods, shape (batch,), 
                tf.float32
            old_log_prob (tf.Tensor): batch old action log likelihoods, shape 
                (batch,), tf.float32

        Returns:
            tf.Tensor: policy loss, tf.float32
        '''
        # normalize advantage, stable baselines: ppo2.py#L265
        advantage = (advantage - tf.math.reduce_mean(advantage)) / (tf.math.reduce_std(advantage) + 1e-8)
        # policy ratio
        ratio = tf.exp(log_prob - old_log_prob)
        # clipped surrogate loss
        policy_loss_1 = advantage * ratio
        policy_loss_2 = advantage * tf.clip_by_value(ratio, 1.-self.clip_range, 1.+self.clip_range)
        policy_loss   = -tf.math.reduce_mean(tf.minimum(policy_loss_1, policy_loss_2))

        return policy_loss
    
    @tf.function
    def value_loss(self, values, old_values, returns):
        '''Compute value loss

        Args:
            values (tf.Tensor): batch values, shape (batch,), tf.float32
            old_values (tf.Tensor): batch old values, shape (batch,), tf.float32
            returns (tf.Tensor): batch returns, shape (batch,), tf.float32

        Returns:
            tf.Tensor: value loss, tf.float32
        '''
        if self.clip_range_vf is None:
            values_pred = values
        else:
            values_pred = old_values + tf.clip_by_value(values-old_values, -self.clip_range_vf, self.clip_range_vf)
        
        return tf.keras.losses.MSE(returns, values_pred)

    @tf.function
    def _train_step(self, obs, actions, old_values, old_log_probs, advantages, returns):
        '''Perform one gradient update

        Args:
            obs (tf.Tensor): batch observations, shape (batch, obs_space.shape),
                tf.uint8 for image observations, tf.float32 for non-image observations
            actions (tf.Tensor): batch actions, shape (batch,), tf.int32 or tf.int64 
                for discrete action space, shape (batch, act_space.shape), tf.float32
                for continuous action space.
            old_values (tf.Tensor): batch old values, shape (batch,), tf.float32
            old_log_probs (tf.Tensor): batch old action log likelihood, shape (batch,)
                tf.float32
            advantages (tf.Tensor): batch advantages, shape (batch,), tf.float32
            returns (tf.Tensor): batch returns, shape (batch,), tf.float32

        Returns:
            tf.Tensor: total loss, tf.float32
            tf.Tensor: policy KL divergence, tf.float32
            tf.Tensor: policy entropy, tf.float32
            tf.Tensor: policy loss, tf.float32
            tf.Tensor: value loss, tf.float32
            tf.Tensor: entropy loss, tf.float32
        '''
        
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)

            # forward
            logits, values = self._forward(obs, training=True)
            distrib   = self.agent.policy_net.get_distribution(logits)

            log_probs = distrib.log_prob(actions)
            entropy   = distrib.entropy()
            kl        = 0.5 * tf.math.reduce_mean(tf.math.square(old_log_probs - log_probs))

            # compute policy loss & value loss
            pi_loss   = self.policy_loss(advantages, log_probs, old_log_probs)
            vf_loss   = self.value_loss(values, old_values, returns)

            # compute entropy loss
            ent_loss  = -tf.math.reduce_mean(entropy)

            # compute total loss
            loss      = pi_loss + self.ent_coef * ent_loss + self.vf_coef * vf_loss
 
        # perform gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss, kl, entropy, pi_loss, vf_loss, ent_loss

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
            
            raw_actions, values, log_probs = self(obs)

            raw_actions  = raw_actions.numpy()
            values       = values.numpy()
            log_probs    = log_probs.numpy()

            # clip action to a valid range (Continuous action)
            if isinstance(self.action_space, gym.spaces.Box):
                actions = np.clip(raw_actions, self.action_space.low, self.action_space.high)
            else:
                actions = raw_actions

            # step environment
            new_obs, rews, dones, infos = self.env.step(actions)
            
            # add to buffer
            self.buffer.add(obs, raw_actions, rews, dones, values, log_probs)
            obs = new_obs

            # update state
            self.num_timesteps += self.n_envs

        return new_obs

    def train(self, n_subepochs, batch_size):
        '''Train one epochs

        Args:
            n_subepochs (int): Number of subepochs.
            batch_size (int): batch size.

        Returns:
            float: mean total loss of the last subepoch, tf.float32
            float: mean policy KL divergence of the last subepoch, tf.float32
            float: mean policy entropy of the last subepoch, tf.float32
            float: mean policy loss of the last subepoch, tf.float32
            float: mean value loss of the last subepoch, tf.float32
            float: mean entropy loss of the last subepoch, tf.float32
            float: mean explained variation of the last subepoch, tf.float32
        '''
        assert self.buffer.ready_for_sampling, "Buffer is not ready for sampling, please call buffer.make() before sampling"

        for epoch in range(n_subepochs):
            all_losses = []

            for replay_data in self.buffer(batch_size):
                # update once
                losses = self._train_step(*replay_data)
                
                # convert to tuple
                if not isinstance(losses, (tuple)):
                    losses = (losses, )
                
                all_losses.append(losses)

                # update state
                self.num_gradsteps += 1

            # update state
            self.num_subepochs += 1

            m_kl = np.mean(np.hstack(np.array(all_kl)))
            # early stop
            if self.target_kl is not None and m_kl > 1.5 * self.target_kl:
                LOG.warning('Early stopping at epoch {} due to reaching max kl: {:.2f}'.format(epoch, m_kl))
                break

        # calculate explained variance
        y_pred  = self.buffer.values.flatten()
        y_true  = self.buffer.returns.flatten()
        var     = np.var(y_true)
        exp_var = np.nan if var == 0 else 1 - np.var(y_true - y_pred) / var

        m_losses = []
        for loss in zip(*all_losses):
            m_losses.append(np.mean(np.hstack(np.asarray(losses, dtype=np.float32))))

        return (*m_losses, exp_var)

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
                    eval_interval:    int = 1000, 
                    eval_episodes:    int = 5, 
                    eval_max_steps:   int = 3000,
                    save_interval:    int = 1000,
                    save_path:        str = None,
                    tb_logdir:        str = None, 
                    reset_timesteps: bool = False):
        '''Train PPO

        Args:
            total_timesteps (int): Total timesteps to train agent.
            log_interval (int, optional): Print log every ``log_interval`` 
                epochs. Defaults to 1.
            eval_env (gym.Env, optional): Environment for evaluation. 
                Defaults to None.
            eval_interval (int, optional): Evaluate every ``eval_interval``
                epochs. Defaults to 1000.
            eval_episodes (int, optional): Evaluate ``eval_episodes`` episodes. 
                for every evaluation. Defaults to 5.
            eval_max_steps (int, optional): maximum steps every evaluation. 
                Defaults to 3000.
            save_interval (int, optional): Save model every ``save_interval``
                epochs. Default to 1000.
            save_path (str, optional): Model saving path. Default to None.
            tb_logdir (str, optional): tensorboard log directory. Defaults to None.
            reset_timesteps (bool, optional): reset timesteps. Defaults to False.

        Returns:
            PPO: self
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
            self.num_subepochs = 0
            self.progress      = 0


        # initialize
        obs        = None
        time_start = time.time()
        time_spent = 0
        timesteps_per_epoch = self.n_steps * self.n_envs
        total_epochs = int(float(total_timesteps - self.num_timesteps) / 
                                        float(timesteps_per_epoch) + 0.5)


        while self.num_timesteps < total_timesteps:
            
            # reset buffer
            self.buffer.reset()
            # collect rollouts
            obs = self._run(steps=self.n_steps, obs=obs)

            # update state
            self.num_epochs += 1
            self.progess = float(self.num_timesteps) / float(total_timesteps)

            # make buffer
            self.buffer.make()
            # train agent
            losses = self.train(self.n_subepochs, batch_size=self.batch_size)

            log_info = {}
            tb_info  = {}

            loss, kl, ent, pi_loss, vf_loss, ent_loss, exp_var = losses
            tb_info['loss']          = log_info['Loss']          = loss
            tb_info['approx_kl']     = log_info['Approx KL']     = kl
            tb_info['entropy']       = log_info['Entropy']       = ent
            tb_info['policy_loss']   = log_info['Policy loss']   = pi_loss
            tb_info['value_loss']    = log_info['Value loss']    = vf_loss
            tb_info['entropy_loss']  = log_info['Entropy loss']  = ent_loss
            tb_info['explained_var'] = log_info['Explained var'] = exp_var

            # write tensorboard
            if self.tb_writer is not None:
                with self.tb_writer.as_default():
                    for field, value in tb_info.items():
                        tf.summary.scalar(field, value, step=self.num_timesteps)

                self.tb_writer.flush()

            # print training log
            if (log_interval is not None) and (self.num_epochs % log_interval == 0):
                # current time
                time_now        = time.time()
                # execution time (one epoch)
                execution_time  = (time_now - time_start) - time_spent
                # total time spent
                time_spent      = time_now - time_start
                # remaining time
                remaining_time  = (time_spent / self.progress)*(1.0-self.progress)
                # eta
                eta             = (datetime.datetime.now() + datetime.timedelta(seconds=remaining_time)).strftime('%Y-%m-%d %H:%M:%S')
                # average steps per second
                fps             = float(self.num_timesteps) / time_spent

                LOG.set_header('Epoch {}/{}'.format(self.num_epochs, total_epochs))
                LOG.add_line()
                LOG.add_row('Timesteps',      self.num_timesteps, total_timesteps, fmt='{}: {}/{}')
                LOG.add_row('Steps/sec',      fps,                                 fmt='{}: {:.2f}')
                LOG.add_row('Progress',       self.progress*100.0,                 fmt='{}: {:.2f}%')

                if self.verbose > 0:
                    LOG.add_row('Execution time', datetime.timedelta(seconds=execution_time))
                    LOG.add_row('Elapsed time',   datetime.timedelta(seconds=time_spent))
                    LOG.add_row('Remaining time', datetime.timedelta(seconds=remaining_time))
                    LOG.add_row('ETA',            eta)
                    LOG.add_line()

                    for field, value in log_info.items():
                        LOG.add_row(field, value, fmt='{}: {:.6f}')
                
                LOG.add_line()
                LOG.flush('INFO')

            # evaluate model
            if (eval_env is not None) (and self.num_epochs % eval_interval == 0):

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

                LOG.set_header('Evaluate {}/{}'.format(episode, total_episode))
                LOG.add_line()
                LOG.add_row('Max rewards',  max_rews)
                LOG.add_row('  Length',     max_steps)
                LOG.add_line()
                LOG.add_row('Mean rewards', mean_rews)
                LOG.add_row('Std rewards',  std_rews, fmt='{}: {:.6f}')
                LOG.add_row('Mean length',  mean_steps)
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

        init_config = { 'learning_rate':   self.learning_rate,
                        'batch_size':      self.batch_size,
                        'n_subepochs':     self.n_subepochs,
                        'n_steps':         self.n_steps,
                        'gamma':           self.gamma,
                        'gae_lambda':      self.gae_lambda,
                        'clip_range':      self.clip_range,
                        'clip_range_vf':   self.clip_range_vf,
                        'ent_coef':        self.ent_coef,
                        'vf_coef':         self.vf_coef,
                        'max_grad_norm':   self.max_grad_norm,
                        'target_kl':       self.target_kl,
                        'shared_net':      self.shared_net,
                        'force_mlp':       self.force_mlp,
                        'verbose':         self.verbose}

        setup_config = {'observation_space': self.observation_space,
                        'action_space': self.action_space}


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
