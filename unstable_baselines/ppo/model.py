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

from unstable_baselines.base import SavableModel
from unstable_baselines.prob import (Categorical, 
                                     MultiNormal)
from unstable_baselines.utils import (set_global_seeds,
                                      normalize,
                                      unnormalize,
                                      to_json_serializable,
                                      from_json_serializable)


# create logger
logger.Config.use(level='DEBUG', colored=True, reset=False)
LOG = logger.getLogger('PPO')

__all__ = [
    'PPO'
]

# === Buffers ===
class GaeBuffer():
    '''
    GAE Buffer
        refer to: https://arxiv.org/abs/1506.02438

    add: add samples
    __call__: sampling, return a generator
    make: compute GAE
    '''
    def __init__(self, gae_lambda=1.0, gamma=0.99):
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
        '''
        Add samples
        
        obs: observations, shape: (n_envs, obs_space.shape)
        action: actions, shape: (n_envs, act_space.shape)
        reward: reward, shape: (n_envs,)
        done: done, shape: (n_envs,)
        value: value, shape: (n_envs,)
        log_prob: log pi, shape: (n_envs,)
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
        '''
        Generate samples

        indices: 1D array
        '''
        return (self.observations[indices], # (batch, obs_space.shape)
                self.actions[indices],      # (batch, act_space.shape)
                self.values[indices],       # (batch, )
                self.log_probs[indices],    # (batch, )
                self.advantages[indices],   # (batch, )
                self.returns[indices])      # (batch, )

    def make(self):

        self.observations = np.asarray(self.observations, dtype=np.float32) # (steps, n_envs, obs_space.shape)
        self.actions      = np.asarray(self.actions,      dtype=np.float32) # (steps, n_envs, act_space.shape)
        self.rewards      = np.asarray(self.rewards,      dtype=np.float32) # (steps, n_envs)
        self.dones        = np.asarray(self.dones,        dtype=np.float32) # (steps, n_envs)
        self.values       = np.asarray(self.values,       dtype=np.float32) # (steps, n_envs)
        self.log_probs    = np.asarray(self.log_probs,    dtype=np.float32) # (steps, n_envs)
        self.advantages   = np.zeros(self.values.shape,   dtype=np.float32) # (steps, n_envs)

        # compute GAE
        last_gae_lam = 0
        buffer_size = len(self)
        next_non_terminal = 1.0 - self.dones[-1]
        next_value = self.values[-1]

        for step in reversed(range(buffer_size)):

            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            
            self.advantages[step] = last_gae_lam
            
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
    
    @classmethod
    def _swap_flatten(cls, v):
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
        '''
        Nature CNN use the same architecture as its origin paper
            "Playing Atari with Deep Reinforcement Learning"
        '''
        super().__init__(**kwargs)

        self._layers = [
            tf.keras.layers.Conv2D(32, 8, 4, name='conv1'),
            tf.keras.layers.ReLU(name='relu1'),
            tf.keras.layers.Conv2D(64, 4, 2, name='conv2'),
            tf.keras.layers.ReLU(name='relu2'),
            tf.keras.layers.Conv2D(64, 3, 1, name='conv3'),
            tf.keras.layers.ReLU(name='relu3'),
            tf.keras.layers.Flatten(name='flatten'),
            tf.keras.layers.Dense(512, name='fc'),
            tf.keras.layers.ReLU(name='relu4')
        ]

    def call(self, inputs, training=False):
        '''
        inputs: observations (batch, obs_space.shape)
        '''
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x


# Mlp feature extractor
class Mlp(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._layers = [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, name='fc1'),
            tf.keras.layers.ReLU(name='relu1'),
            tf.keras.layers.Dense(64, name='fc2'),
            tf.keras.layers.ReLU(name='relu2'),
        ]
    def call(self, inputs, training=False):
        '''
        inputs: observations (batch, obs_space.shape)
        '''
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x


# for Discrete action space
class CategoricalPolicyNet(tf.keras.Model):
    def __init__(self, action_space, **kwargs):
        super().__init__(**kwargs)

        self._layers = [
            tf.keras.layers.Dense(action_space.n)
        ]
    
    def call(self, inputs, training=False):
        '''
        inputs: latent (batch, latent_size)
        '''
        x = inputs
        for layer in self._layers:
            x = layer(x)
        
        return x

    def get_distribution(self, logits):
        return Categorical(logits)

# for Continuous (Box) action space
class DiagGaussianPolicyNet(tf.keras.Model):
    def __init__(self, action_space, **kwargs):
        super().__init__(**kwargs)

        self._layers = [
            tf.keras.layers.Dense(action_space.shape[0])
        ]
        
        self._logstd = tf.Variable(tf.keras.initializers.Zeros()(shape=(action_space.shape[0],)),
                                   shape=(action_space.shape[0],), dtype=tf.float32)

    def call(self, inputs, training=False):
        '''
        inputs: latent (batch, latent_size)
        outputs: [mean, scale]
        '''
        x = inputs
        for layer in self._layers:
            x = layer(x)
        
        std = tf.expand_dims(tf.math.exp(self._logstd), axis=0)
        return x, std

    def get_distribution(self, logits):
        return MultiNormal(logits[0], logits[1])

# Policy network
class PolicyNet(tf.keras.Model):
    def __init__(self, action_space, **kwargs):
        super().__init__(**kwargs)

        if isinstance(action_space, gym.spaces.Discrete):
            self._net = CategoricalPolicyNet(action_space)
        elif isinstance(action_space, gym.spaces.Box):
            self._net = DiagGaussianPolicyNet(action_space)
        else:
            raise NotImplementedError('Action space not supported: {}'.format(type(action_space)))

    def call(self, inputs, training=False):
        '''
        inputs: latent (batch, latent_size)
        '''
        
        return self._net(inputs, training=training)

    def get_distribution(self, logits):

        return self._net.get_distribution(logits)

# Value network
class ValueNet(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._layers = [
            tf.keras.layers.Dense(1)
        ]

    def call(self, inputs, training=False):
        '''
        inputs: latent (batch, latent_size)
        '''
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x


# ==== Agent, Model ===

class Agent(SavableModel):
    def __init__(self, observation_space, action_space, force_mlp=False, **kwargs):
        '''
        force_mlp: Force Mlp
        '''
        super().__init__(**kwargs)

        self.force_mlp = force_mlp

        # --- Initialize ---
        self.observation_space = None
        self.action_space = None
        self.net = None
        self.policy_net = None
        self.value_net = None

        if observation_space is not None and action_space is not None:
            self.setup_model(observation_space, action_space)

    def setup_model(self, observation_space, action_space):

        self.observation_space = observation_space
        self.action_space      = action_space

        # --- setup model ---
        if (len(self.observation_space.shape) == 3) and (not self.force_mlp):
            # Image observation and mlp is False
            self.net = NatureCnn()
        else:
            self.net = Mlp()
            
        self.policy_net = PolicyNet(self.action_space)
        self.value_net  = ValueNet()

        # construct networks
        inputs  = tf.keras.Input(shape=self.observation_space.shape, dtype=tf.float32)
        outputs = self.net(inputs)
        self.policy_net(outputs)
        self.value_net(outputs)

    #@tf.function
    def _forward(self, inputs, training=True):
        '''
        Forward network

        return logits, values
        '''

        # cast and normalize non float32 inputs (e.g. image with uint8)
        if inputs.dtype != tf.float32:
            # cast observations to float32
            inputs = tf.cast(inputs, dtype=tf.float32)
            low = tf.cast(self.observation_space.low, dtype=tf.float32)
            high = tf.cast(self.observation_space.high, dtype=tf.float32)
            # normalize observations
            inputs = normalize(inputs, low=low, high=high)

        # forward network
        latent = self.net(inputs, training=training)
        # forward policy net
        logits = self.policy_net(latent, training=training)
        # forward value net
        values = self.value_net(latent, training=training) # (batch, 1)
        values = tf.squeeze(values, axis=-1)               # (batch, )

        return logits, values

    def call(self, inputs, deterministic=False, training=True):
        '''
        Predict actions

        deterministic: deterministic action
        return actions, values, log_probs
        '''

        # forward 
        logits, values = self._forward(inputs, training=training)
        distrib = self.policy_net.get_distribution(logits)

        if deterministic:
            actions = distrib.mode()
        else:
            actions = distrib.sample()

        log_probs = distrib.log_prob(actions)

        # actions (batch, act_space.shape)
        # values (batch, )
        # log_probs (batch, )
        return actions, values, log_probs

    def predict(self, inputs, clip_action=True, deterministic=True):
        '''
        Predict actions

        inputs: observations, shape: (obs.shape) or (batch, obs.shape)
        clip_action: clip action range (for Continuous action)
        deterministic: deterministic action
        return clipped actions
        '''
        one_sample  = (len(inputs.shape) == len(self.observation_space.shape))

        if one_sample:
            inputs  = np.expand_dims(inputs, axis=0)

        # predict
        outputs, *_ = self(inputs, deterministic=deterministic, training=False)
        outputs     = outputs.numpy()

        if clip_action and isinstance(self.action_space, gym.spaces.Box):
            outputs = np.clip(outputs, self.action_space.low, self.action_space.high)

        if one_sample:
            outputs = np.squeeze(outputs, axis=0)

        # predict
        return outputs

    def get_config(self):

        config = {'observation_space': self.observation_space,
                  'action_space':      self.action_space,
                  'force_mlp':         self.force_mlp}

        return to_json_serializable(config)


class PPO(SavableModel):
    def __init__(self, env, learning_rate: float = 3e-4, 
                            n_steps:         int = 2048, 
                            batch_size:      int = 64, 
                            n_epochs:        int = 10,
                            gamma:         float = 0.99, 
                            gae_lambda:    float = 0.95,
                            clip_range:    float = 0.2, 
                            clip_range_vf: float = None,
                            ent_coef:      float = 0.01, 
                            vf_coef:       float = 0.5, 
                            max_grad_norm: float = 0.5,
                            target_kl:     float = None,
                            force_mlp:      bool = False,
                            verbose:         int = 0, 
                            **kwargs):
        super().__init__(**kwargs)

        self.env = env
        
        self.learning_rate   = learning_rate
        self.batch_size      = batch_size
        self.n_epochs        = n_epochs
        self.n_steps         = n_steps
        self.gamma           = gamma
        self.gae_lambda      = gae_lambda
        self.clip_range      = clip_range
        self.clip_range_vf   = clip_range_vf
        self.ent_coef        = ent_coef
        self.vf_coef         = vf_coef
        self.max_grad_norm   = max_grad_norm
        self.target_kl       = target_kl
        self.force_mlp       = force_mlp
        self.verbose         = verbose  # 0=no, 1=training log, 2=eval log

        self.num_timesteps     = 0
        self.buffer            = None
        self.tb_writer         = None
        self.observation_space = None
        self.action_space      = None
        self.n_envs            = None

        if env is not None:
            self.set_env(env)
            self.setup_model(env.observation_space, env.action_space)
    
    def setup_model(self, observation_space, action_space):

        self.observation_space = observation_space
        self.action_space      = action_space

        # --- setup model ---
        self.buffer     = GaeBuffer(gae_lambda=self.gae_lambda, gamma=self.gamma)
        self.agent      = Agent(self.observation_space, self.action_space, force_mlp=self.force_mlp)
        
        self.optimizer  = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=self.max_grad_norm)

    def set_env(self, env):

        if self.observation_space is not None:
            assert env.observation_space == self.observation_space, 'Observation space mismatch, expect {}, got {}'.format(
                                                                        self.observation_space, env.observation_space)

        if self.action_space is not None:
            assert env.action_space == self.action_space, 'Action space mismatch, expect {}, got {}'.format(
                                                                self.action_space, env.action_space)
        self.env = env
        self.n_envs = env.n_envs

    @tf.function
    def _forward(self, input, training=True):
        '''
        Forward agent

        return logits, values
        '''
        return self.agent._forward(input, training=training)
    
    def call(self, inputs, deterministic=False, training=True):
        '''
        Forward

        return actions, values, log_probs
        '''
        return self.agent(inputs, deterministic=deterministic, training=training)

    def predict(self, inputs, clip_action=True, deterministic=True):
        '''
        Predict actions

        inputs: observations, shape: (obs.shape) or (batch, obs.shape)
        return clipped actions
        '''
        return self.agent.predict(inputs, clip_action=clip_action, deterministic=deterministic)

    def _run(self, steps, obs=None):
        '''
        Run environments, collect rollouts

        steps: number of steps
        obs: initial observations, if None, reset env
        '''

        if obs is None:
            obs = self.env.reset()

        for _ in range(steps):
            
            actions, values, log_probs = self(obs)

            actions   = actions.numpy()
            values    = values.numpy()
            log_probs = log_probs.numpy()

            # clip action range (for Continuous action)
            if isinstance(self.action_space, gym.spaces.Box):
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

            # step environment
            new_obs, rews, dones, infos = self.env.step(actions)
            
            self.buffer.add(obs, actions, rews, dones, values, log_probs)
            obs = new_obs

        return new_obs

    @tf.function
    def policy_loss(self, advantage, log_prob, old_log_prob, clip_range):
        '''
        Compute policy loss (Clipped surrogate loss)
        
        advantages: (batch, )
        log_prob: (batch, )
        old_log_prob: (batch, )
        '''
        # normalize advantage, stable baselines: ppo2.py#L265
        advantage = (advantage - tf.math.reduce_mean(advantage)) / (tf.math.reduce_std(advantage) + 1e-8)
        # policy ratio
        ratio = tf.exp(log_prob - old_log_prob)
        # clipped surrogate loss
        policy_loss_1 = advantage * ratio
        policy_loss_2 = advantage * tf.clip_by_value(ratio, 1-clip_range, 1+clip_range)
        policy_loss   = -tf.math.reduce_mean(tf.minimum(policy_loss_1, policy_loss_2))

        return policy_loss
    
    @tf.function
    def value_loss(self, values, old_values, returns, clip_range_vf):
        '''
        Compute value loss

        values: (batch, )
        old_values: (batch, )
        returns: (batch, )
        '''
        if clip_range_vf is None:
            values_pred = values
        else:
            # clipping
            values_pred = old_values + tf.clip_by_value(values-old_values, -clip_range_vf, clip_range_vf)
        
        return tf.keras.losses.MSE(returns, values_pred)

    def _train_step(self, obs, actions, old_values, old_log_probs, advantages, returns):
        '''
        Update networks (one step gradient)

        obs: (batch, obs_space.shape)
        actions: (batch, act_space.shape)
        old_values: (batch, )
        old_log_probs: (batch, )
        advantages: (batch, )
        returns: (batch,)
        '''

        actions = tf.cast(actions, dtype=tf.int64)
        
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)

            # forward
            logits, values = self._forward(obs, training=True)
            distrib   = self.agent.policy_net.get_distribution(logits)

            log_probs = distrib.log_prob(actions)
            entropy   = distrib.entropy()
            kl        = 0.5 * tf.math.reduce_mean(tf.math.square(old_log_probs - log_probs))

            # compute policy loss & value loss
            pi_loss   = self.policy_loss(advantages, log_probs, old_log_probs, self.clip_range)
            vf_loss   = self.value_loss(values, old_values, returns, self.clip_range_vf)

            # compute entropy loss
            ent_loss  = -tf.math.reduce_mean(entropy)

            # compute total loss
            loss      = pi_loss + self.ent_coef * ent_loss + self.vf_coef * vf_loss
 
        # perform gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss, kl, entropy, pi_loss, vf_loss, ent_loss

    def train(self, epochs, batch_size):
        '''
        Train agent
        '''
        assert self.buffer.ready_for_sampling, "Buffer is not ready for sampling, please call buffer.make() before sampling"

        for epoch in range(epochs):
            all_loss     = []
            all_kl       = []
            all_entropy  = []
            all_pi_loss  = []
            all_vf_loss  = []
            all_ent_loss = []

            for replay_data in self.buffer(batch_size):
                # update once
                loss, kl, entropy, pi_loss, vf_loss, ent_loss = self._train_step(*replay_data)
                
                all_loss.append(loss.numpy())
                all_kl.append(kl.numpy())
                all_entropy.append(entropy.numpy())
                all_pi_loss.append(pi_loss.numpy())
                all_vf_loss.append(vf_loss.numpy())
                all_ent_loss.append(ent_loss.numpy())
                

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

        # mean entropy / loss / policy loss / value loss
        m_loss     = np.mean(np.hstack(np.array(all_loss)))
        m_kl       = np.mean(np.hstack(np.array(all_kl)))
        m_entropy  = np.mean(np.hstack(np.array(all_entropy)))
        m_pi_loss  = np.mean(np.hstack(np.array(all_pi_loss)))
        m_vf_loss  = np.mean(np.hstack(np.array(all_vf_loss)))
        m_ent_loss = np.mean(np.hstack(np.array(all_ent_loss)))
        
        return m_loss, m_kl, m_entropy, m_pi_loss, m_vf_loss, m_ent_loss, exp_var

    def eval(self, env, n_episodes=5, max_steps=10000):
        assert n_episodes > 0, 'n_episodes must be greater than zero'

        eps_rews = []
        eps_steps = []
        for episode in range(n_episodes):
            obs = env.reset()
            total_rews = 0

            for steps in range(max_steps):
                # predict action
                acts = self.predict(obs)

                # step environment
                obs, rew, done, info = env.step(acts)
                total_rews += rew
                if done:
                    break

            if self.verbose > 1:
                LOG.set_header('Eval {}/{}'.format(episode+1, n_episodes))
                LOG.add_line()
                LOG.add_row('Rewards', total_rews)
                LOG.add_row('Steps', steps+1)
                LOG.add_line()
                LOG.flush('INFO')
        
            eps_rews.append(total_rews)
            eps_steps.append(steps+1)

        return eps_rews, eps_steps

        

    def learn(self, total_timesteps:  int, 
                    log_interval:     int = 1,
                    eval_env              = None, 
                    eval_interval:    int = 1, 
                    eval_episodes:    int = 5, 
                    eval_max_steps:   int = 10000,
                    tb_logdir:        str = None, 
                    reset_timesteps: bool = False):
        
        assert self.env is not None, 'Please set env before training'

        # create tensorboard writer
        if tb_logdir is not None:
            self.tb_writer = tf.summary.create_file_writer(tb_logdir)

        # initialize
        obs        = None
        episode    = 0
        progress   = 0
        time_start = time.time()
        time_spent = 0
        timesteps_per_episode = self.n_steps * self.n_envs
        total_episode = int(float(total_timesteps - self.num_timesteps) / float(timesteps_per_episode) + 0.5)

        if reset_timesteps:
            self.num_timesteps = 0

        while self.num_timesteps < total_timesteps:
            
            # reset buffer
            self.buffer.reset()
            # collect rollouts
            obs = self._run(steps=self.n_steps, obs=obs)

            episode += 1
            self.num_timesteps += timesteps_per_episode
            progress = float(self.num_timesteps) / float(total_timesteps)

            # make buffer
            self.buffer.make()
            # update networks
            loss, kl, ent, pi_loss, vf_loss, ent_loss, exp_var = self.train(self.n_epochs, batch_size=self.batch_size)

            # write tensorboard
            if self.tb_writer is not None:
                with self.tb_writer.as_default():
                    tf.summary.scalar('loss',               loss,     step=self.num_timesteps)
                    tf.summary.scalar('approx_kl',          kl,       step=self.num_timesteps)
                    tf.summary.scalar('entropy',            ent,      step=self.num_timesteps)
                    tf.summary.scalar('policy_loss',        pi_loss,  step=self.num_timesteps)
                    tf.summary.scalar('value_loss',         vf_loss,  step=self.num_timesteps)
                    tf.summary.scalar('entropy_loss',       ent_loss, step=self.num_timesteps)
                    tf.summary.scalar('explained_variance', exp_var,  step=self.num_timesteps)
                
                self.tb_writer.flush()

            # print training log
            if log_interval is not None and episode % log_interval == 0:
                # current time
                time_now = time.time()
                # execution time (one epoch)
                execution_time = (time_now - time_start) - time_spent
                # total time spent
                time_spent   = time_now - time_start
                # remaining time
                remaining_time  = (time_spent / progress)*(1.0-progress)
                # end at
                end_at = (datetime.datetime.now() + datetime.timedelta(seconds=remaining_time)).strftime('%Y-%m-%d %H:%M:%S')
                # average steps per second
                fps = float(self.num_timesteps) / time_spent

                LOG.set_header('Episode {}/{}'.format(episode, total_episode))
                LOG.add_line()
                LOG.add_row('Timesteps',      self.num_timesteps, total_timesteps, fmt='{}: {}/{}')
                LOG.add_row('Steps/sec',      fps,                                 fmt='{}: {:.2f}')
                LOG.add_row('Progress',       progress*100.0,                      fmt='{}: {:.2f}%')

                if self.verbose > 0:
                    LOG.add_row('Execution time', datetime.timedelta(seconds=execution_time))
                    LOG.add_row('Elapsed time',   datetime.timedelta(seconds=time_spent))
                    LOG.add_row('Remaining time', datetime.timedelta(seconds=remaining_time))
                    LOG.add_row('End at',         end_at)
                    LOG.add_line()
                    LOG.add_row('Loss',           loss,     fmt='{}: {:.6f}')
                    LOG.add_row('Approx KL',      kl,       fmt='{}: {:.6f}')
                    LOG.add_row('Entropy',        ent,      fmt='{}: {:.6f}')
                    LOG.add_row('Policy loss',    pi_loss,  fmt='{}: {:.6f}')
                    LOG.add_row('Value loss',     vf_loss,  fmt='{}: {:.6f}')
                    LOG.add_row('Entropy loss',   ent_loss, fmt='{}: {:.6f}')
                    LOG.add_row('Explained var',  exp_var,  fmt='{}: {:.6f}')
                
                LOG.add_line()
                LOG.flush('INFO')

            # evaluate PPO
            if eval_env is not None and episode % eval_interval == 0:
                eps_rews, eps_steps = self.eval(env=eval_env, n_episodes=eval_episodes, max_steps=eval_max_steps)
                
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

        return self

    def get_config(self):

        init_config = {'learning_rate':   self.learning_rate,
                        'batch_size':      self.batch_size,
                        'n_epochs':        self.n_epochs,
                        'n_steps':         self.n_steps,
                        'gamma':           self.gamma,
                        'gae_lambda':      self.gae_lambda,
                        'clip_range':      self.clip_range,
                        'clip_range_vf':   self.clip_range_vf,
                        'ent_coef':        self.ent_coef,
                        'vf_coef':         self.vf_coef,
                        'max_grad_norm':   self.max_grad_norm,
                        'target_kl':       self.target_kl,
                        'force_mlp':       self.force_mlp,
                        'verbose':         self.verbose}

        setup_config = {'observation_space': self.observation_space,
                        'action_space': self.action_space}

        state_config = {'num_timesteps': self.num_timesteps}

        init_config  = to_json_serializable(init_config)
        setup_config = to_json_serializable(setup_config)
        state_config = to_json_serializable(state_config)

        return {'init_config': init_config, 
                'setup_config': setup_config,
                'state_config': state_config}

    @classmethod
    def from_config(cls, config):

        assert 'init_config' in config, 'Failed to load {} config, init_config not found'.format(cls.__name__)
        assert 'setup_config' in config, 'Failed to load {} config, setup_config not found'.format(cls.__name__)
        assert 'state_config' in config, 'Failed to load {} config, state_config not found'.format(cls.__name__)

        init_config = from_json_serializable(config['init_config'])
        setup_config = from_json_serializable(config['setup_config'])
        state_config = from_json_serializable(config['state_config'])

        # construct model
        self = cls(env=None, **init_config)
        self.setup_model(**setup_config)

        for attr, v in state_config.items():
            setattr(self, attr, v)

        return self
