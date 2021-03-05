__copyright__ = '''
 The MIT License (MIT)
 Copyright (c) 2017 OpenAI (http://openai.com)
 Copyright (c) 2018-2019 Stable-Baselines Team
 Copyright (c) 2020-2021 Joe Hsiao
 
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

from collections import OrderedDict

# --- 3rd party ---
import gym
import cloudpickle

import numpy as np
import tensorflow as tf

# --- my module ---
from unstable_baselines import logger

from unstable_baselines.envs import *
from unstable_baselines.utils import set_global_seeds


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
        Add samples, (np.array)
        
        obs: observations, shape: (n_envs, obs size)
        action: actions, shape: (n_envs, 1)
        reward: reward, shape: (n_envs)
        done: done, shape: (n_envs)
        value: value, shape: (n_envs)
        log_prob: log pi, shape: (n_envs)
        '''
        

        assert self.ready_for_sampling == False, 'The buffer is ready for sampling, please reset the buffer'

        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1)

        #LOG.debug('observation shape: {}'.format(np.asarray(obs).shape))
        #LOG.debug('action shape: {}'.format(np.asarray(action).shape))
        #LOG.debug('reward shape: {}'.format(np.asarray(reward).shape))
        #LOG.debug('done shape: {}'.format(np.asarray(done).shape))
        #LOG.debug('value shape: {}'.format(value.flatten().shape))
        #LOG.debug('log probs shape: {}'.format(log_prob.shape))

        self.observations.append(np.asarray(obs).copy())
        self.actions.append(np.asarray(action).copy())
        self.rewards.append(np.asarray(reward).copy())
        self.dones.append(np.asarray(done).copy())
        self.values.append(value.flatten().copy())
        self.log_probs.append(log_prob.copy())

    def __len__(self):
        return len(self.observations)

    def __call__(self, batch_size=None):
        assert self.ready_for_sampling, 'The buffer is not ready for sampling, please call make() first'

        if batch_size is None:
            batch_size = self.observations.shape[0]
        
        indices = np.random.permutation(self.observations.shape[0])

        start_idx = 0
        while start_idx < self.observations.shape[0]:
            # return generator
            yield self._get_samples(indices[start_idx:start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, indices):
        return (self.observations[indices],
                self.actions[indices],
                self.values[indices].flatten(),
                self.log_probs[indices].flatten(),
                self.advantages[indices].flatten(),
                self.returns[indices].flatten())

    def make(self):
        last_value = self.values[-1]
        dones = self.dones[-1]

        self.observations = np.asarray(self.observations, dtype=np.float32)
        self.actions      = np.asarray(self.actions, dtype=np.float32)
        self.rewards      = np.asarray(self.rewards, dtype=np.float32)
        self.dones        = np.asarray(self.dones, dtype=np.float32)
        self.values       = np.asarray(self.values, dtype=np.float32)
        self.log_probs    = np.asarray(self.log_probs, dtype=np.float32)
        self.advantages   = np.zeros(self.observations.shape[0:2], dtype=np.float32) # shape:(# of samples, n_envs)

        # compute GAE
        last_gae_lam = 0
        buffer_size = len(self)
        for step in reversed(range(buffer_size)):
            if step == buffer_size - 1:
                next_non_terminal = np.array(1.0 - dones)
                next_value        = last_value.flatten()
            else:
                next_non_terminal = 1.0 - self.dones[step+1]
                next_value        = self.values[step + 1]
            
            delta        = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam

            self.advantages[step] = last_gae_lam
        
        self.returns = self.advantages + self.values

        # flatten 
        self.observations = self._flatten(self.observations) # (n_envs*steps, observation size)
        self.actions      = self._flatten(self.actions)      
        self.values       = self._flatten(self.values)
        self.log_probs    = self._flatten(self.log_probs)
        self.advantages   = self._flatten(self.advantages)
        self.returns      = self._flatten(self.returns)

        self.ready_for_sampling = True
    
    @classmethod
    def _flatten(cls, v):
        '''
        Flatten dimensions

        (n_envs, steps, ...) -> (n_envs*steps, ...)
        '''
        shape = v.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return v.swapaxes(0, 1).reshape(shape[0]*shape[1], *shape[2:])


# === Networks ===

# CNN feature extractor
class NatureCnn(tf.keras.Model):
    def __init__(self):
        '''
        Nature CNN use the same architecture as the original paper
            "Playing Atari with Deep Reinforcement Learning"
        '''
        super().__init__()

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

    @tf.function
    def call(self, inputs, training=False):
        # expand 3d to 4d
        if tf.rank(inputs) == 3:
            inputs = tf.expand_dims(inputs, axis=0)
        
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x

# Policy network (Discrete action)
class PolicyNet(tf.keras.Model):
    def __init__(self, action_space):
        super().__init__()

        self.action_space = action_space
        self._layers = [
            tf.keras.layers.Dense(action_space.n)
        ]

    def call(self, inputs, training=False):
        # forward hidden layers
        x = inputs
        for layer in self._layers:
            x = layer(inputs)

        return x


# Value network
class ValueNet(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self._layers = [
            tf.keras.layers.Dense(1)
        ]

    @tf.function
    def call(self, inputs, training=False):
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x


# === Probability ===

# Categorical probability (Discrete action)
class Categorical(tf.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = tf.convert_to_tensor(logits)

    @tf.function
    def _p_pi(self):
        # softmax (safe softmax)
        pi_       = self.logits - tf.math.reduce_max(self.logits, axis=-1, keepdims=True)
        exp_pi_   = tf.exp(pi_)
        z_exp_pi_ = tf.math.reduce_sum(exp_pi_, axis=-1, keepdims=True)
        p_pi      = exp_pi_ / z_exp_pi_
        return p_pi
    
    @tf.function
    def _logp_pi(self):
        # softmax (safe softmax)
        pi_       = self.logits - tf.math.reduce_max(self.logits, axis=-1, keepdims=True)
        exp_pi_   = tf.exp(pi_)
        z_exp_pi_ = tf.math.reduce_sum(exp_pi_, axis=-1, keepdims=True)
        logp_pi   = pi_ - tf.math.log(z_exp_pi_)
        return logp_pi

    @tf.function
    def sample(self):
        # gumbel reparameterization
        e       = tf.random.uniform(tf.shape(self.logits)) # noise
        it      = self.logits - tf.math.log(-tf.math.log(e))
        samples = tf.math.argmax(it, axis=-1)
        return samples

    @tf.function
    def log_prob(self, x):
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=x, logits=self.logits)
    
    @tf.function
    def entropy(self):
        # softmax (safe softmax)
        pi_       = self.logits - tf.math.reduce_max(self.logits, axis=-1, keepdims=True)
        exp_pi_   = tf.exp(pi_)
        z_exp_pi_ = tf.math.reduce_sum(exp_pi_, axis=-1, keepdims=True)
        p_pi      = exp_pi_ / z_exp_pi_
        # entropy
        logp_pi   = pi_ - tf.math.log(z_exp_pi_)
        ent_pi    = tf.math.reduce_sum(-p_pi * logp_pi, axis=-1)
        return ent_pi

    @tf.function
    def kl(self, target_logits):
        # softmax (safe softmax)
        logits_       = target_logits - tf.math.reduce_max(target_logits, axis=-1, keepdims=True)
        exp_logits_   = tf.exp(logits_)
        z_exp_logits_ = tf.math.reduce_sum(exp_logits_, axis=-1, keepdims=True)
        # kl divergence
        log_qp        = self.logits_ - tf.math.log(z_exp_logits_) - self.pi_ + tf.math.log(self.z_exp_pi_)
        kl            = tf.math.reduce_sum(-self._p_pi() * log_qp, axis=-1)
        return kl


class PPO(tf.keras.Model):
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
                            verbose:         int = 0, 
                            **kwargs):
        super().__init__()
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
        self.verbose         = verbose  # 0=no, 1=training log, 2=eval log

        self.num_timesteps = 0
        self.buffer        = None
        self.tb_writer     = None

        if self.env is not None:
            self.setup_model(self.env)
    
    def setup_model(self, env):

        assert env is not None, 'Env is None'

        self.env               = env
        self.observation_space = env.observation_space
        self.action_space      = env.action_space
        self.n_envs            = env.n_envs

        # --- setup model ---
        self.buffer     = GaeBuffer(gae_lambda=self.gae_lambda, gamma=self.gamma)
        self.net        = NatureCnn()
        self.policy_net = PolicyNet(self.action_space)
        self.value_net  = ValueNet()
        self.optimizer  = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, epsilon=1e-5, clipnorm=self.max_grad_norm)

        # construct networks
        inputs  = tf.keras.Input(shape=self.observation_space.shape, dtype=tf.float32)
        outputs = self.net(inputs)
        self.policy_net(outputs)
        self.value_net(outputs)

    @tf.function
    def _forward(self, inputs, training=False):
        # preprocess inputs
        # cast image inputs (uint8) to float32
        inputs = tf.cast(inputs, dtype=tf.float32)
        # normalize
        inputs = (inputs - self.observation_space.low) / (self.observation_space.high - self.observation_space.low)

        # forward network
        latent = self.net(inputs, training=training)
        # forward policy net
        logits = self.policy_net(latent, training=training)
        # forward value net
        values = self.value_net(latent, training=training)

        return logits, values

    def call(self, inputs, deterministic=True, verbose=False):
        '''
        Predict actions

        verbose: return additional info 
        '''
        # forward
        logits, values = self._forward(inputs)
        
        if deterministic:
            return tf.math.argmax(logits, axis=-1)
        
        distrib = Categorical(logits)
        actions = distrib.sample()
        log_probs = distrib.log_prob(actions)

        if verbose:
            return actions, values, log_probs
        else:
            return actions

    def _run(self, steps, obs=None):
        '''
        Run environments, collect rollouts

        steps: number of steps
        obs: initial observations, if None, reset env
        '''

        if obs is None:
            obs = self.env.reset()

        for _ in range(steps):
            
            actions, values, log_probs = self(obs, deterministic=False, verbose=True)

            actions   = actions.numpy()
            values    = values.numpy()
            log_probs = log_probs.numpy()

            # step environment
            new_obs, rews, dones, infos = self.env.step(actions)
            
            # if action space is Discrete, reshape
            actions = actions.reshape(-1, 1)
            self.buffer.add(obs, actions, rews, dones, values, log_probs)
            obs = new_obs

        return new_obs

    @tf.function
    def policy_loss(self, advantage, log_prob, old_log_prob, clip_range):
        '''
        Compute policy loss (Clipped surrogate loss)
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
        '''
        if clip_range_vf is None:
            values_pred = values
        else:
            # clipping
            values_pred = old_values + tf.clip_by_value(values-old_values, -clip_range_vf, clip_range_vf)
        
        return tf.keras.losses.MSE(returns, values_pred)

    def _train_step(self, obs, actions, old_values, old_log_probs, advantages, returns):
        '''
        Update PPO (one step gradient)
        '''

        actions = tf.cast(actions, dtype=tf.int64)
        actions = tf.reshape(actions, shape=[-1])
        
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)

            # forward
            logits, values = self._forward(obs, training=True)
            distrib = Categorical(logits)

            log_probs = distrib.log_prob(actions)
            entropy   = distrib.entropy()
            kl        = 0.5 * tf.math.reduce_mean(tf.math.square(old_log_probs - log_probs))

            values    = tf.reshape(values, shape=[-1])

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

    def train(self, gradient_steps, batch_size=64):
        assert self.buffer.ready_for_sampling, "Buffer is not ready for sampling, please call buffer.make() before sampling"

        for gradient_step in range(gradient_steps):
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
                LOG.warning('Early stopping at step {} due to reaching max kl: {:.2f}'.format(gradient_step, m_kl))
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
                acts = acts.item()
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
            eps_steps.append(steps)

        return eps_rews, eps_steps

        

    def learn(self, total_timesteps:  int, 
                    log_interval:    bool = 1,
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
        total_episode = int(float(total_timesteps) / float(timesteps_per_episode) + 0.5)

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
                    LOG.add_row('Value loss',     vf_loss,   fmt='{}: {:.6f}')
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

    def export(self, path):
        '''
        Export model

        Call `tf.keras.models.load_model(path)` to load model. (predict only)
        '''
        super().save(path)



def parse_args():

    parser = argparse.ArgumentParser(description='Proximal Policy Optimization')
    parser.add_argument('--logdir',           type=str, default='log/{env_id}/ppo/{rank}',help='Root dir             (args: {env_id}, {rank})')
    parser.add_argument('--logging',          type=str, default='train.log',              help='Log path             (args: {env_id}, {rank})')
    parser.add_argument('--monitor_dir',      type=str, default='monitor',                help='Monitor dir          (args: {env_id}, {rank})')
    parser.add_argument('--tb_logdir',        type=str, default='',                       help='Tensorboard log name (args: {env_id}, {rank})')
    parser.add_argument('--model_dir',        type=str, default='model',                  help='Model export path    (args: {env_id}, {rank})')
    parser.add_argument('--env_id',           type=str, default='BeamRiderNoFrameskip-v0',help='Environment ID')
    parser.add_argument('--num_envs',         type=int, default=4,      help='Number of environments')
    parser.add_argument('--num_episodes',     type=int, default=10000,  help='Number of training episodes (not environment episodes)')
    parser.add_argument('--num_steps',        type=int, default=256,    help='Number of timesteps per episode (interact with envs)')
    parser.add_argument('--num_epochs',       type=int, default=10,     help='Number of epochs per episode (perform gradient update)')
    parser.add_argument('--batch_size',       type=int, default=64,     help='Training batch size')
    parser.add_argument('--verbose',          type=int, default=1,      help='Print more message, 0=less, 1=more train log, 2=more eval log')
    parser.add_argument('--rank',             type=int, default=0,      help='Optional arguments for parallel training')
    parser.add_argument('--seed',             type=int, default=0,      help='Random seed')
    parser.add_argument('--log_interval',     type=int, default=1,      help='Logging interval (episodes)')
    parser.add_argument('--eval_interval',    type=int, default=1000,   help='Evaluation interval (episodes)')
    parser.add_argument('--eval_episodes',    type=int, default=5,      help='Number of episodes each evaluation')
    parser.add_argument('--eval_max_steps',   type=int, default=3000,   help='Maximum timesteps in each evaluation episode')
    parser.add_argument('--eval_seed',        type=int, default=0,      help='Environment seed for evaluation')
    parser.add_argument('--lr',               type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma',            type=float, default=0.99, help='Gamma decay rate')
    parser.add_argument('--gae_lambda',       type=float, default=0.95, help='GAE lambda decay rate')
    parser.add_argument('--clip_range',       type=float, default=0.2,  help='PPO policy clip range (epsilon)')
    parser.add_argument('--clip_range_vf',    type=float, default=None, help='Value clip range')
    parser.add_argument('--ent_coef',         type=float, default=0.01, help='Entropy loss ratio')
    parser.add_argument('--vf_coef',          type=float, default=0.5,  help='Value loss ratio')
    parser.add_argument('--max_grad_norm',    type=float, default=0.5,  help='Max gradient norm')
    parser.add_argument('--target_kl',        type=float, default=None, help='Target kl (early stop)')

    a = parser.parse_args()

    a.logdir      = a.logdir.format(env_id=a.env_id, rank=a.rank)
    a.logging     = os.path.join(a.logdir, a.logging).format(env_id=a.env_id, rank=a.rank)
    a.monitor_dir = os.path.join(a.logdir, a.monitor_dir).format(env_id=a.env_id, rank=a.rank)
    a.tb_logdir   = os.path.join(a.logdir, a.tb_logdir).format(env_id=a.env_id, rank=a.rank)
    a.model_dir   = os.path.join(a.logdir, a.model_dir).format(env_id=a.env_id, rank=a.rank)

    return a


if __name__ == '__main__':

    a = parse_args()

    # === Reset logger ===
    logger.Config.use(filename=a.logging, level='DEBUG', colored=True, reset=True)
    LOG = logger.getLogger()

    # === Print welcome message ===
    LOG.add_row('')
    LOG.add_rows('PPO', fmt='{:@f:ANSI_Shadow}', align='center')
    LOG.add_line()
    LOG.add_rows('{}'.format(__copyright__))
    LOG.flush('INFO')
    time.sleep(1)

    # === Print arguments ===
    LOG.set_header('Arguments')
    LOG.add_row('Log dir',          a.logdir)
    LOG.add_row('Logging path',     a.logging)
    LOG.add_row('Monitor path',     a.monitor_dir)
    LOG.add_row('Tensorboard path', a.tb_logdir)
    LOG.add_row('Model path',       a.model_dir)
    LOG.add_row('Env ID',           a.env_id)
    LOG.add_row('Seed',             a.seed)
    LOG.add_row('Eval seed',        a.eval_seed)
    LOG.add_line()
    LOG.add_row('Num of envs',           a.num_envs)
    LOG.add_row('Num of steps/episode ', a.num_steps)
    LOG.add_row('Num of epochs/episode', a.num_epochs)
    LOG.add_row('Num of episodes',       a.num_episodes)
    LOG.add_row('Log interval',          a.log_interval)
    LOG.add_row('Eval interval',         a.eval_interval)
    LOG.add_row('Eval episodes',         a.eval_episodes)
    LOG.add_row('Eval max steps',        a.eval_max_steps)
    LOG.add_row('Batch size',            a.batch_size)
    LOG.add_row('Verbose',               a.verbose)
    LOG.add_line()
    LOG.add_row('Learning rate',     a.lr)
    LOG.add_row('Gamma',             a.gamma)
    LOG.add_row('Lambda',            a.gae_lambda)
    LOG.add_row('Clip range',        a.clip_range)
    LOG.add_row('Value clip range',  a.clip_range_vf)
    LOG.add_row('Entropy coef',      a.ent_coef)
    LOG.add_row('Value coef',        a.vf_coef)
    LOG.add_row('Max gradient norm', a.max_grad_norm)
    LOG.add_row('Target KL',         a.target_kl)
    LOG.flush('WARNING')

    # === Make envs ===
    # make atari env
    assert 'NoFrameskip' in a.env_id
    def make_env(env_id, rank, log_path, seed=0):
        def _init():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = Monitor(env, directory=log_path, prefix=str(rank),
                        enable_video_recording=True, force=True,
                        video_kwargs={'prefix':'video/train.{}'.format(rank)})
            env = EpisodicLifeEnv(env)
            env = WarpFrame(env)
            env = ClipRewardEnv(env)
            return env
        set_global_seeds(seed)
        return _init

    # make env
    env = SubprocVecEnv([make_env(a.env_id, i, a.monitor_dir, seed=a.seed) for i in range(a.num_envs)])
    env = VecFrameStack(env, 4)

    eval_env = gym.make(a.env_id)
    eval_env.seed(a.eval_seed)
    eval_env = NoopResetEnv(eval_env, noop_max=30)
    eval_env = MaxAndSkipEnv(eval_env, skip=4)
    eval_env = Monitor(eval_env, directory=a.monitor_dir, prefix='eval',
                       enable_video_recording=True, force=True,
                       video_kwargs={'prefix':'video/eval',
                                     'callback': lambda x: True})
    eval_env = WarpFrame(eval_env)
    eval_env = FrameStack(eval_env, 4)
    
    LOG.debug('Action space: {}'.format(env.action_space))
    LOG.debug('Observation space: {}'.format(env.observation_space))


    # === Create model ===
    try:
        model = PPO(env, learning_rate   = a.lr,
                         n_steps         = a.num_steps,
                         batch_size      = a.batch_size,
                         n_epochs        = a.num_epochs, 
                         gamma           = a.gamma,
                         gae_lambda      = a.gae_lambda,
                         clip_range      = a.clip_range,
                         clip_range_vf   = a.clip_range_vf,
                         ent_coef        = a.ent_coef,
                         vf_coef         = a.vf_coef,
                         max_grad_norm   = a.max_grad_norm,
                         target_kl       = a.target_kl,
                         verbose         = a.verbose)
        
        # Total timesteps = num_steps * num_envs * num_episodes (default ~ 10M)
        model.learn(a.num_steps *    a.num_envs * a.num_episodes, 
                    tb_logdir      = a.tb_logdir, 
                    log_interval   = a.log_interval,
                    eval_env       = eval_env, 
                    eval_interval  = a.eval_interval, 
                    eval_episodes  = a.eval_episodes, 
                    eval_max_steps = a.eval_max_steps)
        
        LOG.info('DONE')

        # Export model
        model.export(a.model_dir)

        # Load model
        loaded_model = tf.keras.models.load_model(a.model_dir)

        # Evaluation
        eps_rews  = []
        eps_steps = []
        for episode in range(a.eval_episodes):
            obs = eval_env.reset()
            total_rews = 0

            for steps in range(10000):
                # predict action
                acts = loaded_model.predict(obs)
                acts = acts.item()
                # step environment
                obs, rew, done, info = eval_env.step(acts)
                total_rews += rew
                if done:
                    break

            # === Print episode result ===
            LOG.set_header('Eval {}/{}'.format(episode+1, a.eval_episodes))
            LOG.add_line()
            LOG.add_row('Rewards', total_rews)
            LOG.add_row('Steps', steps+1)
            LOG.add_line()
            LOG.flush('INFO')
        
            eps_rews.append(total_rews)
            eps_steps.append(steps)

        max_idx    = np.argmax(eps_rews)
        max_rews   = eps_rews[max_idx]
        max_steps  = eps_steps[max_idx]
        mean_rews  = np.mean(eps_rews)
        std_rews   = np.std(eps_rews)
        mean_steps = np.mean(eps_steps)

        # === Print evaluation results ===
        LOG.set_header('Evaluate')
        LOG.add_line()
        LOG.add_row('Max rewards',  max_rews)
        LOG.add_row('  Length',     max_steps)
        LOG.add_line()
        LOG.add_row('Mean rewards', mean_rews)
        LOG.add_row('Std rewards',  std_rews, fmt='{}: {:.6f}')
        LOG.add_row('Mean length',  mean_steps)
        LOG.add_line()
        LOG.flush('INFO')

    except:
        LOG.exception('Exception occurred')
        exit(1)
