__copyright__ = '''
 The MIT License (MIT)
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
from unstable_baselines.base import SavableModel
from unstable_baselines.utils import (set_global_seeds,
                                      normalize_action,
                                      unnormalize_action,
                                      to_json_serializable,
                                      from_json_serializable,
                                      tf_soft_update_params)


logger.Config.use(level='DEBUG', colored=True, reset=False)
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
        '''
        Add new samples

        observations: (np.ndarray) shape: (n_envs, obs_space.shape)
        next_observations: (np.ndarray) shape: (n_envs, obs_space.shape)
        actions: (np.ndarray) shape: (n_envs, action_space.shape)
        rewards: (np.ndarray) shape: (n_envs,)
        dones: (np.ndarray) shape: (n_envs,)
        '''

        obss      = np.asarray(observations)
        next_obss = np.asarray(next_observations)
        actions   = np.asarray(actions)
        rewards   = np.asarray(rewards)
        dones     = np.asarray(dones)

        n_env = obss.shape[0]

        if self.obss is None:
            # create spaces
            self.obss      = np.zeros((self.buffer_size, ) + obss.shape[1:],    dtype=np.float32)
            self.acts      = np.zeros((self.buffer_size, ) + actions.shape[1:], dtype=np.float32)
            self.next_obss = np.zeros((self.buffer_size, ) + obss.shape[1:],    dtype=np.float32)
            self.rews      = np.zeros((self.buffer_size, ) + rewards.shape[1:], dtype=np.float32)
            self.dones     = np.zeros((self.buffer_size, ) + dones.shape[1:],   dtype=np.float32)

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
        '''
        Sample batch
        '''
        if batch_size is None:
            batch_size = len(self)

        batch_inds = np.random.randint(0, len(self), size=batch_size)

        return self._get_samples(batch_inds)


    def _get_samples(self, batch_inds):
        return (self.obss[batch_inds],
                self.acts[batch_inds],
                self.next_obss[batch_inds],
                self.dones[batch_inds].reshape(-1 ,1),  # (batch, 1)
                self.rews[batch_inds].reshape(-1, 1))   # (batch, 1)



# === Networks ===

class Actor(tf.keras.Model):
    '''
    Actor network (continuous state, action)
    
    from original paper: https://arxiv.org/abs/1802.09477 (Appendix C)
    '''
    def __init__(self, action_space, **kwargs):
        super().__init__(**kwargs)

        assert isinstance(action_space, gym.spaces.Box)
        self.action_space = action_space

        self._layers = [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(400, name='fc1'),
            tf.keras.layers.ReLU(name='relu1'),
            tf.keras.layers.Dense(300, name='fc2'),
            tf.keras.layers.ReLU(name='relu2'),
            tf.keras.layers.Dense(action_space.shape[0], name='fc3'),
            tf.keras.layers.Activation(activation='tanh', name='tanh')
        ]

    def call(self, inputs, training=False):
        x = inputs
        for layer in self._layers:
            x = layer(x)

        return x

    def update(self, other_actor, polyak=1.):
        '''
        Perform soft update (polyak update)
        '''

        tf_soft_update_params(self.trainable_variables, 
                              other_actor.trainable_variables, 
                              polyak=polyak)


class Critic(tf.keras.Model):
    '''
    Critic network (continuous state, action)

    from original paper: https://arxiv.org/abs/1802.09477 (Appendix C)
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._concat = tf.keras.layers.Concatenate()

        self._layers = [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(400, name='fc1'),
            tf.keras.layers.ReLU(name='relu1'),
            tf.keras.layers.Dense(300, name='fc2'),
            tf.keras.layers.ReLU(name='relu2'),
            tf.keras.layers.Dense(1, name='fc3')
        ]

    def call(self, inputs, training=False):
        # inputs = [observations, actions]

        x = self._concat(inputs)
        for layer in self._layers:
            x = layer(x)

        return x

    def update(self, other_critic, polyak=1.):
        '''
        Perform soft update (polyak update)
        '''

        tf_soft_update_params(self.trainable_variables, 
                              other_critic.trainable_variables, 
                              polyak=polyak)


# === Agent, ALGO ===

class TD3Agent(SavableModel):
    def __init__(self, observation_space, action_space, **kwargs):
        super().__init__(**kwargs)


        # --- Initialize ---
        self.observation_space = None
        self.action_space = None
        self.actor = None
        self.critic_1 = None
        self.critic_2 = None

        if observation_space is not None and action_space is not None:
            self.setup_model(observation_space, action_space)

    def setup_model(self, observation_space, action_space):

        # check observation/action space
        assert isinstance(observation_space, gym.spaces.Box), 'The observation space must be gym.spaces.Box, got {}'.format(type(obs_space))
        assert isinstance(action_space, gym.spaces.Box), 'The action space must be gym.spaces.Box, got {}'.format(type(act_space))

        self.observation_space = observation_space
        self.action_space      = action_space

        # --- setup model ---
        self.actor = Actor(self.action_space)
        self.critic_1 = Critic()
        self.critic_2 = Critic()

        # construct networks
        obs_inputs = tf.keras.Input(shape=self.observation_space.shape, dtype=tf.float32)
        act_inputs = tf.keras.Input(shape=self.action_space.shape, dtype=tf.float32)
        self.actor(obs_inputs)
        self.critic_1([obs_inputs, act_inputs])
        self.critic_2([obs_inputs, act_inputs])

    def _forward(self, inputs):
        return self.actor(inputs)

    def call(self, inputs, normalized=True):
        '''
        Forward actor

        inputs: observations shape: (batch, obs.shape)
        normalized: return normalized actions (batch, act.shape)
        '''

        action = self._forward(inputs)

        if not normalized:
            action = unnormalize_action(action, high=self.action_space.high, low=self.action_space.low)
        
        return action

    def predict(self, inputs, normalized=False):
        '''
        Predict actions

        inputs: observations shape: (obs.shape) or (batch, obs.shape)
        normalized: return normalized actions (act.shape) or (batch, act.shape)
        '''
        one_sample = (len(inputs.shape) == len(self.observation_space.shape))

        if one_sample:
            inputs = np.expand_dims(inputs, axis=0)

        # predict
        outputs = self(inputs, normalized=normalized).numpy()
        if one_sample:
            outputs = outputs.reshape((-1,))

        # predict
        return outputs

    def get_config(self):
        
        config = {'observation_space': self.observation_space, 
                  'action_space': self.action_space}

        return to_json_serializable(config)





class TD3(SavableModel):
    def __init__(self, env, learning_rate:       float = 1e-3,
                            buffer_size:           int = int(1e6),
                            min_buffer:            int = 100,
                            n_steps:               int = 100,
                            gradient_steps:        int = 200,
                            batch_size:            int = 128,
                            policy_delay:          int = 2,
                            gamma:               float = 0.99,
                            tau:                 float = 0.005, # polyak
                            target_policy_noise: float = 0.2,
                            target_noise_clip:   float = 100,
                            action_noise               = None,
                            verbose:               int = 0,
                            **kwargs):
        super().__init__(**kwargs)

        self.env = env

        self.learning_rate       = learning_rate
        self.buffer_size         = buffer_size
        self.min_buffer          = min_buffer
        self.n_steps             = n_steps
        self.gradient_steps      = gradient_steps
        self.batch_size          = batch_size
        self.policy_delay        = policy_delay
        self.gamma               = gamma
        self.tau                 = tau
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip   = target_noise_clip
        self.action_noise        = action_noise
        self.verbose             = verbose

        self.num_timesteps = 0
        self.buffer = None
        self.tb_writer = None
        self.observation_space = None
        self.action_space = None
        self.n_envs = 0

        if env is not None:
            self.set_env(env)
            self.setup_model(env.observation_space, env.action_space)

    def setup_model(self, observation_space, action_space):

        assert isinstance(observation_space, gym.spaces.Box), 'The observation space must be gym.spaces.Box, got {}'.format(type(obs_space))
        assert isinstance(action_space, gym.spaces.Box), 'The action space must be gym.spaces.Box, got {}'.format(type(act_space))

        self.observation_space = observation_space
        self.action_space = action_space

        # --- setup model ---
        self.buffer = ReplayBuffer(buffer_size=self.buffer_size)
        self.agent = TD3Agent(self.observation_space, self.action_space)
        self.agent_target = TD3Agent(self.observation_space, self.action_space)

        self.actor_optimizer  = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # initialize target
        self.agent_target.actor.update(self.agent.actor)
        self.agent_target.critic_1.update(self.agent.critic_1)
        self.agent_target.critic_2.update(self.agent.critic_2)

    def set_env(self, env):

        if self.observation_space is not None:
            assert env.observation_space == self.observation_space, 'Observation space not match, expect {}, got {}'.format(
                                                                        self.observation_space, env.observation_space)

        if self.action_space is not None:
            assert env.action_space == self.action_space, 'Action space not match, expect {}, got {}'.format(
                                                                self.action_space, env.action_space)
        
        self.env = env
        self.n_envs = env.n_envs

    @tf.function
    def _forward(self, inputs):
        '''
        Forward actor
        '''
        return self.agent._forward(inputs)

    def call(self, inputs, normalized=True):
        return self.agent(inputs, normalized=normalized)

    def predict(self, inputs, normalized=False):
        return self.agent.predict(inputs, normalized=normalized)

    def _run(self, steps, obs=None):
        '''
        Run environments, collect rollouts

        steps: number of timesteps
        obs: initial observations, if None, reset env
        '''

        if obs is None:
            obs = self.env.reset()

        if self.action_noise is not None:
            self.action_noise.reset()

        for _ in range(steps):

            if len(self.buffer) < self.min_buffer:
                # random sample (collecting rollouts)
                action = np.array([self.action_space.sample() for n in range(self.n_envs)])
                action = normalize_action(action, high=self.action_space.high, low=self.action_space.low)
            else:
                # sample from policy
                action = self(obs)
            
            # add action noise
            if self.action_noise is not None:
                action = np.clip(action + self.action_noise(), -1, 1)

            # step environment
            raw_action = unnormalize_action(action, high=self.action_space.high, low=self.action_space.low)
            new_obs, reward, done, infos = self.env.step(raw_action)
            
            # add to buffer
            self.buffer.add(obs, new_obs, action, reward, done)
            obs = new_obs

            if self.n_envs == 1 and done[0]:
                if self.action_noise is not None:
                    self.action_noise.reset()

        return new_obs

    @tf.function
    def actor_loss(self, obs):
        '''
        Compute actor loss

        L = -Q(s, mu(s))
        '''

        mu = self.agent.actor(obs)
        return -tf.reduce_mean(self.agent.critic_1([obs, mu]))

    @tf.function
    def critic_loss(self, obs, action, next_obs, done, reward):
        '''
        Compute critic loss

        clipped target
        y = r + gamma * min( Q1*(s', mu*(s') + noise), 
                             Q2*(s', mu*(s') + noise) )

        L = MSE(y, Q1(s, mu(s))) + MSE(y, Q2(s, mu(s)))
        '''

        noise    = tf.random.normal(shape=action.shape) * self.target_policy_noise
        noise    = tf.clip_by_value(noise, -self.target_noise_clip, self.target_noise_clip)
        next_act = tf.clip_by_value(self.agent_target.actor(next_obs) + noise, -1., 1.)

        # compute the target Q value
        Q1       = self.agent_target.critic_1([next_obs, next_act])
        Q2       = self.agent_target.critic_2([next_obs, next_act])
        target_Q = tf.minimum(Q1, Q2)

        y = reward + tf.stop_gradient( (1.-done) * self.gamma * target_Q) # (batch, 1)

        # compute Q estimate
        q1 = self.agent.critic_1([obs, action]) # (batch, 1)
        q2 = self.agent.critic_2([obs, action]) # (batch, 1)
    
        # compute critic loss
        return tf.keras.losses.MSE(q1, y) + tf.keras.losses.MSE(q2, y)
    
    @tf.function
    def update_targets(self):
        self.agent_target.actor.update(self.agent.actor,       polyak=self.tau)
        self.agent_target.critic_1.update(self.agent.critic_1, polyak=self.tau)
        self.agent_target.critic_2.update(self.agent.critic_2, polyak=self.tau)


    def _train_actor(self, obs):
        '''
        Update actor
        '''
        with tf.GradientTape() as tape:
            tape.watch(self.agent.actor.trainable_variables)

            # compute actor loss
            loss = self.actor_loss(obs)

        # perform gradient
        grads = tape.gradient(loss, self.agent.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.agent.actor.trainable_variables))

        return loss

    def _train_critic(self, obs, action, next_obs, done, reward):
        '''
        Update critics
        '''
        # combine two variable list
        variables = self.agent.critic_1.trainable_variables + self.agent.critic_2.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(variables)

            loss = self.critic_loss(obs, action, next_obs, done, reward)

        # perform gradients
        grads = tape.gradient(loss, variables)
        self.critic_optimizer.apply_gradients(zip(grads, variables))

        return loss

    def train(self, steps, batch_size, policy_delay):
        '''
        Train one epoch
        '''

        m_actor_loss = None
        m_critic_loss = None

        all_actor_loss = []
        all_critic_loss = []

        for gradient_steps in range(steps):
            (obs, action, next_obs, done, reward) = self.buffer(batch_size)

            critic_loss = self._train_critic(obs, action, next_obs, done, reward)

            # delayed policy update
            if gradient_steps % policy_delay == 0:

                # update critic
                actor_loss = self._train_actor(obs)

                # update target models
                self.update_targets()

                all_actor_loss.append(actor_loss)
            all_critic_loss.append(critic_loss)

        if len(all_actor_loss) > 0:
            m_actor_loss = np.mean(np.hstack(np.asarray(all_actor_loss)))
        m_critic_loss = np.mean(np.hstack(np.asarray(all_critic_loss)))
        
        return m_actor_loss, m_critic_loss

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
                LOG.add_row(' Length', steps+1)
                LOG.add_line()
                LOG.flush('INFO')
        
            eps_rews.append(total_rews)
            eps_steps.append(steps)

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
        total_episode = int(float(total_timesteps-self.num_timesteps) / float(timesteps_per_episode) + 0.5)

        if reset_timesteps:
            self.num_timesteps = 0
            # reset buffer
            self.buffer.reset()

        while self.num_timesteps < total_timesteps:
            # collect rollouts
            obs = self._run(steps=self.n_steps, obs=obs)

            episode += 1
            self.num_timesteps += timesteps_per_episode
            progress = float(self.num_timesteps) / float(total_timesteps)

            if len(self.buffer) > self.min_buffer:

                # training
                actor_loss, critic_loss = self.train(self.gradient_steps, batch_size=self.batch_size, policy_delay=self.policy_delay)

                # write tensorboard
                if self.tb_writer is not None:
                    with self.tb_writer.as_default():
                        if actor_loss is not None:
                            tf.summary.scalar('actor_loss',  actor_loss,  step=self.num_timesteps)
                        tf.summary.scalar('critic_loss', critic_loss, step=self.num_timesteps)

                    self.tb_writer.flush()

            # print training log
            if log_interval is not None and episode % log_interval == 0:
                # current time
                time_now       = time.time()
                # execution time (one epoch)
                execution_time = (time_now - time_start) - time_spent
                # total time spent
                time_spent     = (time_now - time_start)
                # remaining time
                remaining_time = (time_spent / progress)*(1.0-progress)
                # end at
                end_at         = (datetime.datetime.now() + datetime.timedelta(seconds=remaining_time)).strftime('%Y-%m-%d %H:%M:%S')
                # average steps per second
                fps            = float(self.num_timesteps) / time_spent

                LOG.set_header('Episode {}/{}'.format(episode, total_episode))
                LOG.add_line()
                LOG.add_row('Timesteps', self.num_timesteps, total_timesteps, fmt='{}: {}/{}')
                LOG.add_row('Steps/sec', fps,                                 fmt='{}: {:.2f}')
                LOG.add_row('Progress',  progress*100.0,                      fmt='{}: {:.2f}%')

                if self.verbose > 0:
                    LOG.add_row('Execution time', datetime.timedelta(seconds=execution_time))
                    LOG.add_row('Elapsed time',   datetime.timedelta(seconds=time_spent))
                    LOG.add_row('Remaining time', datetime.timedelta(seconds=remaining_time))
                    LOG.add_row('End at',         end_at)
                    LOG.add_line()

                    if len(self.buffer) > self.min_buffer:
                        if actor_loss is not None:
                            LOG.add_row('Actor loss',     actor_loss,     fmt='{}: {:.6f}')
                        LOG.add_row('Critic loss',    critic_loss,    fmt='{}: {:.6f}')
                    else:
                        LOG.add_row('Collecting rollouts {}/{}'.format(len(self.buffer), self.min_buffer))

                LOG.add_line()
                LOG.flush('INFO')

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
                LOG.add_row('     Length',  max_steps)
                LOG.add_line()
                LOG.add_row('Mean rewards', mean_rews)
                LOG.add_row(' Std rewards', std_rews, fmt='{}: {:.6f}')
                LOG.add_row(' Mean length', mean_steps)
                LOG.add_line()
                LOG.flush('INFO')

        return self

    def get_config(self):
        
        init_config = { 'learning_rate':       self.learning_rate,
                        'buffer_size':         self.buffer_size,
                        'min_buffer':          self.min_buffer,
                        'n_steps':             self.n_steps,
                        'gradient_steps':      self.gradient_steps,
                        'batch_size':          self.batch_size,
                        'policy_delay':        self.policy_delay,
                        'gamma':               self.gamma,
                        'tau':                 self.tau,
                        'target_policy_noise': self.target_policy_noise,
                        'target_noise_clip':   self.target_noise_clip,
                        'action_noise':        self.action_noise,
                        'verbose':             self.verbose}

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

def parse_args():

    parser = argparse.ArgumentParser(description='Twined Delayed Deep Deterministic Policy Gradient (TD3)')
    parser.add_argument('--logdir',           type=str, default='log/{env_id}/ppo/{rank}',help='Root dir             (args: {env_id}, {rank})')
    parser.add_argument('--logging',          type=str, default='train.log',              help='Log path             (args: {env_id}, {rank})')
    parser.add_argument('--monitor_dir',      type=str, default='monitor',                help='Monitor dir          (args: {env_id}, {rank})')
    parser.add_argument('--tb_logdir',        type=str, default='',                       help='Tensorboard log name (args: {env_id}, {rank})')
    parser.add_argument('--model_dir',        type=str, default='model',                  help='Model export path    (args: {env_id}, {rank})')
    parser.add_argument('--env_id',           type=str, default='BeamRiderNoFrameskip-v0',help='Environment ID')
    parser.add_argument('--buffer_size',      type=int, default=1000000, help='Maximum size of replay buffer')
    parser.add_argument('--min_buffer',       type=int, default=10000,   help='Minimum number of samples in replay buffer')
    parser.add_argument('--num_envs',         type=int, default=1,       help='Number of environments')
    parser.add_argument('--num_episodes',     type=int, default=1000,    help='Number of training episodes (not environment episodes)')
    parser.add_argument('--num_steps',        type=int, default=1000,    help='Number of timesteps per episode (interact with envs)')
    parser.add_argument('--gradient_steps',   type=int, default=1000,    help='Number of gradient steps')
    parser.add_argument('--batch_size',       type=int, default=100,     help='Training batch size')
    parser.add_argument('--policy_delay',     type=int, default=2,       help='Delayed gradient steps to update policy')
    parser.add_argument('--verbose',          type=int, default=1,       help='Print more message, 0=less, 1=more train log, 2=more eval log')
    parser.add_argument('--rank',             type=int, default=0,       help='Optional arguments for parallel training')
    parser.add_argument('--seed',             type=int, default=0,       help='Random seed')
    parser.add_argument('--log_interval',     type=int, default=1,       help='Logging interval (episodes)')
    parser.add_argument('--eval_interval',    type=int, default=100,     help='Evaluation interval (episodes)')
    parser.add_argument('--eval_episodes',    type=int, default=5,       help='Number of episodes each evaluation')
    parser.add_argument('--eval_max_steps',   type=int, default=1000,    help='Maximum timesteps in each evaluation episode')
    parser.add_argument('--eval_seed',        type=int, default=0,       help='Environment seed for evaluation')
    parser.add_argument('--action_noise',     type=str, default=None,    help='Action noise, can be one of [\'None\']')
    parser.add_argument('--lr',                  type=float, default=1e-3,  help='Learning rate')
    parser.add_argument('--gamma',               type=float, default=0.99,  help='Gamma decay rate')
    parser.add_argument('--polyak',              type=float, default=0.005, help='Polyak coefficient (tau in original paper)')
    parser.add_argument('--target_policy_noise', type=float, default=0.2,   help='Policy noise range')
    parser.add_argument('--target_noise_clip',   type=float, default=0.5,   help='Policy noise clip range')

    a = parser.parse_args()

    a.logdir      = a.logdir.format(env_id=a.env_id, rank=a.rank)
    a.logging     = os.path.join(a.logdir, a.logging).format(env_id=a.env_id, rank=a.rank)
    a.monitor_dir = os.path.join(a.logdir, a.monitor_dir).format(env_id=a.env_id, rank=a.rank)
    a.tb_logdir   = os.path.join(a.logdir, a.tb_logdir).format(env_id=a.env_id, rank=a.rank)
    a.model_dir   = os.path.join(a.logdir, a.model_dir).format(env_id=a.env_id, rank=a.rank)

    return a

if __name__ == '__main__':
    
    import pybullet_envs

    a = parse_args()

    # === Reset logger ===
    logger.Config.use(filename=a.logging, level='DEBUG', colored=True, reset=True)
    LOG = logger.getLogger()

    # === Print welcome message ===
    LOG.add_row('')
    LOG.add_rows('TD3', fmt='{:@f:ANSI_Shadow}', align='center')
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
    LOG.add_row('Num of gradient steps', a.gradient_steps)
    LOG.add_row('Num of episodes',       a.num_episodes)
    LOG.add_row('Log interval',          a.log_interval)
    LOG.add_row('Eval interval',         a.eval_interval)
    LOG.add_row('Eval episodes',         a.eval_episodes)
    LOG.add_row('Eval max steps',        a.eval_max_steps)
    LOG.add_row('Batch size',            a.batch_size)
    LOG.add_row('Buffer size',           a.buffer_size)
    LOG.add_row('Min buffer size',       a.min_buffer)
    LOG.add_row('Verbose',               a.verbose)
    LOG.add_line()
    LOG.add_row('Learning rate',     a.lr)
    LOG.add_row('Gamma',             a.gamma)
    LOG.add_row('Polya (tau)', a.polyak)
    LOG.add_row('Policy delay', a.policy_delay)
    LOG.add_row('Target policy noise', a.target_policy_noise)
    LOG.add_row('Target noise clip', a.target_noise_clip)
    LOG.add_row('Action noise', a.action_noise)
    LOG.flush('WARNING')

    # === Make envs ===
    # make pybullet/mujoco env
    def make_env(env_id, rank, log_path, seed=0):
        def _init():
            import pybullet_envs

            env = gym.make(env_id)
            env.seed(seed + rank)
            env = Monitor(env, directory=log_path, prefix=str(rank),
                        enable_video_recording=True, force=True,
                        video_kwargs={'prefix':'video/train.{}'.format(rank)})
            return env
        set_global_seeds(seed)
        return _init

    env = SubprocVecEnv([make_env(a.env_id, i, a.monitor_dir, seed=a.seed) for i in range(a.num_envs)])

    # make env for evaluation
    eval_env = gym.make(a.env_id)
    eval_env.seed(a.eval_seed)
    eval_env = Monitor(eval_env, directory=a.monitor_dir, prefix='eval',
                       enable_video_recording=True, force=True,
                       video_kwargs={'prefix':'video/eval',
                                     'width': 640,
                                     'height': 480,
                                     'callback': lambda x: True})
    
    LOG.debug('Action space: {}'.format(env.action_space))
    LOG.debug('Observation space: {}'.format(env.observation_space))

    # === Create model ===
    try:
        model = TD3(env, learning_rate       = a.lr,
                         buffer_size         = a.buffer_size,
                         min_buffer          = a.min_buffer,
                         n_steps             = a.num_steps,
                         gradient_steps      = a.gradient_steps,
                         batch_size          = a.batch_size,
                         policy_delay        = a.policy_delay,
                         gamma               = a.gamma,
                         tau                 = a.polyak,
                         target_policy_noise = a.target_policy_noise,
                         target_noise_clip   = a.target_noise_clip,
                         action_noise        = a.action_noise,
                         verbose             = a.verbose)
        
        # Total timesteps = num_steps * num_envs * num_episodes (default ~ 10M)
        model.learn(a.num_steps *    a.num_envs * a.num_episodes,
                    tb_logdir      = a.tb_logdir,
                    log_interval   = a.log_interval,
                    eval_env       =  eval_env, 
                    eval_interval  = a.eval_interval, 
                    eval_episodes  = a.eval_episodes, 
                    eval_max_steps = a.eval_max_steps)

        LOG.info('DONE')

        # Save complete model (continue training)
        model.save(a.model_dir)
        loaded_model = TD3.load(a.model_dir)

        # set env to continue training
        # loaded_model.set_env(env)
        # loaded_model.learn(a.num_steps *    a.num_envs * a.num_episodes * 2,
        #                     tb_logdir      = a.tb_logdir,
        #                     log_interval   = a.log_interval,
        #                     eval_env       =  eval_env, 
        #                     eval_interval  = a.eval_interval, 
        #                     eval_episodes  = a.eval_episodes, 
        #                     eval_max_steps = a.eval_max_steps)

        # Save agent only
        #model.agent.save(a.model_dir)
        #loaded_model = TD3Agent.load(a.model_dir)


        # Evaluation
        eps_rews  = []
        eps_steps = []
        for episode in range(a.eval_episodes):
            obs = eval_env.reset()
            total_rews = 0

            for steps in range(1000):
                # predict action
                acts = loaded_model.predict(obs)
                # step environment
                obs, rew, done, info = eval_env.step(acts)
                total_rews += rew
                if done:
                    break

            # === Print episode result ===
            LOG.set_header('Eval {}/{}'.format(episode+1, a.eval_episodes))
            LOG.add_line()
            LOG.add_row('Rewards', total_rews)
            LOG.add_row(' Length', steps+1)
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
        LOG.add_row('     Length',  max_steps)
        LOG.add_line()
        LOG.add_row('Mean rewards', mean_rews)
        LOG.add_row(' Std rewards', std_rews, fmt='{}: {:.6f}')
        LOG.add_row(' Mean length', mean_steps)
        LOG.add_line()
        LOG.flush('INFO')

    except:
        LOG.exception('Exception occurred')
    
        env.close()
        eval_env.close()
        exit(1)
    
    env.close()
    eval_env.close()