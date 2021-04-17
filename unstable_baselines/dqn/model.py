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
from unstable_baselines.utils import (set_global_seeds,
                                      normalize,
                                      to_json_serializable,
                                      from_json_serializable,
                                      tf_soft_update_params,
                                      StateObject)



# create logger
logger.Config.use(level='DEBUG', colored=True, reset=False)
LOG = logger.getLogger('DQN')

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
        '''
        inputs: observations with shape (batch, obs_space.shape)
        '''
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x

# Mlp feature extractor
class MlpNet(tf.keras.Model):
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
        '''
        inputs: observations with shape (batch, obs_space.shape)
        '''
        x = inputs
        for layer in self._layers:
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
        '''
        inputs: latent with shape (batch, latent_size)
        '''
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x

# === Agent, Model ===

class Agent(SavableModel):
    def __init__(self, observation_space, action_space, force_mlp=False, **kwargs):
        super().__init__(**kwargs)

        self.force_mlp = force_mlp

        # --- Initialize ---
        self.observation_space = None
        self.action_space      = None
        self.net               = None
        self.q_net             = None

        if observation_space is not None and action_space is not None:
            self.setup_model(observation_space, action_space)

    def setup_model(self, observation_space, action_space):
        
        self.observation_space = observation_space
        self.action_space      = action_space

        # --- setup model ---
        if (len(self.observation_space.shape) == 3) and (not self.force_mlp):
            self.net = NatureCnn()
        else:
            self.net = MlpNet()

        self.q_net = QNet()

        # construct networks
        inputs = tf.keras.Input(shape=self.observation_space.shape, dtype=tf.float32)

        outputs = self.net(inputs)
        self.q_net(outputs)

    @tf.function
    def _forward(self, inputs, training=True):
        '''
        Forwrd network

        inputs: observations, only accepts shape (batch, obs_space.shape)
        
        return values
        '''

        # cast and normalize non-float32 inputs (e.g. image with uint8)
        # TODO: a better way to perform normalization?
        if tf.as_dtype(inputs.dtype) != tf.float32:
            # cast observations to float32
            inputs = tf.cast(inputs, dtype=tf.float32)
            low    = tf.cast(self.observation_space.low, dtype=tf.float32)
            high   = tf.cast(self.observation_space.high, dtype=tf.float32)
            # normalize observations [0, 1]
            inputs = normalize(inputs, low=low, high=high, nlow=0., nhigh=1.)

        # forward network
        latest = self.net(inputs, training=training)
        # forward policy net
        values = self.q_net(latent, training=training)

        return values

    @tf.function
    def call(self, inputs, training=True):
        '''
        Batch predict actions

        inputs: observations, only accepts shape (batch, obs_space.shape)
        '''

        # forward
        values = self._forward(inputs, training=training)

        actions = tf.math.argmax(values, axis=-1)
        
        return actions, values
    
    def predict(self, inputs):
        '''
        Predict actions

        inputs: observations, only accepts shape (batch, obs_space.shape)
        '''

        one_sample = (len(inputs.shape) == len(self.observation_space.shape))

        if one_sample:
            inputs = np.expand_dims(inputs, axis=0)

        # predict
        outputs, *_ = self(inputs, training=False)
        outputs = outputs.numpy()

        if one_sample:
            outputs = np.squeeze(outputs, axis=0)

        # predict
        return outputs

    def get_config(self):
        
        config = {'observation_space': self.observation_space,
                  'action_space':      self.action_space,
                  'force_mlp':         self.force_mlp}

        return to_json_serializable(config)


class DQN(TrainableModel):
    def __init__(self, env, learning_rate:       float = 3e-4,
                            buffer_size:           int = int(1e6),
                            min_buffer:            int = 1000,
                            n_steps:               int = 100,
                            gradient_steps:        int = 200,
                            batch_size:            int = 128,
                            gamma:               float = 0.99,
                            tau:                 float = 1.0,
                            max_grad_norm:       float = 0.5,
                            huber:                bool = True,
                            force_mlp:            bool = False,
                            explore_schedule           = None,
                            verbose:               int = 0,
                            **kwargs):
        '''Double DQN

        Args:
            env (gym.Env): environment
            learning_rate (float, optional): learning rate. Defaults to 3e-4.
            buffer_size (int, optional): maximum size of the replay buffer. Defaults to int(1e6).
            min_buffer (int, optional): minimum size of the replay buffer before training. Defaults to 1000.
            n_steps (int, optional): number of steps of rollouts to collect every epoch. Defaults to 100.
            gradient_steps (int, optional): [description]. Defaults to 200.
            batch_size (int, optional): [description]. Defaults to 128.
            gamma (float, optional): [description]. Defaults to 0.99.
            tau (float, optional): [description]. Defaults to 1.0.
            max_grad_norm (float, optional): [description]. Defaults to 0.5.
            huber (bool, optional): [description]. Defaults to True.
            force_mlp (bool, optional): [description]. Defaults to False.
            explore_schedule ([type], optional): [description]. Defaults to None.
            verbose (int, optional): [description]. Defaults to 0.
        '''
        super().__init__(**kwargs)

        self.env = env

        self.learning_rate    = learning_rate
        self.buffer_size      = buffer_size
        self.min_buffer       = min_buffer
        self.n_steps          = n_steps
        self.gradient_steps   = gradient_steps
        self.batch_size       = batch_size
        self.gamma            = gamma
        self.tau              = tau
        self.max_grad_norm    = max_grad_norm
        self.huber            = huber
        self.force_mlp        = force_mlp
        self.explore_schedule = explore_schedule
        self.verbose          = verbose

        # initialize states
        self.s = StateObject()
        self.s.num_timesteps   = 0
        self.s.num_epochs      = 0
        self.s.num_gradsteps   = 0
        self.s.progress        = 0

        self.buffer            = None
        self.tb_writer         = None
        self.observation_space = None
        self.action_space      = None
        self.n_envs            = 0

        if env is not None:
            self.set_env(env)
            self.setup_model(env.observation_space, env.action_space)

    @property
    def num_timesteps(self):
        return self.s.num_timesteps

    @property
    def num_epochs(self):
        return self.s.num_epochs

    @property
    def num_gradsteps(self):
        return self.s.num_gradsteps

    def set_env(self, env):

        if self.observation_space is not None:
            assert env.observation_space == self.observation_space, 'Observation space mismatch, expect {}, got {}'.format(
                                                                        self.observation_space, env.observation_space)

        if self.action_space is not None:
            assert env.action_space == self.action_space, 'Action space mismatch, expect {}, got {}'.format(
                                                                self.action_space, env.action_space)
        
        self.env = env
        self.n_envs = env.n_envs

    def setup_model(self, observation_space, action_space):
        
        self.observation_space = observation_space
        self.action_space = action_space

        # --- setup model ---
        self.buffer = ReplayBuffer(buffer_size=self.buffer_size)
        self.agent = Agent(self.observation_space, self.action_space, force_mlp=force_mlp)
        self.agent_target = Agent(self.observation_space, self.action_space, force_mlp=force_mlp)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate,
                                                  clipnorm=self.max_grad_norm)

        # initialize target
        self.agent_target.update(self.agent)

        # setup scheduler
        self.explore_schedule = Scheduler.get_scheduler(self.explore_schedule, 
                                                        state_object=self.s)

    @tf.function
    def _forward(self, inputs, training=True):
        '''
        Forward actor

        Return:
            values: q values (batch, act_space.n)
        '''
        return self.agent._forward(inputs, training=training)

    @tf.function
    def call(self, inputs, training=True):
        '''
        Batch predict actions

        Returns:
            actions: actions (batch, )
            values: q values (batch, act_space.n)
        '''
        return self.agent(inputs, training=training)

    def predict(self, inputs):
        '''
        Predict actions

        Returns:
            actions: actions (batch, )
        '''
        return self.agent.predict(inputs)

    def _run(self, steps, obs=None):
        '''
        Run environments, collect rollouts

        steps: number of timesteps
        obs: initial observations, if None, reset env
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
                action, _ = self(obs)

            # step environment
            new_obs, reward, done, infos = self.env.step(action)

            # add to buffer
            self.buffer.add(obs, new_obs, action, reward, done)
            obs = new_obs

            # update state
            self.s.num_timesteps += self.n_envs

        return new_obs

    @tf.function
    def value_loss(self, obs, action, next_obs, done, reward):
        '''
        Compute value loss

        Args:
            obs: observation (batch, obs_space.shape)
            action: action (batch, )
            next_obs: next observation (batch, obs_space.shape)
            done: done (batch, )
            reward: reward (batch, )
        '''

        action = tf.cast(action, dtype=tf.int64)

        next_qs = self.agent_target._forward(next_obs)
        target_q = tf.math.reduce_max(next_qs, axis=-1)

        y = reward + tf.stop_gradient((1.-done) * self.gamma * target_q)

        qs = self._forward(obs)
        q = tf.gather(qs, indices=action, batch_dims=1)

        if self.huber:
            loss = tf.keras.losses.Huber()(q, y)
        else:
            loss = tf.keras.losses.MSE(q, y)

        return loss

    @tf.function
    def _train_step(self, obs, action, next_obs, done, reward):
        '''
        Update q value
        '''

        variables = self.agent.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(variables)

            loss = self.value_loss(obs, action, next_obs, done, reward)

        # perform gradients
        grads = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))

        return loss

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

            loss = self._train_step(obs, action_next_obs, done, reward)

            all_loss.append(loss)

            self.s.num_gradsteps += 1

            # update target networks
            if self.s.num_gradsteps % target_update == 0:
                self.agent_target.update(self.agent, polyak=self.tau)

        m_loss = np.mean(np.hstack(np.asarray(all_loss)))
        
        self.s.num_epochs += 1

        return m_loss

    def eval(self, env, n_episodes=5, max_steps=10000):
        '''Evaluate model (use default evaluation method)

        Args:
            env (gym.Env): the environment for evaluation
            n_episodes (int, optional): number of episodes to evaluate. Defaults to 5.
            max_steps (int, optional): maximum steps. Defaults to 10000.

        Returns:
            list: episode rewards
            list: episode length
        '''

        return super().eval(env, n_episodes=n_episodes, 
                                 max_steps=max_steps)

    def learn(self, total_timesteps:  int,
                    log_interval:     int = 1,
                    eval_env:     gym.Env = None,
                    eval_interval:    int = 1,
                    eval_episodes:    int = 5,
                    eval_max_steps:   int = 10000,
                    target_update:    int = 10000,
                    tb_logdir:        str = None,
                    reset_timesteps: bool = False):
        '''[summary]

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
            target_update (int, optional): Frequency of updating target network.
                update every ``target_update`` gradient steps. Defaults to 10000.
            tb_logdir (str, optional): tensorboard log directory. Defaults to None.
            reset_timesteps (bool, optional): reset timesteps. Defaults to False.

        Returns:
            DQN: self
        '''        

        assert self.env is not None, 'Env not set, call set_env() before training'

        # create tensorboard writer
        if tb_logdir is not None:
            self.tb_writer = tf.summary.create_file_writer(tb_logdir)

        # initialize
        if reset_timesteps:
            self.s.num_timesteps = 0
            self.s.num_gradsteps = 0
            self.s.num_epochs    = 0
            self.s.progress      = 0
            # reset buffer
            self.buffer.reset()

        obs        = None
        time_start = time.time()
        time_spent = 0
        timesteps_per_epoch = self.n_steps * self.n_envs
        total_epochs = int(float(total_timesteps-self.s.num_timesteps) /
                                        float(timesteps_per_epoch) + 0.5)
        

        while self.s.num_timesteps < total_timesteps:
            # collect rollouts
            obs = self._run(steps=self.n_steps, obs=obs)

            # update state
            self.s.num_epochs += 1
            self.s.progress = float(self.s.num_timesteps) / float(total_timesteps)

            if len(self.buffer) > self.min_buffer:

                # training
                loss = self.train(self.gradient_steps, 
                                batch_size=self.batch_size, 
                                policy_delay=self.policy_delay)

                # write tensorboard
                if self.tb_writer is not None:
                    with self.tb_writer.as_default():
                        tf.summary.scalar('loss', loss, step=self.s.num_timesteps)
                        tf.summary.scalar('explore_rate', self.explore_schedule(),
                                                        step=self.s.num_timesteps)

                    self.tb_writer.flush()

            # print training log
            if (log_interval is not None) and (self.s.num_epochs % log_interval) == 0:
                # current time
                time_now       = time.time()
                # execution time (one epoch)
                execution_time = (time_now - time_start) - time_spent
                # total time spent
                time_spent     = (time_now - time_start)
                # remaining time
                remaining_time = (time_spent / self.s.progress)*(1.0-self.s.progress)
                # eta
                eta            = (datetime.datetime.now() + datetime.timedelta(seconds=remaining_time)).strftime('%Y-%m-%d %H:%M:%S')
                # average steps per second
                fps            = float(self.s.num_timesteps) / time_spent

                LOG.set_header('Epoch {}/{}'.format(self.s.num_epochs, total_epochs))
                LOG.add_line()
                LOG.add_row('Timesteps', self.s.num_timesteps, total_timesteps, fmt='{}: {}/{}')
                LOG.add_row('Steps/sec', fps,                                   fmt='{}: {:.2f}')
                LOG.add_row('Progress',  self.s.progress*100.0,                 fmt='{}: {:.2f}%')

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

            if (eval_env is not None) and (self.s.num_epochs % eval_interval) == 0:
                eps_rews, eps_steps = self.eval(env=eval_env, n_episodes=eval_episodes, max_steps=eval_max_steps)
                
                max_idx    = np.argmax(eps_rews)
                max_rews   = eps_rews[max_idx]
                max_steps  = eps_steps[max_idx]
                mean_rews  = np.mean(eps_rews)
                std_rews   = np.std(eps_rews)
                mean_steps = np.mean(eps_steps)

                if self.tb_writer is not None:
                    with self.tb_writer.as_default():
                        tf.summary.scalar('max_rewards',  max_rews,   step=self.s.num_timesteps)
                        tf.summary.scalar('mean_rewards', mean_rews,  step=self.s.num_timesteps)
                        tf.summary.scalar('std_rewards',  std_rews,   step=self.s.num_timesteps)
                        tf.summary.scalar('mean_length',  mean_steps, step=self.s.num_timesteps)

                    self.tb_writer.flush()

                LOG.set_header('Evaluate {}/{}'.format(self.s.num_epochs, total_epochs))
                LOG.add_line()
                LOG.add_row('Max rewards',  max_rews)
                LOG.add_row('     Length',  max_steps)
                LOG.add_line()
                LOG.add_row('Mean rewards', mean_rews)
                LOG.add_row(' Std rewards', std_rews, fmt='{}: {:.3f}')
                LOG.add_row(' Mean length', mean_steps)
                LOG.add_line()
                LOG.flush('INFO')

        return self

    def get_config(self):

        self.learning_rate    = learning_rate
        self.buffer_size      = buffer_size
        self.min_buffer       = min_buffer
        self.n_steps          = n_steps
        self.gradient_steps   = gradient_steps
        self.batch_size       = batch_size
        self.gamma            = gamma
        self.tau              = tau
        self.max_grad_norm    = max_grad_norm
        self.huber            = huber
        self.force_mlp        = force_mlp
        self.explore_schedule = explore_schedule
        self.verbose          = verbose
        
        init_config = { 'learning_rate':       self.learning_rate,
                        'buffer_size':         self.buffer_size,
                        'min_buffer':          self.min_buffer,
                        'n_steps':             self.n_steps,
                        'gradient_steps':      self.gradient_steps,
                        'batch_size':          self.batch_size,
                        'gamma':               self.gamma,
                        'tau':                 self.tau,
                        'max_grad_norm':       self.max_grad_norm,
                        'huber':               self.huber,
                        'force_mlp':           self.force_mlp,
                        'explore_schedule':    self.explore_schedule,
                        'verbose':             self.verbose}

        setup_config = {'observation_space': self.observation_space,
                        'action_space':      self.action_space}

        state_config = dict(self.s)

        init_config  = to_json_serializable(init_config)
        setup_config = to_json_serializable(setup_config)
        state_config = to_json_serializable(state_config)

        return {'init_config': init_config,
                'state_config': state_config,
                'setup_config': setup_config}
    
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

        self.s.update(state_config)

        return self