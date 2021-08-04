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
import datetime

# --- 3rd party ---
import gym

import numpy as np
import tensorflow as tf

# --- my module ---
import unstable_baselines as ub

# === Networks ===

class Actor(ub.nets.PolicyNet):
    '''Actor network'''
    support_spaces = [gym.spaces.Box]
    def __init__(self, action_space, squash=True, **kwargs):
        '''Create actor

        Args:
            action_space (gym.spaces): action space, must be
                one of the types listed in `support_spaces`
            squash (bool, optional): Tanh squashed for gaussian distribution.
                Defaults to True.
        '''        
        super().__init__(action_space, squash=squash, **kwargs)

    def call(self, inputs, training=True):
        '''Forward actor

        Args:
            inputs (tf.Tensor): Expecting a batch latents with shape
                (b, latent), tf.float32
            training (bool, optional): Training mode. Defaults to True.
        
        Returns:
            Distribution: a policy distribution.
        '''
        return super().call(inputs, training=training)

    def get_model(self):
        '''Base model'''
        return tf.keras.Sequential([
            tf.keras.layers.Dense(256),
            ub.patch.ReLU(),
            tf.keras.layers.Dense(256),
            ub.patch.ReLU(),
        ])

class Critic(ub.nets.MultiHeadValueNets):
    '''Critic networks'''
    def __init__(self, action_space, n_critics: int=2, **kwargs):
        '''Create critics, state-action value function

        Args:
            n_critics (int, optional): number of critics. Defaults to 2.
        '''        
        super().__init__(n_critics=n_critics, **kwargs)

        if isinstance(action_space, gym.spaces.Box):
            self.action_dims = 1
        elif isinstance(action_space, gym.spaces.Discrete):
            self.action_dims = action_space.n
        else:
            raise ValueError(f'{type(self).__name__} does not '
                f'support action space of type `{type(action_space)}`')

    def call(self, inputs, training=True):
        '''Forward all critics

        Args:
            inputs (tf.Tensor): Expecting a batch latents with shape
                (b, latent), tf.float32
            training (bool, optional): Training mode. Defaults to True.

        Returns:
            tf.Tensor: (n_critics, b, 1)
        '''
        return super().call(inputs, training=training)

    def get_model(self):
        return tf.keras.Sequential([
            tf.keras.layers.Concatenate(),
            tf.keras.layers.Dense(256),
            ub.patch.ReLU(),
            tf.keras.layers.Dense(256),
            ub.patch.ReLU(),
            tf.keras.layers.Dense(self.action_dims)
        ])


class Agent(ub.base.BaseAgent):
    support_obs_spaces = [gym.spaces.Box]
    support_obs_spaces = [gym.spaces.Box]
    def __init__(self, observation_space, 
                       action_space,
                       squash:    bool = True,
                       n_critics:  int = 2,
                       **kwargs):
        '''SAC Agent

        Args:
            observation_space (gym.spaces): Observation space
            action_space (gym.spaces): Action space
        '''
        super().__init__(observation_space, action_space, **kwargs)

        self.squash    = squash
        self.n_critics = n_critics

        self.actor  = None
        self.critic = None

        if observation_space is not None and action_space is not None:
            self.setup_model()

    def setup_model(self):
        '''Setup agent model

        Args:
            observation_space (gym.spaces): observation space.
            action_space (gym.spaces): action space.
        '''
        # --- setup model ---
        # create actor, critic net
        self.actor  = Actor(action_space, squash=self.squash)
        self.critic = Critic(action_space, n_critics=self.n_critics)
        # construct networks
        obs = tf.keras.Input(shape=self.observation_space.shape, dtype=tf.float32)
        act = tf.keras.Input(shape=self.action_space.shape, dtype=tf.float32)
        self._fwd_actor(obs, proc_obs=False)
        self._fwd_critic([obs, act], proc_obs=False)

    def map_action(self, act):
        '''Map raw action (from policy) to env action range
        '''
        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash and ub.utils.is_bounded(self.action_space):
                act = ub.utils.normalize(act, -1, 1, 
                        self.action_space.low, self.action_space.high)
            else:
                #TODO: how to do in one line for both tensor and non-tensor?
                if tf.is_tensor(act):
                    act = tf.clip_by_value(act, self.action_space.low,
                                                self.action_space.high)
                else:
                    act = np.clip(act, self.action_space.low,
                                       self.action_space.high)
        return act

    def _fwd_actor(self, inputs, proc_obs=True, training=True):
        '''Forward actor

        Args:
            inputs (tf.Tensor): batch observations in shape
                (batch, obs_spce.shape), tf.float32
            proc_obs (bool, optional): preprocess observations. Default
                to True.
            training (bool, optional): Training mode. Defaults to True.

        Returns:
            Distribution: Policy distribution
        '''
        if proc_obs:
            # cast and normalize inputs (eg. image in uint8)
            inputs = self.map_observation(inputs)
        return self.actor(inputs, training=training)

    def _fwd_critic(self, inputs, proc_obs=True, training=True):
        '''Forward critic

        Args:
            inputs (tuple): [obs, act] tuple.
            proc_obs (bool, optional): preprocess observations. Default
                to True.
            training (bool, optional): Training mode. Defaults to True.
        
        Return:
            tf.Tensor: predicted critic values (n_critics, b, 1)
                or (n_critics, b, act_space.n)
        '''
        obs, act = inputs
        if proc_obs:
            # cast and normalize inputs (eg. image in uint8)
            obs = self.map_observation(obs)
        if isinstance(self.action_space, gym.spaces.Box):
            inputs = [obs, act]
        else:
            inputs = [obs]
        return self.critic(inputs, training=training)

    def _fwd(self, inputs, proc_obs=True, training=True):
        return self._fwd_actor(inputs, proc_obs=proc_obs, training=training)

    @tf.function
    def call(self, inputs,
                   proc_obs: bool = True,
                   proc_act: bool = False,
                   det:      bool = False,
                   training: bool = True):
        '''Forward agent

        Args:
            inputs (tf.Tensor): batch observations in shape
                (b, obs_space.shape), tf.float32
            proc_obs (bool, optional): preprocess observations. Default
                to True.
            proc_act (bool, optional): postprocess actions. Default to
                False.
            det (bool, optional): deterministic actions.
                Defaults to False.
            training (bool, optional): Training mode. Defaults to True.

        Returns:
            tf.Tensor: predicted actions in shape (b, act_space.shape)
            tf.Tensor: log probabilities of predicted actions. (b,)
        '''
        # forward actor
        distrib = self._fwd(inputs, proc_obs=proc_obs, training=training)
        # predict actions
        if det:
            actions = distrib.mode()
        else:
            actions = distrib.sample()
        # get log probabilities
        log_probs = distrib.log_prob(actions)
        if proc_act:
            # map actions to env's action space range
            actions = self.map_action(actions)
        # return results
        return actions, log_probs

    def predict(self, inputs, 
                      proc_obs: bool = True,
                      proc_act: bool = True,
                      det:      bool = True):
        '''Predict actions

        Args:
            inputs (np.ndarray): batch observations in shape (b, obs_space.shape)
                or one observation in shape(obs_space.shape).
            proc_obs (bool, optional): preprocess observations. Default
                to True.
            proc_act (bool, optional): postprocess actions. Default to
                True.
            det (bool, optional): deterministic actions. Defaults to True.

        Returns:
            np.ndarray: predicted actions in shape (b, act_space.shape) or
                (act_space.shape) for one sample.
        '''
        one_sample = (len(inputs.shape) == len(self.observation_space.shape))
        if one_sample:
            inputs = np.expand_dims(inputs, axis=0)
        # predict
        outputs, *_ = self(inputs, proc_obs, proc_act, det, training=False)
        outputs     = np.asarray(outputs)
        if one_sample:
            outputs = np.squeeze(outputs, axis=0)
        return outputs

    def get_config(self):
        config = {
            'squash':            self.squash,
            'n_critics':         self.n_critics,
            'observation_space': self.observation_space, 
            'action_space':      self.action_space,
        }
        return config


class SAC(ub.base.OffPolicyModel):
    support_obs_spaces = [gym.spaces.Box]
    support_act_spaces = [gym.spaces.Box]
    def __init__(self, env, 
                    # --- learning parameters ---
                       learning_rate:       float = 3e-4,
                       buffer_size:           int = int(1e6),
                       gamma:               float = 0.99,
                       polyak:              float = 0.005,
                       clipnorm:            float = 0.5,
                       reward_scale:        float = 1.0,
                       weight_decay:        float = 1e-5,
                       target_ent:          float = None,
                    # --- architecture parameters ---
                       squash:               bool = True,
                       n_critics:             int = 2,
                    # --- other parameters ---
                       n_steps:               int = 4,
                       n_gradsteps:           int = 1,
                       wramup_steps:          int = int(1e4),
                       batch_size:            int = 256,
                       verbose:               int = 0,
                       **kwargs):
        '''Soft Actor-Critic (SAC)

        This implementation mainly follows this repo:
        - https://github.com/rail-berkeley/softlearning
        Paper:
        - Soft Actor-Critic Algorithms and Applications, Haarnoja et al.

        The first argument `env` can be `None` for delayed model setup. You
        should call `set_env()` then call `setup_model()` to manually setup
        the model.

        Args:
            env (gym.Env): Training environment. Set to `None` for delayed setup.
            learning_rate (float, Scheduler, optional): Learning rate. 
                Defaults to 3e-4.
            buffer_size (int, optional): Replay buffer capacity. Defaults to 
                int(1e6).
            gamma (float, optional): Discount factor. Defaults to 0.99.
            polyak (float, optional): Soft update parameter. Defaults to 0.005.
            clipnorm (float, optional): Gradient clip range. Defaults to 0.5.
            squash (bool, optional): Squashed actor. Defaults to True.
            n_critics (int, optional): Critic network count. Defaults to 2.
            n_steps (int, optional): Number of steps (samples) per env per 
                epoch. Defaults to 4.
            n_gradsteps (int, optional): Number of gradient updates per epoch. 
                Defaults to 1.
            wramup_steps (int, optional): Number of samples to wramup the replay
                buffer. Defaults to int(1e4).
            batch_size (int, optional): Mini-batch size. Defaults to 256.
            verbose (int, optional): More logging [0, 1, 2]. Defaults to 0.
        '''
        super().__init__(
            env          = env,
            n_steps      = n_steps,
            n_gradsteps  = n_gradsteps,
            wramup_steps = wramup_steps,
            batch_size   = batch_size,
            verbose      = verbose,
            **kwargs
        )

        self.learning_rate = learning_rate
        self.buffer_size   = buffer_size
        self.gamma         = gamma
        self.polyak        = polyak
        self.clipnorm      = clipnorm
        self.reward_scale  = reward_scale
        self.weight_decay  = weight_decay
        self.target_ent    = target_ent
        self.squash        = squash
        self.n_critics     = n_critics

        # initialize states
        self.buffer            = None
        self.agent             = None
        self.agent_target      = None
        self.log_alpha         = None
        self.actor_optimizer   = None
        self.alpha_optimizer   = None
        self.critic_optimizer  = None

        if env is not None:
            self.setup_model()

    def _setup_buffer(self):
        '''Override this method to customize to your replay buffer'''
        self.buffer = ub.data.ReplayBuffer(
            buffer_size = self.buffer_size
        )
    
    def _setup_model(self):
        '''Override this method to customize to your model'''
        self.agent = Agent(
            self.observation_space,
            self.action_space,
            squash    = self.squash,
            n_critics = self.n_critics
        )
        self.agent_target = Agent(
            self.observation_space,
            self.action_space,
            squash    = self.squash,
            n_critics = self.n_critics
        )
        self.log_alpha = tf.Variable(0.0, dtype=tf.float32)
    
    def _setup_optimizer(self):
        '''Override this method to customize to your optimizers'''
        # --- setup scheduler ---
        self.learning_rate = ub.sche.get_scheduler(
            self.learning_rate,
            state = self.state
        )
        # --- optimizers ---
        self.actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate = self.learning_rate(),
            clipnorm      = self.clipnorm
        )
        self.alpha_optimizer  = tf.keras.optimizers.Adam(
            learning_rate = self.learning_rate(),
            clipnorm      = self.clipnorm
        )
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate = self.learning_rate(),
            clipnorm      = self.clipnorm
        )

    def setup_model(self):
        '''Setup model, optimizer and scheduler for training'''
        self._setup_buffer()
        self._setup_model()
        self._setup_optimizer()
        # initialize target
        self.agent_target.update(self.agent)
        # set up target entropy: dim(A)
        if self.target_ent is None:
            self.target_ent = -np.prod(self.action_space.shape)

    @tf.function
    def call(self, inputs, 
                   proc_obs: bool = True,
                   proc_act: bool = False,
                   det:      bool = False,
                   training: bool = True):
        '''See Agent.call'''
        return self.agent(
            inputs, 
            proc_obs = proc_obs,
            proc_act = proc_act,
            det      = det,
            training = training
        )

    def predict(self, inputs, 
                      proc_obs: bool = True,
                      proc_act: bool = True,
                      det:      bool = True):
        '''See Agent.predict'''
        return self.agent.predict(
            inputs, 
            proc_obs = proc_obs,
            proc_act = proc_act,
            det      = det
        )

    def _run_step(self, obs=None):
        '''Collect one sample

        Args:
            obs (np.ndarray, optional): Current observations. Defaults to None.

        Returns:
            np.ndarray: next observations
        '''        
        if self.is_wraming_up():
            # sample random actions
            rawact = np.asarray([self.action_space.sample()
                                 for n in range(self.n_envs)])
            act = rawact
        else:
            # sample actions from policy
            rawact = self.predict(obs, proc_act=False, det=False)
            act = self.agent.map_action(rawact)
        # step environment
        next_obs, rew, done, infos = self.env.step(act)
        # add to buffer
        self.buffer.add(
            obs      = obs,
            act      = rawact,
            next_obs = next_obs,
            done     = done,
            rew      = rew
        )
        return next_obs

    def alpha_loss(self, log_ps):
        '''Compute entropy loss

        loss = E[-a*logp(a|s)-a*H]

        Args:
            log_ps (tf.Tensor): log probabilities with shape (b, 1)

        Returns:
            tf.Tensor: loss
        '''
        alpha = tf.math.exp(self.log_alpha)
        x = tf.stop_gradient(log_ps + self.target_ent)
        return -tf.reduce_mean(tf.alpha * x, axis=0)

    def actor_loss(self, log_ps, qs):
        '''Compute actor loss

        loss = E[a*logp(a|s) - Q(s, mu(a))]

        Args:
            log_ps (tf.Tensor): log probabilities (b, 1)
            qs (tf.Tensor): Q-values (n_critics, b)
        '''
        alpha = tf.stop_gradient(tf.math.exp(self.log_alpha))
        target_q = tf.math.reduce_mean(qs, axis=0)
        return tf.reduce_mean(alpha * log_ps - target_q)

    def critic_loss(self, obs, act, next_obs, done, rew):
        '''Compute critic loss

        V(s') = E[Q(s',a')-a*logp(a'|s')]
        loss = E[0.5*(Q(s,a)-(r+g*V(s')))^2]

        Args:
            obs (tf.Tensor): batch observations
            act (tf.Tensor): batch actions
            next_obs (tf.Tensor): batch next observations
            done (tf.Tensor): batch dones
            rew (tf.Tensor): batch rewards

        Returns:
            tf.Tensor: critic losses, shape (b,)
        '''
        reward = tf.cast(reward, dtype=tf.float32)
        done   = tf.cast(done,   dtype=tf.float32)

        alpha = tf.stop_gradinet(tf.math.exp(self.log_alpha))
        next_act, next_log_p = self.agent_target(
            next_obs, proc_act=False, det=False
        )
        next_qs = self.agent_target._fwd_critic([next_obs, next_act])
        # compute V value for the next states
        next_q  = tf.math.reduce_min(next_qs, axis=0) # (b, 1)
        next_v  = next_q - alpha * next_log_p
        # compute target Q value
        rew  = self.reward_scale * rew
        y    = rew + tf.stop_gradient((1.-done) * self.gamma * next_v)
        # compute Q estimate
        qs   = self.agent._fwd_critic([obs, act]) # (n, b, 1)
        # compute critic loss
        loss = tf.keras.losses.MSE(qs, y) # (n, b)
        loss = 0.5 * tf.math.reduce_sum(losses, axis=0)
        return tf.reduce_mean(loss)

    def reg_loss(self, var_list, loss_type='l2'):
        if not var_list or self.weight_decay is None:
            return 0.
        if loss_type == 'l1':
            reg = tf.keras.regularizers.L1(self.weight_decay)
        else:
            reg = tf.keras.regularizers.L2(self.weight_decay)
        return tf.math.add_n([reg(var) for var in var_list])

    @tf.function
    def _train_actor(self, batch):
        actor_vars = self.agent.actor.trainable_variables
        alpha_vars = [self.log_alpha]
        variables  = actor_vars + alpha_vars
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(variables)
            obs = batch['obs']
            # predict critics and log probs
            act, log_ps = self.agent(obs, proc_act=False, det=False)
            qs = self.agent._fwd_critic([obs, act])
            # compute losses
            actor_loss = self.actor_loss(log_ps, qs)
            alpha_loss = self.alpha_loss(log_ps)
            reg_loss   = self.reg_loss(actor_vars)
            loss = actor_loss + reg_loss
        # perform gradient update
        grads = tape.gradient(loss, actor_vars)
        self.actor_optimizer.apply_gradients(zip(grads, actor_vars))
        grads = tape.gradient(alpha_loss, alpha_vars)
        self.alpha_optimizer.apply_gradients(zip(grads, alpha_vars))
        return {'loss':       loss,
                'actor_loss': actor_loss,
                'reg_loss':   reg_loss,
                'alpha_loss': alpha_loss}

    @tf.function
    def _train_critic(self, batch):
        variables = self.agent.critic.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(variables)
            # compute losses
            critic_loss = self.critic_loss(**batch)
            reg_loss    = self.reg_loss(variables)
            loss        = critic_loss + reg_loss
        # perform gradient update
        grads = tape.gradient(loss, variables)
        self.critic_optimizer.apply_gradients(zip(grads, variables))
        return {'loss':        loss,
                'critic_loss': critic_loss,
                'reg_loss':    reg_loss}

    def _train_step(self, batch_size):
        # sample buffer
        batch = self.buffer(batch_size)
        # update learning rate
        lr = tf.convert_to_tensor(self.learning_rate())
        self.critic_optimizer.lr.assign(lr)
        self.actor_optimizer.lr.assign(lr)
        self.alpha_optimizer.lr.assign(lr)
        # train networks
        losses = ub.utils.flatten_dicts([
            self._train_critic(batch, lr),
            self._train_actor(batch, lr)
        ])
        # aggregate losses
        losses = ub.utils.nested_iter(losses, lambda v: np.asarray(v).sum())
        return losses

    def _update_target(self):
        self.agent_target.update(self.agent, polyak=self.polyck)

    def is_warming_up(self):
        return len(self.buffer) < self.wramup_steps

    def get_config(self):
        config = {
            'learning_rate': self.learning_rate,
            'buffer_size':   self.buffer_size,
            'gamma':         self.gamma,
            'polyak':        self.polyak,
            'clipnorm':      self.clipnorm,
            'reward_scale':  self.reward_scale,
            'weight_decay':  self.weight_decay,
            'target_ent':    self.target_ent,
            'squash':        self.squash,
            'n_critics':     self.n_critics,
        }
        return config