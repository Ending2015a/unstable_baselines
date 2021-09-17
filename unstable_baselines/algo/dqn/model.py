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

class QNet(ub.nets.ValueNet):
    def __init__(self, action_dims: int,
                       dueling: bool = False,
                       net: tf.keras.Model = None,
                       **kwargs):
        '''Q-value net

        Args:
            action_dims (int): Output action size.
            dueling (bool, optional): Whether to use dueling nets. 
                Defaults to False.
            net (tf.keras.Model, optional): Base network, feature extractor. 
                Defaults to None.
        '''
        super().__init__(net=net, **kwargs)
        self.action_dims = action_dims
        self.dueling = dueling

    def call(self, inputs, training=True):
        '''Forward q-value net

        Args:
            inputs (tf.Tensor): Input tensors, shape (b, *)
            training (bool, optional): Training mode. Defaults to True.

        Returns:
            tf.Tensor: value predictions, shape (b, act)
        '''
        x = self._model(inputs, training=training)
        if self.dueling:
            # Dueling DQN
            q = self._value[0](x, training=training)
            v = self._value[1](x, training=training)
            a = q - tf.math.reduce_mean(q, axis=-1, keepdims=True)
            return a + v
        else:
            return self._value(x, training=training)

    def create_value_model(self):
        '''Create value model
        Override this method to customize model arch

        Returns:
            tf.keras.Model, list: Q net. If `dueling`, returns
                a list contains Q net and V net.
        '''
        if self.dueling:
            # Dueling DQN (Q, V)
            return [
                tf.keras.layers.Dense(self.action_dims),
                tf.keras.layers.Dense(1)
            ]
        else:
            return tf.keras.layers.Dense(self.action_dims)

class Agent(ub.base.BaseAgent):
    support_obs_spaces = ub.utils.spaces.All
    support_act_spaces = [gym.spaces.Discrete]
    def __init__(self, observation_space,
                       action_space,
                       dueling:   bool = False,
                       force_mlp: bool = False,
                       mlp_units: list = [64, 64],
                       **kwargs):
        '''DQN agent

        The `observation_space` and `action_space` can be `None` for delayed
        model setup. You should call `set_spaces()` then call `setup()` to 
        manually setup the model.

        Args:
            observation_space (gym.Space): The observation space of the env.
                Set to None for delayed setup.
            action_space (gym.Space): The action space of the env. Set to None 
                for delayed setup.
            force_mlp (bool, optional): Force to use mlp as feature extractors.
                Default to False.
            mlp_units (list, optional): Number of units for each mlp layer. 
                Defaults to [64, 64]
        '''        
        super().__init__(observation_space, action_space, **kwargs)

        self.dueling   = dueling
        self.force_mlp = force_mlp
        self.mlp_units = mlp_units

        self.value = None

        if (observation_space is not None) and (action_space is not None):
            self.setup()
    
    # --- setup group ---
    def _setup_model(self):
        '''Setup models
        Override this method to customize nets
        '''
        # AwesomeNet automatically handles complex observations
        # e.g. Dict or Tuple and creates separate feature extractors
        # either NatureCnn or MlpNet for each space
        base_net = ub.nets.AwesomeNet(
            input_space = self.observation_space,
            force_mlp   = self.force_mlp,
            mlp_units   = self.mlp_units
        )
        # create q net
        self.value = QNet(
            action_dims = self.action_space.n,
            dueling     = self.dueling,
            net         = base_net
        )

    def setup(self):
        '''(Override) Setup agent model'''
        self._setup_model()
        # construct network params
        inputs = ub.utils.input_tensor(self.observation_space)
        self.call(inputs, proc_obs=True)

    def call_value(self, inputs, proc_obs=True, training=True):
        '''Forward value net'''
        if proc_obs:
            inputs = self.proc_observation(inputs)
        values = self.value(inputs, training=training)
        return values

    @tf.function
    def call(self, inputs,
                   proc_obs: bool = True,
                   training: bool = True):
        '''(Override) Forward agent

        Args:
            inputs (tf.Tensor): Batch observations in shape
                (b, *obs.space.shape), tf.float32
            proc_obs (bool, optional): Preprocess observations. Defaults to True.
            training (bool, optional): Training mode. Defaults to True.

        Returns:
            tf.Tensor: Predicted actions, shape (b,)
            tf.Tensor: Predicted q values, shape (b,)
        '''
        # forward value nets
        values  = self.call_value(inputs, proc_obs, training)
        actions = tf.math.argmax(values, axis=-1)
        values  = tf.math.reduce_max(values, axis=-1)
        return actions, values

    def predict(self, inputs, proc_obs: bool=True):
        return super().predict(inputs, proc_obs=proc_obs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'dueling':   self.dueling,
            'force_mlp': self.force_mlp,
            'mlp_units': self.mlp_units
        })
        return config

class DQN(ub.base.OffPolicyModel):
    support_obs_spaces = ub.utils.spaces.All
    support_act_spaces = [gym.spaces.Discrete]
    def __init__(self,
        env,
    # --- hyper parameters ---
        learning_rate: float = 3e-4,
        buffer_size:     int = int(1e5),
        multi_step:      int = 1,
        gamma:         float = 0.99,
        tau:           float = 1.0,
        reg_coef:      float = 0.0,
        clipnorm:      float = 0.5,
        explore_rate:  float = 0.3,
        huber:          bool = True,
        huber_rate:    float = 0.1,
        prioritized:    bool = False,
        prio_alpha:    float = 0.6,
        prio_beta:     float = 0.4,
    # --- architecture parameters ---
        dueling:        bool = False,
        force_mlp:      bool = False,
        mlp_units:      list = [64, 64],
    # --- other parameters ---
        n_steps:         int = 4,
        n_gradsteps:     int = 1,
        warmup_steps:    int = int(1e4),
        batch_size:      int = 128,
        **kwargs
    ):
        '''DQN
        Variants: Double / Dualing / N-step DQN

        The implementation mainly follows its originated paper
        `Deep Reinforcement Learning with Double Q-learning` by Hasselt et al.
        `Dueling Network Architectures for Deep Reinforcement Learning` 
        by Wang et al.

        The first argument `env` can be `None` for delayed model setup. You
        should call `set_env()` then call `setup_model()` to manually setup
        the model.

        Args:
            env (gym.Env): Training environment. Can be `None`
            learning_rate (float, optional): Learning rate can be a float or
                a ub.sche.Scheduler. Defaults to 3e-4.
            buffer_size (int, optional): Replay buffer capacity. Defaults to 
                int(1e6).
            multi_step (int, optional): N-step learning. Defaults to 1.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            tau (float, optional): Soft update parameter. Defaults to 1.0.
            reg_coef (float, optional): Regularization coefficient. Defaults to 0.0.
            clipnorm (float, optional): Gradient clip. Defaults to 0.5.
            huber (bool, optional): Whether to use Huber loss or MSE loss. Defaults to True.
            huber_rate (float, optional): Rate of quadratic and linear in huber loss.
                Defaults to 0.1.
            explore_rate (float, optional): Random exploration rate (epsiln greedy).
                It can be a float for constant exploration or a ub.sche.Scheduler. 
                Defaults to 0.3.
            prioritized (bool, optional): Whether to use prioritized replay buffer.
                Defaults to False.
            prio_alpha (float, optional): Prioritization exponent. Defaults to 0.6.
            prio_beta: (float, optional): Importance sampling exponent, can be a float
                or a ub.sche.Scheduler. Defaults to 0.4.
            dueling (bool, optional): Whether to enable dueling DQN. Defaults to False.
            force_mlp (bool, optional): Force to use MlpNet as the feature extractors. Defaults to False.
            mlp_units (list, optional): Number of units for each mlp layer. Defaults to [64, 64].
            n_steps (int, optional): Number of steps to collect rollouts in
                each epoch. Defaults to 4.
            n_gradsteps (int, optional): Number of gradients in each epoch. Defaults to 1.
            warmup_steps (int, optional): Number of steps in warming up stage. Defaults to int(1e4).
            batch_size (int, optional): Training batch size. Defaults to 128.
        '''        
        super().__init__(
            env          = env,
            n_steps      = n_steps,
            n_gradsteps  = n_gradsteps,
            warmup_steps = warmup_steps,
            batch_size   = batch_size,
            **kwargs
        )
        self.learning_rate = learning_rate
        self.buffer_size   = buffer_size
        self.multi_step    = multi_step
        self.gamma         = gamma
        self.tau           = tau
        self.reg_coef      = reg_coef
        self.clipnorm      = clipnorm
        self.explore_rate  = explore_rate
        self.huber         = huber
        self.huber_rate    = huber_rate
        self.prioritized   = prioritized
        self.prio_alpha    = prio_alpha
        self.prio_beta     = prio_beta
        self.dueling       = dueling
        self.force_mlp     = force_mlp
        self.mlp_units     = mlp_units
        # initialize objects
        self.buffer    = None
        self.agent     = None
        self.agent_tar = None
        self.optimizer = None

        if env is not None:
            self.setup()

    # --- setup group ---
    def _setup_buffer(self):
        '''Setup buffers
        Override this method to customize to your replay buffer
        '''
        self.buffer  = ub.data.ReplayBuffer(self.buffer_size)
        if self.prioritized:
            self.sampler = ub.data.PriorSampler(self.buffer, self.prio_alpha)
        else:
            self.sampler = ub.data.UniformSampler(self.buffer)

    def _setup_model(self):
        '''Setup models, agent, target agent
        Override this method to customize to your model
        '''
        self.agent = Agent(
            self.observation_space,
            self.action_space,
            dueling   = self.dueling,
            force_mlp = self.force_mlp,
            mlp_units = self.mlp_units
        )
        self.agent_tar = Agent(
            self.observation_space,
            self.action_space,
            dueling   = self.dueling,
            force_mlp = self.force_mlp,
            mlp_units = self.mlp_units
        )
        self.agent_tar.update(self.agent, polyak=1.0)

    def _setup_optimizer(self):
        '''Setup optimizers
        Override this method to customize to your optimizer
        '''
        # --- setup scheduler ---
        self.learning_rate = ub.sche.get_scheduler(
            self.learning_rate,
            state = self.state
        )
        self.explore_rate = ub.sche.get_scheduler(
            self.explore_rate,
            state = self.state
        )
        if self.prioritized:
            self.prio_beta = ub.sche.get_scheduler(
                self.prio_beta,
                state = self.state
            )
        # --- optimizers ---
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate = self.learning_rate(),
            clipnorm = self.clipnorm
        )
        # track weights
        ub.utils.set_optimizer_params(
            self.optimizer,
            self.agent.trainable_variables
        )

    def setup(self):
        '''(Override) Setup models, optimizers and schedulers for training'''
        self._setup_buffer()
        self._setup_model()
        self._setup_optimizer()

    # --- forward group ---
    @tf.function
    def call(self, inputs,
                   proc_obs: bool = True,
                   training: bool = True):
        '''(Override) See Agent.call'''
        return self.agent(
            inputs,
            proc_obs = proc_obs,
            training = training
        )

    def predict(self, inputs,
                      proc_obs: bool = True):
        '''(Override) See Agent.predict'''
        return self.agent.predict(
            inputs,
            proc_obs = proc_obs
        )

    # --- collect group ---
    def _collect_step(self, obs):
        '''(Override) Collect one sample
        
        Args:
            obs (np.ndarray, optional): Current observations
        
        Returns:
            np.ndarray: next observations
        '''
        if (self.is_warming_up() or
                np.random.rand() < self.explore_rate()):
            # sample random action
            act = np.asarray([s.sample() for s in self.env.action_spaces])
        else:
            act = self.predict(obs)
        # step environment
        next_obs, rew, done, infos = self.env.step(act)
        # Add data to buffer
        self.sampler.add(
            obs      = obs,
            act      = act,
            done     = done,
            rew      = rew
        )
        return next_obs

    # --- train group ---
    def td_error(self, obs, act, done, rew, next_obs):
        '''Compute N-step TD-error

        Y = r + gamma*N * max(Q(s'))
        td = Y - Q(s)

        Args:
            obs (tf.Tensor): Batch observations, (b, *obs_space.shape)
            act (tf.Tensor): Batch actions, (b,), tf.int64
            done (tf.Tensor): Batch dones, (b,), tf.float32
            rew (tf.Tensor): Batch rewards, (b,), tf.float32
            next_obs (tf.Tensor): Batch next observations, (b, *obs_space.shape)
        
        Returns:
            tf.Tensor: TD error, tf.float32
        '''
        act  = tf.cast(act, dtype=tf.int64)
        rew  = tf.cast(rew, dtype=tf.float32)
        done = tf.cast(done, dtype=tf.float32)
        # calculate target q value
        next_qs = self.agent_tar.call_value(next_obs)
        next_q  = tf.math.reduce_max(next_qs, axis=-1)
        gamma   = self.gamma ** self.multi_step
        y = rew + tf.stop_gradient((1.-done) * gamma * next_q)
        # calculate current q
        qs = self.agent.call_value(obs)
        q  = tf.gather(qs, indices=act, batch_dims=1)
        return y - q

    def td_loss(self, td: tf.Tensor, w: tf.Tensor=None):
        '''Compute TD loss (MSE or Huber)from TD error

        Huber:
            1/2 * TD^2, if |TD| <= delta
            delta * (|TD| - delta/2), otherwise
        MSE:
            TD^2

        Args:
            td (tf.Tensor): TD error, tf.float32
            w (tf.Tensor, optional): Loss weights, tf.float32. Defaults to None.
        
        Returns:
            tf.Tensor: td loss, tf.float32
        '''
        if w is None:
            w = tf.ones_like(td)
        w = tf.cast(w, dtype=tf.float32)
        if self.huber:
            # huber loss
            delta = tf.convert_to_tensor(self.huber_rate)
            losses = tf.where(tf.math.abs(td) <= delta, 
                0.5 * tf.math.square(td),
                delta * tf.math.abs(td) - 0.5 * tf.math.square(delta))
        else:
            # mse loss
            losses = tf.math.square(td)
        return tf.math.reduce_mean(losses * w)

    def reg_loss(self, var_list):
        '''L2 regularization loss

        Args:
            var_list (list): A list of variables need to be regularized.

        Returns:
            tf.Tensor: regularization loss.
        '''
        reg = tf.keras.regularizers.L2(self.reg_coef)
        return tf.math.add_n([reg(var) for var in var_list] + [0.0])

    @tf.function
    def _train_model(self, batch, w=None):
        '''Perform one gradient update
        
        Args:
            batch (dict): A sample of batch.
            w (tf.Tensor, optional): Loss weights.
        '''
        vars = self.agent.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(vars)
            td = self.td_error(**batch)
            # compute losses
            td_loss  = self.td_loss(td, w=w)
            reg_loss = self.reg_loss(vars)
            # aggregate losses
            loss = (td_loss + reg_loss)
        # perform gradients
        grads = tape.gradient(loss, vars)
        self.optimizer.apply_gradients(zip(grads, vars))
        return {
            'loss':     loss,
            'td_loss':  td_loss,
            'reg_loss': reg_loss
        }, tf.math.abs(td)

    def _sample_nstep_batch(self, batch_size):
        '''Sample a batch of replay with n-step rewards (if enabled)
        from replay buffer or prioritized replay buffer
        '''
        if self.prioritized:
            beta = self.prio_beta()
            batch = self.sampler(batch_size, beta=beta)
        else:
            batch = self.sampler(batch_size)
        # get nstep rewards and dones
        if self.multi_step > 1:
            n_batch = self.sampler.rel[:self.multi_step]
            # calculate N-step rewards
            batch['rew'] = ub.data.compute_nstep_rew(
                rew   = n_batch['rew'],
                done  = n_batch['done'],
                gamma = self.gamma
            )
            batch['done'] = np.any(n_batch['done'], axis=0)
        # Get the observation after N steps
        batch['next_obs'] = self.sampler.rel[self.multi_step]['obs']
        return batch

    def _train_step(self, batch_size):
        '''(Override) Train one step'''
        # update learning rate
        lr = tf.convert_to_tensor(self.learning_rate())
        self.optimizer.lr.assign(lr)
        # sample data and train model
        batch = self._sample_nstep_batch(batch_size)
        losses, td = self._train_model(batch, batch.pop('w', None))
        if self.prioritized:
            # update sample weights
            self.sampler.update(w=td)
        return losses

    def _update_target(self):
        '''(Override) Update target networks'''
        self.agent_tar.update(self.agent, polyak=self.tau)

    def get_config(self):
        config = super().get_config()
        config.update({
            'learning_rate': self.learning_rate,
            'buffer_size':   self.buffer_size,
            'multi_step':    self.multi_step,
            'gamma':         self.gamma,
            'tau':           self.tau,
            'reg_coef':      self.reg_coef,
            'clipnorm':      self.clipnorm,
            'explore_rate':  self.explore_rate,
            'huber':         self.huber,
            'huber_rate':    self.huber_rate,
            'prioritized':   self.prioritized,
            'prio_alpha':    self.prio_alpha,
            'prio_beta':     self.prio_beta,
            'dueling':       self.dueling,
            'force_mlp':     self.force_mlp,
            'mlp_units':     self.mlp_units,
        })
        return config