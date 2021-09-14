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

class DiagGaussianPolicyNet(ub.nets.DiagGaussianPolicyNet):
    '''Override Diagonal Gaussian mixture policy net
    with state-independent covariance
    '''
    support_spaces = [gym.spaces.Box]
    def create_logstd_model(self):
        '''(Override) Logstd model'''
        return ub.nets.Constant(tf.Variable(
            np.zeros((self.action_dims,), dtype=np.float32),
            name = 'logstd'
        ))

class PolicyNet(ub.nets.PolicyNet):
    '''Policy network'''
    def __init__(self, action_space,
                       net: tf.keras.Model = None,
                       squash = False, # disabled
                       **kwargs):
        super().__init__(action_space, squash=False, net=net, **kwargs)

    def call(self, inputs, training=True):
        '''Forward policy net

        Args:
            inputs (tf.Tensor): Input tensors, shape (b, *)
            training (bool, optional): Training mode. Defaults to True.

        Returns:
            Distribution: Action distributions.
        '''
        return super().call(inputs, training=training)

    def create_gaussian_policy(self):
        '''(Override) Policy for Box action space'''
        return DiagGaussianPolicyNet(self.action_space, squash=False)

class ValueNet(ub.nets.ValueNet):
    '''Value net'''
    def __init__(self, net: tf.keras.Model=None, **kwargs):
        super().__init__(net=net, **kwargs)

    def call(self, inputs, training=True):
        '''Forward value net

        Args:
            inputs (tf.Tensor): Input tensors, shape (b, *).
            training (bool, optional): Training mode. Defaults to True.

        Returns:
            tf.Tensor: value predictions, (Default) shape (b, 1)
        '''
        return super().call(inputs, training=training)

    def create_value_model(self):
        '''(Override) Value model arch'''
        return tf.keras.layers.Dense(1)

# === Agent ===

class Agent(ub.base.BaseAgent):
    support_obs_spaces = ub.utils.spaces.All
    support_act_spaces = [gym.spaces.Box, gym.spaces.Discrete]
    def __init__(self, observation_space, 
                       action_space, 
                       share_net: bool = False,
                       force_mlp: bool = False,
                       mlp_units: list = [64, 64],
                       **kwargs):
        '''PPO Agent

        The `observation_space` and `action_space` can be `None` for delayed
        model setup. You should call `set_spaces()` then call `setup()` to 
        manually setup the model.

        Args:
            observation_space (gym.Space): The observation space of the env.
                Set to None for delayed setup.
            action_space (gym.Space): The action space of the env. Set to None 
                for delayed setup.
            share_net (bool, optional): Whether to share the feature extractor 
                between policy and value network. Defaults to False.
            force_mlp (bool, optional): Force to use mlp as feature extractors.
                Default to False.
            mlp_units (list, optional): Number of units for each mlp layer. 
                Defaults to [64, 64]
        '''
        super().__init__(observation_space, action_space, **kwargs)

        self.share_net = share_net
        self.force_mlp = force_mlp
        self.mlp_units = mlp_units

        self.policy = None
        self.value  = None

        if (observation_space is not None) and (action_space is not None):
            self.setup()

    # --- setup group ---
    def _setup_model(self):
        '''Setup models
        Override this method to customize nets
        '''
        # AwesomeNet automatically handles complex observations
        # e.g. Dict or Tuple and creates either NatureCnn or 
        # MlpNet for each space
        create_net_fn = lambda: ub.nets.AwesomeNet(
            input_space = self.observation_space,
            force_mlp   = self.force_mlp,
            mlp_units   = self.mlp_units
        )
        # create base nets, feature extractors
        if self.share_net:
            # share base net
            base_net_p = create_net_fn()
            base_net_v = base_net_p
        else:
            # create separate nets
            base_net_p = create_net_fn()
            base_net_v = create_net_fn()
        # create policy, value nets
        self.policy = PolicyNet(self.action_space, base_net_p)
        self.value  = ValueNet(base_net_v)

    def setup(self):
        '''(Override) Setup agent model'''
        self._setup_model()
        # construct network params
        inputs = ub.utils.input_tensor(self.observation_space)
        self.call_policy(inputs, proc_obs=True)
        self.call_value(inputs, proc_obs=True)

    # --- forwrd group ---
    def call_policy(self, inputs, proc_obs=True, training=True):
        '''Forward policy net'''
        if proc_obs:
            # cast and normalize inputs (eg. image in uint8)
            inputs = self.proc_observation(inputs)
        return self.policy(inputs, training=training)

    def call_value(self, inputs, proc_obs=True, training=True):
        '''Forward value net'''
        if proc_obs:
            # cast and normalize inputs (eg. image in uint8)
            inputs = self.proc_observation(inputs)
        outputs = self.value(inputs, training=training)
        return tf.math.reduce_mean(outputs, axis=-1)

    @tf.function
    def call(self, inputs,
                   proc_obs: bool = True,
                   proc_act: bool = False,
                   det:      bool = False,
                   training: bool = True):
        '''(Override) Forward agent

        Args:
            inputs (tf.Tensor): Batch observations in shape 
                (b, *obs_space.shape), tf.float32
            proc_obs (bool, optional): Preprocess observations. Default to True.
            proc_act (bool, optional): Postprocess actions. Default to False.
            det (bool, optional): Deterministic actions. Defaults to False.
            training (bool, optional): Training mode. Defaults to True.

        Returns:
            tf.Tensor: Predicted actions, shape (b, *act_space.shape)
            tf.Tensor: Predicted values, shape (b,)
            tf.Tensor: Action log likelihoods, shape (b,)
        '''
        # forward actor
        distrib = self.call_policy(inputs, proc_obs=proc_obs, training=training)
        values  = self.call_value(inputs, proc_obs=proc_obs, training=training)
        # predict actions
        if det:
            actions = distrib.mode()
        else:
            actions = distrib.sample()
        # get log probabilities
        log_probs = distrib.log_prob(actions)
        if proc_act:
            # map actions to env's action space range
            actions = self.proc_action(actions)
        # return results
        return actions, values, log_probs
    
    def predict(self, inputs,
                      proc_obs: bool = True,
                      proc_act: bool = True,
                      det:      bool = True):
        return super().predict(
            inputs, 
            proc_obs=proc_obs,
            proc_act=proc_act,
            det=det
        )
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'share_net':         self.share_net,
            'force_mlp':         self.force_mlp,
            'mlp_units':         self.mlp_units,
        })
        return config

class PPO(ub.base.OnPolicyModel):
    support_obs_spaces = ub.utils.spaces.All
    support_act_spaces = [gym.spaces.Box, gym.spaces.Discrete]
    def __init__(self,
        env,
    # --- hyper parameters ---
        learning_rate: float = 3e-4,
        gamma:         float = 0.99,
        gae_lambda:    float = 0.95,
        policy_clip:   float = 0.2,
        value_clip:    float = None,
        dual_clip:     float = None,
        ent_coef:      float = 0.01,
        vf_coef:       float = 0.5,
        reg_coef:      float = 0.0,
        clipnorm:      float = 0.5,
        target_kl:     float = None,
    # --- architecture parameters ---
        share_net:      bool = False,
        force_mlp:      bool = False,
        mlp_units:      list = [64, 64],
    # --- other parameters ---
        n_steps:         int = 256,
        n_subepochs:     int = 4,
        batch_size:      int = 128,
        verbose:         int = 0,
        **kwargs
    ):
        '''Proximal policy optimization

        The implementation mainly follows its originated paper
        "Proximal Policy Optimization Algorithms" by Schulman et al.

        The `env` can be `None` for delayed model setup. You should call 
        `set_env()` then call `setup()` to manually setup the model.

        Args:
            env (ub.envs.BaseVecEnv): Training env. Must be a vectorized env.
                Set to `None` for delayed model setup.
            learning_rate (float, optional): Learning rate schedule. Can be a
                float or a ub.sche.Scheduler. Defaults to 3e-4.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            gae_lambda (float, optional): GAE smooth parameter. Defaults to 
                0.95.
            policy_clip (float, optional): Policy ratio clip. Defaults to 0.2.
            value_clip (float, optional): Target value clip. Defaults to None.
            dual_clip (float, optional): Policy dual clip from paper "Mastering
                Complex Control in MOBA Games with Deep Reinforcement Learning".
                Defaults to None.
            ent_coef (float, optional): Entropy loss coefficient. Defaults to 
                0.01.
            vf_coef (float, optional): Value loss coefficient. Defaults to 0.5.
            reg_coef (float, optional): Regularization coefficient. Default to 
                0.0.
            clipnorm (float, optional): Gradient clip. Defaults to 0.5.
            target_kl (float, optional): Target KL to early stop. Defaults 
                to None.
            share_net (bool, optional): Whether to share the feature extractor 
                between policy and value network. Defaults to False.
            force_mlp (bool, optional): Force to use mlp as feature extractors.
                Default to False.
            mlp_units (list, optional): Number of units for each mlp layer. 
                Defaults to [64, 64]
            n_steps (int, optional): Number of steps to collect rollouts in 
                each epoch. Defaults to 256.
            n_subepochs (int, optional): Number of subepochs per epoch. Defaults 
                to 4.
            batch_size (int, optional): Training batch size. Defaults to 128.
            verbose (int, optional): More logging. Defaults to 0.
        '''
        super().__init__(
            env         = env,
            n_steps     = n_steps,
            n_subepochs = n_subepochs,
            batch_size  = batch_size,
            verbose     = verbose,
            **kwargs
        )
        self.learning_rate = learning_rate
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.policy_clip   = policy_clip
        self.value_clip    = value_clip
        self.dual_clip     = dual_clip
        self.ent_coef      = ent_coef
        self.vf_coef       = vf_coef
        self.reg_coef      = reg_coef
        self.clipnorm      = clipnorm
        self.target_kl     = target_kl
        self.share_net     = share_net
        self.force_mlp     = force_mlp
        self.mlp_units     = mlp_units
        # initialize objects
        self.buffer    = None
        self.agent     = None
        self.optimizer = None

        if env is not None:
            self.setup()
    
    # --- setup group ---
    def _setup_buffer(self):
        '''Setup buffers
        Override this method to customize to your replay byffer
        '''
        self.buffer = ub.data.DynamicBuffer()
        self.sampler = ub.data.PermuteSampler(self.buffer)
        
    def _setup_model(self):
        '''Setup models, agents
        Override this method to customize to you model
        '''
        self.agent = Agent(
            self.observation_space,
            self.action_space,
            share_net = self.share_net,
            force_mlp = self.force_mlp,
            mlp_units = self.mlp_units
        )

    def _setup_optimizer(self):
        '''Setup optimizers
        Override this method to customize to your optimizer
        '''
        # --- setup scheduler ---
        self.learning_rate = ub.sche.get_scheduler(
            self.learning_rate,
            state = self.state
        )
        # --- optimizers ---
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate = self.learning_rate(),
            clipnorm      = self.clipnorm
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
                   proc_act: bool = False,
                   det:      bool = False,
                   training: bool = True):
        '''(Override) See Agent.call'''
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
        '''(Override) See Agent.predict'''
        return self.agent.predict(
            inputs,
            proc_obs = proc_obs,
            proc_act = proc_act,
            det      = det
        )
    
    # --- collect group ---
    def _collect_step(self, obs):
        '''(Override) Collect one sample

        Args:
            obs (np.ndarray, optional): Current observations.
        
        Returns:
            np.ndarray: next observations
        '''
        # sample actions from policy
        rawact, values, log_probs = self.call(obs, proc_act=False, det=False)
        act = self.agent.proc_action(rawact)
        # convert to np.ndarray
        rawact    = np.asarray(rawact)
        act       = np.asarray(act)
        values    = np.asarray(values)    # (b,)
        log_probs = np.asarray(log_probs) # (b,)
        # step environment
        next_obs, rew, done, infos = self.env.step(act)
        # add to buffer
        self.buffer.add(
            obs  = obs,
            act  = rawact,
            done = done,
            rew  = rew,
            val  = values,
            logp = log_probs,
        )
        return next_obs
    
    def run(self, steps, obs=None):
        '''(Override) Run rollouts collection procedure'''
        # reset buffer
        self.buffer.reset()
        # collect rollouts
        obs = self.collect(steps, obs=obs)
        # compute gae
        adv = ub.data.compute_advantage(
            rew        = self.buffer.data['rew'],
            val        = self.buffer.data['val'],
            done       = self.buffer.data['done'],
            gamma      = self.gamma,
            gae_lambda = self.gae_lambda
        )
        self.buffer.data['adv'] = adv
        # ready for sampling
        self.buffer.make()
        return obs

    # --- train group ---
    def policy_loss(self, adv, logp, old_logp):
        '''Compute policy loss (clipped surrogate loss)
        Dual-clip from: https://arxiv.org/abs/1912.09729
        Mastering Complex Control in MOBA Games with Deep Reinforcement Learning

        Args:
            adv (tf.Tensor): Batch advantages (GAE), shape (b,), tf.float32
            logp (tf.Tensor): Batch action log likelihoods, shape (b,),
                tf.float32
            old_logp (tf.Tensor): Batch old action log likelihoods, shape 
                (b,), tf.float32

        Returns:
            tf.Tensor: policy loss, tf.float32
        '''
        adv = (adv - tf.math.reduce_mean(adv)) / tf.math.reduce_std(adv)
        # policy ratio
        ratio = tf.exp(logp - old_logp)
        # clipped surrogate loss
        loss1 = adv * ratio
        loss2 = adv * tf.clip_by_value(ratio, 1.-self.policy_clip,
                                              1.+self.policy_clip)
        if self.dual_clip is not None:
            clip1 = tf.minimum(loss1, loss2)
            clip2 = tf.maximum(clip1, self.dual_clip * adv)
            pi_loss = tf.where(adv < 0., clip2, clip1)
        else:
            pi_loss = tf.minimum(loss1, loss2)
        return -tf.math.reduce_mean(pi_loss)

    def value_loss(self, adv, val, old_val):
        '''Compute value loss

        Y = A(s) + V_old(s)
        loss = (V(s)-Y)^2

        Args:
            adv (tf.Tensor): Batch advantages, shape (b,)
            val (tf.Tensor): Batch value predictions, shape (b,)
            old_val (tf.Tensor): Batch old value predictions, shape (b,)

        Returns:
            tf.Tensor: value loss, shape (b,)
        '''
        y = adv + old_val
        if self.value_clip is None:
            vf_loss = tf.math.square(val-y)
        else:
            v = old_val + tf.clip_by_value(val-old_val, -self.value_clip,
                                                         self.value_clip)
            loss1 = tf.math.square(val-y)
            loss2 = tf.math.square(v-y)
            vf_loss = tf.maximum(loss1, loss2)
        return tf.math.reduce_mean(vf_loss)

    def reg_loss(self, var_list):
        '''Regularization loss

        Args:
            var_list (list): A list of variables need to be regularized.

        Returns:
            tf.Tensor: regularization loss.
        '''
        reg = tf.keras.regularizers.L2(self.reg_coef)
        return tf.math.add_n([reg(var) for var in var_list] + [0.0])

    @tf.function
    def _train_model(self, batch):
        '''Perform one gradient update'''
        vars = self.agent.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(vars)
            # forward
            dist = self.agent.call_policy(batch['obs'], training=True)
            val  = self.agent.call_value(batch['obs'], training=True)
            logp = dist.log_prob(batch['act'])
            ent  = dist.entropy()
            kl   = 0.5 * tf.math.reduce_mean(
                            tf.math.square(batch['logp'] - logp))
            # compute policy/value loss
            pi_loss  = self.policy_loss(batch['adv'], logp, batch['logp'])
            vf_loss  = self.value_loss(batch['adv'], val, batch['val'])
            ent_loss = -tf.math.reduce_mean(ent)
            reg_loss = self.reg_loss(vars)
            # aggregate losses
            loss = (pi_loss + self.vf_coef * vf_loss 
                            + self.ent_coef * ent_loss + reg_loss)
        # perform gradients
        grads = tape.gradient(loss, vars)
        self.optimizer.apply_gradients(zip(grads, vars))
        return {
            'loss':     loss,
            'pi_loss':  pi_loss,
            'vf_loss':  vf_loss,
            'ent_loss': ent_loss,
            'reg_loss': reg_loss,
            'kl':       kl
        }, kl

    def _train_step(self, batch):
        '''(Override) Train one step'''
        # update learning rate
        lr = tf.convert_to_tensor(self.learning_rate())
        self.optimizer.lr.assign(lr)
        # train model
        losses, kl = self._train_model(batch)
        return losses, kl

    def train(self, batch_size, subepochs, **kwargs):
        '''(Override) Train one epoch'''
        all_losses = []
        for subepoch in range(subepochs):
            all_kls = []
            # train one subepoch
            for batch in self.sampler(batch_size):
                # train one step
                losses, kl = self._train_step(batch)
                all_losses.append(losses)
                all_kls.append(kl)
                # update state
                self.num_gradsteps += 1
            # update states
            self.num_subepochs += 1
            # compute average kl for early stop
            if self.target_kl is not None:
                kl = np.mean(np.hstack(np.asarray(all_kls)))
                if kl > 1.5 * self.target_kl:
                    self.LOG.warning(f'Early stopping at epoch {self.num_epochs} '
                        f'due to reaching max kl: {kl:.6f}')
                    break
        # aggregate losses
        all_losses = ub.utils.flatten_dicts(all_losses)
        m_losses = {}
        for name, losses in all_losses.items():
            m_losses[name] = np.mean(np.hstack(np.asarray(losses)))
        return m_losses

    def get_config(self):
        config = super().get_config()
        config.update({
            'learning_rate': self.learning_rate,
            'gamma':         self.gamma,
            'gae_lambda':    self.gae_lambda,
            'policy_clip':   self.policy_clip,
            'value_clip':    self.value_clip,
            'dual_clip':     self.dual_clip,
            'ent_coef':      self.ent_coef,
            'vf_coef':       self.vf_coef,
            'reg_coef':      self.reg_coef,
            'clipnorm':      self.clipnorm,
            'target_kl':     self.target_kl,
            'share_net':     self.share_net,
            'force_mlp':     self.force_mlp,
            'mlp_units':     self.mlp_units
        })
        return config