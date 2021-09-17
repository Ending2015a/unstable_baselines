# --- built in ---
import os
import sys
import time
import logging
import unittest
import tempfile

# --- 3rd party ---
import gym
import numpy as np
import tensorflow as tf 
tf.config.run_functions_eagerly(True)
# we only test on cpu
tf.config.set_visible_devices([], 'GPU')

from parameterized import parameterized

# --- my module ---
from unstable_baselines.lib import utils as ub_utils
from unstable_baselines.lib import sche as ub_sche
from unstable_baselines.lib import data as ub_data
from unstable_baselines.lib import nets as ub_nets
from unstable_baselines.lib.envs import vec as ub_vec
from unstable_baselines.algo.dqn import model as dqn_model
from test.utils import TestCase
from test.test_lib.test_envs.utils import FakeImageEnv, FakeContinuousEnv

class TestDQNModel(TestCase):
    def assertVariables(self, tar_list, var_list):
        self.assertEqual(len(tar_list), len(var_list))
        for tar, var in zip(tar_list, var_list):
            self.assertAllClose(tar, var)
    
    def test_q_net(self):
        action_dims = 3
        batch_size = 2
        input_dims = 16
        qnet = dqn_model.QNet(action_dims)
        inputs = np.zeros((batch_size, input_dims), dtype=np.float32)
        outputs = qnet(inputs)
        self.assertArrayEqual((batch_size, action_dims), outputs.shape)
        self.assertEqual(2, len(qnet.trainable_variables))

    def test_q_net_w_net(self):
        action_dims = 4
        batch_size = 2
        w, h, c = 64, 64, 3
        net = ub_nets.NatureCnn()
        qnet = dqn_model.QNet(action_dims, net=net)
        inputs = np.zeros((batch_size, h, w, c), dtype=np.float32)
        outputs = qnet(inputs)
        self.assertArrayEqual((batch_size, action_dims), outputs.shape)
        self.assertEqual(8+2, len(qnet.trainable_variables))

    def test_q_net_dueling(self):
        action_dims = 3
        batch_size = 2
        input_dims = 16
        qnet = dqn_model.QNet(action_dims, dueling=True)
        inputs = np.zeros((batch_size, input_dims), dtype=np.float32)
        outputs = qnet(inputs)
        self.assertArrayEqual((batch_size, action_dims), outputs.shape)
        self.assertEqual(4, len(qnet.trainable_variables))

    @parameterized.expand([
        (False,),
        (True,)
    ])
    def test_dqn_agent_image_obs(self, dueling):
        batch_size = 2
        action_dims = 16
        obs_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), 
                                    dtype=np.uint8)
        act_space = gym.spaces.Discrete(action_dims)
        agent = dqn_model.Agent(obs_space, act_space, dueling=dueling)
        inputs = np.zeros((batch_size,)+obs_space.shape, dtype=obs_space.dtype)
        # test call_value
        outputs = agent.call_value(inputs)
        self.assertArrayEqual((batch_size, action_dims), outputs.shape)
        # test call
        actions, values = agent(inputs)
        self.assertTrue(np.all(actions < action_dims))
        self.assertArrayEqual((batch_size,), actions.shape)
        self.assertArrayEqual((batch_size,), values.shape)
        # test predict
        inputs = np.zeros(obs_space.shape, dtype=obs_space.dtype)
        outputs = agent.predict(inputs)
        self.assertTrue(outputs < action_dims)
        self.assertArrayEqual([], outputs.shape)
        # test get_config
        agent.get_config()

    @parameterized.expand([
        (gym.spaces.Discrete(16),),
        (gym.spaces.MultiBinary(16),),
        (gym.spaces.MultiDiscrete([4, 6]),)
    ])
    def test_dqn_agent_setup_non_box_obs(self, obs_space):
        act_space = gym.spaces.Discrete(3)
        # no dueling, mlp
        agent = dqn_model.Agent(obs_space, act_space, force_mlp=False, 
                                dueling=False, mlp_units=[64, 64, 64])
        self.assertTrue(agent.value is not None)
        # mlp(3) + value
        self.assertEqual(6+2, len(agent.trainable_variables))
        # dueling, mlp
        agent = dqn_model.Agent(obs_space, act_space, force_mlp=False, 
                                dueling=True, mlp_units=[64, 64, 64])
        self.assertTrue(agent.value is not None)
        # mlp(3) + value
        self.assertEqual(6+2+2, len(agent.trainable_variables))

    @parameterized.expand([
        (gym.spaces.Dict({
            'sp1': gym.spaces.Box(low=0, high=255, shape=(64, 64, 3),
                                    dtype=np.uint8),
            'sp2':gym.spaces.Discrete(16)
        }),),
        (gym.spaces.Tuple((
            gym.spaces.Box(low=0, high=255, shape=(64, 64, 3),
                                    dtype=np.uint8),
            gym.spaces.Discrete(16)
        )),)
    ])
    def test_dqn_agent_setup_nested_obs(self, obs_space):
        act_space = gym.spaces.Discrete(3)
        # no share net, mlp
        agent = dqn_model.Agent(obs_space, act_space, dueling=False,
                            force_mlp=False, mlp_units=[64, 64, 64])
        self.assertTrue(agent.value is not None)
        # mlp(3) + nature_cnn + value
        self.assertEqual(6+8+2, len(agent.trainable_variables))
        # share net
        agent = dqn_model.Agent(obs_space, act_space, dueling=True,
                            force_mlp=False, mlp_units=[64, 64, 64])
        self.assertTrue(agent.value is not None)
        # mlp(3) + nature_cnn + value(dueling)
        self.assertEqual(6+8+2+2, len(agent.trainable_variables))

    def test_dqn_agent_delayed_setup(self):
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(64,),
                                dtype=np.float32)
        act_space = gym.spaces.Discrete(16)
        agent = dqn_model.Agent(None, None, force_mlp=False, mlp_units=[64, 64])
        self.assertTrue(agent.value is None)
        self.assertTrue(agent.observation_space is None)
        self.assertTrue(agent.action_space is None)
        agent.set_spaces(obs_space, act_space)
        self.assertTrue(agent.observation_space is obs_space)
        self.assertTrue(agent.action_space is act_space)
        agent.setup()
        self.assertTrue(agent.value is not None)
    
    def test_dqn_setup_image_obs(self):
        envs = [FakeImageEnv() for _ in range(3)]
        env = ub_vec.VecEnv(envs)
        model = dqn_model.DQN(env)
        self.assertEqual(3, model.n_envs)
        self.assertTrue(model.observation_space is not None)
        self.assertTrue(model.action_space is not None)
        # nature_cnn + value
        self.assertEqual((8+2)*2, len(model.trainable_variables))
        # test dueling
        model = dqn_model.DQN(env, dueling=True)
        # nature_cnn + value(dueling)
        self.assertEqual((8+4)*2, len(model.trainable_variables))

    def test_dqn_delayed_setup(self):
        model = dqn_model.DQN(None)
        self.assertTrue(model.observation_space is None)
        self.assertTrue(model.action_space is None)
        self.assertTrue(model.agent is None)
        envs = [FakeImageEnv() for _ in range(3)]
        env = ub_vec.VecEnv(envs)
        model.set_env(env)
        model.setup()
        self.assertTrue(model.observation_space is not None)
        self.assertTrue(model.action_space is not None)
        self.assertEqual((8+2)*2, len(model.trainable_variables))

    def test_dqn_call_predict(self):
        envs = [FakeImageEnv() for _ in range(3)]
        env = ub_vec.VecEnv(envs)
        model = dqn_model.DQN(env)
        batch_size = 3
        obs_space = env.observation_space
        act_space = env.action_space
        act_dims = act_space.n
        obs = np.asarray([obs_space.sample() for _ in range(batch_size)])
        # test call
        act, val = model(obs, proc_obs=True)
        self.assertArrayEqual((batch_size,), act.shape)
        self.assertArrayEqual((batch_size,), val.shape)
        # test predict
        act = model.predict(obs_space.sample())
        self.assertArrayEqual([], act.shape)
        # test dueling
        model = dqn_model.DQN(env, dueling=True)
        act, val = model(obs, proc_obs=True)
        # test call
        self.assertArrayEqual((batch_size,), act.shape)
        self.assertArrayEqual((batch_size,), val.shape)
        # test predict
        act = model.predict(obs_space.sample())
        self.assertArrayEqual([], act.shape)

    def test_dqn_run(self):
        n_envs = 3
        warmup_steps = 50
        buffer_size = 90
        envs = [FakeImageEnv() for _ in range(n_envs)]
        env = ub_vec.VecEnv(envs)
        model = dqn_model.DQN(
            env, 
            buffer_size=buffer_size, 
            warmup_steps=warmup_steps
        )
        obs_shape = env.observation_space.shape
        act_shape = env.action_space.shape
        n_samples = 100
        n_slots = buffer_size // n_envs
        model.run(n_samples)
        buf = model.buffer
        self.assertEqual(n_slots*n_envs, len(buf))
        self.assertTrue(buf.ready_for_sample)
        self.assertTrue(buf.isfull)
        # test buffer contents
        self.assertArrayEqual((n_slots, n_envs, *obs_shape),
                            buf.data['obs'].shape)
        self.assertArrayEqual((n_slots, n_envs, *act_shape),
                            buf.data['act'].shape)
        self.assertArrayEqual((n_slots, n_envs),
                            buf.data['rew'].shape)
        self.assertArrayEqual((n_slots, n_envs),
                            buf.data['done'].shape)

    @parameterized.expand([
        (True,),
        (False,)
    ])
    def test_dqn_train(self, huber):
        n_envs = 3
        envs = [FakeImageEnv() for _ in range(n_envs)]
        env = ub_vec.VecEnv(envs)
        model = dqn_model.DQN(env, huber=huber)
        n_samples = 10
        batch_size = 10
        n_gradsteps = 4
        n_subepochs = 1
        target_update = 2
        exp_gradsteps = n_subepochs * n_gradsteps
        model.run(n_samples)
        model.train(batch_size, n_subepochs, n_gradsteps, target_update)
        self.assertEqual(exp_gradsteps, model.num_gradsteps)
        self.assertEqual(n_subepochs, model.num_subepochs)

    @parameterized.expand([
        (True,),
        (False,)
    ])
    def test_dqn_td_error(self, huber):
        n_envs = 3
        envs = [FakeImageEnv() for _ in range(n_envs)]
        env = ub_vec.VecEnv(envs)
        model = dqn_model.DQN(env, huber=huber)
        obs = env.observation_space.sample()
        obs = obs.reshape(1, *obs.shape)
        next_obs = env.observation_space.sample()
        next_obs = next_obs.reshape(1, *next_obs.shape)
        act = env.action_space.sample()
        act = np.asarray([act], dtype=np.int64)
        done = np.asarray([False], dtype=np.bool_)
        rew = np.asarray([1.0], dtype=np.float32)
        # test td error
        td = model.td_error(obs, act, done, rew, next_obs)
        self.assertArrayEqual((1,), td.shape)
        self.assertFalse(np.all(np.isnan(td)))
        # test td loss
        loss = model.td_loss(td)
        self.assertArrayEqual([], loss.shape)
        self.assertFalse(np.all(np.isnan(loss)))

    def test_dqn_reg_loss(self):
        n_envs = 3
        envs = [FakeImageEnv() for _ in range(n_envs)]
        env = ub_vec.VecEnv(envs)
        model = dqn_model.DQN(env)
        loss = model.reg_loss(model.agent.trainable_variables)
        self.assertArrayEqual([], loss.shape)
        self.assertFalse(np.all(np.isnan(loss)))

    @parameterized.expand([
        (True,),
        (False,)
    ])
    def test_dqn_train_model(self, huber):
        n_envs = 3
        envs = [FakeImageEnv() for _ in range(n_envs)]
        env = ub_vec.VecEnv(envs)
        model = dqn_model.DQN(env, huber=huber)
        n_samples = 10
        batch_size = 2
        model.run(n_samples)
        samp = model.sampler
        batch = samp.sample(batch_size)
        batch['next_obs'] = samp.rel[1]['obs']
        losses, td = model._train_model(batch)
        for key, loss in losses.items():
            self.assertArrayEqual([], loss.shape)
            self.assertFalse(np.all(np.isnan(loss)))
        self.assertArrayEqual((batch_size,), td.shape)
        self.assertFalse(np.all(np.isnan(td)))
        self.assertTrue(np.all(td >= 0))

    def test_dqn_sample_nstep_batch(self):
        ub_utils.set_seed(1)
        n_envs = 3
        gamma = 0.99
        multi_step = 2
        envs = [FakeImageEnv() for _ in range(n_envs)]
        env = ub_vec.VecEnv(envs)
        model = dqn_model.DQN(env, multi_step=multi_step, gamma=gamma)
        n_samples = 10
        batch_size = 1
        model.run(n_samples)
        samp = model.sampler
        batch = model._sample_nstep_batch(batch_size)
        orig_batch = samp.rel[0]
        self.assertArrayEqual(orig_batch['obs'], batch['obs'])
        self.assertArrayEqual(orig_batch['act'], batch['act'])
        self.assertArrayNotEqual(orig_batch['rew'], batch['rew'])
        next_batch = samp.rel[1]
        # depends on random seed
        self.assertAllClose(orig_batch['rew']+gamma*next_batch['rew'],
                             batch['rew'], atol=1e-6)
        nnext_batch = samp.rel[2]
        self.assertArrayEqual(nnext_batch['obs'], batch['next_obs'])

    def test_dqn_save_load(self):
        n_envs = 3
        envs = [FakeImageEnv() for _ in range(n_envs)]
        env = ub_vec.VecEnv(envs)
        env.seed(1)
        ub_utils.set_seed(1)
        model = dqn_model.DQN(env, warmup_steps=5)
        n_samples = 10
        batch_size = 10
        model.run(n_samples)
        # train for some steps
        ub_utils.set_seed(2)
        batch = model.sampler(batch_size)
        batch['next_obs'] = model.sampler.rel[1]['obs']
        model._train_model(batch)
        with tempfile.TemporaryDirectory() as tempdir:
            save_path = tempdir
            # save & load model
            model.save(save_path)
            loaded_model = dqn_model.DQN.load(save_path)
        # check model setup
        self.assertTrue(loaded_model.agent is not None)
        self.assertTrue(loaded_model.buffer is not None)
        self.assertTrue(loaded_model.optimizer is not None)
        # check if config is correctly restored
        model_config = model.get_config()
        loaded_config = loaded_model.get_config()
        self.assertEqual(set(model_config.keys()), set(loaded_config.keys()))
        for key in model_config:
            self.assertEqual(model_config[key], loaded_config[key], key)
        # check if all network variables are correctly restored
        self.assertVariables(model.trainable_variables,
                    loaded_model.trainable_variables)
        # test optimizers
        # load optimizer params
        batches = []
        for i in range(3):
            batch = model.sampler(batch_size)
            batch['next_obs'] = model.sampler.rel[1]['obs']
            batches.append(batch)
        ub_utils.set_seed(1)
        for batch in batches:
            losses1, td1 = model._train_model(batch)
        for batch in batches:
            losses2, td2 = loaded_model._train_model(batch)
        # check if losses are matches
        self.assertEqual(set(losses1.keys()), set(losses2.keys()))
        for key in losses1.keys():
            self.assertEqual(losses1[key], losses2[key])
        self.assertAllClose(td1, td2)
        # check if vars are same
        self.assertVariables(model.trainable_variables,
                    loaded_model.trainable_variables)
        # check if params of the optimizers are same
        self.assertVariables(model.optimizer.variables(),
                    loaded_model.optimizer.variables())

    @parameterized.expand([
        (True,),
        (False,)
    ])
    def test_dqn_prioritized(self, huber):
        n_envs = 3
        ub_utils.set_seed(1)
        envs = [FakeImageEnv() for _ in range(n_envs)]
        env = ub_vec.VecEnv(envs)
        model = dqn_model.DQN(env, prioritized=True, warmup_steps=5)
        self.assertTrue(isinstance(model.prio_beta, ub_sche.Scheduler))
        self.assertTrue(isinstance(model.sampler, ub_data.PriorSampler))
        n_samples = 10
        batch_size = 10
        model.run(n_samples)
        res = model.sampler._weight_tree[:n_samples*n_envs]
        exp = np.ones_like(res, dtype=np.float32)
        self.assertArrayEqual(exp, res)
        model._train_step(batch_size)
        res = model.sampler._weight_tree[:n_samples*n_envs]
        self.assertArrayNotEqual(exp, res)
        self.assertTrue(np.all(res >= 0.0))

    @parameterized.expand([
        (False,),
        (True,),
    ])
    def test_dqn_cartpole(self, prioritized):
        with ub_utils.run_eagerly(False):
            ub_utils.set_seed(1)
            env = ub_vec.VecEnv([gym.make('CartPole-v0') for _ in range(10)])
            env.seed(1)
            eval_env = gym.make('CartPole-v0')
            eval_env.seed(0)
            model = dqn_model.DQN(
                env,
                buffer_size=20000,
                multi_step=3,
                prioritized=prioritized,
                learning_rate=1e-3,
                gamma=0.8,
                batch_size=128,
                n_steps=10
            ).learn(
                150000,
                target_update=100,
                verbose=1
            )
            # evaluate model
            results = model.eval(eval_env, 20, 200)
            metrics = model.get_eval_metrics(results)
            self.assertAllClose(200.0, metrics['mean-reward'])
            env.close()
            eval_env.close()