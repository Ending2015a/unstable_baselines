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
from unstable_baselines.lib import data as ub_data
from unstable_baselines.lib import nets as ub_nets
from unstable_baselines.lib import prob as ub_prob
from unstable_baselines.lib.envs import vec as ub_vec
from unstable_baselines.algo.ppo import model as ppo_model
from test.utils import TestCase
from test.test_lib.test_envs.utils import FakeImageEnv, FakeContinuousEnv


def legacy_gae(rew, val, done, gamma, lam):
    last_gae = 0
    next_nterm = 1.0 - done[-1]
    next_val = val[-1]
    adv = np.zeros_like(val)
    for t in reversed(range(len(done))):
        delta = rew[t] + gamma * next_val * next_nterm - val[t]
        last_gae = delta + gamma * lam * next_nterm * last_gae
        adv[t] = last_gae
        next_nterm = 1.0 - done[t]
        next_val = val[t]
    return adv

class TestPPOModel(TestCase):

    def assertVariables(self, tar_list, var_list):
        self.assertEqual(len(tar_list), len(var_list))
        for tar, var in zip(tar_list, var_list):
            self.assertAllClose(tar, var)

    def test_state_indep_diag_gaussian_policy_net(self):
        batch_size = 5
        obs_dim = 64
        act_dim = 16
        space = gym.spaces.Box(low=-1., high=-1., shape=(act_dim,),
                                dtype=np.float32)
        pi = ppo_model.DiagGaussianPolicyNet(space)
        inputs = tf.zeros((batch_size, obs_dim), dtype=tf.float32)
        dist = pi(inputs)
        self.assertTrue(isinstance(pi._logstd_model, ub_nets.Constant))
        self.assertEqual((act_dim,), pi._logstd_model.constant.shape)
        self.assertEqual(1, len(pi._logstd_model.trainable_variables))
        act = dist.sample()
        self.assertArrayEqual((batch_size, act_dim), act.shape)

    def test_state_indep_diag_gaussian_policy_net_multidim(self):
        batch_size = 5
        obs_dim = 64
        act_dims = (2, 16)
        space = gym.spaces.Box(low=-1., high=-1., shape=act_dims,
                                dtype=np.float32)
        pi = ppo_model.DiagGaussianPolicyNet(space)
        inputs = tf.zeros((batch_size, obs_dim), dtype=tf.float32)
        dist = pi(inputs)
        self.assertTrue(isinstance(pi._logstd_model, ub_nets.Constant))
        self.assertEqual((np.prod(act_dims),), pi._logstd_model.constant.shape)
        self.assertEqual(1, len(pi._logstd_model.trainable_variables))
        act = dist.sample()
        self.assertArrayEqual((batch_size, *act_dims), act.shape)

    def test_policy_net_categorical(self):
        batch_size = 5
        act_dim = 16
        obs_dim = 60
        action_space = gym.spaces.Discrete(act_dim)
        pi = ppo_model.PolicyNet(action_space)
        obs  = tf.zeros((batch_size, obs_dim), dtype=tf.float32)
        dist = pi(obs)
        self.assertTrue(isinstance(dist, ub_prob.Categorical))
        act = dist.sample()
        self.assertArrayEqual((batch_size,), act.shape)
        self.assertEqual(2, len(pi.trainable_variables))

    def test_policy_net_diag_gaussian(self):
        batch_size = 5
        act_dim = 16
        obs_dim = 60
        low = np.ones((act_dim,), dtype=np.float32) * -1.0
        high = np.ones((act_dim,), dtype=np.float32) * 1.0
        action_space = gym.spaces.Box(low=low, high=high)
        pi   = ppo_model.PolicyNet(action_space, squash=False)
        obs  = tf.zeros((batch_size, obs_dim), dtype=tf.float32)
        dist = pi(obs)
        self.assertTrue(isinstance(dist, ub_prob.MultiNormal))
        act = dist.sample()
        self.assertArrayEqual((batch_size, act_dim), act.shape)
        self.assertEqual(3, len(pi.trainable_variables))

    def test_value_net(self):
        batch_size = 5
        obs_dim = 16
        vf = ppo_model.ValueNet()
        inputs = tf.zeros((batch_size, obs_dim), dtype=tf.float32)
        outputs = vf(inputs)
        self.assertArrayEqual(outputs.shape, (batch_size, 1))

    def test_ppo_agent_setup_image_obs(self):
        obs_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3),
                                    dtype=np.uint8)
        act_space = gym.spaces.Discrete(16)
        # no share net, nature cnn
        agent = ppo_model.Agent(obs_space, act_space, share_net=False,
                            force_mlp=False)
        self.assertTrue(agent.policy is not None)
        self.assertTrue(agent.value is not None)
        # nature_cnn + nature_cnn + policy + value
        self.assertEqual(8+8+2+2, len(agent.trainable_variables))
        # test share net
        agent = ppo_model.Agent(obs_space, act_space, share_net=True,
                            force_mlp=False)
        self.assertTrue(agent.policy is not None)
        self.assertTrue(agent.value is not None)
        # nature_cnn + policy + value
        self.assertEqual(8+2+2, len(agent.trainable_variables))
        # test force mlp
        agent = ppo_model.Agent(obs_space, act_space, share_net=False,
                            force_mlp=True, mlp_units=[64, 64, 64])
        self.assertTrue(agent.policy is not None)
        self.assertTrue(agent.value is not None)
        # mlp(3) + mlp(3) + policy + value
        self.assertEqual(6+6+2+2, len(agent.trainable_variables))

    def test_ppo_agent_setup_box_obs(self):
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(64,),
                                dtype=np.float32)
        act_space = gym.spaces.Box(low=-1, high=1, shape=(16,),
                                dtype=np.float32)
        # no share net, mlp
        agent = ppo_model.Agent(obs_space, act_space, share_net=False,
                            force_mlp=False, mlp_units=[64, 64, 64])
        self.assertTrue(agent.policy is not None)
        self.assertTrue(agent.value is not None)
        # mlp(3) + mlp(3) + policy + value
        self.assertEqual(6+6+3+2, len(agent.trainable_variables))
        # share net
        agent = ppo_model.Agent(obs_space, act_space, share_net=True,
                            force_mlp=False, mlp_units=[64, 64, 64])
        self.assertTrue(agent.policy is not None)
        self.assertTrue(agent.value is not None)
        # mlp(3) + mlp(3) + policy + value
        self.assertEqual(6+3+2, len(agent.trainable_variables))
    
    @parameterized.expand([
        (gym.spaces.Discrete(16),),
        (gym.spaces.MultiBinary(16),),
        (gym.spaces.MultiDiscrete([4, 6]),)
    ])
    def test_ppo_agent_setup_non_box_obs(self, obs_space):
        act_space = gym.spaces.Box(low=-1, high=1, shape=(16,),
                                dtype=np.float32)
        # no share net, mlp
        agent = ppo_model.Agent(obs_space, act_space, share_net=False,
                            force_mlp=False, mlp_units=[64, 64, 64])
        self.assertTrue(agent.policy is not None)
        self.assertTrue(agent.value is not None)
        # mlp(3) + mlp(3) + policy + value
        self.assertEqual(6+6+3+2, len(agent.trainable_variables))
        # share net
        agent = ppo_model.Agent(obs_space, act_space, share_net=True,
                            force_mlp=False, mlp_units=[64, 64, 64])
        self.assertTrue(agent.policy is not None)
        self.assertTrue(agent.value is not None)
        # mlp(3) + mlp(3) + policy + value
        self.assertEqual(6+3+2, len(agent.trainable_variables))

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
    def test_ppo_agent_setup_nested_obs(self, obs_space):
        act_space = gym.spaces.Box(low=-1, high=1, shape=(16,),
                                dtype=np.float32)
        # no share net, mlp
        agent = ppo_model.Agent(obs_space, act_space, share_net=False,
                            force_mlp=False, mlp_units=[64, 64, 64])
        self.assertTrue(agent.policy is not None)
        self.assertTrue(agent.value is not None)
        # mlp(3)*2 + nature_cnn*2 + policy + value
        self.assertEqual(6*2+8*2+3+2, len(agent.trainable_variables))
        # share net
        agent = ppo_model.Agent(obs_space, act_space, share_net=True,
                            force_mlp=False, mlp_units=[64, 64, 64])
        self.assertTrue(agent.policy is not None)
        self.assertTrue(agent.value is not None)
        # mlp(3) + mlp(3) + policy + value
        self.assertEqual(6+8+3+2, len(agent.trainable_variables))

    def test_ppo_agent_delayed_setup(self):
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(64,),
                                dtype=np.float32)
        act_space = gym.spaces.Box(low=-1, high=1, shape=(16,),
                                dtype=np.float32)
        agent = ppo_model.Agent(None, None, share_net=False,
                            force_mlp=False, mlp_units=[64, 64])
        self.assertTrue(agent.policy is None)
        self.assertTrue(agent.value is None)
        self.assertTrue(agent.observation_space is None)
        self.assertTrue(agent.action_space is None)
        agent.set_spaces(obs_space, act_space)
        self.assertTrue(agent.observation_space is obs_space)
        self.assertTrue(agent.action_space is act_space)
        agent.setup()
        self.assertTrue(agent.policy is not None)
        self.assertTrue(agent.value is not None)
        
    def test_ppo_agent_reset_spaces_not_match(self):
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(64,),
                                dtype=np.float32)
        act_space = gym.spaces.Box(low=-1, high=1, shape=(16,),
                                dtype=np.float32)
        agent = ppo_model.Agent(None, None, share_net=False,
                            force_mlp=False, mlp_units=[64, 64])
        agent.set_spaces(obs_space, act_space)
        other_space = gym.spaces.Box(low=0., high=1., shape=(64,),
                                dtype=np.float32)
        # space not match
        with self.assertRaises(RuntimeError):
            agent.set_spaces(other_space, act_space)
        with self.assertRaises(RuntimeError):
            agent.set_spaces(obs_space, other_space)
        
    def test_ppo_agent_set_spaces_not_supported(self):
        # ppo supported spaces
        box_space = gym.spaces.Box(low=-1, high=1, shape=(64,),
                                dtype=np.float32)
        discrete_space = gym.spaces.Discrete(6)
        # ppo non supported spaces
        multid_space = gym.spaces.MultiDiscrete([3, 5])
        multib_space = gym.spaces.MultiBinary(6)
        #
        ppo_model.Agent(box_space, box_space)
        ppo_model.Agent(box_space, discrete_space)
        ppo_model.Agent(discrete_space, discrete_space)
        ppo_model.Agent(multid_space, discrete_space)
        ppo_model.Agent(multib_space, discrete_space)
        with self.assertRaises(RuntimeError):
            ppo_model.Agent(box_space, multid_space)
        with self.assertRaises(RuntimeError):
            ppo_model.Agent(box_space, multib_space)
        
    def test_ppo_agent_proc_box(self):
        obs_space = gym.spaces.Box(low=0, high=255, shape=(64,),
                                dtype=np.uint8)
        act_space = gym.spaces.Box(low=-1, high=1, shape=(16,),
                                dtype=np.float32)
        agent = ppo_model.Agent(obs_space, act_space, share_net=False,
                            force_mlp=False, mlp_units=[64, 64])
        obs = obs_space.sample()
        res_obs = agent.proc_observation(obs)
        obs_norm = obs.astype(np.float32)/255.
        self.assertAllClose(obs_norm, res_obs)
        # action out of range (clip)
        act = np.linspace(-2, 2, num=16, dtype=np.float32)
        res_act = agent.proc_action(act)
        act_norm = np.clip(act, act_space.low, act_space.high)
        self.assertAllClose(act_norm, res_act)
        # action out of range (clip) (tensor)
        res_act = agent.proc_action(tf.convert_to_tensor(act))
        self.assertAllClose(act_norm, res_act)

    def test_ppo_agent_proc_discrete(self):
        obs_space = gym.spaces.Box(low=0, high=255, shape=(64,),
                                dtype=np.uint8)
        act_space = gym.spaces.Discrete(6)
        agent = ppo_model.Agent(obs_space, act_space, share_net=False,
                            force_mlp=False, mlp_units=[64, 64])
        obs = obs_space.sample()
        res_obs = agent.proc_observation(obs)
        obs_norm = obs.astype(np.float32)/255.
        self.assertAllClose(obs_norm, res_obs)
        # Do nothing
        act = act_space.sample()
        res_act = agent.proc_action(act)
        self.assertAllClose(act, res_act)
    
    def test_ppo_agent_proc_nested(self):
        obs1_space = gym.spaces.Box(low=0, high=255, shape=(64,),
                                    dtype=np.uint8)
        obs2_space = gym.spaces.Discrete(16)
        obs_space = gym.spaces.Dict({'obs1': obs1_space, 'obs2':obs2_space})
        act_space = gym.spaces.Discrete(6)
        agent = ppo_model.Agent(obs_space, act_space, share_net=False,
                            force_mlp=False, mlp_units=[64, 64])
        obs = obs_space.sample()
        res_obs = agent.proc_observation(obs)
        obs1_norm = obs['obs1'].astype(np.float32)/255.
        obs2_norm = np.zeros((16,), np.float32)
        obs2_norm[obs['obs2']] = 1.0
        self.assertAllClose(obs1_norm, res_obs['obs1'])
        self.assertAllClose(obs2_norm, res_obs['obs2'])

    def test_ppo_agent_call(self):
        batch_size = 3
        obs_dim = 64
        act_dim = 16
        obs_space = gym.spaces.Box(low=0, high=255, shape=(obs_dim,),
                                dtype=np.uint8)
        act_space = gym.spaces.Box(low=-1, high=1, shape=(act_dim,),
                                dtype=np.float32)
        agent = ppo_model.Agent(obs_space, act_space, share_net=False,
                            force_mlp=False, mlp_units=[64, 64])
        obs = np.asarray([obs_space.sample() for _ in range(batch_size)])
        act, val, logp = agent(obs, det=True)
        self.assertArrayEqual((batch_size, act_dim), act.shape)
        self.assertArrayEqual((batch_size,), val.shape)
        self.assertArrayEqual((batch_size,), logp.shape)
        act, val, logp = agent(obs, det=False)
        self.assertArrayEqual((batch_size, act_dim), act.shape)
        self.assertArrayEqual((batch_size,), val.shape)
        self.assertArrayEqual((batch_size,), logp.shape)

    def test_ppo_agent_predict(self):
        obs_dim = 64
        act_dim = 16
        obs_space = gym.spaces.Box(low=0, high=255, shape=(obs_dim,),
                                dtype=np.uint8)
        act_space = gym.spaces.Box(low=-1, high=1, shape=(act_dim,),
                                dtype=np.float32)
        agent = ppo_model.Agent(obs_space, act_space, share_net=False,
                            force_mlp=False, mlp_units=[64, 64])
        obs = obs_space.sample()
        act = agent.predict(obs, det=True)
        self.assertArrayEqual((act_dim,), act.shape)
        act = agent.predict(obs, det=False)
        self.assertArrayEqual((act_dim,), act.shape)
    
    def test_ppo_agent_predict_batch(self):
        batch_size = 3
        obs_dim = 64
        act_dim = 16
        obs_space = gym.spaces.Box(low=0, high=255, shape=(obs_dim,),
                                dtype=np.uint8)
        act_space = gym.spaces.Box(low=-1, high=1, shape=(act_dim,),
                                dtype=np.float32)
        agent = ppo_model.Agent(obs_space, act_space, share_net=False,
                            force_mlp=False, mlp_units=[64, 64])
        obs = np.asarray([obs_space.sample() for _ in range(batch_size)])
        act = agent.predict(obs, det=True)
        self.assertArrayEqual((batch_size, act_dim), act.shape)
        act = agent.predict(obs, det=False)
        self.assertArrayEqual((batch_size, act_dim), act.shape)

    def test_ppo_agent_save_load(self):
        batch_size = 3
        obs_dim = 64
        act_dim = 16
        obs_space = gym.spaces.Box(low=0, high=255, shape=(obs_dim,),
                                dtype=np.uint8)
        act_space = gym.spaces.Box(low=-1, high=1, shape=(act_dim,),
                                dtype=np.float32)
        agent = ppo_model.Agent(obs_space, act_space, share_net=True,
                            force_mlp=False, mlp_units=[64, 64, 64])
        obs = np.asarray([obs_space.sample() for _ in range(batch_size)])
        act, val, logp = agent(obs, det=True)
        with tempfile.TemporaryDirectory() as tempdir:
            save_path = tempdir
            agent.save(save_path)
            loaded_agent = ppo_model.Agent.load(save_path)
        self.assertEqual(agent.share_net, loaded_agent.share_net)
        self.assertEqual(agent.force_mlp, loaded_agent.force_mlp)
        self.assertEqual(agent.mlp_units, loaded_agent.mlp_units)
        self.assertEqual(agent.observation_space, loaded_agent.observation_space)
        self.assertEqual(agent.action_space, loaded_agent.action_space)
        for target_var, var in zip(agent.trainable_variables,
                                 loaded_agent.trainable_variables):
            self.assertAllClose(target_var, var)

    def test_ppo_setup_image_obs(self):
        envs = [FakeImageEnv() for _ in range(3)]
        env = ub_vec.VecEnv(envs)
        model = ppo_model.PPO(env)
        self.assertEqual(3, model.n_envs)
        self.assertTrue(model.observation_space is not None)
        self.assertTrue(model.action_space is not None)
        # nature_cnn + nature_cnn + policy + value
        self.assertEqual(8+8+2+2, len(model.trainable_variables))
        # test share net
        model = ppo_model.PPO(env, share_net=True)
        # nature_cnn + policy + value
        self.assertEqual(8+2+2, len(model.trainable_variables))
        # test force mlp
        model = ppo_model.PPO(env, share_net=False,
                            force_mlp=True, mlp_units=[64, 64, 64])
        # mlp(3) + mlp(3) + policy + value
        self.assertEqual(6+6+2+2, len(model.trainable_variables))

    def test_ppo_setup_non_image_obs(self):
        envs = [FakeContinuousEnv() for _ in range(3)]
        env = ub_vec.VecEnv(envs)
        # no share net, mlp
        model = ppo_model.PPO(env, mlp_units=[64, 64, 64])
        self.assertEqual(3, model.n_envs)
        self.assertTrue(model.observation_space is not None)
        self.assertTrue(model.action_space is not None)
        # mlp(3) + mlp(3) + policy + value
        self.assertEqual(6+6+3+2, len(model.trainable_variables))
        # share net
        model = ppo_model.PPO(env, share_net=True,
                            force_mlp=False, mlp_units=[64, 64, 64])
        # mlp(3) + mlp(3) + policy + value
        self.assertEqual(6+3+2, len(model.trainable_variables))

    def test_ppo_delayed_setup(self):
        model = ppo_model.PPO(None)
        self.assertTrue(model.observation_space is None)
        self.assertTrue(model.action_space is None)
        self.assertTrue(model.agent is None)
        envs = [FakeContinuousEnv() for _ in range(3)]
        env = ub_vec.VecEnv(envs)
        model.set_env(env)
        model.setup()
        self.assertTrue(model.observation_space is not None)
        self.assertTrue(model.action_space is not None)
        self.assertEqual(4+4+3+2, len(model.trainable_variables))
    
    def test_ppo_call(self):
        envs = [FakeContinuousEnv() for _ in range(3)]
        env = ub_vec.VecEnv(envs)
        model = ppo_model.PPO(env)
        batch_size = 3
        obs_space = env.observation_space
        act_space = env.action_space
        act_dim = act_space.shape[0]
        obs = np.asarray([obs_space.sample() for _ in range(batch_size)])
        act, val, logp = model(obs, det=True)
        self.assertArrayEqual((batch_size, act_dim), act.shape)
        self.assertArrayEqual((batch_size,), val.shape)
        self.assertArrayEqual((batch_size,), logp.shape)
        act, val, logp = model(obs, det=False)
        self.assertArrayEqual((batch_size, act_dim), act.shape)
        self.assertArrayEqual((batch_size,), val.shape)
        self.assertArrayEqual((batch_size,), logp.shape)
    
    def test_ppo_predict_batch(self):
        envs = [FakeContinuousEnv() for _ in range(3)]
        env = ub_vec.VecEnv(envs)
        model = ppo_model.PPO(env)
        batch_size = 3
        obs_space = env.observation_space
        act_space = env.action_space
        act_dim = act_space.shape[0]
        obs = np.asarray([obs_space.sample() for _ in range(batch_size)])
        act = model.predict(obs, det=True)
        self.assertArrayEqual((batch_size, act_dim), act.shape)
        act = model.predict(obs, det=False)
        self.assertArrayEqual((batch_size, act_dim), act.shape)

    def test_ppo_run(self):
        n_envs = 3
        envs = [FakeContinuousEnv() for _ in range(n_envs)]
        env = ub_vec.VecEnv(envs)
        model = ppo_model.PPO(env)
        obs_shape = env.observation_space.shape
        act_shape = env.action_space.shape
        n_samples = 100
        model.run(n_samples)
        buf = model.buffer
        self.assertEqual(n_samples*n_envs, len(buf))
        self.assertTrue(buf.ready_for_sample)
        self.assertFalse(buf.isfull)
        # test buffer contents
        self.assertArrayEqual((n_samples, n_envs, *obs_shape), 
                              buf.data['obs'].shape)
        self.assertArrayEqual((n_samples, n_envs, *act_shape),
                              buf.data['act'].shape)
        self.assertArrayEqual((n_samples, n_envs),
                              buf.data['done'].shape)
        self.assertArrayEqual((n_samples, n_envs),
                              buf.data['rew'].shape)
        self.assertArrayEqual((n_samples, n_envs),
                              buf.data['val'].shape)
        self.assertArrayEqual((n_samples, n_envs),
                              buf.data['logp'].shape)

    def test_ppo_train(self):
        n_envs = 3
        envs = [FakeContinuousEnv() for _ in range(n_envs)]
        env = ub_vec.VecEnv(envs)
        model = ppo_model.PPO(env)
        n_samples = 10
        batch_size = 10
        n_subepochs = 4
        exp_gradsteps = n_samples * n_envs * n_subepochs / batch_size
        model.run(n_samples)
        model.train(batch_size, n_subepochs)
        self.assertEqual(exp_gradsteps, model.num_gradsteps)
        self.assertEqual(n_subepochs, model.num_subepochs)

    def test_ppo_train_with_target_kl(self):
        n_envs = 3
        target_kl = 0.1
        envs = [FakeContinuousEnv() for _ in range(n_envs)]
        env = ub_vec.VecEnv(envs)
        env.seed(0)
        ub_utils.set_seed(0)
        model = ppo_model.PPO(env, target_kl=target_kl)
        n_samples = 10
        batch_size = 10
        n_subepochs = 4
        exp_gradsteps = (n_samples * n_envs * n_subepochs) // batch_size
        model.run(n_samples)
        model.train(batch_size, n_subepochs) 
        self.assertTrue(exp_gradsteps > model.num_gradsteps, model.num_gradsteps)
        self.assertTrue(n_subepochs > model.num_subepochs, model.num_subepochs)

    def test_ppo_param_order_non_delayed_vs_delayed(self):
        n_envs = 3
        envs = [FakeContinuousEnv() for _ in range(n_envs)]
        env = ub_vec.VecEnv(envs)
        env.seed(1)
        ub_utils.set_seed(1)
        obs_space = env.observation_space
        act_space = env.action_space
        model = ppo_model.PPO(env)
        n_samples = 10
        batch_size = 10
        model.run(n_samples)
        # train for some steps
        batch = next(iter(model.sampler(batch_size)))
        ub_utils.set_seed(1)
        model._train_model(batch)
        # delayed
        model2 = ppo_model.PPO(None, observation_space=obs_space,
                                     action_space=act_space)
        model2.setup()
        ub_utils.set_seed(1)
        model2._train_model(batch)
        # check trainable variables order
        self.assertVariables(model.trainable_variables,
                            model2.trainable_variables)
        # check optimizer variables order
        self.assertVariables(model.optimizer.variables(),
                            model2.optimizer.variables())

    def test_ppo_save_load(self):
        n_envs = 3
        envs = [FakeContinuousEnv() for _ in range(n_envs)]
        env = ub_vec.VecEnv(envs)
        env.seed(1)
        ub_utils.set_seed(1)
        model = ppo_model.PPO(env)
        n_samples = 10
        batch_size = 10
        model.run(n_samples)
        # train for some steps
        ub_utils.set_seed(2)
        batch = next(iter(model.sampler(batch_size)))
        model._train_model(batch)
        with tempfile.TemporaryDirectory() as tempdir:
            save_path = tempdir
            # save & load model
            model.save(save_path)
            loaded_model = ppo_model.PPO.load(save_path)
        # check model setup
        self.assertTrue(loaded_model.agent is not None)
        self.assertTrue(loaded_model.buffer is not None)
        self.assertTrue(loaded_model.optimizer is not None)
        # check if config is correctly restored
        model_config = model.get_config()
        loaded_config = loaded_model.get_config()
        self.assertEqual(set(model_config.keys()), set(loaded_config.keys()))
        for key in model_config:
            self.assertEqual(model_config[key], loaded_config[key])
        # check if all network variables are correctly restored
        self.assertVariables(model.trainable_variables,
                        loaded_model.trainable_variables)
        # test optimizers
        # load optimizer params
        batches = [batch for batch in model.sampler(batch_size)]
        ub_utils.set_seed(1)
        for batch in batches:
            losses1, kl1 = model._train_step(batch)
        ub_utils.set_seed(1)
        for batch in batches:
            losses2, kl2 = loaded_model._train_step(batch)
        # check if losses are matched
        self.assertEqual(set(losses1.keys()), set(losses2.keys()))
        for key in losses1.keys():
            self.assertEqual(losses1[key], losses2[key])
        self.assertAllClose(kl1, kl2)
        # check if vars are same
        self.assertVariables(model.trainable_variables,
                        loaded_model.trainable_variables)
        # check if params of the optimizer are same
        self.assertVariables(model.optimizer.variables(),
                        loaded_model.optimizer.variables())

    def test_ppo_learn(self):
        n_envs = 4
        n_steps = 125
        n_subepochs = 2
        n_epochs = 2
        batch_size = 50
        total_steps = n_envs * n_steps * n_epochs
        total_gradsteps = (int((n_envs * n_steps)/batch_size+0.5)
                            * n_subepochs * n_epochs)
        envs = [FakeContinuousEnv() for _ in range(n_envs)]
        env = ub_vec.VecEnv(envs)
        eval_env = FakeContinuousEnv()
        env.seed(1)
        ub_utils.set_seed(1)
        model = ppo_model.PPO(
            env, 
            batch_size=batch_size,
            n_steps=n_steps,
            n_subepochs=n_subepochs,
        )
        with tempfile.TemporaryDirectory() as tempdir:
            save_path = tempdir
            model.learn(
                total_steps,
                log_interval=1,
                eval_env=eval_env,
                eval_interval=1,
                eval_episodes=1,
                eval_max_steps=10,
                save_path=save_path,
                save_interval=1,
                tb_logdir=save_path,
                reset_timesteps=True,
                verbose=3
            )
            # test load weights
            ppo_model.PPO.load(save_path)
            # test model state
            self.assertEqual(total_steps, model.num_timesteps)
            self.assertEqual(n_epochs, model.num_epochs)
            self.assertEqual(n_subepochs*n_epochs, model.num_subepochs)
            self.assertEqual(total_gradsteps, model.num_gradsteps)
            self.assertEqual(1.0, model.progress)

    def test_ppo_reset_spaces_conflict(self):
        n_envs = 4
        envs = [FakeContinuousEnv() for _ in range(n_envs)]
        env = ub_vec.VecEnv(envs)
        model = ppo_model.PPO(env)
        envs = [FakeImageEnv() for _ in range(n_envs)]
        env = ub_vec.VecEnv(envs)
        with self.assertRaises(RuntimeError):
            # space conflict
            model.set_env(env)
    
    def test_ppo_not_vec_env(self):
        env = FakeContinuousEnv()
        with self.assertRaises(RuntimeError):
            ppo_model.PPO(env)

    def test_ppo_dual_clip_valu_clip(self):
        n_envs = 4
        envs = [FakeContinuousEnv() for _ in range(n_envs)]
        env = ub_vec.VecEnv(envs)
        model = ppo_model.PPO(env, value_clip=0.1, dual_clip=0.1)
        n_samples = 10
        batch_size = 10
        n_subepochs = 4
        exp_gradsteps = n_samples * n_envs * 1 / batch_size
        model.run(n_samples)
        model.train(batch_size, n_subepochs)

    def test_ppo_gae(self):
        n_envs = 2
        gamma = 0.99
        lam = 0.95
        envs = [FakeImageEnv(max_steps=10) for _ in range(n_envs)]
        env = ub_vec.VecEnv(envs)
        env.seed(1)
        ub_utils.set_seed(1)
        n_samples = 20
        model = ppo_model.PPO(env, gamma=gamma, gae_lambda=lam)
        model.collect(n_samples)
        exp_gae = legacy_gae(
            rew   = model.buffer.data['rew'], 
            val   = model.buffer.data['val'], 
            done  = model.buffer.data['done'], 
            gamma = gamma, 
            lam   = lam
        )
        env.seed(1)
        model.run(n_samples)
        gae = model.buffer.data['adv']
        self.assertAllClose(exp_gae, gae)

    def test_ppo_cartpole(self):
        with ub_utils.run_eagerly(False):
            ub_utils.set_seed(1)
            env = ub_vec.VecEnv([gym.make('CartPole-v0') for _ in range(10)])
            env.seed(1)
            eval_env = gym.make('CartPole-v0')
            eval_env.seed(0)
            model = ppo_model.PPO(
                env,
                learning_rate=1e-3,
                gamma=0.8,
                batch_size=128,
                n_steps=500,
            ).learn(
                20000,
                verbose=1
            )
            results = model.eval(eval_env, 20, 200)
            metrics = model.get_eval_metrics(results)
            self.assertAllClose(200.0, metrics['mean-reward'])
            env.close()
            eval_env.close()
