# --- built in ---
import os
import sys
import time
import logging
import unittest
import tempfile

# --- 3rd party --
import gym
import numpy as np
import pybullet_envs

# --- my module ---
from unstable_baselines.lib import utils as ub_utils
from unstable_baselines.lib import envs as ub_envs
from test.utils import TestCase
from test.test_lib.test_envs.utils import FakeContinuousEnv

class TestWrapModule(TestCase):
    '''Test unstable_baselines.lib.envs.wrap module
    '''
    def test_time_feature_wrapper_wo_time_limit(self):
        max_steps = 10
        test_mode = False
        env = FakeContinuousEnv(max_steps=1000)
        orig_obs_space = env.observation_space
        env = ub_envs.TimeFeatureWrapper(env, max_steps=max_steps, 
                                              test_mode=test_mode)
        obs_space = env.observation_space
        self.assertEqual(max_steps, env._max_steps)
        self.assertEqual(test_mode, env._test_mode)
        self.assertEqual(orig_obs_space.shape[0]+1, obs_space.shape[0])
        env.reset()
        steps = 0
        get_feature = lambda i: 1.-((i%max_steps+1))/max_steps
        for i in range(max_steps+1):
            steps += 1
            feature = 1. - (steps/max_steps)
            obs, rew, done, info = env.step(env.action_space.sample())
            self.assertAllClose(feature, obs[-1])
            if done:
                steps = 0
                env.reset()
    
    def test_time_feature_wrapper_w_time_limit(self):
        max_steps = 10
        test_mode = False
        env = FakeContinuousEnv(max_steps=1000)
        orig_obs_space = env.observation_space
        env = ub_envs.wrap_mujoco(env, time_limit=max_steps,
                                       time_feature=True,
                                       test_mode=False)
        obs_space = env.observation_space
        self.assertEqual(orig_obs_space.shape[0]+1, obs_space.shape[0])
        env.reset()
        steps = 0
        for i in range(max_steps+1):
            steps += 1
            feature = 1. - (steps/max_steps)
            obs, rew, done, info = env.step(env.action_space.sample())
            self.assertAllClose(feature, obs[-1])
            if done:
                steps = 0
                env.reset()
    
    def test_time_feature_wrapper_test_mode(self):
        max_steps = 10
        test_mode = True
        env = FakeContinuousEnv(max_steps=1000)
        orig_obs_space = env.observation_space
        env = ub_envs.TimeFeatureWrapper(env, max_steps=max_steps, 
                                              test_mode=test_mode)
        obs_space = env.observation_space
        self.assertEqual(max_steps, env._max_steps)
        self.assertEqual(test_mode, env._test_mode)
        self.assertEqual(orig_obs_space.shape[0]+1, obs_space.shape[0])
        env.reset()
        for i in range(max_steps+1):
            obs, rew, done, info = env.step(env.action_space.sample())
            self.assertAllClose(1., obs[-1])
            if done:
                env.reset()

    def test_obs_norm_get_rms_opt(self):
        env = FakeContinuousEnv(max_steps=1000)
        env_norm = ub_envs.ObsNorm(env)
        self.assertTrue(isinstance(env_norm.rms_norm, ub_utils.RMSNormalizer))
        self.assertTrue(env_norm.rms_norm.enabled)
        self.assertFalse(env_norm.rms_norm.fixed)
        self.assertTrue(env_norm.update_rms)
        env_norm = ub_envs.ObsNorm(env, rms_norm=False)
        self.assertTrue(isinstance(env_norm.rms_norm, ub_utils.RMSNormalizer))
        self.assertFalse(env_norm.rms_norm.enabled)
        self.assertTrue(env_norm.rms_norm.fixed)
        self.assertTrue(env_norm.update_rms)
        env_norm = ub_envs.ObsNorm(env, rms_norm=True)
        self.assertTrue(isinstance(env_norm.rms_norm, ub_utils.RMSNormalizer))
        self.assertTrue(env_norm.rms_norm.enabled)
        self.assertFalse(env_norm.rms_norm.fixed)
        self.assertTrue(env_norm.update_rms)
        # user created rms norm
        env_norm = ub_envs.ObsNorm(env, ub_utils.RMSNormalizer())
        self.assertTrue(isinstance(env_norm.rms_norm, ub_utils.RMSNormalizer))
        self.assertTrue(env_norm.rms_norm.is_setup)
        self.assertTrue(env_norm.rms_norm.enabled)
        self.assertFalse(env_norm.rms_norm.fixed)
        self.assertTrue(env_norm.update_rms)
        rms_norm = ub_utils.RMSNormalizer(enable=False, fixed=True)
        env_norm = ub_envs.ObsNorm(env, rms_norm)
        self.assertTrue(isinstance(env_norm.rms_norm, ub_utils.RMSNormalizer))
        self.assertTrue(env_norm.rms_norm.is_setup)
        self.assertFalse(env_norm.rms_norm.enabled)
        self.assertTrue(env_norm.rms_norm.fixed)
        self.assertTrue(env_norm.update_rms)
        # rms norm from vectorized envs
        n_envs = 4
        envs = [FakeContinuousEnv(max_steps=1000)
                    for _ in range(n_envs)]
        rms_norm = ub_utils.RMSNormalizer(enable=False, fixed=True)
        vecenv = ub_envs.VecEnv(envs, rms_norm)
        env_norm = ub_envs.ObsNorm(env, vecenv)
        self.assertTrue(isinstance(env_norm.rms_norm, ub_utils.RMSNormalizer))
        self.assertTrue(env_norm.rms_norm.is_setup)
        self.assertFalse(env_norm.rms_norm.enabled)
        self.assertTrue(env_norm.rms_norm.fixed)
        self.assertFalse(env_norm.update_rms)
    
    def test_obs_norm_seed(self):
        env = FakeContinuousEnv(max_steps=1000)
        env_norm = ub_envs.ObsNorm(env, update_rms=False)
        env.seed(1)
        env.reset()
        obs = np.asarray([env.step(env.action_space.sample())[0]
                        for i in range(10)])
        env_norm.seed(1)
        env_norm.reset()
        obs_norm = np.asarray([env_norm.step(env_norm.action_space.sample())[0]
                        for i in range(10)])
        # ensure seed works
        self.assertAllClose(obs, obs_norm)

    def test_obs_norm(self):
        env = FakeContinuousEnv(max_steps=1000)
        env_norm = ub_envs.ObsNorm(env)
        env.seed(1)
        obs_list = [env.reset()]
        for i in range(100):
            obs_list.append(env.step(env.action_space.sample())[0])
        obs = np.asarray(obs_list)

        env_norm.seed(1)
        env_norm.reset()
        for i in range(100):
            obs_norm = env_norm.step(env_norm.action_space.sample())[0]
        self.assertAllClose(env_norm.rms_norm.rms.mean, obs.mean(axis=0))
        self.assertAllClose(env_norm.rms_norm.rms.var, obs.var(axis=0))
        eps = np.finfo(np.float32).eps.item()
        exp_obs = (obs[-1] - obs.mean(axis=0))/np.sqrt(obs.var(axis=0)+eps)
        self.assertAllClose(exp_obs, obs_norm)
        # test save load
        with tempfile.TemporaryDirectory() as tempdir:
            save_path = os.path.join(tempdir, 'rms.json')
            env_norm.save(save_path)
            
            env = FakeContinuousEnv(max_steps=1000)
            env_norm = ub_envs.ObsNorm(env, rms_norm=save_path,
                                    update_rms=False)
            env_norm.seed(1)
            env_norm.reset()
            for i in range(100):
                obs_norm = env_norm.step(env_norm.action_space.sample())[0]
            self.assertAllClose(exp_obs, obs_norm)
            # different way to load rms
            env_norm = ub_envs.ObsNorm(env, update_rms=False).load(save_path)
            env_norm.seed(1)
            env_norm.reset()
            for i in range(100):
                obs_norm = env_norm.step(env_norm.action_space.sample())[0]
            self.assertAllClose(exp_obs, obs_norm)
        










    
