# --- built in ---
import os
import sys
import time
import functools
import multiprocessing

import typing

# --- 3rd party ---
import gym
import numpy as np

from parameterized import parameterized

# --- my module ---
from unstable_baselines.lib.envs.vec import subproc
from unstable_baselines.lib import utils

from test.utils import TestCase
from test.test_lib.test_envs.utils import FakeEnv

def create_fake_env(rank, env_type):
    return FakeEnv(rank, env_type)

class TestSubprocVecEnvModule(TestCase):

    def test_subproc_vec_env_without_rms(self):
        num_envs = 4
        env_fns = [
            functools.partial(create_fake_env, i, 'Box')
            for i in range(num_envs)
        ]
        envs = subproc.SubprocVecEnv(env_fns, rms_norm=False)
        # test properties
        self.assertTrue(isinstance(envs.rms_norm, utils.RMSNormalizer))
        self.assertFalse(envs.rms_norm.enabled)
        self.assertTrue(envs.rms_norm.fixed)
        self.assertEqual(len(envs), num_envs)
        # test getattrs/setattrs
        attr0s = list(range(num_envs))
        attr1s = list(v+1 for v in attr0s)
        self.assertEqual(envs.getattrs('attr0'), attr0s)
        self.assertEqual(envs.getattrs('attr0', id=0), [0])
        self.assertEqual(envs.getattrs('attr0', id=[0, 2]), [0, 2])
        envs.setattrs('attr1', values=attr1s)
        self.assertEqual(envs.getattrs('attr1'), attr1s)
        attr2 = 'a'
        envs.setattrs('attr2', value=attr2)
        self.assertEqual(envs.getattrs('attr2'), [attr2]*num_envs)
        # test exception
        with self.assertRaises(ValueError):
            envs.setattrs('attr2', value=attr2, values=attr1s)
        # test seed, reset
        envs.seed()
        envs.seed(1)
        obs = envs.reset()
        self.assertArrayNotEqual(obs[0], obs[1])
        self.assertArrayNotEqual(obs[1], obs[2])
        self.assertArrayNotEqual(obs[2], obs[3])
        envs.seed([1]*3, id=[0, 1, 3])
        obs = envs.reset()
        self.assertArrayEqual(obs[0], obs[1])
        self.assertArrayEqual(obs[1], obs[3])
        self.assertArrayNotEqual(obs[2], obs[3])
        envs.seed([1]*num_envs)
        obs = envs.reset()
        self.assertArrayEqual(obs[0], obs[1])
        self.assertArrayEqual(obs[1], obs[2])
        self.assertArrayEqual(obs[2], obs[3])
        acts = [space.sample() for space in envs.action_spaces]
        self.assertArrayEqual(acts[0], acts[1])
        self.assertArrayEqual(acts[1], acts[2])
        self.assertArrayEqual(acts[2], acts[3])
        # test step
        obs, rew, done, info = envs.step(acts)
        self.assertEqual(len(obs), num_envs)
        self.assertEqual(len(rew), num_envs)
        self.assertEqual(len(done), num_envs)
        self.assertEqual(len(info), num_envs)
        self.assertArrayEqual(obs[0], obs[1])
        self.assertArrayEqual(obs[1], obs[2])
        self.assertArrayEqual(obs[2], obs[3])
        self.assertArrayEqual(rew, np.arange(4, dtype=np.float32))
        # test close
        envs.render()
        envs.close()
        envs.close() # do nothing

    @parameterized.expand([
        ('Box', True), 
        ('Image', False), 
        ('Discrete', False), 
        ('MultiBinary', False)
    ])
    def test_subproc_vec_env_auto_rms(self, space_type, should_enable):
        num_envs = 4
        env_fns = [
            functools.partial(create_fake_env, i, space_type) 
            for i in range(num_envs)
        ]
        envs = subproc.SubprocVecEnv(env_fns)
        self.assertTrue(isinstance(envs.rms_norm, utils.RMSNormalizer))
        self.assertEqual(envs.rms_norm.enabled, should_enable)
        self.assertEqual(envs.rms_norm.fixed, not should_enable)
        self.assertEqual(len(envs), num_envs)
        envs.seed(1)
        obs = envs.reset()
        self.assertArrayEqual(obs.shape, (num_envs, *envs.observation_space.shape))
        acts = [space.sample() for space in envs.action_spaces]
        obs, rew, done, info = envs.step(acts)
        self.assertArrayEqual(obs.shape, (num_envs, *envs.observation_space.shape))
        envs.render()
        envs.close()