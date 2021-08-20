# --- built in ---
import os
import sys
import time
import json
import logging
import unittest
import functools

# --- 3rd party ---
import gym
import numpy as np

# --- my module ---
from unstable_baselines.lib.envs.vec import dummy
from unstable_baselines.lib import utils
from test.utils import TestCase


class FakeEnv(gym.Env):
    metadata = {'render.modes':[]}
    reward_range = {-float('inf'), float('inf')}
    spec = None
    observation_space: gym.Space
    action_space: gym.Space
    def __init__(self, rank, env_type):
        if env_type == 'Box':
            obs_space = gym.spaces.Box(low=-np.inf,high=np.inf,
                                      shape=(32, 32, 3),
                                      dtype=np.float32)
            act_space = gym.spaces.Box(low=-1,high=1,
                                      shape=(6,),
                                      dtype=np.float32)
        elif env_type == 'Image':
            obs_space = gym.spaces.Box(low=0,high=255,
                                      shape=(32, 32, 3),
                                      dtype=np.uint8)
            act_space = gym.spaces.Discrete(6)
        elif env_type == 'Discrete':
            obs_space = gym.spaces.Discrete(10)
            act_space = gym.spaces.Discrete(6)
        elif env_type == 'MultiBinary':
            obs_space = gym.spaces.MultiBinary([3, 7])
            act_space = gym.spaces.MultiBinary([4, 6])
        else:
            raise ValueError(f'Unknown env type: {env_type}')
        self.observation_space = obs_space
        self.action_space = act_space
        self.timesteps = 0
        self.attr0 = rank
        self.rank = rank
        self.is_closed = False

    def step(self, action):
        reward = float(self.rank)
        self.timesteps += 1
        done = self.timesteps > 10
        return self.observation_space.sample(), reward, done, {}

    def reset(self):
        self.timesteps = 0
        return self.observation_space.sample()

    def render(self, mode='human'):
        pass

    def close(self):
        if not self.is_closed:
            self.is_closed = True

    def seed(self, seed):
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

def create_fake_env(rank, env_type):
    return FakeEnv(rank, env_type)

class TestEnvsVecDummyModule(TestCase):
    def test_dummy_vec_env_without_rms(self):
        num_envs = 4
        env_fns = [
            functools.partial(create_fake_env, i, 'Box') 
            for i in range(num_envs)
        ]
        envs = dummy.DummyVecEnv(env_fns, rms_norm=False)
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

    def test_dummy_vec_env_box_auto_rms(self):
        num_envs = 4
        env_fns = [
            functools.partial(create_fake_env, i, 'Box') 
            for i in range(num_envs)
        ]
        envs = dummy.DummyVecEnv(env_fns)
        self.assertTrue(isinstance(envs.rms_norm, utils.RMSNormalizer))
        self.assertTrue(envs.rms_norm.enabled)
        self.assertFalse(envs.rms_norm.fixed)
        self.assertEqual(len(envs), num_envs)
        envs.seed(1)
        obs = envs.reset()
        self.assertArrayEqual(obs.shape, (num_envs, *envs.observation_space.shape))
        acts = [space.sample() for space in envs.action_spaces]
        obs, rew, done, info = envs.step(acts)
        self.assertArrayEqual(obs.shape, (num_envs, *envs.observation_space.shape))
        envs.render()
        envs.close()
    
    def test_dummy_vec_env_image_auto_rms(self):
        num_envs = 4
        env_fns = [
            functools.partial(create_fake_env, i, 'Image') 
            for i in range(num_envs)
        ]
        envs = dummy.DummyVecEnv(env_fns)
        self.assertTrue(isinstance(envs.rms_norm, utils.RMSNormalizer))
        self.assertFalse(envs.rms_norm.enabled)
        self.assertTrue(envs.rms_norm.fixed)
        self.assertEqual(len(envs), num_envs)
        envs.seed(1)
        obs = envs.reset()
        self.assertArrayEqual(obs.shape, (num_envs, *envs.observation_space.shape))
        acts = [space.sample() for space in envs.action_spaces]
        obs, rew, done, info = envs.step(acts)
        self.assertArrayEqual(obs.shape, (num_envs, *envs.observation_space.shape))
        envs.render()
        envs.close()

    def test_dummy_vec_env_image_with_rms(self):
        num_envs = 4
        env_fns = [
            functools.partial(create_fake_env, i, 'Image') 
            for i in range(num_envs)
        ]
        rms_norm = utils.RMSNormalizer(None, enable=True)
        envs = dummy.DummyVecEnv(env_fns, rms_norm)
        self.assertTrue(isinstance(envs.rms_norm, utils.RMSNormalizer))
        self.assertTrue(envs.rms_norm.enabled)
        self.assertFalse(envs.rms_norm.fixed)
        self.assertEqual(len(envs), num_envs)
        envs.seed(1)
        obs = envs.reset()
        self.assertArrayEqual(obs.shape, (num_envs, *envs.observation_space.shape))
        acts = [space.sample() for space in envs.action_spaces]
        obs, rew, done, info = envs.step(acts)
        self.assertArrayEqual(obs.shape, (num_envs, *envs.observation_space.shape))
        envs.render()
        envs.close()

    def test_dummy_vec_env_discrete(self):
        num_envs = 4
        env_fns = [
            functools.partial(create_fake_env, i, 'Discrete') 
            for i in range(num_envs)
        ]
        envs = dummy.DummyVecEnv(env_fns)
        self.assertTrue(isinstance(envs.rms_norm, utils.RMSNormalizer))
        self.assertFalse(envs.rms_norm.enabled)
        self.assertTrue(envs.rms_norm.fixed)
        self.assertEqual(len(envs), num_envs)
        envs.seed(1)
        obs = envs.reset()
        self.assertArrayEqual(obs.shape, (num_envs, *envs.observation_space.shape))
        acts = [space.sample() for space in envs.action_spaces]
        obs, rew, done, info = envs.step(acts)
        self.assertArrayEqual(obs.shape, (num_envs, *envs.observation_space.shape))
        envs.render()
        envs.close()

    def test_dummy_vec_env_multibinary(self):
        num_envs = 4
        env_fns = [
            functools.partial(create_fake_env, i, 'MultiBinary') 
            for i in range(num_envs)
        ]
        envs = dummy.DummyVecEnv(env_fns)
        self.assertTrue(isinstance(envs.rms_norm, utils.RMSNormalizer))
        self.assertFalse(envs.rms_norm.enabled)
        self.assertTrue(envs.rms_norm.fixed)
        self.assertEqual(len(envs), num_envs)
        envs.seed(1)
        obs = envs.reset()
        self.assertArrayEqual(obs.shape, (num_envs, *envs.observation_space.shape))
        acts = [space.sample() for space in envs.action_spaces]
        obs, rew, done, info = envs.step(acts)
        self.assertArrayEqual(obs.shape, (num_envs, *envs.observation_space.shape))
        envs.render()
        envs.close()

    def test_vec_env(self):
        num_envs = 4
        envs = [
            create_fake_env(i, 'Box')
            for i in range(num_envs)
        ]
        envs = dummy.VecEnv(envs)
        self.assertTrue(isinstance(envs.rms_norm, utils.RMSNormalizer))
        self.assertTrue(envs.rms_norm.enabled)
        self.assertFalse(envs.rms_norm.fixed)
        self.assertEqual(len(envs), num_envs)
        envs.seed(1)
        obs = envs.reset()
        self.assertArrayEqual(obs.shape, (num_envs, *envs.observation_space.shape))
        acts = [space.sample() for space in envs.action_spaces]
        obs, rew, done, info = envs.step(acts)
        self.assertArrayEqual(obs.shape, (num_envs, *envs.observation_space.shape))
        envs.render()
        envs.close()

