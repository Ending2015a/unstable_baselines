# --- bulit in ---
import os
import sys
import time
import logging
import unittest
import tempfile

# --- 3rd party ---
import gym
import numpy as np

from gym.wrappers import TimeLimit

# --- my module ---
from unstable_baselines.lib import utils as ub_utils
from unstable_baselines.lib.envs.wrap import atari
from test.utils import TestCase

TEST_ENV_ID = 'BeamRiderNoFrameskip-v4'

class TestWrapAtariModule(TestCase):
    def test_noop_reset_env(self):
        # runable test
        noop_max = 20
        env = gym.make(TEST_ENV_ID)
        env = TimeLimit(env, 3)
        env = atari.NoopResetEnv(env, noop_max=noop_max)
        env.reset()
        for i in range(20):
            obs, rew, done, info = env.step(env.action_space.sample())
            if done:
                break
    
    def test_max_and_skip_env(self):
        # runable test
        skip = 4
        env = gym.make(TEST_ENV_ID)
        env = TimeLimit(env, 20)
        env = atari.MaxAndSkipEnv(env, skip=skip)
        env.seed(1)
        ub_utils.set_seed(1)
        env.reset()
        for i in range(20):
            obs, rew, done, info = env.step(env.action_space.sample())
            if done:
                break
        self.assertEqual(4, i)
    
    def test_episodic_life_env(self):
        # runable test
        env = gym.make(TEST_ENV_ID)
        env = atari.EpisodicLifeEnv(env)
        env.reset()
        for i in range(20):
            if i == 10:
                env.lives += 1
            obs, rew, done, info = env.step(env.action_space.sample())
            if done:
                break
        self.assertEqual(10, i)
        env.reset()

    def test_fire_reset_env(self):
        # runable test
        env = gym.make(TEST_ENV_ID)
        env = atari.FireResetEnv(env)
        env.reset()
        for i in range(20):
            env.step(env.action_space.sample())
    
    def test_warp_frame(self):
        env = gym.make(TEST_ENV_ID)
        env = atari.WarpFrame(env)
        obs = env.reset()
        self.assertEqual(np.uint8, obs.dtype)
        self.assertArrayEqual((84, 84, 1), obs.shape)
        for i in range(20):
            obs, *_ = env.step(env.action_space.sample())
            self.assertEqual(np.uint8, obs.dtype)
            self.assertArrayEqual((84, 84, 1), obs.shape)

    def test_scaled_float_frame(self):
        env = gym.make(TEST_ENV_ID)
        env = atari.ScaledFloatFrame(env)
        env = atari.MaxAndSkipEnv(env, skip=4)
        obs = env.reset()
        self.assertEqual(np.float32, obs.dtype)
        self.assertArrayEqual((210, 160, 3), obs.shape)
        self.assertTrue(obs.max() <= 1.0 and obs.min() >= 0.0)
        for i in range(20):
            obs, *_ = env.step(env.action_space.sample())
            self.assertEqual(np.float32, obs.dtype)
            self.assertArrayEqual((210, 160, 3), obs.shape)
            self.assertTrue(obs.max() <= 1.0 and obs.min() >= 0.0)

    def test_clip_reward_env(self):
        env = gym.make(TEST_ENV_ID)
        env = atari.ClipRewardEnv(env)
        env = atari.MaxAndSkipEnv(env, skip=4)
        obs = env.reset()
        for i in range(20):
            obs, rew, done, info = env.step(env.action_space.sample())
            self.assertTrue(rew in [-1, 0, 1])

    def test_frame_stack(self):
        env = gym.make(TEST_ENV_ID)
        env = atari.FrameStack(env, n_frames=4)
        obs = env.reset()
        self.assertEqual(np.uint8, obs.dtype)
        self.assertEqual((210, 160, 12), obs.shape)
        for i in range(20):
            obs, *_ = env.step(env.action_space.sample())
            self.assertEqual(np.uint8, obs.dtype)
            self.assertEqual((210, 160, 12), obs.shape)

    def test_warp_frame_frame_stack(self):
        env = gym.make(TEST_ENV_ID)
        env = atari.WarpFrame(env)
        env = atari.FrameStack(env, n_frames=4)
        obs = env.reset()
        self.assertEqual(np.uint8, obs.dtype)
        self.assertEqual((84, 84, 4), obs.shape)
        for i in range(20):
            obs, *_ = env.step(env.action_space.sample())
            self.assertEqual(np.uint8, obs.dtype)
            self.assertEqual((84, 84, 4), obs.shape)

    def test_wrap_deepmind(self):
        env = gym.make(TEST_ENV_ID)
        env = atari.wrap_deepmind(
            env,
            episode_life=True,
            clip_rewards=True,
            frame_stack=4,
            scale=True,
            warp_frame=True
        )
        obs = env.reset()
        self.assertEqual(np.float32, obs.dtype)
        self.assertEqual((84, 84, 4), obs.shape)
        self.assertTrue(obs.max() <= 1.0 and obs.min() >= 0.0)
        for i in range(20):
            obs, rew, done, info = env.step(env.action_space.sample())
            self.assertEqual(np.float32, obs.dtype)
            self.assertEqual((84, 84, 4), obs.shape)
            self.assertTrue(obs.max() <= 1.0 and obs.min() >= 0.0)
            self.assertTrue(rew in [-1, 0, 1])

    def test_make_atari_wrap_deepmind(self):
        env = atari.make_atari(TEST_ENV_ID)
        self.assertEqual(TEST_ENV_ID, env.unwrapped.spec.id)
        env = atari.wrap_deepmind(
            env,
            episode_life=True,
            clip_rewards=True,
            frame_stack=4,
            scale=True,
            warp_frame=True
        )
        env_str = repr(env)
        self.assertEqual(1, env_str.count('NoopResetEnv'))
        self.assertEqual(1, env_str.count('MaxAndSkipEnv'))
        self.assertEqual(1, env_str.count('EpisodicLifeEnv'))
        self.assertEqual(1, env_str.count('FireResetEnv'))
        self.assertEqual(1, env_str.count('WarpFrame'))
        self.assertEqual(1, env_str.count('ScaledFloatFrame'))
        self.assertEqual(1, env_str.count('ClipRewardEnv'))
        self.assertEqual(1, env_str.count('FrameStack'))
        # test runable
        env.reset()
        for i in range(20):
            env.step(env.action_space.sample())