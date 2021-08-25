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
from test.test_lib.test_envs.utils import FakeImageEnv

class NoopMonitorTool(ub_envs.MonitorToolChain):
    def __init__(self):
        super().__init__()


class BrokenImageEnv(FakeImageEnv):
    def render(self, mode='human'):
        return None

class NonRenderableImageEnv(FakeImageEnv):
    metadata = {'render.modes': []}

class TestMonitorModule(TestCase):
    '''Test unstable_baselines.lib.envs.monitor module
    '''
    def test_monitor_wo_video_recorder(self):
        env = FakeImageEnv(max_steps=10)
        with tempfile.TemporaryDirectory() as tempdir:
            root_dir = os.path.join(tempdir, 'monitor/')

            env = ub_envs.Monitor(env, root_dir=root_dir, video=False)
            self.assertEqual(1, len(env.tools))
            self.assertTrue(isinstance(env.tools[0], ub_envs.StatsRecorder))
            self.assertTrue(env.stats is not None)
            # assert csv created
            csv_path = os.path.join(root_dir, ub_envs.StatsRecorder.monitor_ext)
            self.assertTrue(os.path.isfile(csv_path))
            # test add tools
            env.add_tools([NoopMonitorTool(), NoopMonitorTool()])
            self.assertEqual(3, len(env.tools))
            env.reset()
            for i in range(25):
                obs, rew, done, info = env.step(env.action_space.sample())
                if done:
                    env.reset()
            env.reset() # early reset
            with open(csv_path,'r') as f:
                lines = f.readlines()
            self.assertEqual(5, len(lines))
            self.assertTrue('"env_id": "FakeEnv"' in lines[0])
            self.assertTrue('rewards,length,walltime' in lines[1])
            self.assertTrue('55,10' in lines[2])
            self.assertTrue('55,10' in lines[3])
            self.assertTrue('15,5' in lines[4]) # early reset episode
            env.close()
    
    def test_monitor_wo_video_recorder_exception(self):
        env = FakeImageEnv(max_steps=10)
        with tempfile.TemporaryDirectory() as tempdir:
            root_dir = os.path.join(tempdir, 'monitor/')

            env = ub_envs.Monitor(env, root_dir=root_dir, video=False)
            with self.assertRaises(RuntimeError):
                # test exception: need reset
                for i in range(20):
                    env.step(env.action_space.sample())
            env.close()

    def test_video_recorder_capped_cubic_video_schedule(self):
        for i in range(1, 2000+1):
            res = ub_envs.VideoRecorder.capped_cubic_video_schedule(i)
            if i in [0, 1, 8, 27, 64, 125, 216, 343, 512, 729, 1000, 2000]:
                self.assertTrue(res)
            else:
                self.assertFalse(res)

    def test_monitor_w_video_recorder(self):
        env = FakeImageEnv(max_steps=10)
        with tempfile.TemporaryDirectory() as tempdir:
            root_dir = os.path.join(tempdir, 'monitor/')
            # always record
            env = ub_envs.Monitor(env, root_dir=root_dir, video=True,
                                video_kwargs=dict(interval=1))
            self.assertEqual(2, len(env.tools))
            self.assertTrue(isinstance(env.tools[0], ub_envs.StatsRecorder))
            self.assertTrue(isinstance(env.tools[1], ub_envs.VideoRecorder))
            self.assertTrue(env.tools[1].stats is not None)
            
            for ep in range(10):
                env.reset()
                self.assertTrue(env.tools[1].need_record)
                for step in range(20):
                    obs, rew, done, info = env.step(env.action_space.sample())
                    if done:
                        break
            env.close()
            video_path = os.path.join(root_dir, 'videos')
            records = sorted(os.listdir(video_path))
            self.assertEqual(10*2, len(records)) # json + mp4
            env.close()
    
    def test_monitor_w_video_recorder_cubic(self):
        env = FakeImageEnv(max_steps=10)
        with tempfile.TemporaryDirectory() as tempdir:
            root_dir = os.path.join(tempdir, 'monitor/')
            # always record
            env = ub_envs.Monitor(env, root_dir=root_dir, video=True,
                                video_kwargs=None)
            self.assertEqual(2, len(env.tools))
            self.assertTrue(isinstance(env.tools[0], ub_envs.StatsRecorder))
            self.assertTrue(isinstance(env.tools[1], ub_envs.VideoRecorder))
            self.assertTrue(env.tools[1].stats is not None)
            
            for ep in range(10):
                env.reset()
                self.assertEqual(ep+1 in [1,8], env.tools[1].need_record)
                for step in range(20):
                    obs, rew, done, info = env.step(env.action_space.sample())
                    if done:
                        break
            env.close()
            video_path = os.path.join(root_dir, 'videos')
            records = sorted(os.listdir(video_path))
            self.assertEqual(2*2, len(records)) # json + mp4
            env.close()
    
    def test_monitor_prefix(self):
        prefix = 'eval'
        env = FakeImageEnv(max_steps=10)
        with tempfile.TemporaryDirectory() as tempdir:
            root_dir = os.path.join(tempdir, 'monitor/')
            env = ub_envs.Monitor(env, root_dir=root_dir, prefix=prefix,
                                video=False, video_kwargs=None)
            self.assertEqual(1, len(env.tools))
            self.assertTrue(isinstance(env.tools[0], ub_envs.StatsRecorder))
            env.reset()
            for i in range(25):
                obs, rew, done, info = env.step(env.action_space.sample())
                if done:
                    env.reset()
            csv_path = os.path.join(root_dir, 
                        prefix + '.' + ub_envs.StatsRecorder.monitor_ext)
            self.assertTrue(os.path.isfile(csv_path))
            # test auto close statsrecorder
            # env.close()
    
    def test_monitor_broken_image_env(self):
        env = BrokenImageEnv(max_steps=10)
        with tempfile.TemporaryDirectory() as tempdir:
            root_dir = os.path.join(tempdir, 'monitor/')
            # always record
            env = ub_envs.Monitor(env, root_dir=root_dir, video=True,
                                video_kwargs=dict(interval=1))
            self.assertEqual(2, len(env.tools))
            self.assertTrue(isinstance(env.tools[0], ub_envs.StatsRecorder))
            self.assertTrue(isinstance(env.tools[1], ub_envs.VideoRecorder))
            self.assertTrue(env.tools[1].stats is not None)
            env.reset()
            # The brokwn image env returns None when capture the frame
            print('>>> Test broken image env')
            env.step(env.action_space.sample())
            print('<<<')
            self.assertTrue(env.tools[1]._recorder.broken)
            self.assertFalse(env.tools[1]._recorder.functional)
            for i in range(10):
                obs, rew, done, info = env.step(env.action_space.sample())
                if done:
                    break
            # show log: VideoRecorder is broken
            env.reset()
            env.close()

    def test_monitor_non_recordable_image_env(self):
        env = NonRenderableImageEnv(max_steps=10)
        with tempfile.TemporaryDirectory() as tempdir:
            root_dir = os.path.join(tempdir, 'monitor/')
            video_path = os.path.join(root_dir, 'videos')
            # always record
            env = ub_envs.Monitor(env, root_dir=root_dir, video=True,
                                video_kwargs=dict(interval=1))

            self.assertEqual(2, len(env.tools))
            self.assertTrue(isinstance(env.tools[0], ub_envs.StatsRecorder))
            self.assertTrue(isinstance(env.tools[1], ub_envs.VideoRecorder))
            self.assertTrue(env.tools[1].stats is not None)
            env.reset()
            # The non-recordable image env has no 'rgb_array' mode
            print('>>> Test non-recordable image env')
            env.step(env.action_space.sample())
            print('<<<')
            self.assertFalse(env.tools[1]._recorder.enabled)
            self.assertFalse(env.tools[1]._recorder.functional)
            for i in range(10):
                obs, rew, done, info = env.step(env.action_space.sample())
                if done:
                    break
            env.reset()
            env.close()
            self.assertEqual(0, len(os.listdir(video_path)))