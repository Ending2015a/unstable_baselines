# --- built in ---
import os
import re
import abc
import cv2
import csv
import sys
import glob
import json
import time
import random
import shutil
import inspect
import datetime
import tempfile
import distutils
import contextlib
import subprocess
import multiprocessing

from collections import deque

# --- 3rd party ---
import gym 
import cloudpickle

import numpy as np
import tensorflow as tf


# --- my module ---
from unstable_baselines import logger
from unstable_baselines.utils import flatten_obs


__all__ = [
    'NoopResetEnv',
    'MaxAndSkipEnv',
    'EpisodicLifeEnv',
    'ClipRewardEnv',
    'WarpFrame',
    'Monitor',
    'FrameStack',
    'SubprocVecEnv',
    'VecFrameStack',
]

LOG = logger

def set_env_logger(name='envs', level='DEBUG'):
    global LOG
    LOG = logger.getLogger(name=name, level=level)




# === Env wrappers ---
# Stable baselines - wrapper
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """
        Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        :param env: (Gym Environment) the environment to wrap
        :param noop_max: (int) the maximum value of no-ops to run
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """
        Return only every `skip`-th frame (frameskipping)
        :param env: (Gym Environment) the environment
        :param skip: (int) number of `skip`-th frame
        """
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action):
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.
        :param action: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """
        Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        :param env: (Gym Environment) the environment to wrap
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """
        Calls the Gym environment reset, only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        :param kwargs: Extra keywords passed to env.reset() call
        :return: ([int] or [float]) the first observation of the environment
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        """
        clips the reward to {+1, 0, -1} by its sign.
        :param env: (Gym Environment) the environment
        """
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """
        Bin reward to {+1, 0, -1} by its sign.
        :param reward: (float)
        """
        return np.sign(reward)

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        :param env: (Gym Environment) the environment
        """
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, 1),
                                            dtype=env.observation_space.dtype)

    def observation(self, frame):
        """
        returns the current observation from a frame
        :param frame: ([int] or [float]) environment frame
        :return: ([int] or [float]) the observation
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


# Rewrite gym.wrappers.monitoring.StatsRecorder
#   will output Stable baselines style csv records
class StatsRecorder(object):
    EXT='monitor.csv'
    def __init__(self, directory=None, prefix=None, env_id=None):
        '''
        Filename
            {directory}/{prefix}.{EXT}
        directory: directory
        prefix: filename prefix
            e.g. directory='/foo/bar', prefix=1   => /foo/bar/1.monitor.csv

        CSV format
            # {t_start: timestamp, env_id: env id}
            r, l, t
        r: total episodic rewards
        l: episode length
        t: running time (time - t_start)
        '''
        self.t_start = time.time()
        self.env_id = env_id

        # setup csv file
        self.filename = self._make_filename(directory, prefix)
        self.header = json.dumps({"t_start": self.t_start, 'env_id': self.env_id})

        # write header
        self.file_handler = open(self.filename, "wt")
        self.file_handler.write('#{}\n'.format(self.header))
        
        # create csv writer
        self.writer = csv.DictWriter(self.file_handler, fieldnames=('r', 'l', 't')) # rewards/length/time
        self.writer.writeheader()
        self.file_handler.flush()

        # initialize
        self.rewards = None
        self.need_reset = None
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self._closed = False

    @property
    def closed(self):
        return self._closed

    @classmethod
    def _make_filename(cls, directory, prefix):
        # filename = {directory}/{prefix}.{EXT}

        if prefix:
            if not isinstance(prefix, str):
                prefix = str(prefix)
            # {directory}/{prefix}
            filename = os.path.join(directory, prefix)

            if os.path.basename(filename):
                # if prefix is a valid filename
                filename = filename + '.' + cls.EXT
            else:
                # if prefix is a directory '/'
                filename = os.path.join(filename, cls.EXT)
        else:
            filename = os.path.join(directory, cls.EXT)

        # create directories
        dirpath = os.path.dirname(filename)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        
        LOG.info('Writing monitor to: ' + filename)
        return filename

    def before_step(self):

        if self.need_reset:
            raise RuntimeError('Tried to step environment that needs reset')

    def after_step(self, observation, reward, done, info):

        # append reward
        self.rewards.append(reward)

        # episode ended
        if done:
            self.need_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {'r': round(ep_rew, 6), 'l':ep_len, 't': round(time.time() - self.t_start, 6)}

            self.episode_rewards.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            
            # write episode info to csv
            if self.writer:
                self.writer.writerow(ep_info)
                self.file_handler.flush()

            info['episode'] = ep_info
        
        self.total_steps += 1
        return observation, reward, done, info

    def before_reset(self):
        pass

    def after_reset(self):
        self.rewards = []
        self.need_reset = False

    def close(self):
        if self.file_handler is not None:
            self.file_handler.close()

        self._closed = True

    def __del__(self):
        if not self.closed:
            self.close()

    def flush(self):
        pass

    def get_total_steps(self):
        return self.total_steps

    def get_episode_rewards(self):
        return self.episode_rewards

    def get_episode_lengths(self):
        return self.episode_lengths

    def get_episode_times(self):
        return self.episode_times


class VideoEncoder(object):
    def __init__(self, path, frame_shape, video_shape, fps, output_fps):
        '''
        path: output path
        frame_shape: input frame shape (h,w,c)
        video_shape: output video shape (h,w,c)
        fps: input framerate
        output_fps: output video framerate
        '''
        self.path = path
        self.frame_shape = frame_shape
        self.video_shape = video_shape
        self.fps = fps
        self.output_fps = output_fps

        self.proc = None
        h,w,c = self.frame_shape
        oh,ow,_ = self.video_shape
        
        if c != 3 and c != 4:
            raise RuntimeError('Your frame has shape {}, but we require (w,h,3) or (w,h,4)'.format(self.frame_shape))
        
        self.wh = (w,h)
        self.output_wh = (ow,oh)
        self.includes_alpha = (c == 4)

        if distutils.spawn.find_executable('ffmpeg') is not None:
            self.backend = 'ffmpeg'
        else:
            raise RuntimeError('ffmpeg not found')
        
        self._start_encoder()

    def _start_encoder(self):
        self.cmdline = (
            self.backend,
            '-nostats',   # disable encoding progress info
            '-loglevel', 'error', # suppress warnings
            '-y',  # replace existing file

            # input
            '-f', 'rawvideo', # input format
            '-s:v', "{:d}x{:d}".format(*self.wh), # input shape (w,h)
            '-pix_fmt', ('rgb32' if self.includes_alpha else 'rgb24'),
            '-framerate', '{:d}'.format(self.fps), # input framerate,
            '-i', '-', # input from stdin

            # output
            '-vf', 'scale={:d}:{:d}'.format(*self.output_wh),
            '-c:v', 'libx264', # video codec
            '-pix_fmt', 'yuv420p',
            '-r', '{:d}'.format(self.output_fps),
            self.path)
        

        LOG.debug('Starting ffmpeg with cmd "{}"'.format(' '.join(self.cmdline)))

        self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE)
    
    @property
    def version_info(self):
        return {
            'backend': self.backend,
            'version': str(subprocess.check_output([self.backend, '-version'],stderr=subprocess.STDOUT)),
            'cmdline': self.cmdline
        }

    @property
    def video_info(self):
        return {
            'width': self.wh[0],
            'height': self.wh[1],
            'input_fps': self.fps,
            'output_fps': self.output_fps
        }

    def capture_frame(self, frame):
        if not isinstance(frame, (np.ndarray, np.generic)):
            raise RuntimeError('Wrong type {} for {} (must be np.ndarray or np.generic)'.format(type(frame), frame))
        if frame.shape != self.frame_shape:
            raise RuntimeError('Your frame has shape {}, but the VideoRecorder is configured for shape {}'.format(frame.shape, self.frame.shape))
        if frame.dtype != np.uint8:
            raise RuntimeError('Your frame has data type {}, but we require unit8 (i.e. RGB values from 0-255)'.format(frame.dtype))

        self.proc.stdin.write(frame.tobytes())

    def __del__(self):
        self.close()

    def close(self):
        if self.proc is None:
            return
        
        self.proc.stdin.close()
        ret = self.proc.wait()
        if ret != 0:
            LOG.error('VideoRecorder encoder exited with status {}'.format(ret))

        self.proc = None


# Rewrite gym.wrappers.monitoring.video_recorder.VideoRecoder
class VideoRecoder(object):
    def __init__(self, env, path, meta_path, metadata=None, width=None, height=None, fps=None, enabled=True):
        '''
        env: environment
        path: video path
        meta_path: metadata path
        metadata: additional metadata
            content_type, empty, broken, encoder_version, video_info, exc
        width: video width (None=default)
        height: video height (None=default)
        fps: video framerate (None=default)
        enabled:
        '''
        self.env       = env
        self.path      = path
        self.meta_path = meta_path
        self.width     = width
        self.height    = height
        self.fps       = fps
        self.enabled   = enabled
        self.metadata  = metadata or {}
        self._closed = False

        if not self.enabled:
            return

        fps        = env.metadata.get('video.frames_per_second', 30)
        output_fps = env.metadata.get('video.output_frames_per_second', fps)

        self.fps        = self.fps if self.fps else fps
        self.output_fps = self.fps if self.fps else output_fps

        self._async = env.metadadta.get('semantics.async')
        modes       = env.metadata.get('render.modes', [])

        if 'rbg_array' not in modes:
            LOG.info('Disabling video recorder because {} not supports video mode "rgb_array"'.format(env))
            self.enabled = False
            return
        
        self.encoder = None
        self.broken  = False

        # pybullet_envs
        if hasattr(env.unwrapped, '_render_width'):
            env.unwrapped._render_width = width
        if hasattr(env.unwrapped, '_render_height'):
            env.unwrapped._render_height = height

        # Write metadata
        self.metadata['content_type'] = 'video/mp4'
        self.write_metadata()

        self.empty   = True
        

    @property
    def closed(self):
        return self._closed

    @property
    def functional(self):
        return self.enabled and not self.broken and not self.closed

    def capture_frame(self):
        if not self.functional:
            return 

        frame = self.env.render(mode='rgb_array')

        if frame is None:
            if self._async:
                return
            else:
                LOG.info('Env returned None on render(). Disabling further rendering for video recorder by marking as disabled: {}'.format(self.path))
                self.broken = True
        else:
            self._encode_image_frame(frame)


    def close(self):
        if not self.functional:
            return

        if self.encoder:
            LOG.debug('Closing video encoder: {}'.format(self.path))
            self.encoder.close()
            self.encoder = None
        else:
            if os.path.isfile(self.path):
                os.remove(self.path)
            
            self.metadata['empty'] = True

        if self.broken:
            LOG.info('Cleaning up paths for broken video recorder: {}'.format(self.path))

            if os.path.isfile(self.path):
                os.remove(self.path)

            self.metadata['broken'] = True

        self.write_metadata()

        # mark as closed
        self._closed = True

    def write_metadata(self):
        if not self.enabled:
            return
        
        with open(self.meta_path, 'wt') as f:
            json.dump(self.metadata, f)


    def _get_frame_shape(self, frame):
        if len(frame.shape) != 3:
            raise RuntimeError('Receive unknown frame shape: {}'.format(frame.shape))

        return frame.shape # h,w,c

    def _get_video_shape(self, frame):
        if len(frame.shape) != 3:
            raise RuntimeError('Receive unknown frame shape: {}'.format(frame.shape))

        h,w,c = frame.shape

        self.width = self.width if self.width else w
        self.height = self.height if self.height else h

        return (self.height, self.width, c) # h,w,c

    def _encode_image_frame(self, frame):
        if not self.encoder:
            frame_shape = self._get_frame_shape(frame)
            video_shape = self._get_video_shape(frame)
            self.encoder = VideoEncoder(self.filename, frame_shape, video_shape, self.fps, self.output_fps)

            self.metadata['encoder_version'] = self.encoder.version_info
            self.metadata['video_info'] = self.encoder.video_info
        
        try:
            self.encoder.capture_frame(frame)

        except RuntimeError as e:
            LOG.exception('Tried to pass invalid video frame, marking as broken: {}'.format(self.path))
            self.metadata['exc'] = str(e)
            self.broken = True
        else:
            self.empty = False

    def __del__(self):
        self.close()


def capped_cubic_video_schedule(episode):
    if episode < 1000:
        return int(round(episode_id ** (1. / 3))) ** 3 == episode
    else:
        return episode % 1000 == 0

def disable_videos(episode):
    return False

# Rewrite gym.wrappers.monitors.Monitor
class Monitor(gym.Wrapper):
    def __init__(self, env, directory='monitor', prefix=None, ext='monitor.csv', force=False,
                    enable_video_recording=False, video_kwargs={}):
        '''
        directory: base directory
        prefix: monitor file prefix
        ext: monitor file extention
        force: if False, raise Exception if the monitor file exists
        enable_video_recording: whether record video

        video_kwargs:
            prefix: video filename prefix (default: 'video/')
            ext: video filename ext (default: 'ep{episode}.{start_steps}-{total_steps}.video.mp4')
            meta_ext: metadata filename ext (default: 'ep{episode}.{start_steps}-{total_steps}.metadata.json')
            width: video width 
            height: video height
            fps: video framerate
            callback: video callback
            metadata: metadata (default: None)
        '''
        super().__init__(env=env)

        self.directory      = directory
        self.monitor_prefix = prefix
        self.monitor_ext    = ext
        
        self.stats_recorder = None
        self.video_recorder = None

        self.episode     = 0
        self.start_steps = 0
        self.total_steps = 0
        self.env_id      = env.unwrapped.spec.id
        self.enabled     = True

    
        base_ext = 'ep{episode:06d}.{start_steps}-{total_steps}'

        video_prefix   = video_kwargs.get('prefix', 'video/')
        video_ext      = video_kwargs.get('ext', base_ext + '.video.mp4')
        width          = video_kwargs.get('width', None)
        height         = video_kwargs.get('height', None)
        fps            = video_kwargs.get('fps', None)
        meta_ext       = video_kwargs.get('meta_ext', base_ext + '.metadata.json')
        metadata       = video_kwargs.get('metadata', None)
        video_callback = video_kwargs.get('callback', None)
        
        if enable_video_recording == False:
            video_callback = disable_videos 
        elif video_callback is None:
            video_callback = capped_cubic_video_schedule
        elif not callable(video_callback):
            raise RuntimeError('You must provide a function, None, or False for video_callable, not {}: {}'.format(
                                type(video_callback), video_callback))

        self.video_prefix   = video_prefix
        self.video_ext      = video_ext
        self.width          = width
        self.height         = height
        self.fps            = fps
        self.meta_ext       = meta_ext
        self.metadata       = metadata
        self.video_callback = video_callback

        if self.video_prefix:
            self.video_prefix = str(self.video_prefix)

        # create directory
        if force:
            self._clear_monitor(directory)
        else:
            if self._detect_monitor(directory):
                raise RuntimeError('Trying to write to non-empty monitor directory {}'.format(directory))

        if directory:
            os.makedirs(directory, exist_ok=True)
        
        # Create stats recorder
        self.stats_recorder = StatsRecorder(directory=self.directory, 
                                            prefix=self.monitor_prefix, 
                                            ext=self.monitor_ext, 
                                            env_id=self.env_id)

    def step(self, action):
        self._before_step()
        observation, reward, done, info = self.env.step(action)
        return self._after_step(observation, reward, done, info)

    def reset(self, **kwargs):
        self._before_reset()
        observation = self.env.reset(**kwargs)
        self._after_reset(observation)
        return observation

    def close(self):
        super().close()

        if not self.enabled:
            return
        
        self.stats_recorder.close()
        if self.video_recorder:
            self._close_video_recorder()

        self.enabled = False

        LOG.info('Finished writing results')

    def _before_step(self):
        if not self.enabled:
            return

        self.stats_recorder.before_step()


    def _after_step(self, observation, reward, done, info):
        if not self.enabled:
            return observation, reward, done, info
        
        self.video_recorder.capture_frame()

        self.total_steps += 1

        return self.stats_recorder.after_step(observation, reward, done, info)

    def _before_reset(self):
        if not self.enabled:
            return

        self.stats_recorder.before_reset()

    def _after_reset(self):
        if not self.enabled:
            return

        # make video/metadata path
        video_path, meta_path = self._make_video_meta_path()
        # close video recorder and copy tempfile to `filename`
        self._close_and_save_video_recorder(video_path, meta_path)

        self.stats_recorder.after_reset()
        # increase episode number
        self.start_steps = self.total_steps
        self.episode += 1 

        # create new video recorder for new episode
        directory = os.path.dirname(video_path)
        self._new_video_recorder(directory)

    def _make_ext(self, ext):
        return ext.format(episode=self.episode,
                        start_steps=self.start_steps,
                        total_steps=self.total_steps)

    def _make_video_meta_path(self):

        video_ext  = self._make_ext(self.video_ext)
        meta_ext   = self._make_ext(self.meta_ext)

        base_path  = self.directory
        video_path = video_ext
        meta_path  = meta_ext

        if self.video_prefix:
            if os.path.basename(self.video_prefix):
                # {prefix}.{ext}
                video_path = self.video_prefix + '.' + video_ext
                meta_path  = self.video_prefix + '.' + meta_ext

        video_path = os.path.join(base_path, video_path)
        meta_path  = os.path.join(base_path, meta_path)

        # create directories
        video_dir = os.path.dirname(video_path)
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)

        meta_dir = os.path.dirname(meta_path)
        if meta_dir:
            os.makedirs(meta_dir, exist_ok=True)

        return video_path, meta_path

    def _close_and_save_video_recorder(self, video_path, meta_path):
        if not self.video_recorder or not self.video_recorder.enabled:
            return
        
        self.video_recorder.close()

        # update metadata
        monitor_metadata = {
            'episode': self.episode,
            'start_steps': self.start_steps,
            'total_steps': self.total_steps,
            'episode_length': self.total_steps - self.start_steps,
            'episode_rewards': sum(self.stats_recorder.rewards),
        }

        self.video_recorder.metadata['monitor'] = monitor_metadata
        self.video_recorder.write_metadata()

        # copy tempfile to specified location
        # save metadata
        if os.path.isfile(self.video_recorder.meta_path):
            os.rename(self.video_recorder.meta_path, meta_path)
            LOG.info('Saving metadata to: {}'.format(meta_path))
        else:
            LOG.warn('Metadata not found: {}'.format(meta_path))

        # save video
        if os.path.isfile(self.video_recorder.path):
            os.rename(self.video_recorder.path, video_path)
            LOG.info('Saving video to: {}'.format(video_path))
        else:
            if self.video_recorder.broken:
                LOG.warn('Failed to save video, the VideoRecorder is broken, for more info: {}'.format(meta_path))
            else:
                LOG.error('Failed to save video, missing tempfile, for more info: {}'.format(meta_path))

    def _new_video_recorder(self, directory):

        enabled = self.video_enabled()

        if enabled:
            # generate random filename
            with tempfile.NamedTemporaryFile(dir=directory, suffix='.mp4', delete=False) as f:
                video_path = f.name
            with tempfile.NamedTemporaryFile(dir=directory, suffix='.json', delete=False) as f:
                meta_path  = f.name
        else:
            video_path = None
            meta_path  = None

        self.video_recorder = VideoRecorder(env=self.env, 
                                            path=video_path,
                                            meta_path=meta_path,
                                            metadata=self.metadata,
                                            width=self.width, 
                                            height=self.height, 
                                            fps=self.fps,
                                            enabled=enabled)

        self.video_recorder.capture_frame()
    
    def _video_enabled(self):
        return self.video_callback(self.episode)

    def _detect_monitor(self, directory):
        if not directory:
            return True
        
        return (os.path.isdir(directory) and len(os.listdir(directory)) > 0)

    def _clear_monitor(self, directory):
        if not directory:
            return 

        #TODO: clear monitor
        pass


# class Monitor(gym.Wrapper):
#     EXT = "monitor.csv"

#     def __init__(self, env, path=None, prefix=None):
#         '''
#         Filename
#             {path}/{prefix}.{EXT}
#         path: directory
#         prefix: filename prefix
#             e.g. path='/foo/bar', prefix=1   => /foo/bar/1.monitor.csv

#         CSV format
#             # {t_start: timestamp, env_id: env id}
#             r, l, t
#         r: total episodic rewards
#         l: episode length
#         t: running time (time - t_start)
#         '''
#         super(Monitor, self).__init__(env=env)
#         self.t_start = time.time()

#         # setup csv file
#         self.filename = self.make_filename(path, prefix)
#         self.header = json.dumps({"t_start": self.t_start, 'env_id': env.spec and env.spec.id})

#         # write header
#         self.file_handler = open(self.filename, "wt")
#         self.file_handler.write('#{}\n'.format(self.header))
        
#         # create csv writer
#         self.writer = csv.DictWriter(self.file_handler, fieldnames=('r', 'l', 't')) # rewards/length/time
#         self.writer.writeheader()
#         self.file_handler.flush()

#         # initialize
#         self.rewards = None
#         self.need_reset = None
#         self.episode_rewards = []
#         self.episode_lengths = []
#         self.episode_times = []
#         self.total_steps = 0
    
#     @classmethod
#     def make_filename(cls, path, prefix):
#         # filename = [path]/[prefix].[EXT]
#         if prefix is None:
#             filename = cls.EXT
#         else:
#             # convert prefix to str
#             if not isinstance(prefix, str):
#                 prefix = str(prefix)
            
#             filename = prefix + '.' + cls.EXT

#         # create directories
#         if path is not None:
#             os.makedirs(path, exist_ok=True)
#             filename = os.path.join(path, filename)
        
#         LOG.info('Save monitor file: ' + filename)
#         return filename

#     def reset(self, **kwargs):
#         self.rewards = []
#         self.need_reset = False

#         return self.env.reset(**kwargs)


#     def step(self, action):
        
#         if self.need_reset:
#             raise RuntimeError('Tried to step environment that needs reset')

#         # step env
#         observation, reward, done, info = self.env.step(action)
#         self.rewards.append(reward)

#         # episode ended
#         if done:
#             self.need_reset = True
#             ep_rew = sum(self.rewards)
#             ep_len = len(self.rewards)
#             ep_info = {'r': round(ep_rew, 6), 'l':ep_len, 't': round(time.time() - self.t_start, 6)}

#             self.episode_rewards.append(ep_rew)
#             self.episode_lengths.append(ep_len)
#             self.episode_times.append(time.time() - self.t_start)
            
#             # write episode info to csv
#             if self.writer:
#                 self.writer.writerow(ep_info)
#                 self.file_handler.flush()

#             info['episode'] = ep_info
        
#         self.total_steps += 1
#         return observation, reward, done, info

#     def close(self):
#         super().close()
#         if self.file_handler is not None:
#             self.file_handler.close()

#     def get_total_steps(self):
#         return self.total_steps

#     def get_episode_rewards(self):
#         return self.episode_rewards

#     def get_episode_lengths(self):
#         return self.episode_lengths

#     def get_episode_times(self):
#         return self.episode_times

class FrameStack(gym.Wrapper):
    def __init__(self, env, n_frames):
        """Stack n_frames last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        stable_baselines.common.atari_wrappers.LazyFrames
        :param env: (Gym Environment) the environment
        :param n_frames: (int) the number of frames to stack
        """
        gym.Wrapper.__init__(self, env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * n_frames),
                                            dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return np.concatenate(self.frames, axis=2)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return np.concatenate(self.frames, axis=2), reward, done, info

class PyBulletVideoWriter(gym.Wrapper):
    '''
    Save video from pybullet_envs
    '''
    def __init__(self, env, filename='{env_id}_ep_{episode:06d}_{steps}.mp4', width=320, height=240, fps=60, **kwargs):
        '''
        env: environment
        filename: video filename
        width: video width
        height: video height
        fps: video framerate
        '''
        super().__init__(env=env)

        self.filename = filename
        self.width = width
        self.height = height
        self.fps = fps

        env.env._render_width = self.width
        env.env._render_height = self.height

        self._env_id = env.unwrapped.spec.id
        self._episode = 0
        self._steps = 0
        self._filename = None
        self._tmp_filename = None
        self._command = None
        self._proc = None


    def step(self, action):

        if self.proc is None:
            raise RuntimeError('Tried to step environment that needs reset')

        data = self.step(action)

        frame = self.env.render('rgb_array')
        self.proc.stdin.write(frame.tobytes())
        
        return data

    def reset(self, **kwargs):
        
        # close the existing video writer
        self._close_writer()

        filename = self._make_filename()


        # open a new video writer
        self._open_writer()

        # reset env
        data = self.reset(**kwargs)
        
        frame = env.render('rgb_array')

        return 

    def close(self, **kwargs):
        self._close_writer()

        self.env.close()

    def _make_tempfile(self):


    def _make_filename(self, filename):

    def _write_frame(self, frame):

        if self.proc is None:
            raise RuntimeError('FFmpeg process not opened')
        

    def _open_writer(self):
        self.command = [
            'ffmpeg',
            '-y',                       # overwrite files without asking
            '-f', 'rawvideo',           # input format
            '-vcodec', 'rawvideo',      # input codec
            '-s', '{}x{}'.format(self.width, self.height), # set video size
            '-pix_fmt', 'rgb24',        # set pixel format (bgr24 for cv2 image)
            '-r', str(self.fps), # framerate
            '-i', '-',                  # input from stdin
            '-an',                      # disable audio stream
            '-vcodec', 'mpeg4',         # video codec
            '-b:v', '5000k',            # video bitrate
            self.filename,
        ]

        self.proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def _close_writer(self):
        
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.stderr.close()
            self.proc.wait()

        self.proc = None
        self.command = None

    def __del__(self):
        self._close_writer()


# === Vec Env ===

# Stable baselines - VecEnv
def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.env_fn()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                try:
                    observation, reward, done, info = env.step(data)
                    if done:
                        # save final observation where user can get it, then reset
                        info['terminal_observation'] = observation
                        observation = env.reset()
                except:
                    info['terminal_observation'] = observation
                    observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == 'seed':
                remote.send(env.seed(data))
            elif cmd == 'reset':
                observation = env.reset()
                remote.send(observation)
            elif cmd == 'render':
                remote.send(env.render(data))
            elif cmd == 'close':
                env.close()
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError("`{}` is not implemented in the worker".format(cmd))
        except EOFError:
            break

class CloudpickleWrapper(object):
    def __init__(self, env_fn):
        self.env_fn = env_fn

    def __getstate__(self):
        return cloudpickle.dumps(self.env_fn)

    def __setstate__(self, env_fn):
        self.env_fn = cloudpickle.loads(env_fn)

class SubprocVecEnv():
    def __init__(self, env_fns, start_method=None):
        
        self.waiting = False
        self.closed = False
        self.remotes = None
        self.work_remotes = None
        self.processes = []
        self.n_envs = len(env_fns)
        # create subprocesses
        self.setup_subprocesses(env_fns, start_method)

        # recv spaces from subprocess
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()

        self.observation_space = observation_space
        self.action_space = action_space

    def setup_subprocesses(self, env_fns, start_method):

        if start_method is None:
            forkserver_available = 'forkserver' in multiprocessing.get_all_start_methods()
            start_method = 'forkserver' if forkserver_available else 'spawn'

        ctx = multiprocessing.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe(duplex=True) for _ in range(self.n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting=True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def seed(self, seed=None):
        for idx, remote in enumerate(self.remotes):
            remote.send(('seed', seed+idx))
        return [remote.recv() for remote in self.remotes]

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return flatten_obs(obs, self.observation_space)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

    def render(self, mode='human'):
        imgs = self.get_images()

        def tile_images(img_nhwc):
            img_nhwc = np.asarray(img_nhwc)
            n_images, height, width, n_channels = img_nhwc.shape
            # new_height was named H before
            new_height = int(np.ceil(np.sqrt(n_images)))
            # new_width was named W before
            new_width = int(np.ceil(float(n_images) / new_height))
            img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(n_images, new_height * new_width)])
            # img_HWhwc
            out_image = img_nhwc.reshape(new_height, new_width, height, width, n_channels)
            # img_HhWwc
            out_image = out_image.transpose(0, 2, 1, 3, 4)
            # img_Hh_Ww_c
            out_image = out_image.reshape(new_height * height, new_width * width, n_channels)
            return out_image

        bigimg = tile_images(imgs)
        if mode == 'human':
            import cv2
            cv2.imshow('vecenv', bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode =='rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        for pipe in self.remotes:
            pipe.send(('render', 'rgb_array'))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs
    
    def get_attr(self, attr_name, indices=None):
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in target_remotes]
    
    def set_attr(self, attr_name, value, indices=None):
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('set_attr', (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('env_method', (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def getattr_depth_check(self, name, already_found):
        if hasattr(self, name) and already_found:
            return '{0}.{1}'.format(type(self).__module__, type(self).__name__)
        else:
            return None

    def _get_target_remotes(self, indices):
        if indices is None:
            indices = range(self.n_envs)
        elif isinstance(indices, int):
            indices = [indices]
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]

    @property
    def unwrapped(self):
        raise NotImplementedError('Method not implemented')


class VecFrameStack():
    """
    Frame stacking wrapper for vectorized environment
    :param venv: (VecEnv) the vectorized environment to wrap
    :param n_stack: (int) Number of frames to stack
    """

    def __init__(self, venv, n_stack):
        self.venv = venv
        self.n_stack = n_stack
        wrapped_obs_space = venv.observation_space
        low = np.repeat(wrapped_obs_space.low, self.n_stack, axis=-1)
        high = np.repeat(wrapped_obs_space.high, self.n_stack, axis=-1)
        self.stackedobs = np.zeros((venv.n_envs,) + low.shape, low.dtype)
        observation_space = gym.spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        self.n_envs = venv.n_envs
        self.action_space = venv.action_space
        self.observation_space = observation_space
        self.class_attributes = dict(inspect.getmembers(self.__class__))

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        last_ax_size = observations.shape[-1]
        self.stackedobs = np.roll(self.stackedobs, shift=-last_ax_size, axis=-1)
        for i, done in enumerate(dones):
            if done:
                if 'terminal_observation' in infos[i]:
                    old_terminal = infos[i]['terminal_observation']
                    new_terminal = np.concatenate(
                        (self.stackedobs[i, ..., :-last_ax_size], old_terminal), axis=-1)
                    infos[i]['terminal_observation'] = new_terminal
                else:
                    warnings.warn(
                        "VecFrameStack wrapping a VecEnv without terminal_observation info")
                self.stackedobs[i] = 0
        self.stackedobs[..., -observations.shape[-1]:] = observations
        return self.stackedobs, rewards, dones, infos

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs

    def step(self, actions):
        """
        Step the environments with the given action
        :param actions: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode: str = 'human'):
        return self.venv.render(mode=mode)
    
    def get_images(self):
        return self.venv.get_images()

    def seed(self, seed=None):
        return self.venv.seed(seed)

    def get_attr(self, attr_name, indices=None):
        return self.venv.get_attr(attr_name, indices)

    def set_attr(self, attr_name, value, indices=None):
        return self.venv.set_attr(attr_name, value, indices)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)

    def close(self):
        self.venv.close()

    def __getattr__(self, name):
        """Find attribute from wrapped venv(s) if this wrapper does not have it.
        Useful for accessing attributes from venvs which are wrapped with multiple wrappers
        which have unique attributes of interest.
        """
        blocked_class = self.getattr_depth_check(name, already_found=False)
        if blocked_class is not None:
            own_class = "{0}.{1}".format(type(self).__module__, type(self).__name__)
            format_str = ("Error: Recursive attribute lookup for {0} from {1} is "
                          "ambiguous and hides attribute from {2}")
            raise AttributeError(format_str.format(name, own_class, blocked_class))

        return self.getattr_recursive(name)

    def _get_all_attributes(self):
        """Get all (inherited) instance and class attributes
        :return: (dict<str, object>) all_attributes
        """
        all_attributes = self.__dict__.copy()
        all_attributes.update(self.class_attributes)
        return all_attributes

    def getattr_recursive(self, name):
        """Recursively check wrappers to find attribute.
        :param name (str) name of attribute to look for
        :return: (object) attribute
        """
        all_attributes = self._get_all_attributes()
        if name in all_attributes:  # attribute is present in this wrapper
            attr = getattr(self, name)
        elif hasattr(self.venv, 'getattr_recursive'):
            # Attribute not present, child is wrapper. Call getattr_recursive rather than getattr
            # to avoid a duplicate call to getattr_depth_check.
            attr = self.venv.getattr_recursive(name)
        else:  # attribute not present, child is an unwrapped VecEnv
            attr = getattr(self.venv, name)

        return attr

    def getattr_depth_check(self, name, already_found):
        """See base class.
        :return: (str or None) name of module whose attribute is being shadowed, if any.
        """
        all_attributes = self._get_all_attributes()
        if name in all_attributes and already_found:
            # this venv's attribute is being hidden because of a higher venv.
            shadowed_wrapper_class = "{0}.{1}".format(type(self).__module__, type(self).__name__)
        elif name in all_attributes and not already_found:
            # we have found the first reference to the attribute. Now check for duplicates.
            shadowed_wrapper_class = self.venv.getattr_depth_check(name, True)
        else:
            # this wrapper does not have the attribute. Keep searching.
            shadowed_wrapper_class = self.venv.getattr_depth_check(name, already_found)

        return shadowed_wrapper_class