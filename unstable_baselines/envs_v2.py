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
from unstable_baselines.utils_v2 import (flatten_obs,
                                        StateObject)


__all__ = [
    'SeedEnv',
    'NoopResetEnv',
    'MaxAndSkipEnv',
    'EpisodicLifeEnv',
    'ClipRewardEnv',
    'WarpFrame',
    'VideoRecorder',
    'Monitor',
    'FrameStack',
    'SubprocVecEnv',
    'VecFrameStack',
]

LOG = logger.getLogger()


# === Env wrappers ---
class SeedEnv(gym.Wrapper):
    def __init__(self, env, seed):
        super().__init__(env)

        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

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
    def __init__(self, env, skip=4, blend=4):
        """
        Return only every `skip`-th frame (frameskipping)
        :param env: (Gym Environment) the environment
        :param skip: (int) number of `skip`-th frame
        """
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((blend,) + env.observation_space.shape, dtype=env.observation_space.dtype)
        self._skip = skip
        self._blend = blend

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

            # store observations to buffer
            self._obs_buffer[i % self._blend] = obs

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
#   output Stable-baselines style csv records
class _StatsRecorder(object):
    def __init__(self, directory: str = None, 
                       prefix:    str = None, 
                       ext:       str = 'monitor.csv', 
                       env_id:    str = None):
        '''StatsRecorder records environment stats: episodic rewards,
        total timesteps, time spent. And write them to a monitor file
        `{directory}/{prefix}.{ext}` in CSV format.

        CSV contents:
        The first line is a JSON comment indicating the timestamp 
        the environment started, and the name of the environment.
        The next line is the header line, indicating the column
        names [r, l, t] for each columns, where 'r' denotes
        rewards, 'l' episode length, 't' episode running time.
        The following lines are data for each episode.

        Example:
            # {t_start: timestamp, env_id: env_id}
            r, l, t
            320, 1000, 5.199693
            ...

        r: total episodic rewards
        l: episode length
        t: running time (time - t_start)

        Args:
            directory (str, optional): [description]. Defaults to None.
            prefix (str, optional): [description]. Defaults to None.
            ext (str, optional): [description]. Defaults to 'monitor.csv'.
            env_id (str, optional): [description]. Defaults to None.
        '''
        self.t_start = time.time()

        self.directory = directory or './'
        self.prefix    = prefix
        self.ext       = ext or 'monitor.csv'
        self.env_id    = env_id

        # initialize
        self._closed      = False
        self._need_reset  = True
        self.filepath     = None
        self.header       = None
        self.file_handler = None
        self.writer       = None

        self._stats = StateObject()
        self._stats.episodes = 0    # Current episode number
        self._stats.steps = 0       # Current timestep
        self._stats.start_steps = 0 # Timestep when this episode begins
        self._stats.rewards = []    # Cumulative rewards in one episode
        self._stats.ep_rewards = [] # Episode rewards
        self._stats.ep_lengths = [] # Episode lengths
        self._stats.ep_times   = [] # Episode time spent

        self._setup_writer()

    def _setup_writer(self):
        # setup csv file
        self.filepath = self._make_save_path(self.directory, self.prefix, self.ext)
        self.header = json.dumps({'t_start': self.t_start, 'env_id': self.env_id})

        filedir = os.path.dirname(self.filepath)
        relpath = os.path.relpath(self.filepath)

        self._ensure_dir_exists(filedir)
        LOG.debug('Writing monitor to: ' + relpath)

        # write header
        self.file_handler = open(self.filepath, 'wt')
        self.file_handler.write('#{}\n'.format(self.header))

        # create csv writer
        #   r: rewards
        #   l: length
        #   t: timestamp
        self.writer = csv.DictWriter(self.file_handler, fieldnames=('r', 'l', 't'))
        self.writer.writeheader()
        self.file_handler.flush()

    @property
    def stats(self):
        return self._stats

    def _make_save_path(self, base_path, prefix, ext):
        '''Make save path {base_path}/{prefix}.{ext}
        or {base_path}/{ext} if {preffix} is None
        '''
        paths = []
        if prefix:
            paths.append(str(prefix))
        paths.append(str(ext))

        filename = '.'.join(paths)
        path = os.path.join(base_path, filename)

        return os.path.abspath(path)

    def _ensure_dir_exists(self, path):
        '''Ensure nested directories exist
        if not, create them
        '''
        if not path:
            raise RuntimeError('Receive empty path: {}'.format(path))
        try:
            os.makedirs(path, exist_ok=True)
        except:
            pass

    def before_step(self, action):
        if self._need_reset:
            raise RuntimeError('Tried to step environment that needs reset')

        return action

    def after_step(self, obs, rew, done, info):
        # append rewards
        self.stats.rewards.append(rew)

        # episode ended
        if done:
            self._need_reset = True
            ep_rew = sum(self.stats.rewards)
            ep_len = len(self.stats.rewards)
            ep_time = time.time()-self.t_start
            ep_info = {'r': round(ep_rew, 6), 
                       'l': ep_len,
                       't': round(ep_time, 6)}

            self.stats.ep_rewards.append(ep_rew)
            self.stats.ep_lengths.append(ep_len)
            self.stats.ep_times.append(ep_time)

            # write episode info to csv
            if self.writer:
                self.writer.writerow(ep_info)
                self.file_handler.flush()
            
            info['episode'] = ep_info
        
        self.stats.steps += 1
        return obs, rew, done, info

    def before_reset(self, kwargs):
        '''Do nothing'''
        return kwargs

    def after_reset(self, obs):
        '''Reset episodic info'''
        self.stats.rewards = []
        self.stats.start_steps = self.stats.steps
        self.stats.episodes += 1
        self._need_reset = False
        return obs

    def close(self):
        if self.file_handler is not None:
            self.file_handler.close()

        self._closed = True

    def __del__(self):
        if not self._closed:
            self.close()

    def flush(self):
        pass


class _VideoEncoder(object):
    def __init__(self, path, 
                       frame_shape, 
                       video_shape, 
                       in_fps, 
                       out_fps):
        '''
        path: output path
        frame_shape: input frame shape (h,w,c)
        video_shape: output video shape (h,w,c)
        in_fps: input framerate
        out_fps: output video framerate
        '''
        self.path = path
        self.frame_shape = frame_shape
        self.video_shape = video_shape
        self.in_fps = in_fps
        self.out_fps = out_fps

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
            '-framerate', '{:d}'.format(self.in_fps), # input framerate,
            '-i', '-', # input from stdin

            # output
            '-vf', 'scale={:d}:{:d}'.format(*self.output_wh),
            '-c:v', 'libx264', # video codec
            '-pix_fmt', 'yuv420p',
            '-r', '{:d}'.format(self.out_fps),
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
            'input_fps': self.in_fps,
            'output_fps': self.out_fps
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


class _VideoRecorder(object):
    def __init__(self, env:   gym.Env, 
                       path:      str, 
                       meta_path: str, 
                       metadata: dict = None, 
                       width:     int = None, 
                       height:    int = None,
                       in_fps:    int = None,
                       out_fps:   int = None,
                       enabled:  bool = True):
        '''VideoRecorder captures video frames
        from `env` and saves to `path`.

        Args:
            env (gym.Emv): Environment.
            path (str): Video save path.
            meta_path (str): Metadata save path.
            metadata (dict, optional): Additional metadata. Defaults to None.
            width (int, optional): Video width. Defaults to None.
            height (int, optional): Video height. Defaults to None.
            in_fps (int, optional): Environment fps. Defaults to None.
            out_fps (int, optional): Output video fps. Defaults to None.
            enabled (bool, optional): Enable this recorder. Defaults to True.
        '''
        self.env       = env
        self.path      = path
        self.meta_path = meta_path
        self.width     = width
        self.height    = height
        self.in_fps    = in_fps
        self.out_fps   = out_fps
        self.enabled   = enabled
        self.metadata  = metadata or {}
        self._closed   = False
        if not self.enabled:
            return

        if not path:
            raise ValueError("'path' not specified: {}".format(path))
        if not meta_path:
            raise ValueError("'meta_path' not specified: {}".format(meta_path))

        in_fps  = env.metadata.get('video.frames_per_second', 30)
        out_fps = env.metadata.get('video.output_frames_per_second', in_fps)

        self.in_fps  = self.in_fps or self.out_fps or in_fps
        self.out_fps = self.out_fps or self.in_fps or out_fps

        self._async = env.metadata.get('semantics.async')
        modes       = env.metadata.get('render.modes', [])

        if 'rgb_array' not in modes:
            LOG.warn('Disabling video recorder because {} not '
                    'supports video mode "rgb_array"'.format(env))
            self.enabled = False
            return
        
        # pybullet_envs: set render width, height to the environment
        if (width is not None) and hasattr(env.unwrapped, '_render_width'):
            env.unwrapped._render_width = width
        if (height is not None) and hasattr(env.unwrapped, '_render_height'):
            env.unwrapped._render_height = height

        # Create directory
        self._ensure_dir_exists(os.path.abspath(self.path))
        self._ensure_dir_exists(os.path.abspath(self.meta_path))
        
        # Write metadata
        self.metadata['content_type'] = 'video/mp4'
        self.write_metadata()

        self.encoder = None
        self.broken  = False
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
                LOG.error('Env returned None on render(). Disabling further '
                        'rendering for video recorder by marking as disabled: '
                        '{}'.format(self.path))
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
            LOG.warn('Cleaning up paths for broken video recorder: '
                        '{}'.format(self.path))

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
            json.dump(self.metadata, f, indent=4)

    def _ensure_dir_exists(self, path):
        '''Ensure nested directories exist
        if not, create them
        '''
        if not path:
            raise RuntimeError('Receive empty path: {}'.format(path))
        try:
            os.makedirs(path, exist_ok=True)
        except:
            pass
        

    def _get_frame_shape(self, frame):
        if len(frame.shape) != 3:
            raise RuntimeError('Receive unknown frame shape: '
                                '{}'.format(frame.shape))

        return frame.shape # h,w,c

    def _get_video_shape(self, frame):
        if len(frame.shape) != 3:
            raise RuntimeError('Receive unknown frame shape: '
                                '{}'.format(frame.shape))

        h,w,c = frame.shape

        self.width = self.width if self.width else w
        self.height = self.height if self.height else h

        return (self.height, self.width, c) # h,w,c

    def _encode_image_frame(self, frame):
        if not self.encoder:
            frame_shape = self._get_frame_shape(frame)
            video_shape = self._get_video_shape(frame)
            self.encoder = _VideoEncoder(self.path, 
                                         frame_shape, 
                                         video_shape, 
                                         self.in_fps, 
                                         self.out_fps)

            self.metadata['encoder_version'] = self.encoder.version_info
            self.metadata['video_info'] = self.encoder.video_info
        
        try:
            self.encoder.capture_frame(frame)

        except RuntimeError as e:
            LOG.exception('Tried to pass invalid video frame, '
                        'marking as broken: {}'.format(self.path))
            self.metadata['exc'] = str(e)
            self.broken = True
        else:
            self.empty = False

    def __del__(self):
        self.close()

# ===

class MonitorGroupWrapper(gym.Wrapper):
    '''MonitoGroupWrapper provides an interface
    for those inherited subclasses (wrappers) 
    to find each other by simply calling
    get_component(other subclasses)
    '''
    def __init__(self, env, depth: int=None):
        super().__init__(env)

        self.set_component(self, depth=depth)

    def get_component(self, component_type, depth: int=None):
        '''Get the `depth`-th wrapper whose type is 
        `component_type`. If `depth` is None, act
        as same as `depth`=1.
        '''
        if (depth is not None) and (depth <= 0):
            return
        if (component_type is type(self)):
            if depth is None or depth == 1:
                return self
            else:
                depth -= 1
        if hasattr(self.env, 'set_component'):
            return self.env.get_component(component_type, depth=depth)
        return None

    def set_component(self, component, depth: int=None):
        '''Set top `depth` wrappers with `component`. If 
        `depth` is None, set all of the wrappers.
        '''
        if (depth is not None) and (depth <= 0):
            return
        if ((component is not self)
                and isinstance(component, MonitorGroupWrapper)):
            self._set_component(component)
            if depth is not None:
                depth -= 1
        if hasattr(self.env, 'set_component'):
            self.env.set_component(component, depth=depth)

    @abc.abstractmethod
    def _set_component(self, component):
        '''Override this method to receive 
        set_component() signals from other 
        subclasses.
        '''
        
        raise NotImplementedError('Method not implemented')


class VideoRecorder(MonitorGroupWrapper):
    def __init__(self, env:  gym.Env, 
                      directory: str = None, 
                      prefix:    str = None,
                      suffix:    str = None,
                      ext:       str = None,
                      meta_ext:  str = None,
                      width:     int = None,
                      height:    int = None,
                      fps:       int = None,
                      in_fps:    int = None,
                      callback       = None,
                      metadata: dict = None,
                      force:    bool = False):
        super().__init__(env=env)

        default_suffix = 'ep{episode:06d}.{start_steps}-{end_steps}'
        get = lambda v, de: de if v is None else v

        # Default callback
        if callback is None:
            callback = VideoRecorder.capped_cubic_video_schedule
        elif callback is True:
            callback = lambda ep: True
        elif callback is False:
            callback = lambda ep: False
        elif not callable(callback):
            raise RuntimeError('You must provide a function, None, or False for'
                            'callback not {}: {}'.format(type(callback), callback))

        directory = directory or './'
        if force:
            self._clear_dir(directory)
        elif not self._dir_is_empty(directory):
            raise RuntimeError('Trying to write to non-empty video '
                                'directory {}'.format(directory))
        
        self._ensure_dir_exists(directory)

        self._directory = directory
        self._prefix    = prefix
        self._suffix    = suffix or default_suffix
        self._ext       = ext or 'video.mp4'
        self._width     = width
        self._height    = height
        self._fps       = fps
        self._in_fps    = in_fps
        self._meta_ext  = meta_ext or 'metadata.json'
        self._metadata  = metadata
        self._callback  = callback

        # disabled if something error happend
        self._enabled        = True
        self._video_recorder = None
        self._start_steps    = 0
        self._num_steps      = 0
        self._num_episodes   = 0
        self._episode_rewards = 0
        self._is_reset       = True
        self._monitor        = super().get_component(Monitor)
        self._monitor_stats  = None

        self._stats = StateObject()
        self._stats.episodes    = 0   # Current episode number
        self._stats.steps       = 0   # Current timestep
        self._stats.start_steps = 0   # Timestep when this episode begins
        self._stats.rewards     = []  # Cumulative rewards in one episode

    def _set_component(self, component):
        '''Override MonitorGroupWrapper._set_component'''
        if type(component) is Monitor:
            self._monitor = component
        else:
            LOG.debug('Receive unknown component: '
                        '{}'.format(component))

    @staticmethod
    def capped_cubic_video_schedule(episode):
        if episode < 1000:
            return int(round(episode ** (1./3))) ** 3 == episode
        else:
            return episode % 1000 == 0

    @property
    def need_record(self):
        '''Whether we should record this episode'''
        return self._callback(self.stats.episodes)

    def _get_monitor_stats(self):
        if self._monitor_stats is not None:
            return self._monitor_stats

        stats = None
        if self._monitor is not None:
            if hasattr(self._monitor, 'stats'):
                stats = self._monitor.stats
        return stats

    @property
    def stats(self):
        '''Get stats from Monitor or self (read-only)'''
        monitor_stats = self._get_monitor_stats()
        stats = monitor_stats or self._stats
        return stats

    def step(self, action):
        '''Step env'''
        action = self._before_step(action)
        obs, rew, done, info = self.env.step(action)
        return self._after_step(obs, rew, done, info)

    def reset(self, **kwargs):
        '''Reset env'''
        kwargs = self._before_reset(kwargs)
        obs = self.env.reset(**kwargs)
        return self._after_reset(obs)

    def close(self):
        '''Close VideoRecorder'''
        super().close()

        if not self._enabled:
            return
        self._close_and_save_video_recorder()
        # disable VideoRecorder (self)
        self._enabled = False
    
    def _before_step(self, action):
        '''Create a new recorder if the video recorder is reset'''
        if not self._enabled:
            return action

        # Create a new recorder
        if self._is_reset:
            # {base_path}/{prefix}.{ext}.[tempfile].mp4
            temppath = self._make_save_path(base_path=self._directory,
                                            prefix=self._prefix, 
                                            suffix=None,
                                            ext='ep{episode}.')
            self._new_video_recorder(temppath)
            self._is_reset = False

        return action

    def _after_step(self, obs, rew, done, info):
        '''Capture frames, increase step counter'''
        if not self._enabled:
            return obs, rew, done, info

        self._video_recorder.capture_frame()
        self._stats.steps += 1
        self._stats.rewards.append(rew)

        return obs, rew, done, info

    def _before_reset(self, kwargs):
        '''Do nothing'''
        if not self._enabled:
            return kwargs

        # close video recorder and copy tempfile to `filename`
        # stats recorder is reset at after reset
        self._close_and_save_video_recorder()

        return kwargs

    def _after_reset(self, obs):
        '''Save video and increase episode counter'''
        if not self._enabled:
            return obs

        # increate counter
        self._stats.start_steps = self._stats.steps
        self._stats.episodes += 1
        self._stats.rewards = []
        # mark as reset
        # need to create a new video recorder for the 
        # new episode
        self._is_reset = True

        return obs

    def _ensure_dir_exists(self, path):
        '''Ensure nested directories exist
        if not, create them
        '''
        if not path:
            raise RuntimeError('Receive empty path: {}'.format(path))
        try:
            os.makedirs(path, exist_ok=True)
        except:
            pass

    def _format_path(self, path):
        '''Format path
        Format params:
            episode: Episode number
            start_steps: Number of timesteps the episode started
            end_steps: Number of timesteps the epsiode ended
        '''
        return path.format(episode=self.stats.episodes,
                           start_steps=self.stats.start_steps,
                           end_steps=self.stats.steps)

    def _make_save_path(self, base_path, prefix, suffix, ext):
        '''Make save path {base_path}/{prefix}.{suffix}.{ext}
        or {base_path}/{suffix}.{ext} if {preffix} is None
        or {base_path}/{prefix}.{ext} if {suffix} is None
        or {base_path}/{ext} if both {preffix} and {suffix} are None
        '''
        paths = []
        if prefix:
            paths.append(str(prefix))
        if suffix:
            paths.append(str(suffix))
        paths.append(str(ext))

        filename = '.'.join(paths)
        path = os.path.join(base_path, filename)

        abspath = os.path.abspath(path)
        return self._format_path(abspath)

    def _close_and_save_video_recorder(self):
        
        if (not self._video_recorder) or (not self._video_recorder.enabled):
            return

        self._video_recorder.close()
        
        # make video/metadata path
        video_abspath = self._make_save_path(self._directory, self._prefix,
                                        self._suffix, self._ext)
        meta_abspath  = self._make_save_path(self._directory, self._prefix,
                                        self._suffix, self._meta_ext)
        
        video_path = os.path.relpath(video_abspath)
        meta_path  = os.path.relpath(meta_abspath)

        # update metadata
        episode_metadata = {
            'episode': self.stats.episodes,
            'start_steps': self.stats.start_steps,
            'end_steps': self.stats.steps,
            'episode_length': self.stats.steps - self.stats.start_steps,
            'episode_rewards': sum(self.stats.rewards),
        }

        self._video_recorder.metadata['episode_info'] = episode_metadata
        self._video_recorder.write_metadata()

        # ensure the save paths exist
        self._ensure_dir_exists(os.path.dirname(video_abspath))
        self._ensure_dir_exists(os.path.dirname(meta_abspath))

        # copy tempfiles to the specified locations
        # save metadata
        if os.path.isfile(self._video_recorder.meta_path):
            os.rename(self._video_recorder.meta_path, meta_path)
            LOG.debug('Saving metadata to: {}'.format(meta_path))
        else:
            LOG.warn('Metadata not found: {}'.format(meta_path))
        
        # save video
        if os.path.isfile(self._video_recorder.path):
            os.rename(self._video_recorder.path, video_path)
            LOG.debug('Saving video to: {}'.format(video_path))
        else:
            if self._video_recorder.broken:
                LOG.warn('Failed to save video, the VideoRecorder is broken,'
                        'for more info: {}'.format(meta_path))
            else:
                LOG.error('Failed to save video, missing tempfile, '
                        'for more info: {}'.format(meta_path))

    def _new_video_recorder(self, temppath):
        '''Create a new video recorder

        Args:
            temppath (str): Temporary path to save video and metadata (will
                be renamed in self._close_and_save_video_recorder())
        '''

        base_path = os.path.dirname(temppath)
        prefix = os.path.basename(temppath)
        video_path = None
        meta_path = None

        if self._enabled:
            self._ensure_dir_exists(base_path)
            # generate random filename
            with tempfile.NamedTemporaryFile(dir=base_path, prefix=prefix,\
                    suffix='.mp4', delete=False) as f:
                video_path = f.name
            with tempfile.NamedTemporaryFile(dir=base_path, prefix=prefix,\
                    suffix='.json', delete=False) as f:
                meta_path = f.name

        self._video_recorder = _VideoRecorder(env=self.env,
                                              path=video_path,
                                              meta_path=meta_path,
                                              metadata=self._metadata,
                                              width=self._width,
                                              height=self._height,
                                              in_fps=self._in_fps,
                                              out_fps=self._fps,
                                              enabled=self.need_record)
        # capture the first frame
        self._video_recorder.capture_frame()

    def _dir_is_empty(self, directory):
        '''Returns a bool, determine whether the directory is empty.'''
        if not directory:
            return True
        
        # if directory does not exist, or is empty => True
        return ((not os.path.isdir(directory))
                    or (len(os.listdir(directory)) == 0))


    def _clear_dir(self, directory):
        '''Do nothing'''
        if not directory:
            return True
        
        #TODO: clear folder !!danger!!
        pass

class Monitor(MonitorGroupWrapper):
    def __init__(self, env:   gym.Env, 
                       directory: str = 'monitor', 
                       prefix:    str = None, 
                       ext:       str = 'monitor.csv', 
                       force:    bool = False):
        '''Monitor records timesteps and rewards to 
        {directory}/{prefix}.{ext}

        Args:
            env (gym.Env): Enviornment to monitor.
            directory (str, optional): Base directory. Defaults to 'monitor'.
            prefix (str, optional): Filename prefix. Defaults to None.
            ext (str, optional): File extention. Defaults to 'monitor.csv'.
            force (bool, optional): Force to save files. Defaults to False.
        '''
        super().__init__(env)

        self._directory = directory or './'
        self._prefix    = prefix
        self._ext       = ext or 'monitor.csv'

        self.env_id = env.unwrapped.spec.id

        self._stats_recorder = _StatsRecorder(directory=self._directory,
                                              prefix=self._prefix,
                                              ext=self._ext,
                                              env_id=self.env_id)

    def _set_component(self, component):
        '''Override MonitorGroupWrapper._set_component'''
        pass

    @property
    def stats(self):
        return self._stats_recorder._stats

    def step(self, action):
        action = self._before_step(action)
        obs, rew, done, info = self.env.step(action)
        return self._after_step(obs, rew, done, info)

    def reset(self, **kwargs):
        kwargs = self._before_reset(kwargs)
        obs = self.env.reset(**kwargs)
        return self._after_reset(obs)

    def close(self):
        super().close()
        self._stats_recorder.close()

    def _before_step(self, action):
        return self._stats_recorder.before_step(action)

    def _after_step(self, obs, rew, done, info):
        return self._stats_recorder.after_step(obs, rew, done, info)

    def _before_reset(self, kwargs):
        return self._stats_recorder.before_reset(kwargs)

    def _after_reset(self, obs):
        return self._stats_recorder.after_reset(obs)

# ===


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
