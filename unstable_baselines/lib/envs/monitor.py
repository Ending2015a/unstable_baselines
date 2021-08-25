# --- bulid in ---
import os
import abc
import csv
import json
import time
import tempfile
import distutils
import subprocess

# --- 3rd party ---
import gym
import numpy as np

# --- my module ---
from unstable_baselines import logger
from unstable_baselines.lib import utils as ub_utils

LOG = logger.getLogger()

__all__ = [
    'Monitor',
    'MonitorToolChain',
    'StatsRecorder',
    'VideoRecorder',
]

# === Monitor ===

class Monitor(gym.Wrapper):
    def __init__(self, env:            gym.Env,
                       root_dir:           str = './monitor',
                       prefix:             str = None,
                       force:             bool = True,
                       allow_early_reset: bool = True,
                       video:             bool = False,
                       video_kwargs:      dict = {}):
        super().__init__(env)

        self.root_dir = root_dir or './'
        self.prefix   = prefix
        self.env_id   = env.unwrapped.spec.id
        self.allow_early_reset = allow_early_reset

        self._stats_recorder = StatsRecorder(
            root_dir          = self.root_dir,
            prefix            = self.prefix,
            ext               = self.monitor_ext,
            allow_early_reset = self.allow_early_reset
        )
        self._tools = []

        self.add_tool(self._stats_recorder)
        if video:
            self.add_tool(VideoRecorder(**video_kwargs))
    
    @property
    def stats(self):
        '''Return StateRecorder'''
        return self._stats_recorder._stats

    @property
    def tools(self):
        '''Return Toolchain'''
        return self._tools

    def add_tool(self, tool: 'MonitorToolChain'):
        '''Add monitor tool'''
        if hasattr(tool, 'set_monitor'):
            tool.set_monitor(self)
        self._tools.append(tool)
        return self

    def add_tools(self, tool_list: list):
        '''Add monitor toolchain'''
        for tool in tool_list:
            self.add_tool(tool)
        return self

    def step(self, act):
        act = self._before_step(act)
        obs, rew, done, info = self.env.step(act)
        return self._after_step(obs, rew, done, info)

    def reset(self, **kwargs):
        kwargs = self._before_reset(**kwargs)
        obs = self.env.reset(**kwargs)
        return self._after_reset(obs)

    def close(self):
        super().close()
    
    def _before_step(self, act):
        for tool in self._tools:
            act = tool._before_step(act)
        return act
    
    def _after_step(self, *data):
        for tool in self._tools:
            data = tool._after_step(*data)
        return data

    def _before_reset(self, **kwargs):
        for tool in self._tools:
            kwargs = tool._before_reset(**kwargs)
        return kwargs

    def _after_reset(self, obs):
        for tool in self._tools:
            obs = tool._after_reset(obs)
        return obs



class MonitorToolChain(metaclass=abc.ABCMeta):
    def __init__(self):
        self._monitor = None
        self._stats   = None
        self._env     = None

    def set_monitor(self, monitor: 'Monitor'):
        self._monitor = monitor
        self._stats   = monitor.stats
        self._env     = monitor.env

    def _before_step(self, act):
        return act

    def _after_step(self, *data):
        return data

    def _before_reset(self, **kwargs):
        return kwargs

    def _after_reset(self, obs):
        return obs

    def close(self):
        pass


# === Stats Recorder ===

class Stats(ub_utils.StateObject):
    def __init__(self):
        super().__init__()
        self.episodes    = 0  # Cirrent episode number
        self.steps       = 0  # Current step number
        self.start_steps = 0  # Step when this episode begins
        self.rewards     = 0  # Cumulative rewards in one episode
        self.ep_rewards  = [] # Rewards per episode
        self.ep_lengths  = [] # Episode lengths per episode
        self.ep_times    = [] # Episode time per episode

class StatsRecorder(MonitorToolChain):
    monitor_ext = 'monitor.csv'
    def __init__(self, root_dir:           str = None, 
                       prefix:             str = None, 
                       ext:                str = None,
                       allow_early_reset: bool = True):
        '''StatsRecorder records environment stats: episodic rewards,
        total timesteps, time spent. And write them to a monitor file
        `{root_dir}/{prefix}.{ext}` in CSV format.

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
            root_dir (str, optional): [description]. Defaults to None.
            prefix (str, optional): [description]. Defaults to None.
            ext (str, optional): [description]. Defaults to 'monitor.csv'.
            env_id (str, optional): [description]. Defaults to None.
        '''
        self.t_start = time.time()

        self.root_dir = root_dir
        self.prefix   = prefix
        self.ext      = ext
        self.env_id   = env_id
        self.allow_early_reset = allow_early_reset

        # initialize
        self._closed      = False
        self._need_reset  = True
        self.filepath     = None
        self.header       = None
        self.file_handler = None
        self.writer       = None

        self._stats = Stats()

        self._setup_writer()

    def set_monitor(self, monitor: Monitor):
        self._monitor = monitor
        self.root_dir = self.root_dir or monitor.root_dor
        self.prefix   = self.prefix or monitor.prefix
        self.ext      = self.ext or self.monitor_ext
        self.env_id   = self.env_id or monitor.env_id

    def _setup_writer(self):
        # setup csv file
        self.filepath = self._make_save_path(self.root_dir, self.prefix, self.ext)
        self.header = json.dumps({'t_start': self.t_start, 'env_id': self.env_id})

        filedir = os.path.dirname(self.filepath)
        relpath = os.path.relpath(self.filepath)

        self._create_path(filedir)
        LOG.debug('Writing monitor to: ' + relpath)

        # write header
        self.file_handler = open(self.filepath, 'wt')
        self.file_handler.write('#{}\n'.format(self.header))

        # create csv writer
        #   r: rewards
        #   l: length
        #   t: timestamp
        self.writer = csv.DictWriter(self.file_handler, 
                    fieldnames=('rewards', 'length', 'walltime'))
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

    def _create_path(self, path):
        '''Ensure nested directories exist
        if not, create them
        '''
        os.makedirs(path, exist_ok=True)

    def _before_step(self, act):
        if self._need_reset:
            raise RuntimeError('Tried to step environment that needs reset')

        return act

    def _after_step(self, obs, rew, done, info):
        # append rewards
        self.stats.rewards += rew
        self.stats.steps += 1

        # episode ended
        if done:
            self._need_reset = True
            ep_info = self.write_results()
            info['episode'] = ep_info
        
        return obs, rew, done, info

    def _before_reset(self, **kwargs):
        '''Do nothing'''
        return kwargs

    def _after_reset(self, obs):
        '''Reset episodic info'''
        if not self._need_reset:
            if self.allow_early_reset:
                self._write_results()
        
        self.stats.rewards = 0
        self.stats.start_steps = self.stats.steps
        self.stats.episodes += 1
        self._need_reset = False
        return obs

    def _write_results(self):
        self._need_reset = True
        ep_rew = self.stats.rewards
        ep_len = self.stats.steps - self.stats.start_steps
        ep_time = time.time()-self.t_start
        ep_info = {'rewards': round(ep_rew, 6), 
                    'length': ep_len,
                    'walltime': round(ep_time, 6)}

        self.stats.ep_rewards.append(ep_rew)
        self.stats.ep_lengths.append(ep_len)
        self.stats.ep_times.append(ep_time)

        # write episode info to csv
        if self.writer:
            self.writer.writerow(ep_info)
            self.file_handler.flush()
        return ep_info

    def close(self):
        if self.file_handler is not None:
            self.file_handler.close()

        self._closed = True

    def __del__(self):
        if not self._closed:
            self.close()

    def flush(self):
        pass


# === Video Recorder ===


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
        self._create_path(os.path.abspath(self.path))
        self._create_path(os.path.abspath(self.meta_path))
        
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

    def _create_path(self, path):
        '''Ensure nested directories exist
        if not, create them
        '''
        os.makedirs(path, exist_ok=True)

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


class VideoRecorder(MonitorToolChain):
    video_ext    = 'video.mp4'
    metadata_ext = 'metadata.json'
    video_suffix = 'ep{episode:06d}.{start_steps}-{steps}'
    def __init__(self, root_dir:  str = None,
                       prefix:    str = None,
                       width:     int = None,
                       height:    int = None,
                       fps:       int = None,
                       env_fps:   int = None,
                       interval:  int = None,
                       metadata: dict = None,
                       force:    bool = True):
        super().__init__()

        # Default callback
        if interval is None:
            schedule = VideoRecorder.capped_cubic_video_schedule
        elif isinstance(interval, int):
            schedule = lambda ep: (ep+1)%interval==0
        elif not callable(interval):
            raise RuntimeError('You must provide a function or int for '
                f'`interval` not {type(interval)}: {interval}')

        self._root_dir  = root_dir
        self._prefix    = prefix
        self._suffix    = self.video_suffix
        self._video_ext = self.video_ext
        self._width     = width
        self._height    = height
        self._fps       = fps
        self._env_fps   = env_fps
        self._meta_ext  = self.metadata_ext
        self._metadata  = metadata
        self._schedule  = schedule

        self._is_reset  = False # Whether the env is reset
        self._env       = None
        self._enabled   = True
        self._recorder  = None
        self._monitor   = None
        self._stats     = None

        # initialized in `set_monitor`

    def set_monitor(self, monitor: 'Monitor'):
        self._monitor = monitor
        self._stats   = monitor.stats
        self._env     = monitor.env
        # Default save root: {monitor.root_dir}/videos/
        self._root_dir = (self._root_dir 
            or os.path.join(self._monitor.root_dir, 'videos/'))
        # Default prefix: {monitor.prefix}
        self._prefix = (self._prefix or self._monitor.prefix)
    
    @staticmethod
    def capped_cubic_video_schedule(episode):
        if episode < 1000:
            return int(round(episode ** (1./3))) ** 3 == episode
        else:
            return episode % 1000 == 0
    
    @property
    def stats(self):
        return self._monitor.stats

    @property
    def need_record(self):
        '''Whether we should record this episode'''
        return self._schedule(self.stats.episodes)

    def close():
        '''Close VideoRecorder'''
        if not self._enabled:
            return
        self._close_and_save_video_recorder()
        # disable VideoRecorder
        self._enabled = False

    def _before_step(self, act):
        if not self._enabled:
            return act

        # If the env is reset, create a new recorder
        if self._is_reset:
            self._new_recorder()
            # capture the first frame
            self._recorder.capture_frame()
            self._is_reset = False
        return act

    def _after_step(self, *data):
        if not self._enabled:
            return data
        self._recorder.capture_frame()
        return data

    def _before_reset(self, **kwargs):
        if not self._enabled:
            return kwargs
        
        self._close_and_save_recorder()
        return kwargs

    def _after_reset(self, obs):
        if not self._enabled:
            return obs
        self._is_reset = True
        return obs

    def _create_path(self, path):
        os.makedirs(path, exist_ok=True)

    def _get_temp_path(self, dir, prefix, suffix):
        with tempfile.NamedTemporaryFile(dir=dir, prefix=prefix,
                suffix=suffix, delete=False) as f:
            path = f.name
        return path

    def _make_path(self, root_dir, prefix, suffix, ext):
        paths = []
        if prefix:
            paths.append(str(prefix))
        if suffix:
            paths.append(str(suffix))
        paths.append(str(ext))

        filename = '.'.join(paths)
        path = os.path.join(root_dir, filename)
        abspath = os.path.abspath(path)
        return abspath.format(**self.stats)

    def _close_and_save_recorder(self):
        if((not self._recorder)
                or (not self._recorder.functional)):
            return

        self._recorder.close()

        video_path = self._make_path(
            root_dir = self._root_dir,
            prefix   = self._prefix,
            suffix   = self._suffix,
            ext      = self._video_ext
        )
        meta_path  = self._make_path(
            root_dir = self._root_dir,
            prefix   = self._prefix,
            suffix   = self._suffix,
            ext      = self._meta_ext
        )

        video_relpath = os.path.relpath(video_path)
        meta_relpath  = os.path.relpath(meta_path)

        # update metadata
        episode_metadata = {
            'episode': self.stats.episodes,
            'start_steps': self.stats.start_steps,
            'end_steps': self.stats.steps,
            'episode_length': self.stats.steps - self.stats.start_steps,
            'episode_rewards': self.stats.rewards
        }

        self._recorder.metadata['episode_info'] = episode_metadata
        self._recorder.write_metadata()
        # create save paths
        self._create_path(os.path.dirname(video_path))
        self._create_path(os.path.dirname(meta_path))

        # copy tempfiles to the specified locations
        if os.path.isfile(self._recorder.meta_path):
            os.rename(self._recorder.meta_path, meta_path)
            LOG.debug(f'Saving metadata to: {meta_path}')
        else:
            LOG.warn(f'Metadata not found: {meta_path}')

        if os.path.isfile(self._recorder.path):
            os.rename(self._recorder.path, video_path)
            LOG.debug(f'Saving video to: {video_path}')
        else:
            if self._recorder.broken:
                LOG.warn('Failed to save video, the VideoRecorder is broken,'
                        f'for more info: {meta_path}')
            else:
                LOG.error('Failed to save video, missing tempfile, '
                        f'for more info: {meta_path}')

    def _new_recorder(self):
        '''Create a new video recorder
        '''

        temppath = self._make_path(
            root_dir = self._root_dir,
            prefix   = self._prefix,
            suffix   = None,
            ext      ='ep{episodes:06d}'
        )

        root_dir   = os.path.dirname(temppath)
        prefix     = os.path.basename(temppath)
        video_path = None
        meta_path  = None

        if self.need_record:
            self._create_path(base_path)
            video_path = self._get_temp_path(root_dir, prefix, '.mp4')
            meta_path  = self._get_temp_path(root_dir, prefix, '.json')

        self._recorder = _VideoRecorder(
            env       = self._env,
            path      = video_path,
            meta_path = meta_path,
            metadata  = self._metadata,
            width     = self._width,
            height    = self._height,
            in_fps    = self._env_fps,
            out_fps   = self._fps,
            enabled   = self.need_record
        )
    
    def _path_is_empty(self, path):
        if not path:
            return True
        
        return ((not os.path.isdir(path))
            or (len(os.listdir(path)) == 0))
