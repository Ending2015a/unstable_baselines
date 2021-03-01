# --- built in ---
import os
import sys
import time
import random
import logging

# --- 3rd party ---
import gym 

import numpy as np

# --- my module ---
from .utils import flatten_obs

__all__ = [
    'NoopResetEnv',
    'MaxAndSkipEnv',
    'EpisodicLifeEnv',
    'ClipRewardEnv',
    'WarpFrame',
    'Monitor',
    'SubprocVecEnv',
    'VecFrameStack',
]

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





# Stable baselines - Monitor
class Monitor(gym.Wrapper):
    EXT = "monitor.csv"

    def __init__(self, env, path=None, prefix=None):
        super(Monitor, self).__init__(env=env)
        self.t_start = time.time()

        # setup csv file
        self.filename = self.make_filename(path, prefix)
        self.file_handler = open(self.filename, "wt")
        self.file_handler.write('#%s\n' % json.dumps({"t_start": self.t_start, 'env_id': env.spec and env.spec.id}))
        self.writer = csv.DictWriter(self.file_handler, fieldnames=('r', 'l', 't')) # rewards/length/time
        self.writer.writeheader()
        self.file_handler.flush()

        # init
        self.rewards = None
        self.need_reset = None
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
    
    @classmethod
    def make_filename(cls, path, prefix):
        # filename = [path]/[prefix].[EXT]
        if prefix is None:
            filename = cls.EXT
        else:
            filename = prefix + '.' + cls.EXT

        # create directories
        if path is not None:
            os.makedirs(path, exist_ok=True)
            filename = os.path.join(path, filename)
        
        LOG.info('Monitor: ' + filename)
        return filename

    def reset(self, **kwargs):
        self.rewards = []
        self.need_reset = False

        return self.env.reset(**kwargs)


    def step(self, action):
        
        if self.need_reset:
            raise RuntimeError('Tried to step environment that needs reset')

        # step env
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)

        # if the episode ended
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

    def close(self):
        super().close()
        if self.file_handler is not None:
            self.file_handler.close()

    def get_total_steps(self):
        return self.total_steps

    def get_episode_rewards(self):
        return self.episode_rewards

    def get_episode_lengths(self):
        return self.episode_lengths

    def get_episode_times(self):
        return self.episode_times

# === Vec Env ===

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
        self.num_envs = len(env_fns)
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

        self.remotes, self.work_remotes = zip(*[ctx.Pipe(duplex=True) for _ in range(self.num_envs)])
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
            indices = range(self.num_envs)
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
        self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)
        observation_space = gym.spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        self.num_envs = venv.num_envs
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