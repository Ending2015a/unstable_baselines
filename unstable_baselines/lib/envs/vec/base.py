# --- built in ---
import os
import abc
import sys

from typing import Union

# --- 3rd party ---
import gym
import numpy as np

# --- my module ---
from unstable_baselines.lib import utils as ub_utils

[lambda i=j:j for j in range(10)]

# The implementation mainly follows tianshou/venvs.py
# habitat-lab/vector_env.py and stable-baselines3/subproc_vec_env.py
class BaseEnvWorker(metaclass=abc.ABCMeta):
    def __init__(self, env_fn, auto_reset: bool):
        self._env_fn = env_fn
        self._auto_reset = auto_reset

    def setup(self):
        self.observation_space = self.getattr('observation_space')
        self.action_space      = self.getattr('action_space')
        self.metadata          = self.getattr('metadata')
        self.reward_range      = self.getattr('reward_range')
        self.spec              = self.getattr('spec')

    @abc.abstractmethod
    def getattr(self, attrname: str):
        raise NotImplementedError

    @abc.abstractmethod
    def setattr(self, attrname: str, value):
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def step_async(self, act):
        raise NotImplementedError

    @abc.abstractmethod
    def step_wait(self):
        raise NotImplementedError

    @abc.abstractmethod
    def seed(self, seed):
        self.action_space.seed(seed)

    @abc.abstractmethod
    def render(self):
        raise NotImplementedError

    @abc.abstractmethod
    def close_async(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def close_wait(self):
        raise NotImplementedError

# The implementation mainly follows tianshou/venvs.py
# habitat-lab/vector_env.py and stable-baselines3/subproc_vec_env.py
class BaseVecEnv(gym.Env):
    def __init__(self,
        env_fns:      list,
        worker_class: BaseEnvWorker,
        rms_norm:     Union[str, bool, ub_utils.RMSNormalizer] = None,
        auto_reset:   bool = True
    ):
        self._worker_class = worker_class
        self.workers       = [worker_class(fn, auto_reset) for fn in env_fns]
        self.n_envs        = len(self.workers)
        self._closed       = False
        # setup workers
        [w.setup() for w in self.workers]

        # init properties
        self.observation_spaces = [w.observation_space for w in self.workers]
        self.action_spaces      = [w.action_space for w in self.workers]
        self.metadatas          = [w.metadata for w in self.workers]
        self.reward_ranges      = [w.reward_range for w in self.workers]
        self.specs              = [w.spec for w in self.workers]

        self.observation_space = self.observation_spaces[0]
        self.action_space      = self.action_spaces[0]
        self.metadata          = self.metadatas[0]
        self.reward_range      = self.reward_ranges[0]
        self.spec              = self.specs[0]

        self._rms_norm = self._get_rms_norm_opt(rms_norm)

    @property
    def rms_norm(self):
        return self._rms_norm

    @property
    def closed(self):
        return self._closed

    def __len__(self):
        return self.n_envs

    def getattrs(self, attrname: str, id: list=None):
        '''Retrieve attributes from each environment

        Args:
            attrname (str): Attribute name
            id (list, optional): Environment indices. Defaults to None.

        Returns:
            list: A list of retrieved attributes
        '''
        self._assert_not_closed()
        ids = self._get_ids(id)
        return [self.workers[id].getattr(attrname) for id in ids]

    def setattrs(self, attrname:str, value=None, values: list=None, id: list=None):
        '''Set attributes for each environments

        Args:
            attrname (str): Attribute name
            value (Any, optional): Attribute value. This value will be passed
                to all envs specified by `id`. Defaults to None.
            values (list, optional): A list of attribute values. Defaults to None.
            id (list, optional): Environment indices. Defaults to None.
        
        Returns:
            list: Returned values
        '''
        self._assert_not_closed()
        if values is not None and value is not None:
            raise ValueError('Both `value` and `values` are set. '
                'Only one of them can be specified.')
        ids = self._get_ids(id)
        if values is not None:
            assert len(values) == len(ids)
            return [self.workers[id].setattr(attrname, v) 
                    for id, v in zip(ids, values)]
        return [self.workers[id].setattr(attrname, value) for id in ids]

    def reset(self, id: list=None, **kwargs):
        self._assert_not_closed()
        ids = self._get_ids(id)
        obs_list = [self.workers[id].reset(**kwargs) for id in ids]
        obs = self._flatten_obs(obs_list)
        self.rms_norm.update(obs)
        return self.rms_norm(obs)

    def step(self, acts: np.ndarray, id:list=None):
        self._assert_not_closed()
        ids = self._get_ids(id)
        assert len(acts) == len(ids)
        # step workers
        [self.workers[id].step_async(act) for act, id in zip(acts, ids)]
        results = [self.workers[id].step_wait() for id in ids]
        obs_list,rew_list,done_list,info_list = zip(*results)
        # stack results
        obs  = self._flatten_obs(obs_list)
        rew  = np.stack(rew_list)
        done = np.stack(done_list)
        # rms norm
        self.rms_norm.update(obs)
        return self.rms_norm(obs), rew, done, info_list

    def seed(self, seed:list=None, id:list=None):
        self._assert_not_closed()
        ids = self._get_ids(id)
        if seed is None:
            seed_list = [None] * self.n_envs
        elif isinstance(seed, int):
            seed_list = [seed+id for id in ids]
        else:
            # list, tuple, np.ndarray
            assert len(seed) == len(ids)
            seed_list = seed
        return [self.workers[id].seed(s) for id, s in zip(ids, seed_list)]

    def render(self, **kwargs):
        self._assert_not_closed()
        return [w.render(**kwargs) for w in self.workers]

    def close(self):
        if self.closed:
            return
        [w.close_async() for w in self.workers]
        [w.close_wait()  for w in self.workers]
        self._closed = True

    def _flatten_obs(self, obs_list):
        obs = ub_utils.nested_iter_tuple(
            tuple(obs_list), lambda obs: np.stack(obs))
        return obs

    def _get_ids(self, id: list=None):
        if id is None:
            return list(range(self.n_envs))
        return [id] if np.isscalar(id) else id

    def _assert_not_closed(self):
        assert not self.closed, "This env is already closed"

    def _get_rms_norm_opt(self, opt):
        if isinstance(opt, ub_utils.RMSNormalizer):
            # setup RMS normalizer
            if not opt.is_setup:
                opt.setup(self.observation_space)
            return opt
        # select default RMS normalizer
        if isinstance(self.observation_space, 
                (gym.spaces.Dict, gym.spaces.Tuple)):
            _rms_class = ub_utils.NestedRMSNormalizer
        else:
            _rms_class = ub_utils.RMSNormalizer
        enable = None
        if isinstance(opt, bool):
            enable = opt
        # Create RMSNormalizer
        rms = _rms_class(space  = self.observation_space, 
                         enable = enable)
        if isinstance(opt, str):
            rms.load(opt)
        return rms