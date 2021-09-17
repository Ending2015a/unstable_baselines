# --- bulid in ---
import os
import sys
import time

from typing import Union

# --- 3rd party ---
import gym
import numpy as np

from gym.wrappers import TimeLimit

# --- my module ---
from unstable_baselines.lib import utils as ub_utils
from unstable_baselines.lib.envs import vec as ub_vec

__all__ = [
    'TimeLimit',
    'TimeFeatureWrapper',
    'wrap_mujoco',
    'ObsNorm'
]

# === Wrappers for Continuous Tasks ===

# Borrowed from Stable baselines
class TimeFeatureWrapper(gym.Wrapper):
    """
    Add remaining time to observation space for fixed length episodes.
    See https://arxiv.org/abs/1712.00378 and https://github.com/aravindr93/mjrl/issues/13.
    :param env: (gym.Env)
    :param max_steps: (int) Max number of steps of an episode
        if it is not wrapped in a TimeLimit object.
    :param test_mode: (bool) In test mode, the time feature is constant,
        equal to zero. This allow to check that the agent did not overfit this feature,
        learning a deterministic pre-defined sequence of actions.
    """
    def __init__(self, env, max_steps=1000, test_mode=True):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)

        low, high = env.observation_space.low, env.observation_space.high
        low, high = np.concatenate((low, [0.])), np.concatenate((high, [1.]))
        
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        if (isinstance(env, TimeLimit)):
            self._max_steps = env._max_episode_steps
        else:
            self._max_steps = max_steps
        
        self._current_step = 0
        self._test_mode = test_mode

    def reset(self, **kwargs):
        self._current_step = 0
        obs = self.env.reset(**kwargs)
        return self._get_obs(obs)

    def step(self, action):
        self._current_step += 1
        obs, reward, done, info = self.env.step(action)
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs):
        """
        Concatenate the time feature to the current observation.
        :param obs: (np.ndarray)
        :return: (np.ndarray)
        """
        # Remaining time is more general
        time_feature = 1 - (self._current_step / self._max_steps)
        if self._test_mode:
            time_feature = 1.0
        # Optionnaly: concatenate [time_feature, time_feature ** 2]
        return np.concatenate((obs, [time_feature])).astype(obs.dtype)


def wrap_mujoco(env:       gym.Env,
                time_limit:    int = 1000,
                time_feature: bool = True,
                test_mode:    bool = False):
    if time_limit:
        env = TimeLimit(env, time_limit)
    if time_feature:
        env = TimeFeatureWrapper(env, test_mode=test_mode)
    return env

# === Other wrappers ===

class ObsNorm(gym.Wrapper):
    def __init__(self,
        env: gym.Env,
        rms_norm: Union[str, bool, ub_utils.RMSNormalizer,
                        ub_vec.BaseVecEnv] = None,
        update_rms: bool = None,
    ):
        '''Observation normalization wrapper. Normalize observations with
        newly created running mean std or the existing RMSNormalizer from
        other vectorized environments. Note that if the RMSNormalizer is 
        disabled or fixed, then this wrapper is omitted.

        Args:
            env (gym.Emv): Environment to wrap.
            rms_norm (Union[str, ub_utils.RunningMeanStd, ub_vec.BaseVecEnv],
                optional): Running mean and std for normalizing observations.
                `str` is the path to the RunningMeanStd file. If it is a
                BaseVecEnv then this wrapper calls BaseVecEnv's `normalize` 
                method to perform normalizations. In this case, if BaseVecEnv's 
                `enable_norm` is disabled then this wrapper is also disabled. 
                Defaults to None.
            update_rms (bool, optional): This value is used to prevent from 
                updating others' rms normalizer. But you still can set this
                value to True if you want to. In default, if `rms` is a 
                BaseVecEnv, this value is set to False, otherwise to True.
        '''
        super().__init__(env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._rms_norm = self._get_rms_opt(rms_norm)
        # prevent from updating others' rms normalizer
        self.update_rms = (not isinstance(rms_norm, ub_vec.BaseVecEnv)
            if update_rms is None else bool(update_rms))

    @property
    def rms_norm(self):
        return self._rms_norm

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        if self.update_rms:
            self.rms_norm.update(obs)
        return self.rms_norm(obs)

    def step(self, act):
        obs, rew, done, info = super().step(act)
        if self.update_rms:
            self.rms_norm.update(obs)
        return self.rms_norm(obs), rew, done, info

    def load(self, path:str):
        '''load rms valu from path'''
        self.rms_norm.load(path)
        return self

    def save(self, path:str):
        self.rms_norm.save(path)

    def _get_rms_opt(self, opt):
        # using existing RMSNormalizer from vectorized environments
        if isinstance(opt, ub_vec.BaseVecEnv):
            opt = opt.rms_norm
        if isinstance(opt, ub_utils.RMSNormalizer):
            # setup RMS normalizer
            if not opt.is_setup:
                opt.setup(self.observation_space)
            return opt
        # select default RMS normalizer
        if isinstance(self.observation_space, 
                (gym.spaces.Dict, gym.spaces.Tuple)):
            # TODO _rms_class = ub_utils.NestedRMSNormalizer
            raise NotImplementedError
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