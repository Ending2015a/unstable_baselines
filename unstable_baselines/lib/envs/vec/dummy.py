# --- built in ---
import os
import sys
import time
import logging

from typing import Union

# --- 3rd party ---
import gym
import numpy as np

# --- my module ---
from unstable_baselines.lib import utils as ub_utils
from unstable_baselines.lib.envs.vec import base as vec_base

__all__ = [
    'DummyVecEnv',
    'VecEnv'
]

class EnvWorker(vec_base.BaseEnvWorker):
    def __init__(self, env_fn):
        self.env = env_fn()
        self._res = None
        super().__init__(env_fn)
    
    def getattr(self, attrname: str):
        return getattr(self.env, attrname)

    def setattr(self, attrname: str, value):
        return setattr(self.env, attrname, value)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step_async(self, act):
        self._res = self.env.step(act)

    def step_wait(self):
        return self._res

    def seed(self, seed):
        super().seed(seed)
        return self.env.seed(seed)

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close_async(self):
        self.env.close()

    def close_wait(self):
        pass


class DummyVecEnv(vec_base.BaseVecEnv):
    def __init__(self,
        env_fns: list, 
        rms_norm: Union[str, bool, ub_utils.RMSNormalizer] = None
    ):
        super().__init__(env_fns, EnvWorker, rms_norm)


class VecEnv(DummyVecEnv):
    def __init__(self, 
        envs: list,
        rms_norm: Union[str, bool, ub_utils.RMSNormalizer] = None
    ):
        env_fns = [lambda i=j: envs[i] for j in range(len(envs))]
        super().__init__(env_fns, rms_norm)