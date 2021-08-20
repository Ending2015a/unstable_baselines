# --- built in ---
import os
import sys
import enum
import time
import logging
import multiprocessing

from typing import Union

# --- 3rd party ---
import gym
import cloudpickle
import numpy as np

# --- my module ---
from unstable_baselines.lib import utils as ub_utils
from unstable_baselines.lib.envs.vec import base as vec_base

__all__ = [
    'SubprocVecEnv'
]


class CloudpickleWrapper():
    def __init__(self, fn):
        self.fn = fn

    def __getstate__(self):
        return cloudpickle.dumps(self.fn)

    def __setstate__(self, fn):
        self.fn = cloudpickle.loads(fn)

# Commands
class CMD:
    getattr = 1
    setattr = 2
    reset   = 3
    step    = 4
    seed    = 5
    render  = 6
    close   = 7

def _worker(_p, p, env_fn_wrapper):
    _p.close()
    env = env_fn_wrapper.fn()
    try:
        while True:
            try:
                cmd, data = p.recv()
            except EOFError: # the pipe has been closed
                p.close()
                break
            if cmd == CMD.getattr:
                p.send(getattr(env, data[0], None))
            elif cmd == CMD.setattr:
                p.send(setattr(env, data[0], data[1]))
            elif cmd == CMD.reset:
                p.send(env.reset(**data[0]))
            elif cmd == CMD.step:
                obs, rew, done, info = env.step(data[0])
                p.send((obs, rew, done, info))
            elif cmd == CMD.seed:
                p.send(env.seed(data[0]))
            elif cmd == CMD.render:
                p.send(env.render(**data[0]))
            elif cmd == CMD.close:
                p.send(env.close())
                p.close()
                break
            else:
                p.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        p.close()

class SubprocEnvWorker(vec_base.BaseEnvWorker):
    def __init__(self, env_fn):
        methods = multiprocessing.get_all_start_methods()
        start_method = 'spawn'
        if 'forkserver' in methods:
            start_method = 'forkserver'
        ctx = multiprocessing.get_context(start_method)
        self.p, _p = ctx.Pipe()
        args = (
            self.p, _p, CloudpickleWrapper(env_fn)
        )
        self.process = ctx.Process(target=_worker, args=args, daemon=True)
        self.process.start()
        self._waiting_cmd = None
        _p.close()
        super().__init__(env_fn)
        
    def getattr(self, attrname: str):
        return self._cmd(CMD.getattr, attrname)

    def setattr(self, attrname: str, value):
        return self._cmd(CMD.setattr, attrname, value)

    def reset(self, **kwargs):
        return self._cmd(CMD.reset, kwargs)

    def step_async(self, act):
        return self._cmd(CMD.step, act, wait=False)

    def step_wait(self):
        return self._wait(CMD.step)

    def seed(self, seed):
        super().seed(seed)
        return self._cmd(CMD.seed, seed)

    def render(self, **kwargs):
        return self._cmd(CMD.render, kwargs)

    def close_async(self):
        return self._cmd(CMD.close, wait=False)

    def close_wait(self):
        return self._wait(CMD.close)
    
    def _cmd(self, cmd, *args, wait=True):
        if self._waiting_cmd:
            raise RuntimeError('Another command was sent when '
                f'waiting for the reply. CMD: {self._waiting_cmd}')
        self.p.send([cmd, args])
        self._waiting_cmd = cmd # marked as waiting reply
        if wait:
            return self._wait(cmd)

    def _wait(self, cmd):
        if self._waiting_cmd != cmd:
            raise RuntimeError
        res = self.p.recv()
        self._waiting_cmd = None #unmarked
        return res


class SubprocVecEnv(vec_base.BaseVecEnv):
    def __init__(self,
        env_fns: list,
        rms_norm: Union[str, bool, ub_utils.RMSNormalizer] = None
    ):
        super().__init__(env_fns, SubprocEnvWorker, rms_norm)