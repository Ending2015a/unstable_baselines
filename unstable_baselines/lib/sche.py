# --- built in ---
import os
import re
import abc
import cv2
import csv
import sys
import enum
import glob
import json
import time
import base64
import pickle
import random
import inspect
import datetime
import tempfile
import multiprocessing

# --- 3rd party ---
import gym 
import cloudpickle

import numpy as np

# --- my module ---
from unstable_baselines.lib import utils as ub_utils



__all__ = [
    'get_scheduler',
    'Scheduler',
    'ConstantScheduler',
    'ConstScheduler',
    'LinearScheduler',
    'ExponentialScheduler',
    'ExpScheduler',
    'MultiScheduler'
]

# === Schedulers ===


class _RegisteredSchedulerClass(ub_utils.StateObject):

    default = None

    def get_type(self, scheduler_type):

        if (type(scheduler_type) is type and 
                issubclass(scheduler_type, Scheduler)):
            return scheduler_type
        elif scheduler_type in self.keys():
            return self[scheduler_type]
        else:
            raise ValueError('Got unknown scheduler type: {}'.format(scheduler_type))


_RegisteredScheduler = _RegisteredSchedulerClass()


def get_scheduler(*args, **kwargs):
    return Scheduler.get_scheduler(*args, **kwargs)

class Scheduler(ub_utils.StateObject):
    __slots__ = ['state']
    
    @enum.unique
    class Unit(str, enum.Enum):
        timestep = 'timestep'
        epoch    = 'epoch'
        gradstep = 'gradstep'
        progress = 'progress'

    
    def __new__(cls, *args, **kwargs):

        sche = super().__new__(cls, *args, **kwargs)
        sche.type = cls.__name__

        return sche

    def __init__(self, unit='timestep', state=None):

        if isinstance(unit, str):
            unit = self.Unit[unit]
        
        self.unit = unit
        self.state = None

        if state is not None:
            self.bind(state)

    def bind(self, state):
        assert state is not None

        self.state = state

    @abc.abstractmethod
    def calc(self, steps):
        '''Calculate variables

        Args:
            steps (int, float): timestep, epoch, gradstep or progress

        Returns:
            float: interpolated value
        '''
        raise NotImplementedError('Method not implemented')

    def get_steps(self):
        if self.state is None:
            raise RuntimeError('The `state` is empty, call '
                                    'bind() to set `state`.')
        
        if self.unit == self.Unit.timestep:
            steps = self.state.num_timesteps
        elif self.unit == self.Unit.epoch:
            steps = self.state.num_epochs
        elif self.unit == self.Unit.gradstep:
            steps = self.state.num_gradsteps
        elif self.unit == self.Unit.progress:
            steps = self.state.progress
        else:
            raise RuntimeError('Unknown unit, unit must be either of '
                        '["timestep", "epoch", "gradstep", "progress"]'
                        ', got {}'.format(self.unit))

        return steps

    def __call__(self, steps=None):
        
        if steps is None:
            steps = self.get_steps()

        return self.calc(steps)

    @classmethod
    def register(cls, names=[], default=False):

        def _register(scheduler_type):

            name_list = names

            if not isinstance(name_list, (tuple, list)):
                name_list = [name_list]

            for name in name_list:
                if name in _RegisteredScheduler.keys():
                    raise RuntimeError('Scheduler already registered: {}'.format(name))

                _RegisteredScheduler[name] = scheduler_type

            if hasattr(scheduler_type, '__name__'):
                name = scheduler_type.__name__
                _RegisteredScheduler[name] = scheduler_type

            # set default scheduler
            if default:
                _RegisteredScheduler.default = scheduler_type

            return scheduler_type

        return _register

    @classmethod
    def get_scheduler(cls, *args, type=None, state=None, **kwargs):
        sche = None

        if len(args) == 1:

            if isinstance(args[0], Scheduler):
                # Scheduler(LinearScheduler(...))
                sche = args[0]
            elif isinstance(args[0], dict):
                # Scheduler({'type': 'linear', 'start_value': 1e-3...})
                args[0].update(kwargs)
                sche = cls.get_scheduler(**args[0])
        
        if sche is None:
            if type is None:
                type = _RegisteredScheduler.default

            type = _RegisteredScheduler.get_type(type)
            sche = type(*args, **kwargs)
            

        if sche is not None:
            if state is not None:
                sche.bind(state)

            return sche

        raise ValueError('Unknown scheduler params: args={}, type={}, '
                                    'kwargs={}'.format(args, type, kwargs))



@Scheduler.register(['const', 'constant',
                    'ConstScheduler',
                    'Constant'], default=True)
class ConstantScheduler(Scheduler):
    def __init__(self, value, **kwargs):
        '''Constant scheduler

        Args:
            value (float): variable values.
        '''
        super().__init__(**kwargs)
        self.value = value

    def __call__(self, v):
        return self.value

# alis
ConstScheduler = ConstantScheduler
Constant = ConstantScheduler

@Scheduler.register(['linear', 'Linear'])
class LinearScheduler(Scheduler):
    def __init__(self, start_value, decay_steps, stop_value=0., **kwargs):
        '''LinearScheduler

        Perform linear decay

        Args:
            start_value (float): Initial value.
            decay_steps (int, float): total steps to stop decay.
            stop_value (float, optional): Final value. Defaults to 0.0.
        '''        
        super().__init__(**kwargs)

        self.start_value = start_value
        self.decay_steps = decay_steps
        self.stop_value  = stop_value

    def calc(self, steps):
        progress = min(1., float(steps)/float(self.decay_steps))
        return self.start_value + (self.stop_value - self.start_value) * progress


# alias
Linear = LinearScheduler

@Scheduler.register(['exp', 'ExpScheduler',
                    'Exponential'])
class ExponentialScheduler(Scheduler):
    def __init__(self, start_value, decay_steps, decay_rate, stop_value=None, **kwargs):
        '''[summary]

        Args:
            start_value ([type]): [description]
            decay_steps ([type]): [description]
            decay_rate ([type]): [description]
            stop_value ([type], optional): [description]. Defaults to None.
        '''        
        super().__init__(**kwargs)

        self.start_value = start_value
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.stop_value = stop_value

    def calc(self, steps):
        decay = self.decay_rate**(float(steps)/self.decay_steps)
        v = self.start_value * decay
        if self.stop_value is not None:
            return np.maximum(self.stop_value, v)
        else:
            return v
            

# alias
ExpScheduler = ExponentialScheduler
Exponential = ExponentialScheduler

@Scheduler.register(['multi'])
class MultiScheduler(Scheduler):
    def __init__(self, schedulers, op='min', **kwargs):
        super().__init__(**kwargs)
        self.schedulers = [
            Scheduler.get_scheduler(sche)
            for sche in schedulers
        ]
        self.op = op.lower()
        assert self.op in ['max', 'min', 'mean']
        assert len(self.schedulers) > 0

    def bind(self, state):
        super().bind(state)
        for sche in self.schedulers:
            sche.bind(state)

    def calc(self, steps):
        vals = [sche(steps) for sche in self.schedulers]
        if self.op == 'max':
            return np.max(vals)
        elif self.op == 'min':
            return np.min(vals)
        elif self.op == 'mean':
            return np.mean(vals)
        else:
            raise NotImplementedError('Op not implemented: {}'.format(self.op))