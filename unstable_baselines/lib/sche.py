# --- built in ---
import os
import abc
import sys
import enum
import time
import functools

# --- 3rd party ---
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

# === Units ===
class MetaEnum(enum.EnumMeta):
    def __contains__(cls, item):
        '''
        Allow to check if the Enum
        contains an enum by name
        '''
        try:
            cls(item)
        except ValueError:
            return False
        return True

@enum.unique
class Unit(str, enum.Enum, 
            metaclass=MetaEnum):
    '''Note: the enum value must match
    to its name, because json.dumps will
    dump enum's value.
    '''
    timestep = 'timestep'
    epoch    = 'epoch'
    subepoch = 'subepoch'
    gradstep = 'gradstep'
    progress = 'progress'

# Registry class
class _RegisteredSchedulerClass(ub_utils.StateObject):
    '''Storing registered scheduler types'''
    # Default scheduler type
    default: 'Scheduler' = None
    def get_type(self, scheduler_type):
        '''Retrieve registered scheduler type by name or
        type

        Args:
            scheduler_type (str, type): scheduler type to retrieve

        Raises:
            ValueError: raised if `scheduler_type` is not contain in
                the registry.

        Returns:
            type: type of scheduler
        '''
        if (type(scheduler_type) is type and 
                issubclass(scheduler_type, Scheduler)):
            return scheduler_type
        elif scheduler_type in self.keys():
            return self[scheduler_type]
        else:
            raise ValueError('Got unknown scheduler type: {}'.format(scheduler_type))

_RegisteredScheduler = _RegisteredSchedulerClass()

def get_scheduler(*args, **kwargs):
    '''Create or retrieve schedulers
    a shortcut of Scheduler.get_scheduler
    '''
    return Scheduler.get_scheduler(*args, **kwargs)

def get_scheduler_type(typename):
    '''Retrieve scheduler types by scheduler name or type
    A shortcut of _RegisteredScheduler.get_type
    '''
    return _RegisteredScheduler.get_type(typename)

# === Schedulers helpers ====

def _get_unit_enum(item):
    if isinstance(item, Unit):
        return item
    if isinstance(item, str):
        return Unit(item) # may raises KeyError
    raise ValueError(f'Unknown type of unit: {type(item)}')

def _get_state_key(unit: Unit):
    '''Retrieve the state key corresponding to the unit'''
    if unit is Unit.timestep:
        key = 'num_timesteps'
    elif unit is Unit.epoch:
        key = 'num_epochs'
    elif unit is Unit.subepoch:
        key = 'num_subepochs'
    elif unit is Unit.gradstep:
        key = 'num_gradsteps'
    elif unit is Unit.progress:
        key = 'progress'
    else:
        #TODO: custom units
        # first check if unit is a Unit type. If not, 
        # check if it is a custom units.
        all_units = list(key.name for key in Unit)
        raise RuntimeError('Unknown unit, unit must be either of '
            f'{all_units}, got {unit}')
    return key

def _check_state_and_unit(unit: Unit, state: ub_utils.StateObject):
    '''Check if the state key corresponds to unit does exist
    '''
    if state is None:
        raise ValueError('State is empty, call '
            'bind() to set state.')
    if not isinstance(state, ub_utils.StateObject):
        raise TypeError('State must be a StateObject, '
            f'got {type(state)}')
    state_key = _get_state_key(unit)
    if state_key not in state.keys():
        raise KeyError(f'State does not contain key for {unit}')

def _get_steps_from_state(key: str, state: ub_utils.StateObject):
    '''Retrieve state info by state key
    Note that is function does not handle the exception of
    missing key, so ensure you have called `_check_state_and_unit`
    before calling this function.
    '''
    if state is None:
        raise ValueError('State is empty, call '
            'bind() to set state.')
    return state[key] # raise KeyError

# === Schedulers ===

class Scheduler(ub_utils.StateObject):
    '''Base scheduler class
    To create a new scheduler class, you need to implement the 
    following methods:
    * calc(self, steps):
    '''
    # do not serialize these members
    __slots__ = ['state', 'state_key']
    
    def __new__(cls, *args, **kwargs):
        sche = super().__new__(cls, *args, **kwargs)
        sche.type = cls.__name__

        return sche

    def __init__(self, unit: str='timestep', 
                       state: ub_utils.StateObject=None):
        '''Initialize scheduler

        Args:
            unit (str, optional): Unit. Defaults to 'timestep'.
            state (StateObject, optional): state object. Defaults to None.
        '''
        self.unit = _get_unit_enum(unit)
        self.state_key = _get_state_key(self.unit) # not serializable
        self.state = None # not serializable

        if state is not None:
            self.bind(state)

    def bind(self, state: ub_utils.StateObject):
        '''Bind state object'''
        _check_state_and_unit(self.unit, state)
        self.state = state

    @abc.abstractmethod
    def calc(self, steps):
        '''Calculate variables

        Args:
            steps (int, float): timestep, epoch, gradstep or progress

        Returns:
            float: interpolated value
        '''
        raise NotImplementedError

    def get_steps(self):
        if self.state is None:
            raise RuntimeError('The `state` is empty, call '
                                    'bind() to set `state`.')
        return _get_steps_from_state(self.state_key, self.state)

    def __call__(self, steps=None):
        
        if steps is None:
            steps = self.get_steps()

        return self.calc(steps)

    @classmethod
    def register(cls, names=[], default=False):
        '''Register a new type of Scheduler

        Args:
            names (list, optional): Name alias. Defaults to [].
            default (bool, optional): Set as default scheduler. 
                Defaults to False.
        '''
        def _register(scheduler_type):
            name_list = names
            if not isinstance(name_list, (tuple, list)):
                name_list = [name_list]
            for name in name_list:
                if name in _RegisteredScheduler.keys():
                    raise ValueError(f'The name {name} is used by scheduler'
                        f'{type(_RegisteredScheduler[name])}')

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
        '''Get scheduler
        The first argument can be a defined Scheduler or a serialized
        Scheduler (mostly it's a `dict`). This is for restoring scheduler.
        Or you can specify `type` to create a new scheduler.

        Args:
            type (str, type, optional): Scheduler name or type. 
                Defaults to None.
            state (StateObject, optional): state object to bind. 
                Defaults to None.

        Returns:
            Scheduler: scheduler
        '''
        sche = None
        # Restore scheduler
        if len(args) == 1:
            if isinstance(args[0], Scheduler):
                # Scheduler(LinearScheduler(...))
                sche = args[0]
            elif isinstance(args[0], dict):
                # Scheduler({'type': 'linear', 'start_value': 1e-3...})
                params = args[0]
                params.update(kwargs)
                sche = cls.get_scheduler(**params)
        # Create a new scheduler
        if sche is None:
            if type is None:
                type = _RegisteredScheduler.default
            # get scheduler class
            type = _RegisteredScheduler.get_type(type)
            # create scheduler
            sche = type(*args, **kwargs)
        # bind state object
        if sche is not None:
            if isinstance(state, ub_utils.StateObject):
                sche.bind(state)
            return sche
        raise ValueError('Unknown scheduler params:'
            f'args={args}, type={type}, kwargs={kwargs}')

# === Build-in schedulers ===

@Scheduler.register(['const', 'constant',
                    'ConstScheduler',
                    'Constant'], default=True)
class ConstantScheduler(Scheduler):
    def __init__(self, value: float, **kwargs):
        '''Constant scheduler

        Args:
            value (float): variable values.
        '''
        super().__init__(**kwargs)
        self.value = value

    def __call__(self, steps=None):
        return self.value

# alias
ConstScheduler = ConstantScheduler
Constant = ConstantScheduler

@Scheduler.register(['linear', 'Linear'])
class LinearScheduler(Scheduler):
    def __init__(self, start_value: float,
                       decay_steps: int,
                       stop_value: float=0.0,
                       **kwargs):
        '''LinearScheduler
        Perform linear decay

        Args:
            start_value (float): Initial value.
            decay_steps (int, float): total steps to stop decay.
            stop_value (float, optional): minimum value. Default to 0.0
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
    def __init__(self, start_value: float, 
                       decay_steps: int, 
                       decay_rate: float, 
                       stop_value: float=None, 
                       **kwargs):
        '''ExponentialScheduler
        Perform exponential decay

        Args: TODO
            start_value (float): [description]
            decay_steps (int): [description]
            decay_rate (float): [description]
            stop_value (float, optional): minimum value. Defaults to None.
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
    __slots__ = ['_op_func']
    def __init__(self, schedulers, op='min', **kwargs):
        '''Multiple schedulers
        This scheduler allows you to combine multiple schedulers.
        Note that the serializable operations `op` only supports
        'min', 'max' and 'mean'. If you are using custom operations,
        don't forget to provide them on restoring schedulers.
        Note that `unit` is discarded.

        Args:
            schedulers (list): A list of schedulers
            op (str, Callable, optional): operation to calculate
                final values. Defaults to 'min'.
        '''        
        super().__init__(**kwargs)
        self.schedulers = [
            Scheduler.get_scheduler(sche)
            for sche in schedulers
        ]
        assert len(self.schedulers) > 0
        if isinstance(op, str): # min, max, mean
            op = op.lower()
            self.op = op
            self._op_func = self.get_op_func(op)
        elif callable(op): # custom function
            self.op = None
            self._op_func = op
        else:
            raise ValueError(f'`op` must be str or callable, got {type(op)}')

    def get_op_func(self, op: str):
        assert op in ['max', 'min', 'mean']
        self.op = op
        if op == 'max':
            _op_func = np.max
        elif op == 'min':
            _op_func = np.min
        elif op == 'mean':
            _op_func = np.mean
        return functools.partial(_op_func, axis=0)

    def bind(self, state):
        super().bind(state)
        for sche in self.schedulers:
            sche.bind(state)

    def __call__(self, steps=None):
        return self.calc(steps)

    def calc(self, steps):
        vals = [sche(steps) for sche in self.schedulers]
        return self._op_func(vals)