# --- built in ---
import os
import sys
import time
import logging
import unittest

# --- 3rd party ---
import numpy as np

# --- my module ---
from unstable_baselines.lib import utils
from unstable_baselines.lib import sche
from test.utils import TestCase

@sche.Scheduler.register(['my_sche'])
class MyScheduler(sche.Scheduler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def calc(self, steps):
        return steps * 2.


def create_empty_state():
    state = utils.StateObject()
    state.num_timesteps = 0
    state.num_epochs = 0
    state.num_subepochs = 0
    state.num_gradsteps = 0
    state.progress = 0.0
    return state


class TestScheModule(TestCase):
    '''Test unstable_baselines.lib.sche Module
    '''
    def test_get_unit_enum(self):
        # get by Enum
        res = sche._get_unit_enum(sche.Unit.timestep)
        self.assertTrue(res is sche.Unit.timestep)
        # get by str
        res = sche._get_unit_enum('timestep')
        self.assertTrue(res is sche.Unit.timestep)
        # raise exception
        with self.assertRaises(ValueError):
            sche._get_unit_enum('NotUnit')
        with self.assertRaises(ValueError):
            sche._get_unit_enum(1)

    def test_get_state_key(self):
        self.assertEqual(sche._get_state_key(sche.Unit.timestep), 'num_timesteps')
        self.assertEqual(sche._get_state_key(sche.Unit.epoch), 'num_epochs')
        self.assertEqual(sche._get_state_key(sche.Unit.subepoch), 'num_subepochs')
        self.assertEqual(sche._get_state_key(sche.Unit.gradstep), 'num_gradsteps')
        self.assertEqual(sche._get_state_key(sche.Unit.progress), 'progress')
        # raise exception
        with self.assertRaises(RuntimeError):
            sche._get_state_key('NotUnit')

    def test_check_state_and_unit(self):
        state = create_empty_state()
        sche._check_state_and_unit(sche.Unit.timestep, state)
        with self.assertRaises(ValueError):
            sche._check_state_and_unit(sche.Unit.timestep, None)
        with self.assertRaises(TypeError):
            state = dict(create_empty_state())
            sche._check_state_and_unit(sche.Unit.timestep, state)
        with self.assertRaises(KeyError):
            state = utils.StateObject()
            sche._check_state_and_unit(sche.Unit.timestep, state)

    def test_get_steps_from_state(self):
        state = create_empty_state()
        state.num_timesteps = 100
        key = sche._get_state_key(sche.Unit.timestep)
        res = sche._get_steps_from_state(key, state)
        self.assertEqual(res, 100)
        with self.assertRaises(ValueError):
            sche._get_steps_from_state(key, None)
        with self.assertRaises(KeyError):
            sche._get_steps_from_state(key, utils.StateObject())

    def test_constant_scheduler_without_state(self):
        constant = 1.0
        const_sche = sche.ConstantScheduler(constant)
        self.assertEqual(const_sche(100), constant)

    def test_constant_scheduler_with_state(self):
        state = create_empty_state()
        constant = 1.0
        const_sche = sche.ConstantScheduler(constant, unit='timestep')
        self.assertEqual(const_sche(100), constant)

    def test_linear_scheduler_without_state(self):
        start_value = 100.0
        decay_steps = 100
        stop_value = 0
        lin_sche = sche.LinearScheduler(start_value, decay_steps, stop_value)
        self.assertAlmostEqual(lin_sche(0), start_value)
        self.assertAlmostEqual(lin_sche(100), stop_value)
        self.assertAlmostEqual(lin_sche(101), stop_value)
        self.assertAlmostEqual(lin_sche(50), 50.0)

    def test_linear_scheduler_with_state(self):
        state = create_empty_state()
        start_value = 100.0
        decay_steps = 100
        stop_value = 0
        lin_sche = sche.LinearScheduler(start_value, decay_steps, stop_value, 
                        unit='timestep')
        lin_sche.bind(state)
        self.assertAlmostEqual(lin_sche(), start_value)
        state.num_timesteps = 100
        self.assertAlmostEqual(lin_sche(), stop_value)
        state.num_timesteps = 50
        self.assertAlmostEqual(lin_sche(), 50.0)

    def test_exponential_scheduler_without_state(self):
        start_value = 100.0
        decay_steps = 10
        decay_rate = 0.1
        stop_value = 0.1
        exp_sche = sche.ExponentialScheduler(start_value, decay_steps, decay_rate,
                        stop_value)
        self.assertAlmostEqual(exp_sche(0), start_value)
        self.assertAlmostEqual(exp_sche(30), stop_value)
        self.assertAlmostEqual(exp_sche(100), stop_value)
        self.assertAlmostEqual(exp_sche(20), 1.0)

    def test_exponential_scheduler_with_state(self):
        state = create_empty_state()
        start_value = 100.0
        decay_steps = 10
        decay_rate = 0.1
        stop_value = 0.1
        exp_sche = sche.ExponentialScheduler(start_value, decay_steps, decay_rate,
                        stop_value, unit='timestep')
        exp_sche.bind(state)
        self.assertAlmostEqual(exp_sche(), start_value)
        state.num_timesteps = 30
        self.assertAlmostEqual(exp_sche(), stop_value)
        state.num_timesteps = 100
        self.assertAlmostEqual(exp_sche(), stop_value)
        state.num_timesteps = 20
        self.assertAlmostEqual(exp_sche(), 1.0)

    def test_exponential_scheduler_no_stop_value(self):
        start_value = 100.0
        decay_steps = 10
        decay_rate = 0.1
        exp_sche = sche.ExponentialScheduler(start_value, decay_steps, decay_rate)
        self.assertAlmostEqual(exp_sche(50), 1e-3)

    def test_multi_scheduler_without_state(self):
        start_value = 100.0
        decay_steps = 100
        stop_value = 0
        lin_sche = sche.LinearScheduler(start_value, decay_steps, stop_value)
        constant = 50.0
        const_sche = sche.ConstantScheduler(constant)
        m_sches = sche.MultiScheduler([lin_sche, const_sche], op='min')
        self.assertAlmostEqual(m_sches(0), constant)
        self.assertAlmostEqual(m_sches(51), 49.0)

    def test_multi_scheduler_with_state(self):
        state = create_empty_state()
        start_value = 100.0
        decay_steps = 100
        stop_value = 0
        lin_sche = sche.LinearScheduler(start_value, decay_steps, stop_value, 
                        unit='timestep')
        constant = 50.0
        const_sche = sche.ConstantScheduler(constant, unit='timestep')
        m_sches = sche.MultiScheduler([lin_sche, const_sche], op='min') # discard unit
        m_sches.bind(state)
        self.assertTrue(lin_sche.state is state)
        self.assertTrue(const_sche.state is state)
        state.num_timesteps = 0
        self.assertAlmostEqual(m_sches(), constant)
        state.num_timesteps = 51
        self.assertAlmostEqual(m_sches(), 49.0)

    def test_multi_scheduler_with_state_and_different_units(self):
        pass

    def test_scheduler_dump(self):
        pass

    def test_get_scheduler_restore(self):
        pass

    def test_get_scheduler_create(self):
        pass

    def test_register_scheduler(self):
        scheduler = sche.get_scheduler(type='my_sche')
        self.assertTrue(isinstance(scheduler, MyScheduler))

    def test_scheduler_no_bind_exception(self):
        pass

    def test_scheduler_unknown_unit_exception(self):
        pass