# --- built in ---
import os
import sys
import time

# --- 3rd party ---
import gym
import numpy as np

# --- my module ---
from unstable_baselines.lib import utils as ub_utils

class _FakeEnv(gym.Env):
    metadata = {'render.modes': []}
    reward_range = {-float('inf'), float('inf')}
    spec = ub_utils.StateObject(id='FakeEnv')
    observation_space: gym.Space
    action_space: gym.Space
    def __init__(self, 
        rank: int, 
        obs_space: gym.Space, 
        act_space: gym.Space,
        max_steps: int = 1000
    ):
        assert isinstance(obs_space, gym.Space)
        assert isinstance(act_space, gym.Space)
        self.observation_space = obs_space
        self.action_space = act_space
        self.timesteps = 0
        self.max_steps = max_steps
        self.is_closed = False
        self.rank = rank

    def step(self, action):
        self.timesteps += 1
        obs = self._get_obs()
        rew = self._get_rew()
        done = self._get_done()
        return obs, rew, done, {}

    def reset(self):
        self.timesteps = 0
        return self._get_obs()

    def render(self, mode='human'):
        pass

    def close(self):
        if not self.is_closed:
            self.is_closed = True

    def seed(self, seed):
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

    def _get_obs(self):
        return self.observation_space.sample()

    def _get_rew(self):
        return self.timesteps
    
    def _get_done(self):
        return self.timesteps >= self.max_steps

class FakeContinuousEnv(_FakeEnv):
    def __init__(self, 
        rank: int = 0, 
        obs_space: gym.Space = None, 
        act_space: gym.Space = None, 
        max_steps: int = 1000
    ):
        if obs_space is None:
            obs_space = gym.spaces.Box(low=-np.inf,high=np.inf,
                            shape=(64,), dtype=np.float32)
        if act_space is None:
            act_space = gym.spaces.Box(low=-1.,high=-1.,
                            shape=(16,), dtype=np.float32)
        super().__init__(rank, obs_space, act_space, max_steps)

class FakeImageEnv(_FakeEnv):
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self,
        rank: int = 0,
        obs_space: gym.Space = None,
        act_space: gym.Space = None,
        max_steps: int = 1000
    ):
        if obs_space is None:
            obs_space = gym.spaces.Box(low=0,high=255,
                            shape=(64, 64, 3), dtype=np.uint8)
        if act_space is None:
            act_space = gym.spaces.Discrete(6)
        super().__init__(rank, obs_space, act_space, max_steps)
        self._temp_obs = None

    def _get_obs(self):
        obs = super()._get_obs()
        self._temp_obs = obs.copy()
        return obs
    
    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self._temp_obs
        pass

class FakeEnv(gym.Env):
    metadata = {'render.modes':[]}
    reward_range = {-float('inf'), float('inf')}
    spec = None
    observation_space: gym.Space
    action_space: gym.Space
    def __init__(self, rank, env_type):
        if env_type == 'Box':
            obs_space = gym.spaces.Box(low=-np.inf,high=np.inf,
                                      shape=(32, 32, 3),
                                      dtype=np.float32)
            act_space = gym.spaces.Box(low=-1,high=1,
                                      shape=(6,),
                                      dtype=np.float32)
        elif env_type == 'Image':
            obs_space = gym.spaces.Box(low=0,high=255,
                                      shape=(32, 32, 3),
                                      dtype=np.uint8)
            act_space = gym.spaces.Discrete(6)
        elif env_type == 'Discrete':
            obs_space = gym.spaces.Discrete(10)
            act_space = gym.spaces.Discrete(6)
        elif env_type == 'MultiBinary':
            obs_space = gym.spaces.MultiBinary([3, 7])
            act_space = gym.spaces.MultiBinary([4, 6])
        else:
            raise ValueError(f'Unknown env type: {env_type}')
        self.observation_space = obs_space
        self.action_space = act_space
        self.timesteps = 0
        self.attr0 = rank
        self.rank = rank
        self.is_closed = False

    def step(self, action):
        reward = float(self.rank)
        self.timesteps += 1
        done = self.timesteps > 10
        return self.observation_space.sample(), reward, done, {}

    def reset(self):
        self.timesteps = 0
        return self.observation_space.sample()

    def render(self, mode='human'):
        pass

    def close(self):
        if not self.is_closed:
            self.is_closed = True

    def seed(self, seed):
        self.observation_space.seed(seed)
        self.action_space.seed(seed)