# --- built in ---
import os
import sys
import time

# --- 3rd party ---
import gym
import numpy as np

# --- my module ---

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