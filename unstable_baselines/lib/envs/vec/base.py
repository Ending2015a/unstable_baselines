# --- built in ---
import os
import sys

# --- 3rd party ---
import gym
import numpy as np

# --- my module ---


class BaseVecEnvWorker():


class BaseVecEnv(gym.Env):
    def __init__(self, 
        env_fns: list, 
        worker_class: BaseVecEnvWorker,
        vec_norm: bool = False,
    ):
        metadata
        reward_range
        spec
        action_space
        observation_space




class VecEnvWorker(BaseVecEnvWorker):

class VecEnv(BaseVecEnv):
    def __init__(self, envs):



class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):