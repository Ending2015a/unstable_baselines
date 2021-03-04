# --- built in ---
import os
import re
import abc
import cv2
import csv
import sys
import glob
import json
import time
import random
import inspect
import datetime
import tempfile
import multiprocessing

# --- 3rd party ---
import gym 
import cloudpickle

import numpy as np
import tensorflow as tf

# --- my module ---


__all__ = [
    'set_global_seeds',
    'flatten_obs'
]

# === Utils ===
def set_global_seeds(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def flatten_obs(obs, space):
    assert isinstance(obs, (list, tuple)), 'expected list or tuple of observations per environment'
    assert len(obs) > 0, 'need observations from at least one environment'

    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), 'Dict space must have ordered subspaces'
        assert isinstance(obs[0], dict), 'non-dict observation for environment with Dict observation space'
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(obs[0], tuple), 'non-tuple observation for environment with Tuple observation space'
        obs_len = len(space.spaces)
        return tuple((np.stack([o[i] for o in obs]) for i in range(obs_len)))
    else: # Discrete, Box
        return np.stack(obs)
