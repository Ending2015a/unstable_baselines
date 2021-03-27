# --- built in ---
import os
import sys
import time
import random
import datetime

from distutils.version import StrictVersion as Version

# --- 3rd party ---
import gym
import cloudpickle

import numpy as np
import tensorflow as tf

# --- my module ---

__all__ = [
    'ReLU'
]




# --- Some TensorFlow internal bugs fixes ---

# TODO: TensorFlow 2.4.1 Internal bugs (or caused by CUDA 11?)
# tf.keras.layers.ReLU will cause memory leaking, so use LeakyReLU(0.001) instead
# refer to this github issue: https://github.com/tensorflow/tensorflow/issues/46475
if Version(tf.__version__) > '2.4.0':
    print('WARNING: Tensorflow version {} > 2.4.0, use LeakyReLU instead of ReLU to avoid memory leaking. This may hurt performance.'.format(tf.__version__))
    class ReLU(tf.keras.layers.LeakyReLU):
        def __init__(self, **kwargs):
            super().__init__(alpha=0.001, **kwargs)
else:
    class ReLU(tf.keras.layers.ReLU):
        pass

