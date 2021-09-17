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

# TODO: TensorFlow 2.4 Internal bugs (or caused by CUDA 11?)
# tf.keras.layers.ReLU will cause memory leaking, so use tf.maximum instead
# refer to this github issue: https://github.com/tensorflow/tensorflow/issues/46475
if Version(tf.__version__) > '2.4.0':
    class ReLU(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def build(self, input_shape):
            pass
    
        def call(self, inputs, training=False):
            return tf.math.maximum(inputs, 0.)
else:
    ReLU = tf.keras.layers.ReLU

