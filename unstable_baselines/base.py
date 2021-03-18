# --- built in ---
import os
import abc
import sys
import time
import json
import logging

# --- 3rd party ---
import numpy as np
import tensorflow as tf

# --- my module ---
from unstable_baselines.utils import from_json_serializable

__all__ = ['SavableModel']

# === Helper ===

def _dump_json_dict(filepath, d):

    with open(filepath, 'w') as f:
        json.dump(d, f, indent=4, ensure_ascii=False)


def _load_json_dict(filepath):

    with open(filepath, 'r') as f:
        json_dict = json.load(f)

    return json_dict

# === SavableModel ===

class SavableModel(tf.keras.Model, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_config(self):
        '''
        Return JSON serializable dictionary
        '''
        raise NotImplementedError('get_config method not implemented')

    @classmethod
    def from_config(cls, config):
        '''
        Construct model from config
        '''
        return cls(**from_json_serializable(config))

    def save_config(self, filepath):
        '''
        Save config
        '''
        config = self.get_config()

        _dump_json_dict(filepath, config)

    @classmethod
    def load_config(cls, filepath):
        '''
        Load config
        '''

        config = _load_json_dict(filepath)

        return cls.from_config(config)

    def save(self, filepath, weights_name='weights', config_name='config.json'):
        '''
        Save model weights and config
        '''

        weights_path = os.path.join(filepath, weights_name)
        config_path = os.path.join(filepath,  config_name)
        
        os.makedirs(filepath, exist_ok=True)
        self.save_weights(weights_path)
        self.save_config(config_path)
    
    @classmethod
    def load(cls, filepath, weights_name='weights', config_name='config.json'):
        '''
        Load model weights and config
        '''
        weights_path = os.path.join(filepath, weights_name)
        config_path = os.path.join(filepath,  config_name)

        self = cls.load_config(config_path)
        self.load_weights(weights_path)

        return self
