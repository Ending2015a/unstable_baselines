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
from unstable_baselines import logger
from unstable_baselines.utils import (from_json_serializable,
                                    tf_soft_update_params)

__all__ = ['SavableModel']

# === Helper ===

def _dump_json_dict(filepath, d):

    with open(filepath, 'w') as f:
        json.dump(d, f, indent=4, ensure_ascii=False)


def _load_json_dict(filepath):

    with open(filepath, 'r') as f:
        json_dict = json.load(f)

    return json_dict

def _get_checkpoint_manager(checkpoint, filepath, checkpoint_name='weights', max_to_keep=None):

    manager = tf.train.CheckpointManager(
        checkpoint, directory=filepath, max_to_keep=max_to_keep, 
        checkpoint_name=checkpoint_name
    )
    return manager

def _get_latest_checkpoint_number(latest_checkpoint):
    '''
    Get latest checkpoint number from latest_checkpoint

    if latest_checkpoint is None or failed to get number, return None
    '''

    checkpoint_number = None
    if latest_checkpoint and isinstance(latest_checkpoint, str):
        if '-' in latest_checkpoint:
            number_str = latest_checkpoint.rsplit('-', 1)[1]
            if number_str.isdigit():
                checkpoint_number = int(number_str)

    return checkpoint_number


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

    def save(self, filepath, weights_name='weights', checkpoint_number=None):
        '''
        Save model weights and config

        The weights are saved to `{filepath}/{weights_name}-{checkpoint_number}.xxxxxx`
        and config saved to `{filepath}/{weights_name}-{checkpoint_number}-config.xxxxxx`

        Args:
            filepath: (str) base directory to save weights and config
            weights_name: (str) filename
            checkpoint_number: (int) checkpoint number, 
        '''

        # create checkpoint
        checkpoint = tf.train.Checkpoint(model=self)

        # create checkpoint manager
        manager = _get_checkpoint_manager(checkpoint, filepath, weights_name)

        # get latest checkpoint number
        latest_checkpoint = manager.latest_checkpoint
        latest_checkpoint_number = _get_latest_checkpoint_number(latest_checkpoint)

        if latest_checkpoint_number is None:
            latest_checkpoint_number = 0

        if checkpoint_number is None:
            checkpoint_number = latest_checkpoint_number + 1

        # save weights
        manager.save(checkpoint_number=checkpoint_number)

        # save config
        config_path = manager.latest_checkpoint + '-config.json'
        self.save_config(config_path)

        return manager.latest_checkpoint


    @classmethod
    def load(cls, filepath, weights_name='weights'):
        '''
        Load model weights and config
        '''

        # get latest checkpoint
        latest_checkpoint = tf.train.latest_checkpoint(filepath)
        
        if latest_checkpoint is None:
            latest_checkpoint = os.path.join(os.path.abspath(filepath),
                                            weights_name)

        config_path = latest_checkpoint + '-config.json'


        if not os.path.isfile(config_path):
            raise RuntimeError('Failed to restore model, config file not '
                    'found: {}, weights_name: {}'.format(filepath, weights_name))
        
        logger.debug('Restore weights from: {}'.format(latest_checkpoint))

        # restore config
        self = cls.load_config(config_path)

        # restore weights
        status = tf.train.Checkpoint(model=self).restore(latest_checkpoint)

        return self

    def update(self, other_model, polyak=1.0, all_vars=False):
        '''
        Update parameters from other model

        all_vars: update all variables. If False: only update 
            trainable variables
        '''

        if not all_vars:
            var = self.trainable_variables
            target_var = other_model.trainable_variables
        else:
            var = self.variables
            target_var = other_model.variables
        
        tf_soft_update_params(var, target_var, polyak=polyak)



class TrainableModel(SavableModel):
    @abs.abstractmethod
    def predict(self, inputs):
        '''
        Predict actions

        return actions
        '''
        raise NotImplementedError('Method not implemented')

    @abc.abstractmethod
    def train(self):
        '''
        Train one epoch
        '''
        raise NotImplementedError('Method not implemented')

    @abc.abstractmethod
    def eval(self, env, n_episodes, max_steps):
        '''
        Evaluate model (default)

        Return:
            eps_rews: (list) episode rewards
            eps_steps: (list) episode length
        '''

        if n_episodes == 0:
            return [], []

        eps_info = []

        for episode in range(n_episodes):
            obs = env.reset()
            total_rews = 0

            for steps in range(max_steps):
                # predict action
                acts = self.predict(obs)

                # step environment
                obs, rew, done, info= env.step(acts)
                total_rews += rew
                if done:
                    break

            total_steps = steps +1

            if self.verbose > 1:
                LOG.set_header('Eval {}/{}'.format(episode+1, n_episodes))
                LOG.add_line()
                LOG.add_row('Rewards', total_rews)
                LOG.add_row(' Length', total_steps)
                LOG.add_line()
                LOG.flush('INFO')
        
            eps_info.append([total_rews, total_steps])

        eps_rews, eps_steps = zip(*eps_info)

        return eps_rews, eps_steps

    @abc.abstractmethod
    def learn(self):
        
        raise NotImplementedError('Method not implemented')
