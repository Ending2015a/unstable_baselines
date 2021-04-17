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


# === Const params ===
DEFAULT_WEIGHTS_NAME='weights'
CONFIG_SUFFIX='.config.json'

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
    '''SavableModel

    SavableModel provides a common interface to handle 
    model save/load operations.
    '''    

    @abc.abstractmethod
    def get_config(self):
        '''Get model configuration (abstract)

        Returns:
            dict: A JSON serializable dict object. Use 
                `to_json_serializable()` to convert anything
                into JSON serializable dict object.
        '''
        raise NotImplementedError('get_config method not implemented')

    @classmethod
    def from_config(cls, config):
        '''Reconstruct model from configuration

        Args:
            config (dict): A JSON serializable dict object.

        Returns:
            SavableModel: reconstructed model
        ''' 
        return cls(**from_json_serializable(config))

    def save_config(self, filepath):
        '''Save model configuration to `filepath`

        Args:
            filepath (str): path to save configuration
        '''
        config = self.get_config()

        _dump_json_dict(filepath, config)

    @classmethod
    def load_config(cls, filepath):
        '''Load and reconstruct model from configuration

        Args:
            filepath (str): path to load configuration

        Returns:
            : [description]
        '''

        config = _load_json_dict(filepath)

        return cls.from_config(config)

    def save(self, filepath: str, checkpoint_number: int=None):
        '''Save model weights and config

        The weights are saved to `{filepath}-{checkpoint_number}.xxxxxx`
        and config saved to `{filepath}-{checkpoint_number}-config.xxxxxx`

        Args:
            filepath (str): path to save weights and config. If a directory
                is given (path followed by a single '/'), the weights are 
                saved to `{filepath}/weights.xxxxxx`.
            checkpoint_number (int, optional): checkpoint number.
                Defaults to None.
        '''

        # is a directory
        if filepath.endswith('/'):
            filepath = os.path.join(filepath, DEFAULT_WEIGHTS_NAME)

        # base path
        filepath = os.path.abspath(filepath)
        filedir  = os.path.dirname(filepath)
        filename = os.path.basename(filepath)

        # create checkpoint 
        checkpoint = tf.train.Checkpoint(model=self)

        # create checkpoint manager
        manager = _get_checkpoint_manager(checkpoint, filedir, filename)

        # get latest checkpoint number
        latest_checkpoint = manager.latest_checkpoint
        latest_checkpoint_number = _get_latest_checkpoint_number(latest_checkpoint)

        if checkpoint_number is None:
            checkpoint_number = latest_checkpoint_number + 1

        # save weights
        manager.save(checkpoint_number=checkpoint_number)

        # save config
        config_path = manager.latest_checkpoint + CONFIG_SUFFIX
        self.save_config(config_path)

        return manager.latest_checkpoint

    @classmethod
    def load(self, filepath: str):
        '''Restore model

        Restore model from the given `filepath`. `filepath` can 
        be either the root directory or the full checkpoint name. 
        For example, the checkpoint directory has the following
        structure:

        model/
        ├── checkpoint  (*)
        ├── weights-5.config.json
        ├── weights-5.data-00000-of-00001
        ├── weights-5.index
        ├── weights-10.config.json
        ├── weights-10.data-00000-of-00001
        ├── weights-10.index
        ├── weights-15.config.json
        ├── weights-15.data-00000-of-00001
        └── weights-15.index

        If `model/` is given, the latest checkpoint is loaded
        acccording to the `checkpoint` file (*), which, in this case,
        is `weights-15`. If `model/weights-10` is given, then 
        `model.weights-10` is loaded.

        Args:
            filepath (str): path to load weights and config. 

        Raises:
            RuntimeError: Config file not found

        Returns:
            DQN: restored model
        '''

        filepath = os.path.abspath(filepath)

        if os.path.isdir(filepath):
            # get latest checkpoint. (filepath is a dir)
            latest_checkpoint = tf.train.latest_checkpoint(filepath)
        else:
            # get latest checkpoint (full checkpoint name)
            if latest_checkpoint is None:
                latest_checkpoint = filepath

        # determines whether the file exists
        config_path = latest_checkpoint + CONFIG_SUFFIX

        if not os.path.isfile(config_path):
            raise RuntimeError('Failed to restore model, config file not '
                    'found: {}'.format(config_path))
        
        logger.debug('Restore weights from: {}'.format(latest_checkpoint))

        # restore config & reconstruct model
        self   = cls.from_config(config_path)

        # restore weights
        status = tf.train.Checkpoint(model=self).restore(latest_checkpoint)

        return self



    # def save(self, filepath: str, weights_name: str='weights', checkpoint_number: int=None):
    #     '''Save model weights and config

    #     The weights are saved to `{filepath}/{weights_name}-{checkpoint_number}.xxxxxx`
    #     and config saved to `{filepath}/{weights_name}-{checkpoint_number}-config.xxxxxx`

    #     Args:
    #         filepath (str): base directory to save weights and config
    #         weights_name (str, optional): weights filename. Defaults to 'weights'.
    #         checkpoint_number (int, optional): checkpoint number. Defaults to None.

    #     Returns:
    #         str: saved path
    #     '''

    #     # create checkpoint
    #     checkpoint = tf.train.Checkpoint(model=self)

    #     # create checkpoint manager
    #     manager = _get_checkpoint_manager(checkpoint, filepath, weights_name)

    #     # get latest checkpoint number
    #     latest_checkpoint = manager.latest_checkpoint
    #     latest_checkpoint_number = _get_latest_checkpoint_number(latest_checkpoint)

    #     if latest_checkpoint_number is None:
    #         latest_checkpoint_number = 0

    #     if checkpoint_number is None:
    #         checkpoint_number = latest_checkpoint_number + 1

    #     # save weights
    #     manager.save(checkpoint_number=checkpoint_number)

    #     # save config
    #     config_path = manager.latest_checkpoint + '-config.json'
    #     self.save_config(config_path)

    #     return manager.latest_checkpoint


    # @classmethod
    # def load(cls, filepath, weights_name='weights'):
    #     '''
    #     Load model weights and config
    #     '''

    #     # get latest checkpoint
    #     latest_checkpoint = tf.train.latest_checkpoint(filepath)
        
    #     if latest_checkpoint is None:
    #         latest_checkpoint = os.path.join(os.path.abspath(filepath),
    #                                         weights_name)

    #     config_path = latest_checkpoint + '-config.json'


    #     if not os.path.isfile(config_path):
    #         raise RuntimeError('Failed to restore model, config file not '
    #                 'found: {}, weights_name: {}'.format(filepath, weights_name))
        
    #     logger.debug('Restore weights from: {}'.format(latest_checkpoint))

    #     # restore config
    #     self = cls.load_config(config_path)

    #     # restore weights
    #     status = tf.train.Checkpoint(model=self).restore(latest_checkpoint)

    #     return self

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
        '''Predict actions

        Returns:
            np.ndarray: Predicted actions.
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

        Args:
            env (gym.Env): Environment for evaluation.
            n_episodes (int): number of episodes to evaluate.
            max_steps (int): Maximum steps in one episode.

        Return:
            eps_rews: (list) episode rewards.
            eps_steps: (list) episode length.
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
        '''Train model

        Returns:
            TrainableModel: self
        '''        
        
        raise NotImplementedError('Method not implemented')
