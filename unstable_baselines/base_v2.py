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

from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.lib.io import file_io
from tensorflow.python.framework import errors

# --- my module ---
from unstable_baselines import logger
from unstable_baselines import utils_v2 as utils

__all__ = ['SavableModel',
           'TrainableModel']


LOG = logger.getLogger()

# === Const params ===
WEIGHTS_NAME='weights'
CONFIG_SUFFIX='.config.json'
BEST_CHECKPOINT='best_checkpoint'


# === Base model utils ===

def _get_best_checkpoint_filename(save_dir, latest_filename):
    if latest_filename is None:
        latest_filename = BEST_CHECKPOINT
    return os.path.join(save_dir, latest_filename)

def _generate_best_checkpoint_state(save_dir,
                                    best_checkpoint_path,
                                    best_checkpoint_metrics=None,
                                    last_preserved_timestamp=None):
    if not os.path.isabs(save_dir):
        if not os.path.isabs(best_checkpoint_path):
            best_checkpoint_path = os.path.relpath(best_checkpoint_path, save_dir)

    coord_best_checkpoint_state = utils.StateObject(
        best_checkpoint_path=best_checkpoint_path,
        best_checkpoint_metrics=best_checkpoint_metrics,
        last_preserved_timestamp=last_preserved_timestamp)

    return coord_best_checkpoint_state

def _update_best_checkpoint_state(save_dir,
                                  best_checkpoint_path,
                                  best_checkpoint_metrics=None,
                                  latest_filename=None,
                                  save_relative_paths=True,
                                  last_preserved_timestamp=None):
    # Writes the "best_checkpoint" file for the coordinator for later restoration.
    coord_checkpoint_filename = _get_best_checkpoint_filename(save_dir, latest_filename)
    if save_relative_paths:
        if os.path.isabs(best_checkpoint_path):
            rel_best_checkpoint_path = os.path.relpath(
                best_checkpoint_path, save_dir)
        else:
            rel_best_checkpoint_path = best_checkpoint_path
        
        ckpt = _generate_best_checkpoint_state(
            save_dir,
            rel_best_checkpoint_path,
            best_checkpoint_metrics=best_checkpoint_metrics,
            last_preserved_timestamp=last_preserved_timestamp)
    else:
        ckpt = _generate_best_checkpoint_state(
            save_dir,
            best_checkpoint_path,
            best_checkpoint_metrics=best_checkpoint_metrics,
            last_preserved_timestamp=last_preserved_timestamp)

    if coord_checkpoint_filename == ckpt.best_checkpoint_path:
        raise RuntimeError('Save path \'{}\' conflicts with path used for '
                            'best checkpoint state.  Please use a different save path.'.format(
                                best_checkpoint_path))
    
    file_io.atomic_write_string_to_file(coord_checkpoint_filename,
                                        ckpt.tostring())

def _get_best_checkpoint_state(checkpoint_dir, latest_filename=None):

    ckpt = None
    coord_checkpoint_filename = _get_best_checkpoint_filename(checkpoint_dir,
                                                             latest_filename)
    try:
        if file_io.file_exists(coord_checkpoint_filename):
            file_content = file_io.read_file_to_string(
                coord_checkpoint_filename)
            ckpt = utils.StateObject.fromstring(file_content)
            if not ckpt.get('best_checkpoint_path', None):
                ckpt.best_checkpoint_path = ''
            if not os.path.isabs(ckpt.best_checkpoint_path):
                ckpt.best_checkpoint_path = os.path.join(checkpoint_dir,
                                                        ckpt.best_checkpoint_path)
    except Exception as e:
        LOG.warning('{}: {}'.format(type(e).__name__), e)
        LOG.warning('{}: Best checkpoint ignored'.format(coord_checkpoint_filename))
    
    return ckpt


def _prefix_to_checkpoint_path(prefix, format_version):
    if format_version == saver_pb2.SaverDef.V2:
        return prefix + '.index'
    return prefix


def _checkpoint_exists(checkpoint_path):
    v2_path = _prefix_to_checkpoint_path(checkpoint_path,
                                         saver_pb2.SaverDef.V2)
    v1_path = _prefix_to_checkpoint_path(checkpoint_path,
                                         saver_pb2.SaverDef.V1)
    if file_io.get_matching_files(v2_path) or file_io.get_matching_files(
            v1_path):
        return True
    return False

def _get_best_checkpoint(checkpoint_dir, latest_filename=None):
    '''Find the filename of the best checkpoint file.

    Args:
        checkpoint_dir: Directory where the variables were saved.
        latest_filename: Optional name for the checkpoint state file 
            that contains the best checkpoint filenames.
            See the corresponding argument to `v1.train.Saver.save`.

    Returns:
        The full path to the best checkpoint or `None` if no checkpoint was found.
    '''
    ckpt = _get_best_checkpoint_state(checkpoint_dir, latest_filename)
    if ckpt and ckpt.best_checkpoint_path:
        if _checkpoint_exists(ckpt.best_checkpoint_path):
            return ckpt.best_checkpoint_path
        else:
            LOG.error("Couldn't match files for checkpoint {}".format(
                ckpt.best_checkpoint_path))
    return None

def _delete_file_if_exists(filespec):
    '''Deletes files matching `filespec`'''
    for pathname in file_io.get_matching_files(filespec):
        try:
            file_io.delete_file(pathname)
        except errors.NotFoundError:
            LOG.warning(
                "Hit NotFoundError when deleting '{}', possibly because another "
                "process/thread is also deleting/moving the same file".format(
                    pathname))



def _get_checkpoint_manager(checkpoint, 
                            directory,
                            max_to_keep=None,
                            checkpoint_name=WEIGHTS_NAME,
                            metrics_compare_fn=None,
                            **kwargs):
    
    # manager = tf.train.CheckpointManager(
    #         checkpoint,
    #         directory=directory,
    #         checkpoint_name=checkpoint_name
    #         max_to_keep=max_to_keep,
    #         **kwargs)
    manager = CheckpointManager(checkpoint=checkpoint,
                                directory=directory,
                                max_to_keep=max_to_keep,
                                checkpoint_name=checkpoint_name,
                                metrics_compare_fn=metrics_compare_fn,
                                **kwargs)

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


def _default_metric_compare_fn(last_metrics, new_metrics):
    '''Compare two metrics

    Return:
        True if new_metrics is equal to or better than
            last_metrics. False otherwise.
    '''
    if last_metrics is None:
        return True
    if new_metrics is None:
        return False
    try:
        better = not (last_metrics > new_metrics)
        return better
    except Exception as e:
        LOG.warning('{}: {}'.format(type(e).__name__, e))
        LOG.warning('Metrics are not comparable: {} vs {}'.format(
            type(last_metrics).__name__, type(new_metrics).__name__))
    return False

class CheckpointManager(tf.train.CheckpointManager):
    def __init__(self,
                checkpoint,
                directory,
                max_to_keep,
                keep_checkpoint_every_n_hours=None,
                checkpoint_name=WEIGHTS_NAME,
                step_counter=None,
                checkpoint_interval=None,
                init_fn=None,
                metrics_compare_fn=None):
        super().__init__(
            checkpoint,
            directory,
            max_to_keep,
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
            checkpoint_name=checkpoint_name,
            step_counter=step_counter,
            checkpoint_interval=checkpoint_interval,
            init_fn=init_fn)
        
        if metrics_compare_fn is None:
            metrics_compare_fn = _default_metric_compare_fn
        self._metrics_compare_fn = metrics_compare_fn

        # Restore best checkpoint state
        recovered_state = _get_best_checkpoint_state(directory)
        current_clock = time.time()
        if recovered_state is None:
            self._best_checkpoint = None
            self._best_checkpoint_metrics = None
            # The `last preserved timestamp` has already been set in
            # supper().__init__. So we don't need to set it here
            # self._last_preserved_timestamp
        else:
            self._best_checkpoint = recovered_state.best_checkpoint_path
            self._best_checkpoint_metrics = recovered_state.best_checkpoint_metrics
            del recovered_state

    @property
    def best_checkpoint(self):
        return self._best_checkpoint

    @property
    def best_checkpoint_metrics(self):
        return self._best_checkpoint_metrics

    def _record_state(self):
        '''This function is used to overwrite its original
        functionality, which is inhertied from the parent
        class, to perform nothing.
        '''
        pass

    def _sweep(self):
        '''This function is used to overwrite its original
        functionality, which is inhertied from the parent
        class, to perform nothing.
        '''
        pass

    def _record_checkpoint_state(self):
        '''Replaces super()._record_state'''
        super()._record_state()
        # save the best checkpoint infomation
        _update_best_checkpoint_state(
            self._directory,
            best_checkpoint_path=self.best_checkpoint,
            best_checkpoint_metrics=self.best_checkpoint_metrics,
            last_preserved_timestamp=self._last_preserved_timestamp,
            save_relative_paths=True)

    def _sweep_checkpoints(self):
        '''Replaces super()._sweep'''
        if not self._max_to_keep:
            # Does not update self._last_preserved_timestamp, since everything is kept
            # in the active set.
            return
        while len(self._maybe_delete) > self._max_to_keep:
            filename, timestamp = self._maybe_delete.popitem(last=False)

            if (self._keep_checkpoint_every_n_hours
                    and (timestamp - self._keep_checkpoint_every_n_hours * 3600.
                        >= self._last_preserved_timestamp)):
                self._last_preserved_timestamp = timestamp
                continue
            
            # skip best checkpoint
            if (self.best_checkpoint and filename == self.best_checkpoint):
                continue

            _delete_file_if_exists(filename + ".index")
            _delete_file_if_exists(filename + ".data-?????-of-?????")

    def save(self, checkpoint_number=None,
                   checkpoint_metrics=None,
                   check_interval=True):

        # save checkpoints
        save_path = super().save(checkpoint_number=checkpoint_number,
                                 check_interval=check_interval)
        
        # TODO: Deepcopy checkpoint metrics
        _checkpoint_metrics = checkpoint_metrics
        # save the best checkpoint
        # If best_checkpoint is empty -> replace it
        if (not self.best_checkpoint
                or self._metrics_compare_fn is None):
            self._best_checkpoint = save_path
            self._best_checkpoint_metrics = _checkpoint_metrics
        else:
            # If the new checkpoint performs better than the old one
            # than replace that.
            better = self._metrics_compare_fn(
                last_metrics=self._best_checkpoint_metrics,
                new_metrics=checkpoint_metrics)

            if better:
                self._best_checkpoint = save_path
                self._best_checkpoint_metrics = _checkpoint_metrics

        self._record_checkpoint_state()
        # Delete old checkpoints
        self._sweep_checkpoints()
        self._record_checkpoint_state()
        return save_path


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
            dict: A JSON serializable dictionary.
        '''
        raise NotImplementedError('get_config method not implemented')

    @classmethod
    def from_config(cls, config):
        '''Reconstruct model from configuration

        Args:
            config (dict): A dictionary deserialized from JSON.

        Returns:
            SavableModel: reconstructed model
        ''' 
        return cls(**config)

    def save_config(self, filepath):
        '''Save model configuration to `filepath`

        Args:
            filepath (str): path to save configuration
        '''
        config = self.get_config()

        utils.safe_json_dump(filepath, config)

    @classmethod
    def load_config(cls, filepath):
        '''Load and reconstruct model from configuration

        Args:
            filepath (str): path to load configuration

        Returns:
            : [description]
        '''
        config = utils.safe_json_load(filepath)

        return cls.from_config(config)

    def save(self, directory: str,
                   weights_name: str=WEIGHTS_NAME,
                   checkpoint_number: int=None,
                   checkpoint_metrics=None,
                   metrics_compare_fn=None,
                   max_to_keep=None,
                   **kwargs):
        '''Save model weights and config

        Default the weights are saved to 
          `{directory}/{weights_name}-{checkpoint_number}.index`
          `{directory}/{weights_name}-{checkpoint_number}.data-?????-of-?????`
        and config files are saved to 
          `{directory}/{weights_name}-{checkpoint_number}.config.json`

        In addition, `checkpoint` and `best_checkpoint` are created.
        The former is used to store the checkpoint information (latest 
        checkpoint, checkpoint paths, timestamps). The latter is used 
        to store the best checkpoint information (best checkpoint path,
        checkpoint metrics). To select the best model, `checkpoint_metrics`
        must be provided. If `checkpoint_metrics` are not comparable 
        using built-in Python comparison method, you must provide a 
        customized function with a signature:
            compare_fn(last_metrics, new_metrics)
        and pass to `metrics_compare_fn`. The `metrics_compare_fn`
        returns True if `new_metrics` is better than `last_metrics`.

        Example:
        
        >>> model.save('./my_model', 'weights')

        This creates a checkpoint structure:

        my_model/
        ├── checkpoint
        ├── best_checkpoint
        ├── weights-1.config.json
        ├── weights-1.data-00000-of-00001
        └── weights-1.index

        Args:
            directory (str): Root path to save weights and config
            weights_name (str, optional): Filename prefix. Defaults to WEIGHTS_NAME.
            checkpoint_number (int, optional): checkpoint number. Defaults to None.
            checkpoint_metrics (Any, optional): Model metrics used to determine the
                best model. Defaults to None.
            metrics_compare_fn (function, optional): Comparison function used to
                define the comparison method of two metrics. Defaults to None.
            max_to_keep (int, optional): maximum number of checkpoints to
                keep. Defaults to None.

        Returns:
            str: Path the checkpoint saved to
        '''

        # create checkpoint
        checkpoint = tf.train.Checkpoint(model=self)

        # create checkpoint manager
        manager = _get_checkpoint_manager(checkpoint=checkpoint,
                                        directory=directory,
                                        max_to_keep=max_to_keep,
                                        checkpoint_name=weights_name,
                                        metrics_compare_fn=metrics_compare_fn,
                                        **kwargs)

        # get latest checkpoint number
        latest_checkpoint = manager.latest_checkpoint
        latest_checkpoint_number = _get_latest_checkpoint_number(latest_checkpoint)

        if latest_checkpoint_number is None:
            latest_checkpoint_number = 0
        
        if checkpoint_number is None:
            checkpoint_number = latest_checkpoint_number + 1

        # save weights
        save_path = manager.save(checkpoint_number=checkpoint_number,
                                 checkpoint_metrics=checkpoint_metrics)

        # save config
        config_path = save_path + CONFIG_SUFFIX
        self.save_config(config_path)

        return save_path
    
    @classmethod
    def _preload(cls,  file_or_dir: str,
                       weights_name: str=WEIGHTS_NAME,
                       restore_best: bool=False):
        '''This method returns the aboluste path of the best 
        matched checkpoint file.

        `file_or_dir` can be the root directory of all checkpoints
        (containing `checkpoint` and `best_checkpoint` file), or 
        can be the full path to a specific checkpoint file (Eg. 
        './my_model/weights-10'). The checkpoint searching priority
        is as follows:

        1. If a directory is provided, it trys to retrieve the checkpoint 
            path from `checkpoint` or `best_checkpoint`. In this case, 
            `weights_name` is ignored.
        2. If step 1 failed, it tests if {file_or_dir}/{weights_name} 
            is a valid checkpoint path.
        3. If `file_or_dir` is not a directory or step 2 failed, it tests
            if {file_or_dir} is a valid checkpoint path. `weights_name`
            is also ignored in this case.

        Example:
        Assume we have the following checkpoint structure

            my_model/
            ├── checkpoint
            ├── best_checkpoint
            ├── weights-5.config.json
            ├── weights-5.data-00000-of-00001
            ├── weights-5.index
            ├── weights-10.config.json
            ├── weights-10.data-00000-of-00001
            ├── weights-10.index
            ├── weights-15.config.json
            ├── weights-15.data-00000-of-00001
            └── weights-15.index

        Case 1: A valid checkpoint root directory is provided
        >>> SavableModel._preload('./my_model')
        {absolute path}/my_model/weights-15
        >>> SavableModel._preload('./my_model', weights_name='whatever')
        {absolute path}/my_model/weights-15
        
        Case 2: Not a valid checkpoint directory (`checkpoint` or 
            `best_checkpoint` is deleted)
        >>> SavableModel._preload('./my_model', weights_name='weights-5')
        {absolute path}/my_model/weights-5
        >>> SavableModel._preload('./my_model', weights_name='weights-10')
        {absolute path}/my_model/weights-10

        Case 3: A full checkpoint path is provided
        >>> SavableModel._preload('./my_model/weights-10')
        {absolute path}/my_model/weights-10
        >>> SavableModel._preload('./my_model/weights-15')
        {absolute path}/my_model/weights-15

        Args:
            file_or_dir (str): A path to the checkpoint directory or
                a full checkpoint path.
            weights_name (str, optional): Weights name. Defaults to WEIGHTS_NAME.
            restore_best (bool, optional): Whether to restore the best
                checkpoint. Defaults to False.

        Raises:
            RuntimeError: Couldn't find a valid checkpoint path.

        Returns:
            str: Checkpoint path.
        '''        

        checkpoint_path = None
        file_or_dir = os.path.abspath(file_or_dir)

        if os.path.isdir(file_or_dir):
            directory = file_or_dir
            # resotre best checkpoint or latest checkpoint
            # if checkpoint state or best checkpoint state
            # files exist, get checkpoint path from files.
            # Eg. directory = model/
            if restore_best:
                checkpoint_path = _get_best_checkpoint(directory)
            else:
                checkpoint_path = tf.train.latest_checkpoint(directory)
            # If state files do not exist, try 
            # {directory}/{weights_name}
            # Eg. directory = model, weights_name = weights-10
            #   => model/weights-10
            if checkpoint_path is None:
                full_checkpoint_path = os.path.join(directory, weights_name)

                if _checkpoint_exists(full_checkpoint_path):
                    checkpoint_path = full_checkpoint_path
        # If file_or_directory is not a directory, or
        # we failed to get checkpoint path at the above steps
        # check if {directory} is a checkpoint path
        # Eg. direcory = model/weights-10
        if checkpoint_path is None:
            if _checkpoint_exists(file_or_dir):
                # Eg. directory = model/weights-10
                checkpoint_path = file_or_dir
            else:
                # If we still failed to get checkpoint path, raise Error
                raise RuntimeError("Couldn't find the checkpoint file for "
                                "checkpoint: {}".format(directory))
        # Get config file path
        config_path = checkpoint_path + CONFIG_SUFFIX
        # check if config path exists
        if not file_io.file_exists(config_path):
            raise RuntimeError("Couldn't find the config file for "
                                "checkpoint: {}".format(checkpoint_path))

        return checkpoint_path

    @classmethod
    def load(cls,  file_or_dir: str,
                   weights_name: str=WEIGHTS_NAME,
                   restore_best: bool=False):
        
        # find checkpoint path
        checkpoint_path = cls._preload(file_or_dir,
                                        weights_name,
                                        restore_best)

        LOG.debug('Restore weights from: {}'.format(checkpoint_path))

        # Get config path
        config_path = checkpoint_path + CONFIG_SUFFIX

        # Reconstruct model from config
        self = cls.load_config(config_path)

        # restore weights
        status = tf.train.Checkpoint(model=self).restore(checkpoint_path)

        return self

    # def save(self, filepath: str, 
    #                 checkpoint_number: int=None):
    #     '''Save model weights and config

    #     The weights are saved to `{filepath}-{checkpoint_number}.xxxxxx`
    #     and config saved to `{filepath}-{checkpoint_number}-config.xxxxxx`

    #     Args:
    #         filepath (str): path to save weights and config. If a directory
    #             is given (path followed by a single '/'), the weights are 
    #             saved to `{filepath}/weights.xxxxxx`.
    #         checkpoint_number (int, optional): checkpoint number.
    #             Defaults to None.
    #     '''

    #     # is a directory
    #     if filepath.endswith('/'):
    #         filepath = os.path.join(filepath, WEIGHTS_NAME)

    #     # base path
    #     filepath = os.path.abspath(filepath)
    #     filedir  = os.path.dirname(filepath)
    #     filename = os.path.basename(filepath)

    #     # create checkpoint 
    #     checkpoint = tf.train.Checkpoint(model=self)

    #     # create checkpoint manager
    #     manager = _get_checkpoint_manager(checkpoint, filedir, filename)

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
    #     config_path = manager.latest_checkpoint + CONFIG_SUFFIX
    #     self.save_config(config_path)

    #     return manager.latest_checkpoint

    # @classmethod
    # def load(cls, filepath: str):
    #     '''Restore model

    #     Restore model from the given `filepath`. `filepath` can 
    #     be either the root directory or the full checkpoint name. 
    #     For example, the checkpoint directory has the following
    #     structure:

    #     model/
    #     ├── checkpoint  (*)
    #     ├── weights-5.config.json
    #     ├── weights-5.data-00000-of-00001
    #     ├── weights-5.index
    #     ├── weights-10.config.json
    #     ├── weights-10.data-00000-of-00001
    #     ├── weights-10.index
    #     ├── weights-15.config.json
    #     ├── weights-15.data-00000-of-00001
    #     └── weights-15.index

    #     If `model/` is given, the latest checkpoint is loaded
    #     acccording to the `checkpoint` file (*), which, in this case,
    #     is `weights-15`. If `model/weights-10` is given, then 
    #     `model/weights-10` is loaded.

    #     Args:
    #         filepath (str): path to load weights and config. 

    #     Raises:
    #         RuntimeError: Config file not found

    #     Returns:
    #         DQN: restored model
    #     '''

    #     filepath = os.path.abspath(filepath)

    #     if os.path.isdir(filepath):
    #         # get latest checkpoint. (filepath is a dir)
    #         latest_checkpoint = tf.train.latest_checkpoint(filepath)
    #     else:
    #         # get latest checkpoint (full checkpoint name)
    #         latest_checkpoint = filepath

    #     # determines whether the file exists
    #     config_path = latest_checkpoint + CONFIG_SUFFIX

    #     if not os.path.isfile(config_path):
    #         raise RuntimeError('Failed to restore model, config file not '
    #                 'found: {}'.format(config_path))
        
    #     LOG.debug('Restore weights from: {}'.format(latest_checkpoint))

    #     # restore config & reconstruct model
    #     self   = cls.load_config(config_path)

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
        
        utils.soft_update(var, target_var, polyak=polyak)



class TrainableModel(SavableModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize state object
        #  store training states
        s = utils.StateObject()
        s.num_timesteps = 0
        s.num_epochs    = 0
        s.num_subepochs = 0
        s.num_gradsteps = 0
        s.progress      = 0
        
        self._state = s

    @abc.abstractmethod
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
            max_steps (int): Maximum steps in one episode. Set to -1 
                to run episodes until done.

        Return:
            eps_rews: (list) episode rewards.
            eps_steps: (list) episode length.
        '''

        if n_episodes <= 0:
            return [], []

        eps_info = []

        for episode in range(n_episodes):
            obs = env.reset()
            total_rews  = 0
            total_steps = 0

            while total_steps != max_steps:
                # predict action
                acts = self.predict(obs)

                # step environment
                obs, rew, done, info = env.step(acts)
                
                total_rews += rew
                total_steps += 1

                if done:
                    break
        
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

    def save_config(self, filepath):
        '''Save model config to `filepath`

        Args:
            filepath (str): path to save configuration
        '''
        _config = self.get_config()

        config = {
            'state': self.state,
            'config': _config
        }

        utils.safe_json_dump(filepath, config)

    @classmethod
    def load_config(cls, filepath):
        '''Load config file and reconstruct model

        Args:
            filepath (str): path to load config file

        Returns:
            TrainableModel: reconstrcted model
        '''
        config = utils.safe_json_load(filepath)

        _config = config.get('config', {})
        state   = config.get('state', {})

        self = cls.from_config(_config)
        self.state.update(state)

        return self

    @property
    def state(self):
        return self._state

    @property
    def num_timesteps(self):
        return self._state.num_timesteps

    @property
    def num_epochs(self):
        return self._state.num_epochs

    @property
    def num_subepochs(self):
        return self._state.num_subepochs

    @property
    def num_gradsteps(self):
        return self._state.num_gradsteps

    @property
    def progress(self):
        return self._state.progress

    @num_timesteps.setter
    def num_timesteps(self, value):
        self._state.num_timesteps = value

    @num_epochs.setter
    def num_epochs(self, value):
        self._state.num_epochs = value

    @num_subepochs.setter
    def num_subepochs(self, value):
        self._state.num_subepochs = value

    @num_gradsteps.setter
    def num_gradsteps(self, value):
        self._state.num_gradsteps = value

    @progress.setter
    def progress(self, value):
        self._state.progress = value