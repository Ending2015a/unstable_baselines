# --- built in ---
import os
import abc
import copy
import math
import time
import datetime
import collections

# --- 3rd party ---
import gym
import tqdm
import numpy as np
import tensorflow as tf

from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.lib.io import file_io
from tensorflow.python.framework import errors

# --- my module ---
from unstable_baselines import logger
from unstable_baselines.lib import utils as ub_utils
from unstable_baselines.lib.envs import vec as ub_vec

__all__ = ['SavableModel', 'TrainableModel']

LOG = logger.getLogger()

# === Const params ===
WEIGHTS_NAME = 'weights'
CONFIG_SUFFIX = '.config.json'
BEST_CHECKPOINT = 'best_checkpoint'

# === Base model utils ===


def _get_best_checkpoint_filename(save_dir, latest_filename):
    if latest_filename is None:
        latest_filename = BEST_CHECKPOINT
    return os.path.join(save_dir, latest_filename)


def _generate_best_checkpoint_state(
    save_dir,
    best_checkpoint_path,
    best_checkpoint_metrics=None,
    all_checkpoint_paths=None,
    all_checkpoint_metrics=None,
    last_preserved_timestamp=None
):
    if all_checkpoint_paths is None:
        all_checkpoint_paths = []
    if all_checkpoint_metrics is None:
        all_checkpoint_metrics = []

    if not os.path.isabs(save_dir):
        if not os.path.isabs(best_checkpoint_path):
            best_checkpoint_path = os.path.relpath(best_checkpoint_path, save_dir)
        for i, p in enumerate(all_checkpoint_paths):
            if not os.path.isabs(p):
                all_checkpoint_paths[i] = os.path.relpath(p, save_dir)

    all_paths = all_checkpoint_paths
    all_metrics = all_checkpoint_metrics
    all_checkpoint_metrics = list(zip(all_paths, all_metrics))

    coord_best_checkpoint_state = ub_utils.StateObject(
        best_checkpoint_path=best_checkpoint_path,
        best_checkpoint_metrics=best_checkpoint_metrics,
        all_checkpoint_metrics=all_checkpoint_metrics,
        last_preserved_timestamp=last_preserved_timestamp
    )

    return coord_best_checkpoint_state


def _update_best_checkpoint_state(
    save_dir,
    best_checkpoint_path,
    best_checkpoint_metrics=None,
    all_checkpoint_paths=None,
    all_checkpoint_metrics=None,
    latest_filename=None,
    save_relative_paths=True,
    last_preserved_timestamp=None
):
    # Writes the "best_checkpoint" file for the coordinator for later restoration.
    coord_checkpoint_filename = _get_best_checkpoint_filename(
        save_dir, latest_filename
    )
    if save_relative_paths:
        if os.path.isabs(best_checkpoint_path):
            rel_best_checkpoint_path = os.path.relpath(
                best_checkpoint_path, save_dir
            )
        else:
            rel_best_checkpoint_path = best_checkpoint_path
        rel_all_checkpoint_paths = []
        for p in all_checkpoint_paths:
            if os.path.isabs(p):
                rel_all_checkpoint_paths.append(os.path.relpath(p, save_dir))
            else:
                rel_all_checkpoint_paths.append(p)
        ckpt = _generate_best_checkpoint_state(
            save_dir,
            rel_best_checkpoint_path,
            best_checkpoint_metrics=best_checkpoint_metrics,
            all_checkpoint_paths=rel_all_checkpoint_paths,
            all_checkpoint_metrics=all_checkpoint_metrics,
            last_preserved_timestamp=last_preserved_timestamp
        )
    else:
        ckpt = _generate_best_checkpoint_state(
            save_dir,
            best_checkpoint_path,
            best_checkpoint_metrics=best_checkpoint_metrics,
            all_checkpoint_paths=all_checkpoint_paths,
            all_checkpoint_metrics=all_checkpoint_metrics,
            last_preserved_timestamp=last_preserved_timestamp
        )

    if coord_checkpoint_filename == ckpt.best_checkpoint_path:
        raise RuntimeError(
            'Save path \'{}\' conflicts with path used for '
            'best checkpoint state.  Please use a different save path.'
            .format(best_checkpoint_path)
        )

    file_io.atomic_write_string_to_file(coord_checkpoint_filename, ckpt.tostring(2))


def _get_best_checkpoint_state(checkpoint_dir, latest_filename=None):

    ckpt = None
    coord_checkpoint_filename = _get_best_checkpoint_filename(
        checkpoint_dir, latest_filename
    )
    try:
        if file_io.file_exists(coord_checkpoint_filename):
            file_content = file_io.read_file_to_string(coord_checkpoint_filename)
            ckpt = ub_utils.StateObject.fromstring(file_content)
            if not ckpt.get('best_checkpoint_path', None):
                ckpt.best_checkpoint_path = ''
            if not ckpt.get('all_checkpoint_metrics', None):
                ckpt.all_checkpoint_metrics = []
            if not os.path.isabs(ckpt.best_checkpoint_path):
                ckpt.best_checkpoint_path = os.path.join(
                    checkpoint_dir, ckpt.best_checkpoint_path
                )
            all_checkpoint_paths = []
            all_checkpoint_metrics = []
            for filename, metrics in ckpt.all_checkpoint_metrics:
                if not os.path.isabs(filename):
                    filename = os.path.join(checkpoint_dir, filename)
                all_checkpoint_paths.append(filename)
                all_checkpoint_metrics.append(metrics)
            ckpt.all_checkpoint_paths = all_checkpoint_paths
            ckpt.all_checkpoint_metrics = all_checkpoint_metrics
    except Exception as e:
        LOG.warning(
            'Failed to load checkpoint states from '
            f'{coord_checkpoint_filename}'
        )
        LOG.warning(f'  {type(e).__name__}: {e}')
    return ckpt


def _prefix_to_checkpoint_path(prefix, format_version):
    if format_version == saver_pb2.SaverDef.V2:
        return prefix + '.index'
    return prefix


def _checkpoint_exists(checkpoint_path):
    v2_path = _prefix_to_checkpoint_path(checkpoint_path, saver_pb2.SaverDef.V2)
    v1_path = _prefix_to_checkpoint_path(checkpoint_path, saver_pb2.SaverDef.V1)
    if file_io.get_matching_files(v2_path) or file_io.get_matching_files(v1_path):
        return True
    return False


def _get_best_checkpoint(checkpoint_dir, latest_filename=None):
    """Find the filename of the best checkpoint file.

    Args:
        checkpoint_dir: Directory where the variables were saved.
        latest_filename: Optional name for the checkpoint state file
            that contains the best checkpoint filenames.

    Returns:
        The full path to the best checkpoint or `None` if no checkpoint was found.

    """
    ckpt = _get_best_checkpoint_state(checkpoint_dir, latest_filename)
    if ckpt and ckpt.best_checkpoint_path:
        if _checkpoint_exists(ckpt.best_checkpoint_path):
            return ckpt.best_checkpoint_path
        else:
            LOG.error(
                "Couldn't match files for checkpoint {}".format(
                    ckpt.best_checkpoint_path
                )
            )
    return None


def _delete_file_if_exists(filespec):
    """Deletes files matching `filespec`"""
    for pathname in file_io.get_matching_files(filespec):
        try:
            file_io.delete_file(pathname)
        except errors.NotFoundError:
            LOG.warning(
                "Hit NotFoundError when deleting '{}', possibly because another "
                "process/thread is also deleting/moving the same file"
                .format(pathname)
            )


def _get_checkpoint_manager(
    checkpoint,
    directory,
    max_to_keep=None,
    checkpoint_name=WEIGHTS_NAME,
    metrics_compare_fn=None,
    **kwargs
):
    manager = CheckpointManager(
        checkpoint=checkpoint,
        directory=directory,
        max_to_keep=max_to_keep,
        checkpoint_name=checkpoint_name,
        metrics_compare_fn=metrics_compare_fn,
        **kwargs
    )
    return manager


def _get_latest_checkpoint_number(latest_checkpoint):
    """Get latest checkpoint number from latest_checkpoint.
    Return None if latest_checkpoint is None or failed to get number
    """
    checkpoint_number = None
    if latest_checkpoint and isinstance(latest_checkpoint, str):
        if '-' in latest_checkpoint:
            number_str = latest_checkpoint.rsplit('-', 1)[1]
            if number_str.isdigit():
                checkpoint_number = int(number_str)
    return checkpoint_number


def _default_metric_compare_fn(last_metrics, new_metrics):
    """Compare two metrics

    Args:
        last_metrics: previous metrics. A comparable object.
        new_metrics: new metrics. A comparable object.

    Returns:
        True if new_metrics is equal to or better than the last_metrics.
            False, otherwise.
    """
    if last_metrics is None:
        return True
    if new_metrics is None:
        return False
    try:
        better = not (last_metrics > new_metrics)
        return better
    except TypeError:
        raise TypeError(
            'Metrics are not comparable: {} vs {}'.format(
                type(last_metrics).__name__, type(new_metrics).__name__
            )
        )
    except Exception:
        # unknown error
        raise
    return False


class CheckpointManager(tf.train.CheckpointManager):
    """An extension to tensorflow CheckpointManager. This supports
    managing metrics of each checkpoint, if provided, to keeping
    the best performed model.

    Example usage:

    ```python
    import unstable_baselines as ub
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = ub.base.CheckpointManager(
        checkpoint, directory="/tmp/model", max_to_keep=5)
    status = checkpoint.restore(manager.best_checkpoint)
    while True:
        # train
        manager.save(checkpoint_metrics=accuracy)
    ```
    """
    def __init__(
        self,
        checkpoint,
        directory,
        max_to_keep,
        keep_checkpoint_every_n_hours=None,
        checkpoint_name=WEIGHTS_NAME,
        step_counter=None,
        checkpoint_interval=None,
        init_fn=None,
        metrics_compare_fn=None
    ):
        super().__init__(
            checkpoint,
            directory,
            max_to_keep,
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
            checkpoint_name=checkpoint_name,
            step_counter=step_counter,
            checkpoint_interval=checkpoint_interval,
            init_fn=init_fn
        )

        if metrics_compare_fn is None:
            metrics_compare_fn = _default_metric_compare_fn
        self._metrics_compare_fn = metrics_compare_fn

        # Restore best checkpoint state
        recovered_state = _get_best_checkpoint_state(directory)
        self._preserved_metrics = collections.OrderedDict()
        if recovered_state is None:
            self._best_checkpoint = None
            self._best_checkpoint_metrics = None
            # The `last preserved timestamp` has already been set in
            # supper().__init__. So we don't need to set it here
            # self._last_preserved_timestamp
        else:
            self._best_checkpoint = recovered_state.best_checkpoint_path
            self._best_checkpoint_metrics = recovered_state.best_checkpoint_metrics
            all_paths = recovered_state.all_checkpoint_paths
            all_metrics = recovered_state.all_checkpoint_metrics
            del recovered_state
            for (filename, metrics) in zip(all_paths, all_metrics):
                self._preserved_metrics[filename] = metrics

    @property
    def best_checkpoint(self):
        return self._best_checkpoint

    @property
    def best_checkpoint_metrics(self):
        return self._best_checkpoint_metrics

    def _record_state(self):
        """This function is used to overwrite its original functionality, which
        inhertied from the parent class, to perform nothing.
        """
        pass

    def _sweep(self):
        """This function is used to overwrite its original functionality, which
        inhertied from the parent class, to perform nothing.
        """
        pass

    def _record_checkpoint_state(self):
        """Replaces super()._record_state"""
        super()._record_state()
        filenames, metrics = zip(*self._preserved_metrics.items())
        # save the best checkpoint infomation
        _update_best_checkpoint_state(
            self._directory,
            best_checkpoint_path=self.best_checkpoint,
            best_checkpoint_metrics=self.best_checkpoint_metrics,
            all_checkpoint_paths=filenames,
            all_checkpoint_metrics=metrics,
            last_preserved_timestamp=self._last_preserved_timestamp,
            save_relative_paths=True
        )

    def _sweep_checkpoints(self):
        """Replaces super()._sweep"""
        if not self._max_to_keep:
            # Does not update self._last_preserved_timestamp, since everything is
            # kept in the active set.
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
        # only preserve those checkpoints not been deleted
        preserved_metrics = collections.OrderedDict()
        for filename in self._maybe_delete.keys():
            if filename in self._preserved_metrics.keys():
                preserved_metrics[filename] = self._preserved_metrics[filename]
        # update preserved metrics
        del self._preserved_metrics
        self._preserved_metrics = preserved_metrics

    def save(
        self, checkpoint_number=None, checkpoint_metrics=None, check_interval=True
    ):
        # save checkpoints
        save_path = super().save(
            checkpoint_number=checkpoint_number, check_interval=check_interval
        )
        # Deepcopy checkpoint metrics
        _checkpoint_metrics = copy.deepcopy(checkpoint_metrics)
        # save the best checkpoint
        # If best_checkpoint is empty -> replace it
        if (not self.best_checkpoint or self._metrics_compare_fn is None):
            self._best_checkpoint = save_path
            self._best_checkpoint_metrics = _checkpoint_metrics
        else:
            # If the new checkpoint performs better than the old one
            # than replace that.
            better = self._metrics_compare_fn(
                last_metrics=self._best_checkpoint_metrics,
                new_metrics=checkpoint_metrics
            )
            # replace the old checkpoint info
            if better:
                self._best_checkpoint = save_path
                self._best_checkpoint_metrics = _checkpoint_metrics
        # insert the new checkpoint into perserved_metrics
        if save_path in self._preserved_metrics:
            # reinsert it to make sure it goes to the end of
            # the queue.
            del self._preserved_metrics[save_path]
        timestamp = self._maybe_delete[save_path]
        self._preserved_metrics[save_path] = {
            'metrics': _checkpoint_metrics, 'timestamp': timestamp
        }
        self._record_checkpoint_state()
        # Delete old checkpoints
        self._sweep_checkpoints()
        self._record_checkpoint_state()
        return save_path


# === SavableModel ===


class SavableModel(tf.keras.Model, metaclass=abc.ABCMeta):
    """Provides a common interface to handle model save/load operations.

    Subclassing to make the weights and configurations of a keras model
    can be easily saved/loaded by calling `save` and `load`. One
    should also override `get_config` method to support this feature.
    `get_config` returns a dictionary representing the configuration of
    the model and will be passed into `__init__` of your model class on
    loading saved model. One can also override `from_config` to customize
    model creation on loading.

    Example usage:

    ```python
    class MyModel(ub.base.SavableModel):
        def __init__(self, param):
            self.param = param

        def get_config(self):
            return {
                'param': self.param
            }

    model = MyModel(10)
    model.save("/tmp/my_model")
    loaded_model = MyModel.load("/tmp/my_model")
    print(loaded_model.param)
    ```
    This will get:
    ```
    10
    ```

    If your model has something that can't be saved or should be determined after
    the model loaded you can override `from_config` method:
    ```python
    class MyModel(ub.base.SavableModel):
        def __init__(self, dont_save, param):
            self.dont_save = dont_save
            self.param = param

        def get_config(self):
            return {
                'param': self.param
            }

        @classmethod
        def from_config(cls, config):
            # Create a MyModel instance
            self = cls(dont_save=None, **config)
            # ... Create your dont_save object
            self.dont_save = dont_save
            return self
    ```
    """
    @abc.abstractmethod
    def get_config(self):
        """Get model configuration

        Returns:
            dict: A JSON serializable dictionary.
        """
        return {}

    @classmethod
    def from_config(cls, config):
        """Reconstruct model from configuration

        Args:
            config (dict): dictionary deserialized from JSON.

        Returns:
            SavableModel: reconstructed model
        """
        return cls(**config)

    def save_config(self, filepath):
        """Save model configuration to `filepath`

        Args:
            filepath(str): path to save configuration
        """
        config = self.get_config()
        ub_utils.safe_json_dump(filepath, config)

    @classmethod
    def load_config(cls, filepath):
        """Load and reconstruct model from configuration

        Args:
            filepath(str): path to load configuration

        Returns:
            SavableModel: new model from loaded configurations

        """
        config = ub_utils.safe_json_load(filepath)
        return cls.from_config(config)

    def save(
        self,
        directory: str,
        weights_name: str = WEIGHTS_NAME,
        checkpoint_number: int = None,
        checkpoint_metrics=None,
        metrics_compare_fn=None,
        max_to_keep=None,
        **kwargs
    ):
        """Save model weights and configurations

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

        Example usage:

        ```python
        model.save("/tmp/my_model", "weights")
        ```

        This creates a checkpoint structure:

        /tmp/my_model/
        ├── checkpoint
        ├── best_checkpoint
        ├── weights-1.config.json
        ├── weights-1.data-00000-of-00001
        └── weights-1.index

        Args:
            directory (str): root path to save weights and config
            weights_name (str): filename prefix. Defaults to "weights".
            checkpoint_number (int): checkpoint number. Defaults to None.
            checkpoint_metrics (Any, optional): model metrics used to determine
                the best model. Defaults to None.
            metrics_compare_fn (Callable, optional): a callback taking two
                arguments (last_metrics, new_metrics) defines the comparison
                method of two metrics. Return True if `new_metrics` is better
                than or equal to `last_metrics`. Defaults to None.
            max_to_keep (int, optional): maximum number of checkpoints to
                keep. Defaults to None.

        Returns:
          str: Path to the checkpoint where it is saved
        """

        # create checkpoint
        checkpoint = tf.train.Checkpoint(model=self)

        # create checkpoint manager
        manager = _get_checkpoint_manager(
            checkpoint=checkpoint,
            directory=directory,
            max_to_keep=max_to_keep,
            checkpoint_name=weights_name,
            metrics_compare_fn=metrics_compare_fn,
            **kwargs
        )

        # get latest checkpoint number
        latest_checkpoint = manager.latest_checkpoint
        latest_checkpoint_number = _get_latest_checkpoint_number(latest_checkpoint)

        if latest_checkpoint_number is None:
            latest_checkpoint_number = 0

        if checkpoint_number is None:
            checkpoint_number = latest_checkpoint_number + 1

        # save weights
        save_path = manager.save(
            checkpoint_number=checkpoint_number,
            checkpoint_metrics=checkpoint_metrics
        )

        # save config
        config_path = save_path + CONFIG_SUFFIX
        self.save_config(config_path)

        return save_path

    @classmethod
    def _preload(
        cls,
        file_or_dir: str,
        weights_name: str = WEIGHTS_NAME,
        best: bool = False
    ):
        """This method returns the aboluste path of the checkpoint file
        that best matched to the input arguments.

        `file_or_dir` can be the root directory of all checkpoints
        (containing `checkpoint` and `best_checkpoint` file), or
        can be the full path to a specific checkpoint file (Eg.
        './my_model/weights-10'). The checkpoint searching priority
        is:

        1. If a directory is provided, it trys to retrieve the checkpoint
            path from `checkpoint` or `best_checkpoint`. In this case,
            `weights_name` is ignored.
        2. It tests if `{file_or_dir}/{weights_name}` is a valid checkpoint
            path. If yes, return this path.
        3. If `file_or_dir` is not a directory, it tests if `{file_or_dir}`
            is a valid checkpoint path. `weights_name` is also ignored in
            this case.

        Example:

        Assume we have the following checkpoint structure

            /tmp/my_model/
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
        ```python
        >>> SavableModel._preload('/tmp/my_model')
        /tmp/my_model/weights-15
        >>> SavableModel._preload('/tmp/my_model', weights_name='whatever')
        /tmp/my_model/weights-15
        ```

        Case 2: Not a valid checkpoint directory (`checkpoint` or
            `best_checkpoint` is deleted)
        ```python
        >>> SavableModel._preload('/tmp/my_model', weights_name='weights-5')
        /tmp/my_model/weights-5
        ```

        Case 3: A full checkpoint path is provided
        ```python
        >>> SavableModel._preload('/tmp/my_model/weights-10')
        /tmp/my_model/weights-10
        ```

        Args:
            file_or_dir (str): path to the checkpoint directory or
                a full checkpoint path.
            weights_name (str): weights name. Defaults to "weights".
            best (bool): whether to restore the best checkpoint.
                Defaults to False.

        Returns:
            str: checkpoint path (absolute path).

        Raises:
            RuntimeError: couldn't find a valid checkpoint path.
        """

        checkpoint_path = None
        file_or_dir = os.path.abspath(file_or_dir)

        if os.path.isdir(file_or_dir):
            directory = file_or_dir
            # resotre best checkpoint or latest checkpoint
            # if checkpoint state or best checkpoint state
            # files exist, get checkpoint path from files.
            # Eg. directory = model/
            if best:
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
                raise RuntimeError(
                    "Couldn't find the checkpoint file for "
                    f"checkpoint: {file_or_dir}"
                )
        # Get config file path
        config_path = checkpoint_path + CONFIG_SUFFIX
        # check if config path exists
        if not file_io.file_exists(config_path):
            raise RuntimeError(
                "Couldn't find the config file for "
                f"checkpoint: {checkpoint_path}"
            )
        return checkpoint_path

    @classmethod
    def load(
        cls,
        file_or_dir: str,
        weights_name: str = WEIGHTS_NAME,
        best: bool = False
    ):
        """Load model from checkpoint. See SavableModel._preload

        Args:
            file_or_dir (str): path to the checkpoint directory or
                a full checkpoint path.
            weights_name (str): weights name. Defaults to "weights".
            best (bool): whether to restore the best checkpoint.
                Defaults to False.

        Returns:
            SavableModel: loaded model.
        """
        # find checkpoint path
        checkpoint_path = cls._preload(file_or_dir, weights_name, best)
        LOG.debug(f"Restore weights from: {checkpoint_path}")
        # Get config path
        config_path = checkpoint_path + CONFIG_SUFFIX
        # Reconstruct model from config
        self = cls.load_config(config_path)
        # restore weights
        tf.train.Checkpoint(model=self).restore(checkpoint_path)
        return self

    def update(self, other_model, polyak=1.0, all_vars=False):
        """Update model parameters from other model

        This updates model parameters with the following formula:
        ```
        param = param * (1.0 - polyak) + other_param * polyak
        ```

        Args:
            other_model (tf.keras.Model): other model.
            polyak (float, optional): update ratio. Defaults to 1.0.
            all_vars (bool, optional): update all variables. If False,
                only updates trainable variables. Defaults to False.
        """
        if not all_vars:
            var = self.trainable_variables
            target_var = other_model.trainable_variables
        else:
            var = self.variables
            target_var = other_model.variables
        ub_utils.soft_update(var, target_var, polyak=polyak)


class TrainableModel(SavableModel):
    """Provides a common interface for training, evaluating and
    inferencing the model.
    """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        # Initialize state object
        #  store training states
        s = ub_utils.StateObject()
        s.num_timesteps = 0
        s.num_epochs = 0
        s.num_subepochs = 0
        s.num_gradsteps = 0
        s.progress = 0

        self._state = s

    def reset(self):
        """Reset training states"""
        self.num_timesteps = 0
        self.num_epochs = 0
        self.num_subepochs = 0
        self.num_gradsteps = 0
        self.progress = 0

    @abc.abstractmethod
    def predict(self):
        """Make predictions for the input samples"""
        raise NotImplementedError

    @abc.abstractmethod
    def train(self):
        """Train the model for one epoch. The entry point of a complete
        training procedure is defined in TrainableModel.learn
        """
        raise NotImplementedError

    def eval(self):
        """Evaluate the model"""
        raise NotImplementedError

    @abc.abstractmethod
    def learn(self):
        """Run the complete training procedure to train the model. It should
        return it self after the training done.

        Returns:
            TrainableModel: self
        """
        raise NotImplementedError

    def metrics_compare_fn(self, last_metrics, new_metrics):
        """Compare two metrics. This method is used to perform
        model selection when the user saved a new checkpoint with
        its metrics.

        Args:
            last_metrics: metrics of current the best checkpoint.
            new_metrics: metrics of the new checkpoint.

        Returns:
            True if `new_metrics` is better than or equal to `last_metrics`
        """
        return _default_metric_compare_fn(last_metrics, new_metrics)

    def save(
        self,
        directory: str,
        weights_name: str = WEIGHTS_NAME,
        checkpoint_number: int = None,
        checkpoint_metrics=None,
        metrics_compare_fn=None,
        max_to_keep=None,
        **kwargs
    ):
        """Save model checkpoint. See `SavableModel.save()` for details.

        Args:
            directory (str): root path to save weights and config
            weights_name (str): filename prefix. Defaults to "weights".
            checkpoint_number (int): checkpoint number. Defaults to None.
            checkpoint_metrics (Any, optional): model metrics used to determine
                the best model. Defaults to None.
            metrics_compare_fn (Callable, optional): a callback taking two
                arguments (last_metrics, new_metrics) defines the comparison
                method of two metrics. Return True if `new_metrics` is better
                than or equal to `last_metrics`. Defaults to None.
            max_to_keep (int, optional): maximum number of checkpoints to
                keep. Defaults to None.

        Returns:
            str: path to the checkpoint where it is saved
        """
        if metrics_compare_fn is None:
            metrics_compare_fn = self.metrics_compare_fn
        return super().save(
            directory=directory,
            weights_name=weights_name,
            checkpoint_number=checkpoint_number,
            checkpoint_metrics=checkpoint_metrics,
            metrics_compare_fn=metrics_compare_fn,
            max_to_keep=max_to_keep,
            **kwargs
        )

    def save_config(self, filepath):
        """Save the model configurations to `filepath`

        Args:
            filepath (str): path to save configuration
        """
        _config = self.get_config()

        config = {'state': self.state, 'config': _config}

        ub_utils.safe_json_dump(filepath, config)

    @classmethod
    def load_config(cls, filepath):
        """Load model configurations and reconstruct a new model

        Args:
            filepath (str): path to load config file

        Returns:
            TrainableModel: loaded model.
        """
        config = ub_utils.safe_json_load(filepath)

        _config = config.get('config', {})
        state = config.get('state', {})

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


# === RL Model ===


class BaseAgent(SavableModel):
    """A general template for model-free RL agent

    Attributes:
        support_obs_spaces (list): the support list of observation space types.
        support_act_spaces (list): the support list of action space types.
    """
    support_obs_spaces = []
    support_act_spaces = []

    def __init__(self, observation_space, action_space, **kwargs):
        """Create an agent and check if the input observation space and action
        space is supported by this class of agent, if they are provided.

        Args:
            observation_space (gym.Space): observation space of the environment.
            action_space (gym.Space): action space of the environment.
        """
        super().__init__(**kwargs)

        self.observation_space = None
        self.action_space = None

        if observation_space is not None and action_space is not None:
            self.set_spaces(observation_space, action_space)

    def set_spaces(self, observation_space, action_space):
        """Check spaces types before setting them to the agent

        Args:
            observation_space (gym.Space): observation space of the environment.
            action_space (gym.Space): action space of the environment.
        """
        # check observation/action spaces
        if self.observation_space is not None:
            if observation_space != self.observation_space:
                raise RuntimeError(
                    'Observation space does not match, expected '
                    f'{self.observation_space}, got {observation_space}'
                )
        if self.action_space is not None:
            if action_space != self.action_space:
                raise RuntimeError(
                    'Action space does not match, expected '
                    f'{self.action_space}, got {action_space}'
                )
        # check if observation/action spaces supportted
        if not isinstance(observation_space, tuple(self.support_obs_spaces)):
            raise RuntimeError(
                f'{type(self).__name__} does not support the '
                f'observation spaces of type {type(observation_space)}'
            )
        if not isinstance(action_space, tuple(self.support_act_spaces)):
            raise RuntimeError(
                f'{type(self).__name__} does not support the '
                f'actions spaces of type {type(action_space)}'
            )

        self.observation_space = observation_space
        self.action_space = action_space

    @abc.abstractmethod
    def setup(self):
        """Setup agent, defines the model setup procedure."""
        raise NotImplementedError

    def proc_observation(self, obs):
        """Preprocess observations before forwarding into the nets
        e.g. normalizing image observations. A nested space is also
        supported, e.g. `gym.spaces.Tuple`, `gym.spaces.Dict`.

        Note that in default, this normalizes the observation only if
        they are not in tf.float32 or tf.float64. For example, tf.uint8
        is normalized if its observation space has a limited low/high bounds.
        And all of them are cast to tf.float32.

        See `ub.utils.preprocess_observation` for more info.

        Args:
            obs (Any): a sample or a batch samples of observations.

        Returns:
            Any: normalized observations.
        """
        # sortkey to ensure key order. In default, gym sorts spaces by keys
        return ub_utils.map_nested_tuple(
            (obs, self.observation_space),
            lambda obs_and_space: ub_utils.preprocess_observation(*obs_and_space),
            sortkey=True
        )

    def proc_action(self, act):
        """Postprocess actions predicted from policy, e.g. denormalizing or
        clipping continuous actions.

        Args:
            act (Any): predicted actions.
        """
        if isinstance(self.action_space, gym.spaces.Box):
            # TODO: can this be done in one line for both tensor and
            # non-tensor types?
            if tf.is_tensor(act):
                act = tf.clip_by_value(
                    act, self.action_space.low, self.action_space.high
                )
            else:
                act = np.clip(act, self.action_space.low, self.action_space.high)
        return act

    @abc.abstractmethod
    def call(self, inputs, training=True):
        """Forward the agent, this method is called from `__call__`"""
        raise NotImplementedError

    def predict(self, inputs, *args, **kwargs):
        """Make action predictions from one input sample or one batch samples

        Args:
            inputs (np.ndarray): batch observations in shape (b, *obs.shape)
                or (*obs.shape) for one input.

        Returns:
            np.ndarray: predicted actions in shape (b, *act.shape) or
                (*act.shape) for one input.
        """
        one_sample = (len(inputs.shape) == len(self.observation_space.shape))
        if one_sample:
            inputs = np.expand_dims(inputs, axis=0)
        # predict actions
        outputs, *_ = self(inputs, *args, training=False, **kwargs)
        outputs = np.asarray(outputs)
        if one_sample:
            outputs = np.squeeze(outputs, axis=0)
        return outputs

    def get_config(self):
        """Base agent configurations. Don't forget to include these
        base configurations to your override configurations.

        Example usage:

        You can add configurations by calling the super method
        ```python
        class MyAgent(ub.base.BaseAgent):
            def get_config(self):
                config = super().get_config()
                config.update({
                    # MY CONFIGS
                })
        ```
        or just add these configs to your configs
        ```python
        class MyAgent(ub.base.BaseAgent):
            def get_config(self):
                return {
                    # MY CONFIGS,
                    'observation_space': self.observation_space,
                    'action_space: self.action_space
                }
        ```
        """
        return {
            'observation_space': self.observation_space,
            'action_space': self.action_space
        }

    @classmethod
    def from_config(cls, config):
        """Restore model from the given configurations

        Args:
            config (dict): configurations

        Returns:
            self
        """

        # construct method
        self = cls(**config)
        # setup model
        self.setup()

        return self

    def save_config(self, filepath):
        """Save model config to `filepath`

        Args:
            filepath (str): path to save configuration
        """
        config = self.get_config()
        ub_utils.safe_json_dump(filepath, config)


class BaseRLModel(TrainableModel):
    """Provides a common interface for either
    off-policy or on-policy RL algos.

    The main differences between off/on policy are:
    - On-policy has no replay buffer warming up stage while off-policy has.
        So `is_wraming_up` method and `warmup_steps` are diabled in default
        for on-policy algorithms, but one can still access or override them.
    - On-policy may need to precompute rollout returns, this can
        be implemented in `run()`.
    - Some on-policy algorithms may need to iterate through all rollout
        samples for many times, this can be implemented by setting
        `n_subepochs`

    The training loop has the following structure:
    1. The training loop, invoked by calling `learn`, trains the model for
        several epochs.
    2. In each epoch, the model calls `run` to collects `n_steps` times the
        number of vectorized environments of samples, and saves them to the
        replay buffer. The sample collection method is defined in `_collect_step`.
    3. Run `n_subepochs` times of subepochs in each epoch.
    4. Each subepoch runs `n_gradsteps` times of gradient updates, each with one
        minibatch sampled from the replay buffer. The `batch_size` defines the
        size of the minibatch. One gradient update is defined in `_train_step`.

    Grouping methods by its behaviour:
    - Initialization: `__init__`, `setup`, `set_env`
    - Inferencing: `call`, `predict`
    - Collecting: `run`, `collect`, `_collect_step`
    - Evaluation: `eval`, `get_eval_metrics`
    - Training: `learn`, `train`, `_train_step`, `_update_target`
    - Logging: `log_train`, `log_eval`, `log_save`
    - Save/Load: `save`, `load`, `get_save_metrics`

    Attributes:
        support_obs_spaces (list): the support list of observation space types.
        support_act_spaces (list): the support list of action space types.
    """
    support_obs_spaces = []
    support_act_spaces = []

    def __init__(
        self,
        env: ub_vec.BaseVecEnv,
        n_steps: int,
        n_subepochs: int,
        n_gradsteps: int,
        warmup_steps: int,
        batch_size: int,
        observation_space=None,
        action_space=None,
        **kwargs
    ):
        """Initialize training procedure

        `env` can be None for delayed setup. In this case, you should call
        `set_env` then call `setup` to manually setup the model.

        Args:
            env (ub.envs.BaseVecEnv): vectorized environments.
            n_steps (int): number of steps collected in each epoch.
            n_subepochs (int): number of subepochs in each epoch.
            n_gradsteps (int): number of steps of gradient updates in
                each subepochs.
            warmup_steps (int): number of steps before starting the training loop
            batch_size (int): size of minibatch
            observation_space (gym.Space, optional): observation space. This
                argument is used when constructing from the checkpoint. So it
                can be omitted. Defaults to None.
            action_space (gym.Space, optional): action space. This argument is
                used when constructing from the checkpoint. So it can be
                omitted. Defaults to None.
        """
        super().__init__(**kwargs)

        self.n_steps = n_steps
        self.n_subepochs = n_subepochs
        self.n_gradsteps = n_gradsteps
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        # initialize states
        self.env = None
        self.n_envs = 0
        self.tb_writer = None    # tensorboard summary writer

        self.observation_space = observation_space
        self.action_space = action_space
        # create default logger
        if not hasattr(self, 'LOG') or self.LOG is None:
            self.LOG = logger.getLogger(f'{type(self).__name__}')
        if env is not None:
            self.set_env(env)

    def set_env(self, env):
        """Set the training environments.

        If the environment is already set, you can call this function
        to change the environment. But the observation space and action
        space must be consistent with the original one.

        Args:
            env (ub.envs.BaseVecEnv): training environment.
        """
        if not isinstance(env, ub_vec.BaseVecEnv):
            raise RuntimeError('Envrionement must be a vectorized env')
        # check observation/action spaces
        if self.observation_space is not None:
            if env.observation_space != self.observation_space:
                raise RuntimeError(
                    'Observation space does not match, expected '
                    f'{self.observation_space}, got {env.observation_space}'
                )
        if self.action_space is not None:
            if env.action_space != self.action_space:
                raise RuntimeError(
                    'Action space does not match, expected '
                    f'{self.action_space}, got {env.action_space}'
                )
        # check if observation/action spaces supportted
        if not isinstance(env.observation_space, tuple(self.support_obs_spaces)):
            raise RuntimeError(
                'This algorithm does not support observation '
                f'spaces of type {type(env.observation_space)}'
            )
        if not isinstance(env.action_space, tuple(self.support_act_spaces)):
            raise RuntimeError(
                'This algorithm does not support action space '
                f'of type {type(env.action_space)}'
            )

        self.env = env
        self.n_envs = env.n_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    @abc.abstractmethod
    def setup(self):
        """Defines the procedure to setup this model for training"""
        raise NotImplementedError

    @abc.abstractmethod
    def call(self):
        """Forward model"""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self):
        """Make action predictions for one sample or one batch of samples"""
        raise NotImplementedError

    @abc.abstractmethod
    def _collect_step(self, obs):
        """Collect one step rollout from the environment.

        Args:
            obs (np.ndarray): current observations.

        Returns:
            np.ndarray: next observations
        """
        raise NotImplementedError

    def collect(self, steps, obs=None):
        """Collect `steps` of rollouts. Reset the environments if
        `obs` is not given.

        Args:
            steps (int): number of steps to collect
            obs (np.ndarray, optional): current observations. Defaults to None.
                set to None to reset the envs.

        Returns:
            np.ndarray: next observations
        """
        if obs is None:
            obs = self.env.reset()
        for _ in range(steps):
            obs = self._collect_step(obs)
            # update state
            self.num_timesteps += self.n_envs
        return obs

    def run(self, steps, obs=None):
        """Run the rollout collection procedures can do some
        buffer preprocessing, e.g. compute GAE here.

        Args:
            steps (int): number of steps to collect
            obs (np.ndarray, optional): current observations. Defaults to None.

        Returns:
            np.ndarray: current observations
        """
        return self.collect(steps, obs=obs)

    def get_eval_metrics(self, results):
        """Compute evaluation metrics from evaluation results

        Args:
            results(dict): evaluation results

        Returns:
            dict: evaluation metrics
        """
        if not results:
            return None

        results = ub_utils.flatten_dicts(results)

        rews = results['rewards']
        steps = results['steps']

        metrics = {
            'mean-reward': np.mean(rews).item(),
            'std-reward': np.std(rews).item(),
            'max-reward': np.max(rews).item(),
            'min-reward': np.min(rews).item(),
            'mean-steps': np.mean(steps).item(),
            'std-steps': np.std(steps).item(),
            'max-steps': np.max(steps).item(),
            'min-steps': np.min(steps).item()
        }
        return metrics

    def get_save_metrics(self, metrics):
        """Compute metrics for model selection when saving the model checkpoint.
        Default is the mean of episode rewards.

        Args:
            metrics(dict): evaluation metrics, usually generated by calling
                `self.get_eval_metrics`.

        Returns:
            Any: metrics for selecting the best model.
        """
        if (not isinstance(metrics, dict) or 'mean-reward' not in metrics):
            return None
        return metrics['mean-reward']

    def eval(self, env, n_episodes, max_steps, render=False):
        """Evaluate model for `n_episodes` episodes and limits the maximum steps
        to `max_steps`

        Args:
            env (gym.Env): a gym environment for evaluation.
            n_episodes (int): number of episodes to evaluate.
            max_steps (int): the maximum number of steps. Set to -1 to run forever
                until the environment done.
            render (bool, optional): whether to render the environment screen.
                Defaults to False.

        Returns:
            list: a list of dict containing the evaluation results per episode.
        """
        if n_episodes <= 0:
            return None

        eps_info = []
        for episode in range(n_episodes):
            obs = env.reset()
            total_rews = 0
            total_steps = 0
            # eval one episode
            while total_steps != max_steps:
                # predict action
                acts = self.predict(obs)
                # step environment
                obs, rew, done, info = env.step(acts)
                total_rews += rew
                total_steps += 1
                if render:
                    env.render('human')
                if done:
                    break
            # append eval results
            eps_info.append({'rewards': total_rews, 'steps': total_steps})
        return eps_info

    @abc.abstractmethod
    def _train_step(self, batch_size):
        """Train for one step (one gradient update).

        Args:
            batch_size(int): mini-batch size.

        Returns:
            dict: loss dict
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _update_target(self):
        """Update target networks, this is called for every `target_update`
        (defined in `learn`) steps of gradient updates.
        """
        raise NotImplementedError

    def train(self, batch_size, subepochs, gradsteps, target_update):
        """Train one epoch

        Args:
            batch_size (int): size of minibatch.
            subepochs (int): number of subepochs
            gradsteps (int): Number of steps of gradient updates per subepoch
            target_update (int): target networks update frequency.

        Returns:
            dict: loss dict
        """
        all_losses = []
        for subepoch in range(subepochs):
            for gradstep in range(gradsteps):
                # train one step
                losses = self._train_step(batch_size)
                all_losses.append(losses)
                # update state
                self.num_gradsteps += 1
                # update target networks
                if self.num_gradsteps % target_update == 0:
                    self._update_target()
            # update states
            self.num_subepochs += 1
        # aggregate losses
        all_losses = ub_utils.flatten_dicts(all_losses)
        m_losses = {}
        for name, losses in all_losses.items():
            m_losses[name] = np.mean(np.hstack(np.asarray(losses)))
        return m_losses

    def log_train(
        self,
        total_timesteps: int,
        total_epochs: int,
        time_start: float,
        last_time_spent: float,
        losses: dict,
        verbose: int = 2
    ):
        """Print training log

        Verbosity:
        0 = silent
        1 = silent
        2 = print number of epochs, timesteps, progress... etc
        3 = print full training messages, includes due time, losses...

        Args:
            total_timesteps (int): total training timesteps
            total_epochs (int): total training epochs
            time_start (float): timestamp when the learning started (sec)
            last_time_spent (float): time spent from the learning started
            losses (dict): loss dictionary
            verbose (int, optional): verbosity. Defaults to 2.
        """
        # current time
        time_now = time.time()
        # total time spent
        time_spent = (time_now - time_start)
        # execution time (one epoch)
        execution_time = time_spent - last_time_spent
        # remaining time
        remaining_time = (time_spent / self.progress) * (1.0 - self.progress)
        # eta
        eta = datetime.datetime.now() + datetime.timedelta(seconds=remaining_time)
        # average steps per second
        fps = float(self.num_timesteps) / time_spent

        if verbose > 1:
            self.LOG.set_header(f'Epoch {self.num_epochs}/{total_epochs}')
            self.LOG.add_line()
            self.LOG.add_row(f'Timesteps: {self.num_timesteps}/{total_timesteps}')
            self.LOG.add_row(f'Steps/sec: {fps:.2f}')
            self.LOG.add_row(f'Progress: {self.progress*100.0:.2f}%')
            if verbose > 2:
                self.LOG.add_row(
                    'Execution time', datetime.timedelta(seconds=execution_time)
                )
                self.LOG.add_row(
                    'Elapsed time', datetime.timedelta(seconds=time_spent)
                )
                self.LOG.add_row(
                    'Remaining time', datetime.timedelta(seconds=remaining_time)
                )
                self.LOG.add_row('ETA', eta.strftime('%Y-%m-%d %H:%M:%S'))
                self.LOG.add_line()

                if self.is_warming_up():
                    self.LOG.add_row('Collecting rollouts')
                else:
                    name = '\n'.join(map(str, losses.keys()))
                    loss = '\n'.join(map('{:.6f}'.format, losses.values()))
                    self.LOG.add_rows(
                        fmt='{name} {||} {loss}', name=name, loss=loss
                    )

            self.LOG.add_line()
            self.LOG.flush('INFO')

    def log_eval(self, results, metrics, verbose=2):
        """Print evaluation log

        Verbosity:
        0 = silent
        1 = silent
        2 = print evaluation metrics
        3 = print episode results and metrics

        Args:
            results (list): evaluaition results, returned from `eval`
            metrics (dict): evaluation metrics, returned from `get_eval_metrics`
            verbose (int, optional): verbosity. Defaults to None.
        """
        if verbose > 1:
            if verbose > 2 and results is not None:
                self.LOG.set_header('Evaluation results')
                n_episodes = len(results)
                for ep in range(n_episodes):
                    self.LOG.subgroup(f'Episode {ep+1}')
                    labels = '\n'.join(results[ep].keys())
                    values = '\n'.join(map('{}'.format, results[ep].values()))

                    self.LOG.add_rows(
                        fmt='{labels} {||} {values}', labels=labels, values=values
                    )
                self.LOG.flush('INFO')

            if metrics is not None:
                self.LOG.set_header('Evaluation metrics')
                labels = '\n'.join(metrics.keys())
                values = '\n'.join(map('{:.3f}'.format, metrics.values()))
                self.LOG.add_rows(
                    fmt='{labels} {||} {values}', labels=labels, values=values
                )
                self.LOG.add_line()
                self.LOG.flush('INFO')

    def log_save(self, save_path, saved_path, verbose=2):
        """Print model checkpoint messages

        Verbosity:
        0 = silent
        1 = silent
        2 = print message
        3 = print message

        Args:
            save_path (str): root path to save the checkpoint
            saved_path (str): full path where the checkpoint actually saved
            verbose (int, optional): verbosity. Defaults to None.
        """
        if verbose > 1:
            # find the best model path
            best_path = self._preload(save_path, best=True)
            if best_path == os.path.abspath(saved_path):
                self.LOG.info(f'[Best] Checkpoint saved to: {saved_path}')
            else:
                self.LOG.info(f'Checkpoint saved to: {saved_path}')

    def is_warming_up(self):
        """Determine whether it is in the warmming up stage"""
        return self.num_timesteps < self.warmup_steps

    def learn(
        self,
        total_timesteps: int,
        log_interval: int = 100,
        eval_env: gym.Env = None,
        eval_interval: int = 100,
        eval_episodes: int = 5,
        eval_max_steps: int = 1000,
        save_interval: int = 100,
        save_path: str = None,
        target_update: int = 2,
        tb_logdir: str = None,
        reset_timesteps: bool = False,
        verbose: int = 1
    ):
        """Start the training procedure

        The total number of epochs is calculated by:
        ```
        timesteps_per_epoch = n_steps * n_envs
        total_epochs = total_timesteps / timesteps_per_epoch
        ```

        Args:
            total_timesteps (int): Total number of timesteps to train the model.
            log_interval (int, optional): log frequency (unit: epoch)
                Defaults to 100.
            eval_env (gym.Env, optional): gym environment for evaluation.
                Defaults to None.
            eval_interval (int, optional): evaluation frequency (unit: epoch).
                Defaults to 100.
            eval_episodes (int, optional): evaluation episodes. Defaults to 5.
            eval_max_steps (int, optional): maximum number of steps of each
                episode when evaluating. Defaults to 1000.
            save_interval (int, optional): checkpoint frequency (unit: epoch).
                Defaults to 100.
            save_path (str, optional): root path or directory to save checkpoints.
                Defaults to None.
            target_update (int, optional): frequency of updating the target model.
                (unit: gradstep) Defaults to 2.
            tb_logdir (str, optional): path to store tensorboard files. None to
                disable logging. Defaults to None.
            reset_timesteps (bool, optional): reset the training states.
                Defaults to False.
            verbose (int, optional): more logging. 0=silent, 1=progress bar,
                2=short, 3=full. Defaults to 1.

        Raises:
            RuntimeError: if the training `env` is not set.

        Returns:
            self
        """
        if self.env is None:
            raise RuntimeError('Env not set, call `set_env` before training')

        # create tensorboard writer
        if tb_logdir is not None:
            self.tb_writer = tf.summary.create_file_writer(tb_logdir)

        # initialize state
        if reset_timesteps:
            self.reset()

        # initialze
        obs = None
        time_start = time.time()
        time_spent = 0
        timesteps_per_epoch = self.n_steps * self.n_envs
        total_epochs = math.ceil(
            float(total_timesteps - self.num_timesteps)
            / float(timesteps_per_epoch)
        ) + self.num_epochs
        progress_bar = tqdm.tqdm(total=total_epochs, disable=(verbose != 1))
        progress_bar.update(self.num_epochs)

        while self.num_timesteps < total_timesteps:
            # collect rollouts
            obs = self.run(steps=self.n_steps, obs=obs)

            # update state
            self.num_epochs += 1
            self.progress = float(self.num_timesteps) / float(total_timesteps)
            progress_bar.update()
            # train one epoch
            losses = {}
            if not self.is_warming_up():
                # training
                losses = self.train(
                    batch_size=self.batch_size,
                    subepochs=self.n_subepochs,
                    gradsteps=self.n_gradsteps,
                    target_update=target_update
                )
                # write tensorboard
                if self.tb_writer is not None:
                    with self.tb_writer.as_default():
                        for name, value in losses.items():
                            tf.summary.scalar(
                                f'train/{name}', value, step=self.num_timesteps
                            )
                    self.tb_writer.flush()
            # print training log
            if (log_interval is not None) and (self.num_epochs % log_interval == 0):
                # print log
                self.log_train(
                    total_timesteps,
                    total_epochs,
                    time_start,
                    time_spent,
                    losses,
                    verbose
                )
                # update time_spent
                time_spent = time.time() - time_start

            # evaluate model
            eval_metrics = None
            if (eval_env is not None) and (self.num_epochs % eval_interval == 0):
                eval_results = self.eval(
                    env=eval_env,
                    n_episodes=eval_episodes,
                    max_steps=eval_max_steps
                )

                eval_metrics = self.get_eval_metrics(eval_results)

                if self.tb_writer is not None:
                    with self.tb_writer.as_default():
                        for name, value in eval_metrics.items():
                            tf.summary.scalar(
                                f'eval/{name}', value, step=self.num_timesteps
                            )
                    self.tb_writer.flush()

                self.log_eval(eval_results, eval_metrics, verbose)

            # save model
            if ((save_path is not None) and (save_interval is not None)
                    and (self.num_epochs % save_interval) == 0):
                # compute metrics
                checkpoint_metrics = self.get_save_metrics(eval_metrics)
                saved_path = self.save(
                    save_path,
                    checkpoint_number=self.num_epochs,
                    checkpoint_metrics=checkpoint_metrics
                )
                self.log_save(save_path, saved_path, verbose)
        progress_bar.close()
        return self

    def get_config(self):
        """Get constructor configurations."""
        config = super().get_config()
        config.update({
            'n_steps': self.n_steps,
            'n_subepochs': self.n_subepochs,
            'n_gradsteps': self.n_gradsteps,
            'batch_size': self.batch_size,
            'observation_space': self.observation_space,
            'action_space': self.action_space
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Restore model from the given configurations

        Args:
            config(dict): configurations
        """

        # construct method
        self = cls(env=None, **config)
        # setup model
        self.setup()

        return self

    def save_config(self, filepath):
        """Save model config to `filepath`

        Args:
            filepath(str): path to save configuration
        """
        _config = self.get_config()
        config = {'state': self.state, 'config': _config}

        ub_utils.safe_json_dump(filepath, config)


class OffPolicyModel(BaseRLModel):
    """Defines the default arguments for off-policy models"""
    def __init__(
        self,
        env,
        n_steps: int = 4,
        n_subepochs: int = 1,
        n_gradsteps: int = 1,
        warmup_steps: int = 10000,
        batch_size: int = 128,
        observation_space=None,
        action_space=None,
        **kwargs
    ):
        super().__init__(
            env=env,
            n_steps=n_steps,
            n_subepochs=n_subepochs,
            n_gradsteps=n_gradsteps,
            warmup_steps=warmup_steps,
            batch_size=batch_size,
            observation_space=observation_space,
            action_space=action_space,
            **kwargs
        )


class OnPolicyModel(BaseRLModel):
    """Defines the default arguments for on-policy models.
    Note that in default the target model updates, replay
    buffer warming up stage and `n_gradsteps` is disabled
    for on-policy models. But one can still access them
    by overriding them.
    """
    def __init__(
        self,
        env,
        n_steps: int = 256,
        n_subepochs: int = 4,
        n_gradsteps: int = None,
        warmup_steps: int = None,
        batch_size: int = 128,
        observation_space=None,
        action_space=None,
        **kwargs
    ):
        super().__init__(
            env=env,
            n_steps=n_steps,
            n_subepochs=n_subepochs,
            n_gradsteps=n_gradsteps,
            warmup_steps=warmup_steps,
            batch_size=batch_size,
            observation_space=observation_space,
            action_space=action_space,
            **kwargs
        )

    def _update_target(self):
        """Do nothing"""
        pass

    def is_warming_up(self):
        """On policy algo has no wram up stage"""
        return False
