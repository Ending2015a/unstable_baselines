# --- built in ---
import os
import sys
import time
import json
import logging
import tempfile
import unittest

# --- 3rd party ---
import numpy as np
import tensorflow as tf

# --- my module ---
from unstable_baselines import base
from unstable_baselines import utils


class A():
    def __init__(self, a, b):
        self.a = a
        self.b = b

class SimpleSavableModel(base.SavableModel):
    def __init__(self, delayed_param: float,
                       hparam1: int=1,
                       hparam2: float=2.4,
                       hparam3: str='hello',
                       hparam4: utils.StateObject=None,
                       hparam5: A=None):
        super().__init__()
        
        self.delayed_param = None # 10.0
        self.hparam1 = hparam1 # 1
        self.hparam2 = hparam2 # 2.4
        self.hparam3 = hparam3 # 'hello'
        self.hparam4 = hparam4 # StateObject(a=10, b=100, c=np.array([1,2,3]))
        self.hparam5 = hparam5 # A(3, 'abc')
    
        if delayed_param is not None:
            self.setup_model(delayed_param)

    def setup_model(self, delayed_param):
        self.delayed_param = delayed_param

        vparam1 = np.array([self.hparam1, self.hparam1], dtype=np.uint8)
        self.vparam1 = tf.Variable(vparam1, trainable=True, 
                                  name='vparam1')
        self.vparam2 = tf.Variable([self.delayed_param], trainable=False,
                                    name='vparam2', dtype=tf.float32)

    def get_config(self):
        init_config = {
            'hparam1': self.hparam1,
            'hparam2': self.hparam2,
            'hparam3': self.hparam3,
            'hparam4': self.hparam4,
            'hparam5': self.hparam5
        }

        delayed_config = {
            'delayed_param': self.delayed_param
        }

        config = {
            'init_config': init_config,
            'delayed_config': delayed_config
        }

        return config

    @classmethod
    def from_config(cls, config):
        init_config = config.get('init_config', {})
        delayed_config = config.get('delayed_config', {})

        self = cls(None, **init_config)
        self.setup_model(**delayed_config)

        return self


class TestBaseModule(unittest.TestCase):
    '''Test unstable_baselines.base module

    Class list:
    [ ] CheckpointManager
        [ ] best_checkpoint
        [ ] best_checkpoint_metrics
        [ ] _record_checkpoint_state
        [ ] _sweep_checkpoints
        [ ] save
    [ ] SavableModel
        [x] get_config
        [x] from_config
        [x] save_config
        [x] load_config
        [x] save
        [x] _preload
        [x] load
        [ ] update
    [ ] TrainableModel
        [ ] eval
        [ ] save_config
        [ ] load_config

    Function list:
    [ ] _get_best_checkpoint_filename
    [ ] _generate_best_checkpoint_state
    [ ] _update_best_checkpoint_state
    [ ] _get_best_checkpoint_state
    [ ] _prefix_to_checkpoint_path
    [ ] _check_checkpoint_exists
    [ ] _get_best_checkpoint
    [ ] _delete_file_if_exists
    [x] _get_checkpoint_manager
    [x] _get_latest_checkpoint_number
    [ ] _default_metric_compare_fn
    '''
    def test_savablemodel(self):
        s = utils.StateObject(a=10, b=100, c=np.array([1,2,3]))
        a = A(3, 'abc')
        model = SimpleSavableModel(10, hparam3='wow', hparam4=s, hparam5=a)

        def check_save(save_path, weights_name, checkpoint_number, expected_number,
                    checkpoint_metrics=None, metrics_compare_fn=None):
            saved_path = model.save(save_path,
                            weights_name=weights_name,
                            checkpoint_number=checkpoint_number,
                            checkpoint_metrics=checkpoint_metrics,
                            max_to_keep=5)

            expected_saved_path = os.path.join(save_path,
                                    '{}-{}'.format(weights_name, expected_number))

            self.assertEqual(saved_path, expected_saved_path)


            # check saved file exist
            filenames = [
                os.path.join(save_path, 'checkpoint'),
                os.path.join(save_path, 'best_checkpoint'),
                saved_path + '.index',
                saved_path + base.CONFIG_SUFFIX,
                saved_path + '.data-00000-of-00001'
            ]

            for fname in filenames:
                try:
                    self.assertTrue(os.path.isfile(fname))
                except:
                    print('{} got False'.format(fname))
                    raise

            # check latest checkpoint
            latest_checkpoint = tf.train.latest_checkpoint(save_path)
            self.assertEqual(latest_checkpoint, expected_saved_path)

            # test _get_latest_checkpoint_number
            latest_number = base._get_latest_checkpoint_number(latest_checkpoint)
            self.assertEqual(latest_number, expected_number)

        def check_preload(save_path, weights_name, expected_number,
                          best=False):
            
            checkpoint_path = SimpleSavableModel._preload(save_path, weights_name, best=best)
            basename = os.path.basename(checkpoint_path)
            if '-' in weights_name:
                exp_weights_name = '{}-{}'.format(weights_name.split('-')[0],
                                            expected_number)
            else:
                exp_weights_name = '{}-{}'.format(weights_name, expected_number)

            self.assertEqual(basename, exp_weights_name)

        def check_load(save_path, weights_name, expected_number,
                        best=False):

            checkpoint_path = SimpleSavableModel._preload(save_path, weights_name, best=best)
            
            basename = os.path.basename(checkpoint_path)
            if '-' in weights_name:
                exp_weights_name = '{}-{}'.format(weights_name.split('-')[0],
                                            expected_number)
            else:
                exp_weights_name = '{}-{}'.format(weights_name, expected_number)
            
            self.assertEqual(basename, exp_weights_name)
            
            new_model = SimpleSavableModel.load(save_path, weights_name, best=best)

            return new_model

        # test save/load
        with tempfile.TemporaryDirectory() as tempdir:
            save_path = tempdir

            # save 
            check_save(save_path, weights_name='simple_weights',
                    checkpoint_number=None,
                    expected_number=1)
            check_save(save_path, weights_name='simple_weights',
                    checkpoint_number=None,
                    expected_number=2)
            check_save(save_path, weights_name='simple_weights',
                    checkpoint_number=None,
                    expected_number=3,
                    checkpoint_metrics=0.8) # best
            check_save(save_path, weights_name='simple_weights',
                    checkpoint_number=10,
                    expected_number=10)
            check_save(save_path, weights_name='simple_weights',
                    checkpoint_number=None,
                    expected_number=11) # latest
            # load 
            check_preload(save_path, weights_name='simple_weights',
                    expected_number=11) # load latest
            check_preload(save_path, weights_name='simple_weights',
                    expected_number=3, best=True) # load best
            # save 
            check_save(save_path, weights_name='simple_weights',
                    checkpoint_number=None,
                    expected_number=12,
                    checkpoint_metrics=0.9) # new best
            # load
            check_preload(save_path, weights_name='simple_weights',
                    expected_number=12, best=True) # load best
            # save
            check_save(save_path, weights_name='simple_weights',
                    checkpoint_number=None,
                    expected_number=13,
                    checkpoint_metrics='f') # metrics not comparable (false)
            # load
            check_preload(save_path, weights_name='simple_weights',
                    expected_number=12, best=True) # load best
            
            os.remove(os.path.join(save_path, 'checkpoint'))
            os.remove(os.path.join(save_path, 'best_checkpoint'))
            check_preload(save_path, weights_name='simple_weights-3',
                    expected_number=3)
            check_preload(save_path, weights_name='simple_weights-10',
                    expected_number=10)
            check_preload(save_path, weights_name='simple_weights-13',
                    expected_number=13)
            check_preload(os.path.join(save_path, 'simple_weights-3'),
                    weights_name='simple_weights',
                    expected_number=3)
            check_preload(os.path.join(save_path, 'simple_weights-10'),
                    weights_name='simple_weights',
                    expected_number=10)
            check_preload(os.path.join(save_path, 'simple_weights-13'),
                    weights_name='simple_weights',
                    expected_number=13)
            
            # save
            model.vparam1.assign([2, 2])
            check_save(save_path, weights_name='simple_weights',
                    checkpoint_number=None,
                    expected_number=1,
                    checkpoint_metrics=0.95)


            model.vparam2.assign([11.0])
            check_save(save_path, weights_name='simple_weights',
                    checkpoint_number=None,
                    expected_number=2,
                    checkpoint_metrics=0.90)
            
            # load
            model2 = check_load(save_path, weights_name='simple_weights',
                    expected_number=2, best=False)
            
            for old_var, new_var in zip(model.variables, model2.variables):
                print('Compare var {} == {}'.format(old_var.name, new_var.name))
                self.assertTrue(np.array_equal(old_var.numpy(), new_var.numpy()))


            print('Compare var1 {} == {}'.format(model2.vparam1.numpy(), 
                                    np.array([2, 2], dtype=np.uint8)))
            self.assertTrue(np.array_equal(model2.vparam1.numpy(),
                                np.array([2, 2], dtype=np.uint8)))

            print('Compare var2 {} == {}'.format(model2.vparam2.numpy(),
                                    np.array([11.0], dtype=np.float32)))
            
            self.assertTrue(np.array_equal(model2.vparam2.numpy(),
                                    np.array([11.0], dtype=np.float32)))

            # load 
            model1 = check_load(save_path, weights_name='simple_weights',
                    expected_number=1, best=True)
            
            print('Compare var1 {} == {}'.format(model1.vparam1.numpy(), 
                                    np.array([2, 2], dtype=np.uint8)))
            self.assertTrue(np.array_equal(model1.vparam1.numpy(),
                                np.array([2, 2], dtype=np.uint8)))

            print('Compare var2 {} == {}'.format(model1.vparam2.numpy(),
                                    np.array([10.0], dtype=np.float32)))
            
            self.assertTrue(np.array_equal(model1.vparam2.numpy(),
                                    np.array([10.0], dtype=np.float32)))


            self.assertEqual(model2.delayed_param, 10.0)
            self.assertEqual(model2.hparam1, 1)
            self.assertEqual(model2.hparam2, 2.4)
            self.assertEqual(model2.hparam3, 'wow')
            target_hparam4 = utils.StateObject(a=10, b=100,c=np.array([1,2,3]))
            self.assertEqual(model2.hparam4.keys(), target_hparam4.keys())
            self.assertEqual(model2.hparam4['a'], target_hparam4.a)
            self.assertEqual(model2.hparam4['b'], target_hparam4.b)
            self.assertTrue(np.array_equal(model2.hparam4['c'], target_hparam4.c))
            self.assertTrue(isinstance(model2.hparam5, A))
            self.assertEqual(model2.hparam5.a, A(3, 'abc').a)
            self.assertEqual(model2.hparam5.b, A(3, 'abc').b)
            

if __name__ == '__main__':
    unittest.main()