__copyright__ = '''
 The MIT License (MIT)
 Copyright (c) 2021 Joe Hsiao
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
 OR OTHER DEALINGS IN THE SOFTWARE.
'''
__license__ = 'MIT'

# --- built in ---
import os
import sys
import time
import argparse
import functools

# --- 3rd party ---
import gym
import tensorflow as tf

# --- my module ---
import unstable_baselines as ub
from unstable_baselines import logger
from unstable_baselines.algo.s.sac import SAC

def parse_args():
    parser = argparse.ArgumentParser(description='Soft Actor-Critic')
    parser.add_argument('--root',      type=str, default='log/sac/')
    parser.add_argument('--env_id',    type=str, default='HalfCheetahBulletEnv-v0')
    parser.add_argument('--seed',      type=int, default=1)
    parser.add_argument('--eval_seed', type=int, default=0)
    parser.add_argument('--n_envs',    type=int, default=1)
    args = parser.parse_args()
    # Configurations
    a = ub.utils.StateObject()
    a.ARGS = ub.utils.StateObject()
    a.ARGS.logging             = os.path.join(args.root, 'training.log')
    a.ARGS.log_level           = 'DEBUG'
    a.ARGS.seed                = args.seed
    a.ARGS.eval_seed           = args.eval_seed
    a.ARGS.n_envs              = args.n_envs
    # Env/Monitor parameters
    a.ENV = ub.utils.StateObject()
    a.ENV.env_id               = args.env_id
    a.ENV.monitor_dir          = os.path.join(args.root, 'monitor')
    a.ENV.enable_video         = True
    # Hyper parameters
    a.MODEL = ub.utils.StateObject()
    a.MODEL.learning_rate      = 3e-4
    a.MODEL.buffer_size        = int(1e6)
    a.MODEL.gamma              = 0.99
    a.MODEL.polyak             = 0.005
    a.MODEL.clipnorm           = 0.5
    a.MODEL.reward_scale       = 1.0
    a.MODEL.weight_decay       = 1e-5
    a.MODEL.target_ent         = None
    a.MODEL.squash             = True
    a.MODEL.n_critics          = 2
    a.MODEL.n_steps            = 1
    a.MODEL.n_gradsteps        = 1
    a.MODEL.wramup_steps       = int(1e4)
    a.MODEL.batch_size         = 256
    a.MODEL.verbose            = 2
    # Training parameters
    a.LEARN = ub.utils.StateObject()
    a.LEARN.total_timesteps    = int(1e6)
    a.LEARN.log_interval       = 10
    a.LEARN.eval_interval      = int(1e5)
    a.LEARN.eval_episodes      = 5
    a.LEARN.eval_max_steps     = 1000
    a.LEARN.save_interval      = int(1e5)
    a.LEARN.save_path          = os.path.join(args.root, 'save')
    a.LEARN.target_update      = 1
    a.LEARN.tb_logdir          = args.root
    return a

def print_args(LOG, a, group):
    LOG.subgroup(group)
    label = '\n'.join(map(str, a.keys()))
    value = '\n'.join(map(str, a.values()))
    LOG.add_rows(fmt='{label:15s} {||} {value}', label=label, value=value)

def make_env(a, seed=0, rank=0, eval=False):
    import pybullet_envs
    # some params for monitering
    prefix = 'eval' if eval else f'{rank}.train'    # filename prefix
    video_kwargs = {'interval':1 if eval else None} # record every n episodes
    # make environment
    env = gym.make(a.env_id)
    env = ub.envs.Monitor(env, root_dir=a.monitor_dir, prefix=prefix,
                        enable_video=a.enable_video, video_kwargs=video_kwargs)
    env = ub.envs.wrap_mujoco(env, test_mode=eval)
    env.seed(seed+rank)
    return env

if __name__ == '__main__':
    a = parse_args()
    # === Reset logger ===
    logger.Config.use(filename=a.ARGS.logging, level=a.ARGS.log_level, 
                      colored=True, reset=False)
    LOG = logger.getLogger('SAC')
    # === Print welcome message ===
    LOG.add_row('')
    LOG.add_rows('SAC', fmt='{:@f:ANSI_Shadow}', align='center')
    LOG.add_line()
    LOG.add_rows(__copyright__)
    LOG.flush('INFO')
    time.sleep(1)
    # === Print parameters ===
    print_args(LOG, a.ARGS,  'ARGS')
    print_args(LOG, a.ENV,   'ENV')
    print_args(LOG, a.MODEL, 'MODEL')
    print_args(LOG, a.LEARN, 'LEARN')
    LOG.flush('WARN')
    ub.utils.set_seed(a.seed)
    # === Make envs ===
    env = ub.envs.SubprocVecEnv([
        functools.partial(make_env, a, seed=a.ARGS.seed, rank=rank, eval=False)
        for rank in range(a.ARGS.n_envs)
    ])
    eval_env = make_env(a, seed=a.ARGS.eval_seed, eval=False)
    LOG.debug(f'Action space: {env.action_space}')
    LOG.debug(f'Observation space: {env.observation_space}')
    exit(0)
    # === Train model ===
    try:
        # ---Setup model & Train ---
        model        = SAC(env, **a.MODEL).learn(eval_env=eval_env, **a.LEARN)
        LOG.info('DONE')
        # Save model
        saved_path   = model.save(a.LEARN.save_path)
        LOG.info(f'Saving model to {saved_path}')
        # --- Load model from the latest checkpoint ---
        loaded_model = SAC.load(a.LEARN.save_path)
        # Evaluate model
        LOG.info('Evaluating the latest model...')
        results      = loaded_model.eval(evan_env, n_episodes=20)
        metrics      = loaded_model.get_eval_metrics(results)
        loaded_model.log_eval(20, results, metrics)
        # --- Load model from the best checkpoint ---
        loaded_model = SAC.load(a.LEARN.save_path, best=True)
        # Evaluate model
        LOG.info('Evaluating the best model...')
        results      = loaded_model.eval(evan_env, n_episodes=20)
        metrics      = loaded_model.get_eval_metrics(results)
        loaded_model.log_eval(20, results, metrics)
    except:
        LOG.exception('Exception occurred')
        env.close()
        eval_env.close()
        exit(1)
    env.close()
    eval_env.close()