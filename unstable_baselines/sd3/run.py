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
import re
import sys
import time
import logging
import argparse
import datetime

# --- 3rd party ---
import gym

import numpy as np
import tensorflow as tf

# --- my module ---
from unstable_baselines import logger

from unstable_baselines.envs import *
from unstable_baselines.utils_v2 import (set_global_seeds,
                                         NormalActionNoise)

from unstable_baselines.sd3.model import SD3
from unstable_baselines.sd3.model import Agent as SD3Agent


def parse_args():

    parser = argparse.ArgumentParser(description='Softmax Deep Double Deterministic Policy Gradient (SD3)')
    parser.add_argument('--logdir',              type=str,   default='log/{env_id}/ppo/{rank}',help='Root dir             (args: {env_id}, {rank})')
    parser.add_argument('--logging',             type=str,   default='train.log',              help='Log path             (args: {env_id}, {rank})')
    parser.add_argument('--log_level',           type=str,   default='INFO',                   help='Log level')
    parser.add_argument('--monitor_dir',         type=str,   default='monitor',                help='Monitor dir          (args: {env_id}, {rank})')
    parser.add_argument('--tb_logdir',           type=str,   default='',                       help='Tensorboard log name (args: {env_id}, {rank})')
    parser.add_argument('--model_dir',           type=str,   default='model/weights',          help='Model save path      (args: {env_id}, {rank})')
    parser.add_argument('--env_id',              type=str,   default='HalfCheetahBulletEnv-v0',help='Environment ID')
    parser.add_argument('--num_envs',            type=int,   default=1,       help='Number of environments')
    parser.add_argument('--num_epochs',          type=int,   default=10000,   help='Number of training epochs')
    parser.add_argument('--num_steps',           type=int,   default=100,     help='Number of timesteps per epoch (interact with envs)')
    parser.add_argument('--num_gradsteps',       type=int,   default=100,     help='Number of gradient steps')
    parser.add_argument('--batch_size',          type=int,   default=100,     help='Training batch size')
    parser.add_argument('--buffer_size',         type=int,   default=1000000, help='Maximum size of replay buffer')
    parser.add_argument('--min_buffer',          type=int,   default=10000,   help='Minimum number of samples in replay buffer')
    parser.add_argument('--policy_update',       type=int,   default=2,       help='Delayed update to policy network (gradsteps)')
    parser.add_argument('--target_update',       type=int,   default=2,       help='Target network update frequency (gradsteps)')
    parser.add_argument('--verbose',             type=int,   default=1,       help='Print more message, 0=less, 1=more train log, 2=more eval log')
    parser.add_argument('--rank',                type=int,   default=0,       help='Optional arguments for parallel training')
    parser.add_argument('--seed',                type=int,   default=0,       help='Random seed')
    parser.add_argument('--log_interval',        type=int,   default=10,      help='Logging interval (epochs)')
    parser.add_argument('--eval_interval',       type=int,   default=1000,    help='Evaluation interval (epochs)')
    parser.add_argument('--eval_episodes',       type=int,   default=5,       help='Number of episodes each evaluation')
    parser.add_argument('--eval_max_steps',      type=int,   default=1000,    help='Maximum timesteps in each evaluation episode')
    parser.add_argument('--eval_seed',           type=int,   default=0,       help='Environment seed for evaluation')
    parser.add_argument('--save_interval',       type=int,   default=1000,    help='Model checkpoint interval (epochs)')
    parser.add_argument('--action_samples',      type=int,   default=50,      help='Number of action samples used to estimate softmax')
    parser.add_argument('--lr',                  type=float, default=1e-3,    help='Learning rate')
    parser.add_argument('--beta',                type=float, default=1e-3,    help='Softmax temperature')
    parser.add_argument('--gamma',               type=float, default=0.99,    help='Discount factor')
    parser.add_argument('--tau',                 type=float, default=0.005,   help='Polyak update coefficient (tau in original paper)')
    parser.add_argument('--max_grad_norm',       type=float, default=0.5,     help='Gradient norm clip range')
    parser.add_argument('--action_noise',        type=float, default=0.2,     help='Noise scale added to target actions')
    parser.add_argument('--action_noise_clip',   type=float, default=0.5,     help='Noise range added to target actions')
    parser.add_argument('--explore_noise_mean',  type=float, default=0,       help='Mean of normal action noise')
    parser.add_argument('--explore_noise_scale', type=float, default=0.1,     help='Scale of normal action noise')
    parser.add_argument('--explore_noise',       action='store_true',         help='Enable exploration noise')
    parser.add_argument('--force_mlp',           action='store_true',         help='Use MLP network')
    parser.add_argument('--importance_sampling', action='store_true',         help='Enable importance sampling in softmax operator')
    parser.add_argument('--record_video',        action='store_true',         help='Enable video recording')
    a = parser.parse_args()

    a.logdir      = a.logdir.format(env_id=a.env_id, rank=a.rank)
    a.logging     = os.path.join(a.logdir, a.logging).format(env_id=a.env_id, rank=a.rank)
    a.monitor_dir = os.path.join(a.logdir, a.monitor_dir).format(env_id=a.env_id, rank=a.rank)
    a.tb_logdir   = os.path.join(a.logdir, a.tb_logdir).format(env_id=a.env_id, rank=a.rank)
    a.model_dir   = os.path.join(a.logdir, a.model_dir).format(env_id=a.env_id,  rank=a.rank)

    return a

def make_env(a, eval=False):
    '''
    Make non-Atari env (Pybullet)
    '''
    import pybullet_envs

    if not eval:
        def _make_env(rank, a):
            def _init():
                logger.Config.use(filename=a.logging, level=a.log_level, 
                                  colored=True, reset=True)
                import pybullet_envs
                env = gym.make(a.env_id)
                env.seed(a.seed + rank)
                env = Monitor(env, directory=a.monitor_dir, prefix=str(rank),
                            enable_video_recording=a.record_video, force=True,
                            video_kwargs={'prefix':'video/train.{}'.format(rank)})
                return env
            set_global_seeds(a.seed)
            return _init
        env = SubprocVecEnv([_make_env(i, a) for i in range(a.num_envs)])
    else:
        env = gym.make(a.env_id)
        env.seed(a.eval_seed)
        env = Monitor(env, directory=a.monitor_dir, prefix='eval',
                        enable_video_recording=a.record_video, force=True,
                        video_kwargs={'prefix':'video/eval',
                                        'callback': lambda x: True})
    return env


if __name__ == '__main__':
    
    a = parse_args()

    # === Reset logger ===
    logger.Config.use(filename=a.logging, level=a.log_level, colored=True, reset=True)
    LOG = logger.getLogger('SD3')

    # === Print welcome message ===
    LOG.add_row('')
    LOG.add_rows('SD3', fmt='{:@f:ANSI_Shadow}', align='center')
    LOG.add_line()
    LOG.add_rows('{}'.format(__copyright__))
    LOG.flush('INFO')
    time.sleep(1)

    # === Print arguments ===
    LOG.set_header('Arguments')
    LOG.add_row('Log dir',               a.logdir)
    LOG.add_row('Logging path',          a.logging)
    LOG.add_row('Monitor path',          a.monitor_dir)
    LOG.add_row('Tensorboard path',      a.tb_logdir)
    LOG.add_row('Model path',            a.model_dir)
    LOG.add_row('Env ID',                a.env_id)
    LOG.add_row('Seed',                  a.seed)
    LOG.add_row('Eval seed',             a.eval_seed)
    LOG.add_row('Record video',          a.record_video)
    LOG.add_line()
    LOG.add_row('Num of envs',           a.num_envs)
    LOG.add_row('Num of steps/epoch',    a.num_steps)
    LOG.add_row('Num of gradient steps', a.num_gradsteps)
    LOG.add_row('Num of epochs',         a.num_epochs)
    LOG.add_row('Log interval',          a.log_interval)
    LOG.add_row('Eval interval',         a.eval_interval)
    LOG.add_row('Eval episodes',         a.eval_episodes)
    LOG.add_row('Eval max steps',        a.eval_max_steps)
    LOG.add_row('Save interval',         a.save_interval)
    LOG.add_row('Batch size',            a.batch_size)
    LOG.add_row('Buffer size',           a.buffer_size)
    LOG.add_row('Min buffer size',       a.min_buffer)
    LOG.add_row('Verbose',               a.verbose)
    LOG.add_line()
    LOG.add_row('Force MLP',             a.force_mlp)
    LOG.add_row('Learning rate',         a.lr)
    LOG.add_row('Beta',                  a.beta)
    LOG.add_row('Gamma',                 a.gamma)
    LOG.add_row('Tau (Polyak)',          a.polyak)
    LOG.add_row('Policy update freq',    a.policy_update)
    LOG.add_row('Target update freq',    a.target_update)
    LOG.add_row('Action samples',        a.action_samples)
    LOG.add_row('Action noise',          a.action_noise)
    LOG.add_row('Action noise clip',     a.action_noise_clip)
    LOG.add_row('Explore noise',         a.explore_noise)
    LOG.add_row('Explore noise mean',    a.explore_noise_mean)
    LOG.add_row('Explore noise scale',   a.explore_noise_scale)
    LOG.add_row('Importance sampling',   a.importance_sampling)

    LOG.flush('WARNING')

    # === Make envs ===
    
    env      = make_env(a, eval=False)
    eval_env = make_env(a, eval=True)
    
    LOG.debug('Action space: {}'.format(env.action_space))
    LOG.debug('Observation space: {}'.format(env.observation_space))

    # === Create action noise ===
    if a.explore_noise:
        explore_noise = NormalActionNoise(a.explore_noise_mean, 
                                          a.explore_noise_scale)
    else:
        explore_noise = None

    # === Create model ===
    try:
        model = SD3(env, learning_rate       = a.lr,
                         buffer_size         = a.buffer_size,
                         min_buffer          = a.min_buffer,
                         n_steps             = a.num_steps,
                         n_gradsteps         = a.n_gradsteps,
                         batch_size          = a.batch_size,
                         policy_delay        = a.policy_delay,
                         gamma               = a.gamma,
                         tau                 = a.tau,
                         beta                = a.beta,
                         max_grad_norm       = a.max_grad_norm,
                         action_samples      = a.action_samples, 
                         action_noise        = a.action_noise,
                         action_noise_clip   = a.action_noise_clip,
                         explore_noise       =   explore_noise,
                         importance_sampling = a.importance_sampling,
                         force_mlp           = a.force_mlp,
                         verbose             = a.verbose)
        
        # Total timesteps = num_steps * num_envs * num_episodes (default ~ 1M)
        model.learn(a.num_steps *    a.num_envs * a.num_episodes,
                    tb_logdir      = a.tb_logdir,
                    log_interval   = a.log_interval,
                    eval_env       =   eval_env, 
                    eval_interval  = a.eval_interval, 
                    eval_episodes  = a.eval_episodes, 
                    eval_max_steps = a.eval_max_steps,
                    save_interval  = a.save_interval,
                    save_path      = a.model_dir,
                    target_update  = a.target_update)

        LOG.info('DONE')

        # Save complete model (continue training)
        saved_path = model.save(a.model_dir)
        LOG.info('Saving model to: {}'.format(saved_path))

        # load the "latest" checkpoint
        loaded_model = SD3.load(a.model_dir)
        # or you can directly load from saved_path
        # loaded_model = SD3.load(saved_path)

        # set env to continue training
        # loaded_model.set_env(env)
        # loaded_model.learn(a.num_steps *    a.num_envs * a.num_episodes * 2,
        #                     tb_logdir      = a.tb_logdir,
        #                     log_interval   = a.log_interval,
        #                     eval_env       =  eval_env, 
        #                     eval_interval  = a.eval_interval, 
        #                     eval_episodes  = a.eval_episodes, 
        #                     eval_max_steps = a.eval_max_steps)

        # Save agent only
        # saved_path = model.agent.save(a.model_dir)
        # LOG.info('Saving model to: {}'.format(saved_path))
        # loaded_model = SD3Agent.load(saved_path)

        # Evaluation
        LOG.info('Evaluating model (Latest checkpoint)')
        
        eps_rews, eps_steps = loaded_model.eval(eval_env, n_episodes=20)

        max_idx    = np.argmax(eps_rews)
        max_rews   = eps_rews[max_idx]
        max_steps  = eps_steps[max_idx]
        mean_rews  = np.mean(eps_rews)
        std_rews   = np.std(eps_rews)
        mean_steps = np.mean(eps_steps)

        # === Print evaluation results ===
        LOG.set_header('Final Evaluation Results')
        LOG.add_line()
        LOG.add_row('Max rewards',  max_rews)
        LOG.add_row('  Length',     max_steps)
        LOG.add_line()
        LOG.add_row('Mean rewards', mean_rews.round(3))
        LOG.add_row('Std rewards',  std_rews, fmt='{}: {:.6f}')
        LOG.add_row('Mean length',  mean_steps)
        LOG.add_line()
        LOG.flush('INFO')


        # load the "best" checkpoints
        loaded_model = SD3.load(a.model_dir, best=True)

        LOG.info('Evaluating model (Best checkpoint)')
        eps_rews, eps_steps = loaded_model.eval(eval_env, n_episodes=20)

        max_idx    = np.argmax(eps_rews)
        max_rews   = eps_rews[max_idx]
        max_steps  = eps_steps[max_idx]
        mean_rews  = np.mean(eps_rews)
        std_rews   = np.std(eps_rews)
        mean_steps = np.mean(eps_steps)

        # === Print evaluation results ===
        LOG.set_header('Final Evaluation Results')
        LOG.add_line()
        LOG.add_row('Max rewards',  max_rews)
        LOG.add_row('  Length',     max_steps)
        LOG.add_line()
        LOG.add_row('Mean rewards', mean_rews.round(3))
        LOG.add_row('Std rewards',  std_rews, fmt='{}: {:.6f}')
        LOG.add_row('Mean length',  mean_steps)
        LOG.add_line()
        LOG.flush('INFO')

    except:
        LOG.exception('Exception occurred')
    
        env.close()
        eval_env.close()
        exit(1)
    
    env.close()
    eval_env.close()