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
import random
import logging
import argparse
import datetime

# --- 3rd party ---
import gym
import cloudpickle

import numpy as np
import tensorflow as tf

# --- my module ---
from unstable_baselines import logger

from unstable_baselines.envs_v2 import *
from unstable_baselines.utils_v2 import set_global_seeds

from unstable_baselines.ppo.model import PPO
from unstable_baselines.ppo.model import Agent as PPOAgent

def parse_args():

    parser = argparse.ArgumentParser(description='Proximal Policy Optimization')
    parser.add_argument('--logdir',           type=str, default='log/{env_id}/ppo/{rank}',help='Root dir             (args: {env_id}, {rank})')
    parser.add_argument('--logging',          type=str, default='train.log',              help='Log path             (args: {env_id}, {rank})')
    parser.add_argument('--log_level',        type=str, default='INFO',                   help='Log level')
    parser.add_argument('--monitor_dir',      type=str, default='monitor',                help='Monitor dir          (args: {env_id}, {rank})')
    parser.add_argument('--tb_logdir',        type=str, default='',                       help='Tensorboard log name (args: {env_id}, {rank})')
    parser.add_argument('--model_dir',        type=str, default='model',                  help='Model save path      (args: {env_id}, {rank})')
    parser.add_argument('--env_id',           type=str, default='BeamRiderNoFrameskip-v0',help='Environment ID')
    parser.add_argument('--num_envs',         type=int, default=4,      help='Number of environments')
    parser.add_argument('--num_epochs',       type=int, default=10000,  help='Number of training epochs')
    parser.add_argument('--num_steps',        type=int, default=256,    help='Number of timesteps per epoch (interact with envs)')
    parser.add_argument('--num_subepochs',    type=int, default=10,     help='Number of subepochs per epoch (perform gradient update)')
    parser.add_argument('--batch_size',       type=int, default=64,     help='Training batch size')
    parser.add_argument('--verbose',          type=int, default=1,      help='Print more message, 0=less, 1=more train log, 2=more eval log')
    parser.add_argument('--rank',             type=int, default=0,      help='Optional arguments for parallel training')
    parser.add_argument('--seed',             type=int, default=0,      help='Random seed')
    parser.add_argument('--log_interval',     type=int, default=1,      help='Logging interval (episodes)')
    parser.add_argument('--eval_interval',    type=int, default=1000,   help='Evaluation interval (episodes)')
    parser.add_argument('--eval_episodes',    type=int, default=5,      help='Number of episodes each evaluation')
    parser.add_argument('--eval_max_steps',   type=int, default=3000,   help='Maximum timesteps in each evaluation episode')
    parser.add_argument('--eval_seed',        type=int, default=0,      help='Environment seed for evaluation')
    parser.add_argument('--save_interval',    type=int, default=1000,   help='Model checkpoint interval (epochs)')
    parser.add_argument('--lr',               type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma',            type=float, default=0.99, help='Gamma decay rate')
    parser.add_argument('--gae_lambda',       type=float, default=0.95, help='GAE lambda decay rate')
    parser.add_argument('--clip_range',       type=float, default=0.2,  help='PPO policy clip range (epsilon)')
    parser.add_argument('--clip_range_vf',    type=float, default=None, help='Value clip range')
    parser.add_argument('--ent_coef',         type=float, default=0.01, help='Entropy loss ratio')
    parser.add_argument('--vf_coef',          type=float, default=0.5,  help='Value loss ratio')
    parser.add_argument('--max_grad_norm',    type=float, default=0.5,  help='Max gradient norm')
    parser.add_argument('--target_kl',        type=float, default=None, help='Target kl (early stop)')
    parser.add_argument('--shared_net',       action='store_true',      help='Share backbone network')
    parser.add_argument('--force_mlp',        action='store_true',      help='Use MLP network')
    parser.add_argument('--record_video',     action='store_true',      help='Enable video recording')

    a = parser.parse_args()

    a.logdir      = a.logdir.format(env_id=a.env_id, rank=a.rank)
    a.logging     = os.path.join(a.logdir, a.logging).format(env_id=a.env_id, rank=a.rank)
    a.monitor_dir = os.path.join(a.logdir, a.monitor_dir).format(env_id=a.env_id, rank=a.rank)
    a.tb_logdir   = os.path.join(a.logdir, a.tb_logdir).format(env_id=a.env_id, rank=a.rank)
    a.model_dir  = os.path.join(a.logdir, a.model_dir).format(env_id=a.env_id,  rank=a.rank)

    return a


def make_atari(a, eval=False):
    '''
    Make Atari-like env (Image observation)
    '''
    if not eval:
        def _make_env(rank, a):
            def _init():
                logger.Config.use(filename=a.logging, level=a.log_level, 
                                  colored=True, reset=True)
                set_global_seeds(a.seed)
                env = gym.make(a.env_id)
                env = SeedEnv(env, seed=a.seed+rank)
                env = NoopResetEnv(env, noop_max=30)
                if a.record_video:
                    env = VideoRecorder(env, os.path.join(a.monitor_dir, 'video/'),
                                    prefix='train.{}'.format(rank), fps=60, force=True)
                env = MaxAndSkipEnv(env, skip=4)
                env = Monitor(env, a.monitor_dir, prefix=str(rank), force=True)
                env = EpisodicLifeEnv(env)
                env = WarpFrame(env)
                env = ClipRewardEnv(env)
                return env
            return _init
        env = SubprocVecEnv([_make_env(i, a) for i in range(a.num_envs)])
        env = VecFrameStack(env, 4)
    else:
        env = gym.make(a.env_id)
        env = SeedEnv(env, seed=a.eval_seed)
        env = NoopResetEnv(env, noop_max=30)
        if a.record_video:
            env = VideoRecorder(env, os.path.join(a.monitor_dir, 'video/'), fps=60,
                                 prefix='eval', callback=True, force=True)
        env = MaxAndSkipEnv(env, skip=4)
        env = Monitor(env, directory=a.monitor_dir, prefix='eval', force=True)
        env = WarpFrame(env)
        env = FrameStack(env, 4)
    
    return env


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
                set_global_seeds(a.seed)
                import pybullet_envs
                env = gym.make(a.env_id)
                env = SeedEnv(env, seed=a.seed+rank)
                if a.record_video:
                    env = VideoRecorder(env, os.path.join(a.monitor_dir, 'video/'),
                                      prefix='train.{}'.format(rank), force=True)
                env = Monitor(env, a.monitor_dir, prefix=str(rank), force=True)
                return env
            return _init
        env = SubprocVecEnv([_make_env(i, a) for i in range(a.num_envs)])
    else:
        env = gym.make(a.env_id)
        env = SeedEnv(env, seed=a.eval_seed)
        if a.record_video:
            env = VideoRecorder(env, os.path.join(a.monitor_dir, 'video/'),
                            prefix='eval', callback=True, force=True)
        env = Monitor(env, a.monitor_dir, prefix='eval',force=True)
    return env


if __name__ == '__main__':

    a = parse_args()

    # === Reset logger ===
    logger.Config.use(filename=a.logging, level=a.log_level, colored=True, reset=True)
    LOG = logger.getLogger('PPO')

    # === Print welcome message ===
    LOG.add_row('')
    LOG.add_rows('PPO', fmt='{:@f:ANSI_Shadow}', align='center')
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
    LOG.add_row('Num of epochs',         a.num_epochs)
    LOG.add_row('Num of subepochs',      a.num_subepochs)
    LOG.add_row('Log interval',          a.log_interval)
    LOG.add_row('Eval interval',         a.eval_interval)
    LOG.add_row('Eval episodes',         a.eval_episodes)
    LOG.add_row('Eval max steps',        a.eval_max_steps)
    LOG.add_row('Save interval',         a.save_interval)
    LOG.add_row('Batch size',            a.batch_size)
    LOG.add_row('Verbose',               a.verbose)
    LOG.add_line()
    LOG.add_row('Shared network',        a.shared_net)
    LOG.add_row('Force MLP',             a.force_mlp)
    LOG.add_row('Learning rate',         a.lr)
    LOG.add_row('Gamma',                 a.gamma)
    LOG.add_row('Lambda',                a.gae_lambda)
    LOG.add_row('Clip range',            a.clip_range)
    LOG.add_row('Value clip range',      a.clip_range_vf)
    LOG.add_row('Entropy coef',          a.ent_coef)
    LOG.add_row('Value coef',            a.vf_coef)
    LOG.add_row('Max gradient norm',     a.max_grad_norm)
    LOG.add_row('Target KL',             a.target_kl)
    LOG.flush('WARNING')

    set_global_seeds(a.seed)

    # === Make envs ===

    if 'NoFrameskip' in a.env_id:
        # Atari env
        env      = make_atari(a, eval=False)
        eval_env = make_atari(a, eval=True)
    else:
        # Pybullet env
        env      = make_env(a, eval=False)
        eval_env = make_env(a, eval=True)
    
    LOG.debug('Action space: {}'.format(env.action_space))
    LOG.debug('Observation space: {}'.format(env.observation_space))


    # === Create model ===
    try:
        model = PPO(env, learning_rate   = a.lr,
                         n_steps         = a.num_steps,
                         batch_size      = a.batch_size,
                         n_subepochs     = a.num_subepochs, 
                         gamma           = a.gamma,
                         gae_lambda      = a.gae_lambda,
                         clip_range      = a.clip_range,
                         clip_range_vf   = a.clip_range_vf,
                         ent_coef        = a.ent_coef,
                         vf_coef         = a.vf_coef,
                         max_grad_norm   = a.max_grad_norm,
                         target_kl       = a.target_kl,
                         shared_net      = a.shared_net,
                         force_mlp       = a.force_mlp,
                         verbose         = a.verbose)
        
        # Total timesteps = num_steps * num_envs * num_epochs (default ~ 10M)
        model.learn(a.num_steps *    a.num_envs * a.num_epochs, 
                    tb_logdir      = a.tb_logdir, 
                    log_interval   = a.log_interval,
                    eval_env       =   eval_env, 
                    eval_interval  = a.eval_interval, 
                    eval_episodes  = a.eval_episodes, 
                    eval_max_steps = a.eval_max_steps,
                    save_interval  = a.save_interval,
                    save_path      = a.model_dir)
        
        LOG.info('DONE')

        # Save complete model (continue training)
        saved_path = model.save(a.model_dir)
        LOG.info('Saving model to: {}'.format(saved_path))

        # load the "latest" checkpoint
        loaded_model = PPO.load(a.model_dir)
        # or you can directly load from saved_path
        # loaded_model = PPO.load(saved_path)

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
        # loaded_model = PPOAgent.load(saved_path)

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
        loaded_model = PPO.load(a.model_dir, best=True)

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
