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
import logging
import argparse

# --- 3rd party ---
import gym

import numpy as np

# --- my module ---
from unstable_baselines import logger

from unstable_baselines.envs import *
from unstable_baselines.utils import set_global_seeds
from unstable_baselines.sche import LinearScheduler

from unstable_baselines.d.c51.model import C51
from unstable_baselines.d.c51.model import Agent as C51Agent

def parse_args():

    parser = argparse.ArgumentParser(description='Categorical DQN (C51)')
    parser.add_argument('--logdir',           type=str, default='log/{env_id}/c51/{rank}',help='Root dir             (args: {env_id}, {rank})')
    parser.add_argument('--logging',          type=str, default='train.log',              help='Log path             (args: {env_id}, {rank})')
    parser.add_argument('--log_level',        type=str, default='INFO',                   help='Log level')
    parser.add_argument('--monitor_dir',      type=str, default='monitor',                help='Monitor dir          (args: {env_id}, {rank})')
    parser.add_argument('--tb_logdir',        type=str, default='',                       help='Tensorboard log name (args: {env_id}, {rank})')
    parser.add_argument('--model_path',       type=str, default='model/weights',          help='Model save path      (args: {env_id}, {rank})')
    parser.add_argument('--env_id',           type=str, default='BeamRiderNoFrameskip-v0',help='Environment ID')
    parser.add_argument('--num_envs',         type=int, default=4,      help='Number of environments')
    parser.add_argument('--num_epochs',       type=int, default=625000, help='Number of training epochs')
    parser.add_argument('--num_steps',        type=int, default=4,      help='Number of timesteps per epoch (interact with envs)')
    parser.add_argument('--num_gradsteps',    type=int, default=1,      help='Number of gradient steps per epoch (perform gradient update)')
    parser.add_argument('--num_atoms',        type=int, default=51,     help='Number of categories of Q value')
    parser.add_argument('--target_update',    type=int, default=2500,   help='Target network update frequency (gradsteps)')
    parser.add_argument('--batch_size',       type=int, default=64,     help='Training batch size')
    parser.add_argument('--buffer_size',      type=int, default=1000000,help='Replay buffer capacity')
    parser.add_argument('--min_buffer',       type=int, default=50000,  help='Minimum size of replay buffer before start training')
    parser.add_argument('--verbose',          type=int, default=1,      help='Print more message, 0=less, 1=more train log, 2=more eval log')
    parser.add_argument('--rank',             type=int, default=0,      help='Optional arguments for parallel training')
    parser.add_argument('--seed',             type=int, default=0,      help='Random seed')
    parser.add_argument('--log_interval',     type=int, default=1000,   help='Logging interval (epochs)')
    parser.add_argument('--eval_interval',    type=int, default=10000,  help='Evaluation interval (epochs)')
    parser.add_argument('--eval_episodes',    type=int, default=5,      help='Number of episodes each evaluation')
    parser.add_argument('--eval_max_steps',   type=int, default=3000,   help='Maximum timesteps in each evaluation episode')
    parser.add_argument('--eval_seed',        type=int, default=0,      help='Environment seed for evaluation')
    parser.add_argument('--save_interval',    type=int, default=10000,  help='Model checkpoint interval (epochs)')
    parser.add_argument('--v_min',            type=float, default=-10., help='Minimum of value')
    parser.add_argument('--v_max',            type=float, default=10.,  help='Maximum of value')
    parser.add_argument('--lr',               type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma',            type=float, default=0.99, help='Gamma decay rate')
    parser.add_argument('--tau',              type=float, default=1.0,  help='Polyak update rate')
    parser.add_argument('--max_grad_norm',    type=float, default=0.5,  help='Gradient clipping range')
    parser.add_argument('--explore_rate',     type=float, default=1.0,  help='Initial exploration rate for epsilon greedy')
    parser.add_argument('--explore_final',    type=float, default=0.05, help='Final exploration rate for epsilon greedy')
    parser.add_argument('--explore_progress', type=float, default=0.1,  help='Progress to stop decaying exploration rate')
    parser.add_argument('--force_mlp',        action='store_true',      help='Use MLP network')
    parser.add_argument('--record_video',     action='store_true',      help='Enable video recording')

    a = parser.parse_args()

    a.logdir      = a.logdir.format(env_id=a.env_id, rank=a.rank)
    a.logging     = os.path.join(a.logdir, a.logging).format(env_id=a.env_id,     rank=a.rank)
    a.monitor_dir = os.path.join(a.logdir, a.monitor_dir).format(env_id=a.env_id, rank=a.rank)
    a.tb_logdir   = os.path.join(a.logdir, a.tb_logdir).format(env_id=a.env_id,   rank=a.rank)
    a.model_path  = os.path.join(a.logdir, a.model_path).format(env_id=a.env_id,  rank=a.rank)

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
                env = gym.make(a.env_id)
                env.seed(a.seed + rank)
                env = NoopResetEnv(env, noop_max=30)
                env = MaxAndSkipEnv(env, skip=4)
                env = Monitor(env, directory=a.monitor_dir, prefix=str(rank),
                            enable_video_recording=a.record_video, force=True,
                            video_kwargs={'prefix':'video/train.{}'.format(rank)})
                env = EpisodicLifeEnv(env)
                env = WarpFrame(env)
                env = ClipRewardEnv(env)
                return env
            set_global_seeds(a.seed)
            return _init
        env = SubprocVecEnv([_make_env(i, a) for i in range(a.num_envs)])
        env = VecFrameStack(env, 4)
    else:
        env = gym.make(a.env_id)
        env.seed(a.eval_seed)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = Monitor(env, directory=a.monitor_dir, prefix='eval',
                        enable_video_recording=a.record_video, force=True,
                        video_kwargs={'prefix':'video/eval',
                                        'callback': lambda x: True})
        env = WarpFrame(env)
        env = FrameStack(env, 4)
    
    return env



if __name__ == '__main__':

    a = parse_args()

    # === Reset logger ===
    logger.Config.use(filename=a.logging, level=a.log_level, colored=True, reset=True)
    LOG = logger.getLogger('C51')

    # === Print welcome message ===
    LOG.add_row('')
    LOG.add_rows('C51', fmt='{:@f:ANSI_Shadow}', align='center')
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
    LOG.add_row('Model path',            a.model_path)
    LOG.add_row('Env ID',                a.env_id)
    LOG.add_row('Seed',                  a.seed)
    LOG.add_row('Eval seed',             a.eval_seed)
    LOG.add_row('Record video',          a.record_video)
    LOG.add_line()
    LOG.add_row('Num of envs',           a.num_envs)
    LOG.add_row('Num of steps/epoch',    a.num_steps)
    LOG.add_row('Num of epochs',         a.num_epochs)
    LOG.add_row('Num of gradsteps',      a.num_gradsteps)
    LOG.add_row('Target update freq',    a.target_update)
    LOG.add_row('Log interval',          a.log_interval)
    LOG.add_row('Eval interval',         a.eval_interval)
    LOG.add_row('Eval episodes',         a.eval_episodes)
    LOG.add_row('Eval max steps',        a.eval_max_steps)
    LOG.add_row('Save interval',         a.save_interval)
    LOG.add_row('Batch size',            a.batch_size)
    LOG.add_row('Buffer size',           a.buffer_size)
    LOG.add_row('Min buffer',            a.min_buffer)
    LOG.add_row('Verbose',               a.verbose)
    LOG.add_line()
    LOG.add_row('Force MLP',             a.force_mlp)
    LOG.add_row('Num of atoms',          a.num_atoms)
    LOG.add_row('Value min',             a.v_min)
    LOG.add_row('Value max',             a.v_max)
    LOG.add_row('Learning rate',         a.lr)
    LOG.add_row('Gamma',                 a.gamma)
    LOG.add_row('Tau (Polyak)',          a.tau)
    LOG.add_row('Max gradient norm',     a.max_grad_norm)
    LOG.add_row('Explore rate',          a.explore_rate)
    LOG.add_row('Final explore rate',    a.explore_final)
    LOG.add_row('Explore progress',      a.explore_progress)
    LOG.flush('WARNING')

    # === Make envs ===

    # Atari env
    env      = make_atari(a, eval=False)
    eval_env = make_atari(a, eval=True)
    
    LOG.debug('Action space: {}'.format(env.action_space))
    LOG.debug('Observation space: {}'.format(env.observation_space))

    # === Create scheduler ===
    Unit = LinearScheduler.Unit
    explore_schedule = LinearScheduler(start_value = a.explore_rate,
                                       stop_value  = a.explore_final,
                                       decay_steps = a.explore_progress,
                                       unit        = Unit.progress)

    # === Create model ===
    try:
        model = C51(env, learning_rate    = a.lr,
                         buffer_size      = a.buffer_size,
                         min_buffer       = a.min_buffer,
                         n_atoms          = a.num_atoms,
                         v_min            = a.v_min,
                         v_max            = a.v_max,
                         n_steps          = a.num_steps,
                         n_gradsteps      = a.num_gradsteps,
                         batch_size       = a.batch_size,
                         gamma            = a.gamma,
                         tau              = a.tau,
                         max_grad_norm    = a.max_grad_norm,
                         force_mlp        = a.force_mlp,
                         explore_schedule =   explore_schedule,
                         verbose          = a.verbose)
        
        # Total timesteps = num_steps * num_envs * num_epochs (default ~ 10M)
        model.learn(a.num_steps *    a.num_envs * a.num_epochs, 
                    tb_logdir      = a.tb_logdir, 
                    log_interval   = a.log_interval,
                    eval_env       =   eval_env, 
                    eval_interval  = a.eval_interval, 
                    eval_episodes  = a.eval_episodes, 
                    eval_max_steps = a.eval_max_steps,
                    save_interval  = a.save_interval,
                    save_path      = a.model_path,
                    target_update  = a.target_update)
        
        LOG.info('DONE')

        # Save complete model (continue training)
        saved_path = model.save(a.model_path)
        LOG.info('Saving model to: {}'.format(saved_path))

        loaded_model = C51.load(saved_path)

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
        # saved_path = model.agent.save(a.model_path)
        # LOG.info('Saving model to: {}'.format(saved_path))
        # loaded_model = C51Agent.load(saved_path)

        # Evaluate model
        LOG.info('Evaluating model')
        
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
        LOG.add_row('Mean rewards', mean_rews)
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