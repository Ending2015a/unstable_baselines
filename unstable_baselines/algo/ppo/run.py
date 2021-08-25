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
from unstable_baselines.algo.ppo import PPO

def parse_args():
    parser = argparse.ArgumentParser(description='Proximal Policy Optimization')
    parser.add_argument('--root',      type=str, default='log/ppo/')
    parser.add_argument('--env_id',    type=str, default='HalfCheetahBulletEnv-v0')
    parser.add_argument('--seed',      type=int, default=1)
    parser.add_argument('--eval_seed', type=int, default=0)
    parser.add_argument('--n_envs',    type=int, default=8)
    parser.add_argument('--mlp',       action='store_true')
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
    a.ENV.video                = True
    # Hyper parameters
    a.MODEL = ub.utils.StateObject()
    a.MODEL.learning_rate      = 3e-4
    a.MODEL.gamma              = 0.99
    a.MODEL.gae_lambda         = 0.95
    a.MODEL.policy_clip        = 0.2
    a.MODEL.value_clip         = None
    a.MODEL.dual_clip          = None
    a.MODEL.ent_coef           = 0.01
    a.MODEL.vf_coef            = 0.5
    a.MODEL.clipnorm           = 0.5
    a.MODEL.target_kl          = None
    a.MODEL.share_net          = not bool(args.mlp)
    a.MODEL.force_mlp          = bool(args.mlp)
    a.MODEL.mlp_units          = [256, 256]
    a.MODEL.n_steps            = 125
    a.MODEL.n_subepochs        = 8
    a.MODEL.batch_size         = 256
    a.MODEL.verbose            = 2
    # Training parameters
    a.LEARN = ub.utils.StateObject()
    a.LEARN.total_timesteps    = a.ARGS.n_envs * a.MODEL.n_steps * 10000 # ~10M
    a.LEARN.log_interval       = 1    # epoch
    a.LEARN.eval_interval      = 1000 # epoch
    a.LEARN.eval_episodes      = 5
    a.LEARN.eval_max_steps     = 1000
    a.LEARN.save_interval      = 1000 # epoch
    a.LEARN.save_path          = os.path.join(args.root, 'save')
    a.LEARN.tb_logdir          = args.root
    return a

def print_args(LOG, a, group):
    '''Pretty print args'''
    LOG.subgroup(group)
    label = '\n'.join(map('{:15s}'.format, a.keys()))
    value = '\n'.join(map(str, a.values()))
    LOG.add_rows(fmt='{label} {||} {value}', label=label, value=value)

def make_atari_env(a, eval, **monitor_params):
    '''Make atari environment'''
    env = ub.envs.make_atari(a.env_id)
    env = ub.envs.Monitor(env, root_dir=a.monitor_dir, video=a.video,
                          **monitor_params)
    env = ub.envs.wrap_deepmind(env, episode_life=not eval,
                                     clip_rewards=not eval)
    return env

def make_pybullet_env(a, eval, **monitor_params):
    '''Make pybullet environment'''
    import pybullet_envs
    env = gym.make(a.env_id)
    env = ub.envs.Monitor(env, root_dir=a.monitor_dir, video=a.video,
                          **monitor_params)
    env = ub.envs.wrap_mujoco(env, test_mode=eval)
    return env

def make_env(a, rank=0, eval=False):
    # some params for monitering
    monitor_params = dict(
        # filename prefix
        prefix = 'eval' if eval else f'{rank}.train',
        # record every n episodes, None for cubic schedule
        video_kwargs = dict(interval=1 if eval else None)
    )
    if 'NoFrameskip' in a.env_id:
        # make atari env
        env = make_atari_env(a, eval=eval, **monitor_params)
    else:
        # make pybullet env
        env = make_pybullet_env(a, eval=eval, **monitor_params)
    return env

if __name__ == '__main__':
    a = parse_args()
    # === Reset logger ===
    logger.Config.use(filename=a.ARGS.logging, level=a.ARGS.log_level,
                    colored=True, reset=False)
    LOG = logger.getLogger('PPO')
    # === Print welcome message ===
    LOG.add_row('')
    LOG.add_rows('PPO', fmt='{:@f:ANSI_Shadow}', align='center')
    LOG.add_line()
    LOG.add_rows(ub.__copyright__)
    LOG.flush('INFO')
    time.sleep(1)
    # === Print parameters ===
    print_args(LOG, a.ARGS,  'ARGS')
    print_args(LOG, a.ENV,   'ENV')
    print_args(LOG, a.MODEL, 'MODEL')
    print_args(LOG, a.LEARN, 'LEARN')
    LOG.flush('WARN')
    ub.utils.set_seed(a.ARGS.seed)
    # === Make envs ===
    env = ub.envs.SubprocVecEnv([
        functools.partial(make_env, a.ENV, rank=rank, eval=False)
        for rank in range(a.ARGS.n_envs)
    ])
    eval_env = make_env(a.ENV, eval=True)
    eval_env = ub.envs.ObsNorm(eval_env, rms_norm=env, update_rms=False)
    env.seed(a.ARGS.seed) # seed ~ seed+n_envs
    eval_env.seed(a.ARGS.eval_seed)
    LOG.debug(f'Action space: {env.action_space}')
    LOG.debug(f'Observation space: {env.observation_space}')
    # === Train model ===
    try:
        # --- Setup model & train ---
        model = PPO(env, **a.MODEL).learn(eval_env=eval_env, **a.LEARN)
        LOG.info('DONE')
        # Save model
        saved_path = model.save(a.LEARN.save_path)
        LOG.info(f'Saving model to {saved_path}')
        # --- Load model from the latest checkpoint ---
        loaded_model = PPO.load(a.LEARN.save_path)
        # Evaluate model
        LOG.info('Evaluating the latest model ...')
        results = loaded_model.eval(eval_env, n_episodes=20)
        metrics = loaded_model.get_eval_metrics(results)
        loaded_model.log_eval(20, results, metrics)
        # --- Load model from the best checkpoint ---
        loaded_model = PPO.load(a.LEARN.save_path, best=True)
        # Evaluate model
        LOG.info('Evaluating the best model ...')
        results = loaded_model.eval(eval_env, n_episodes=20)
        metrics = loaded_model.get_eval_metrics(results)
        loaded_model.log_eval(20, results, metrics)
    except:
        LOG.exception('Exception occurred')
        env.close()
        eval_env.close()
        exit(1)
    env.close()
    eval_env.close()