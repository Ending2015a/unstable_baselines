# This example shows how to train an PPO agent on atari domain
# For complete experiments, please refer to 
#     unstable_baselines.algo.ppo.run
# --- built in ---
import os
import time
import argparse
import functools
# --- 3rd party ---
import gym
# --- my module ---
import unstable_baselines as ub
from unstable_baselines.algo.ppo import PPO

def parse_args():
    root_path = 'log/ppo/BeamRiderNoFrameskip-v4'
    # Configurations
    a = ub.utils.StateObject()
    a.ARGS = ub.utils.StateObject()
    a.ARGS.logging          = os.path.join(root_path, 'training.log')
    a.ARGS.log_level        = 'INFO'
    a.ARGS.seed             = 1
    a.ARGS.eval_seed        = 0
    a.ARGS.n_envs           = 8
    # Env/Monitor parameters
    a.ENV = ub.utils.StateObject()
    a.ENV.env_id            = 'BeamRiderNoFrameskip-v4'
    a.ENV.monitor_dir       = os.path.join(root_path, 'monitor')
    a.ENV.video             = True   # record video
    # Hyper parameters
    a.MODEL = ub.utils.StateObject()
    a.MODEL.learning_rate   = 3e-4
    a.MODEL.gamma           = 0.99
    a.MODEL.gae_lambda      = 0.95
    a.MODEL.policy_clip     = 0.1
    a.MODEL.value_clip      = None
    a.MODEL.dual_clip       = None
    a.MODEL.ent_coef        = 0.01
    a.MODEL.vf_coef         = 0.5
    a.MODEL.reg_coef        = 0.0
    a.MODEL.clipnorm        = 0.5
    a.MODEL.target_kl       = None
    a.MODEL.share_net       = False
    a.MODEL.force_mlp       = False
    a.MODEL.mlp_units       = [64, 64]
    a.MODEL.n_steps         = 125
    a.MODEL.n_subepochs     = 8
    a.MODEL.batch_size      = 256
    a.MODEL.verbose         = 2
    # Training parameters
    a.LEARN = ub.utils.StateObject()
    a.LEARN.total_timesteps = a.ARGS.n_envs * a.MODEL.n_steps * 5 # ~10M
    a.LEARN.log_interval    = 1    # epoch
    a.LEARN.eval_interval   = 1000 # epoch
    a.LEARN.eval_episodes   = 10
    a.LEARN.eval_max_steps  = 5000
    a.LEARN.save_interval   = 1000 # epoch
    a.LEARN.save_path       = os.path.join(root_path, 'save')
    a.LEARN.tb_logdir       = root_path
    # Performance evaluations
    a.EVAL = ub.utils.StateObject()
    a.EVAL.n_episodes       = 1
    a.EVAL.max_steps        = 10000
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

def make_env(a, rank=0, eval=False):
    # some params for monitering
    monitor_params = dict(
        # filename prefix
        prefix = 'eval' if eval else f'{rank}.train',
        # record every n episodes, None for cubic schedule
        video_kwargs = dict(interval=1 if eval else None)
    )
    env = make_atari_env(a, eval=eval, **monitor_params)
    return env

def main():
    a = parse_args()
    # =============== Reset logger ==============
    ub.logger.Config.use(filename=a.ARGS.logging, level=a.ARGS.log_level,
                    colored=True, reset=False)
    LOG = ub.logger.getLogger('PPO')
    # ========== Print welcome message ==========
    LOG.add_row('')
    LOG.add_rows('PPO', fmt='{:@f:ANSI_Shadow}', align='center')
    LOG.add_line()
    LOG.add_rows(ub.__copyright__)
    LOG.flush('INFO')
    time.sleep(1)
    # ============ Print parameters =============
    print_args(LOG, a.ARGS,  'ARGS')
    print_args(LOG, a.ENV,   'ENV')
    print_args(LOG, a.MODEL, 'MODEL')
    print_args(LOG, a.LEARN, 'LEARN')
    print_args(LOG, a.EVAL, 'EVAL')
    LOG.flush('WARN')
    ub.utils.set_seed(a.ARGS.seed)
    # ================ Make envs ================
    env = ub.envs.SubprocVecEnv([
        functools.partial(make_env, a.ENV, rank=rank, eval=False)
        for rank in range(a.ARGS.n_envs)
    ])
    eval_env = make_env(a.ENV, eval=True)
    env.seed(a.ARGS.seed) # seed ~ seed+n_envs
    eval_env.seed(a.ARGS.eval_seed)
    LOG.debug(f'Action space: {env.action_space}')
    LOG.debug(f'Observation space: {env.observation_space}')
    # =============== Train model ===============
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
        results = loaded_model.eval(eval_env, **a.EVAL)
        metrics = loaded_model.get_eval_metrics(results)
        loaded_model.log_eval(a.EVAL.n_episodes, results, metrics)
        # --- Load model from the best checkpoint ---
        loaded_model = PPO.load(a.LEARN.save_path, best=True)
        # Evaluate model
        LOG.info('Evaluating the best model ...')
        results = loaded_model.eval(eval_env, **a.EVAL)
        metrics = loaded_model.get_eval_metrics(results)
        loaded_model.log_eval(a.EVAL.n_episodes, results, metrics)
    except:
        LOG.exception('Exception occurred')
        env.close()
        eval_env.close()
        exit(1)
    env.close()
    eval_env.close()

if __name__ == '__main__':
    main()