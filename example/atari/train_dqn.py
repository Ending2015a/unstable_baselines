# This example shows how to train an PPO agent on atari domain
# For complete experiments, please refer to 
#     experiments/dqn/run.py
# --- built in ---
import os
import time
import argparse
import functools
# --- 3rd party ---
import gym
# --- my module ---
import unstable_baselines as ub
from unstable_baselines.algo.dqn import DQN

def parse_config(env_id, root_path):
    env_id    = env_id
    root_path = os.path.join(root_path, env_id)
    # Create config sections (you can use python dict instead)
    Config = ub.utils.StateObject
    a = Config()
    a.ARGS  = Config()
    a.ENV   = Config()
    a.MODEL = Config()
    a.LEARN = Config()
    a.EVAL  = Config()
    # Parameters
    a.ARGS.logging          = f'{root_path}/training.log'
    a.ARGS.log_level        = 'INFO'
    a.ARGS.seed             = 1
    a.ARGS.eval_seed        = 0
    a.ARGS.n_envs           = 8
    # Env/Monitor parameters
    a.ENV.env_id            = env_id
    a.ENV.monitor_dir       = f'{root_path}/monitor'
    a.ENV.video             = True   # record video
    # Hyper parameters
    a.MODEL.learning_rate   = 3e-4
    a.MODEL.buffer_size     = int(1e6)
    a.MODEL.multi_step      = 1
    a.MODEL.gamma           = 0.99
    a.MODEL.tau             = 1.0
    a.MODEL.reg_coef        = 0.0
    a.MODEL.clipnorm        = 0.5
    a.MODEL.explore_rate    = 0.3
    a.MODEL.huber           = True
    a.MODEL.huber_rate      = 0.1
    a.MODEL.prioritized     = False
    a.MODEL.prio_alpha      = 0.6
    a.MODEL.prio_beta       = 0.4
    a.MODEL.dueling         = False
    a.MODEL.force_mlp       = False
    a.MODEL.mlp_units       = [64, 64]
    a.MODEL.n_steps         = 4
    a.MODEL.n_gradsteps     = 1
    a.MODEL.warmup_steps    = int(1e4)
    a.MODEL.batch_size      = 256
    # Training parameters
    a.LEARN.total_timesteps = a.ARGS.n_envs * a.MODEL.n_steps * 312500 # ~10M
    a.LEARN.target_update   = 625  # gradstep
    a.LEARN.log_interval    = 1000 # epoch
    a.LEARN.eval_interval   = 10000 # epoch
    a.LEARN.eval_episodes   = 5
    a.LEARN.eval_max_steps  = 5000
    a.LEARN.save_interval   = 10000 # epoch
    a.LEARN.save_path       = f'{root_path}/save'
    a.LEARN.tb_logdir       = root_path
    a.LEARN.verbose         = 3
    # Performance evaluations
    a.EVAL.n_episodes       = 10
    a.EVAL.max_steps        = 10000
    a.EVAL.export_path      = f'{root_path}/export'
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

def evaluate_and_export_final_model(model, eval_env, a):
    results = model.eval(eval_env, a.n_episodes, a.max_steps)
    metrics = model.get_eval_metrics(results)
    model.log_eval(results, metrics, verbose=3)
    # export PPO agents (only inference mode)
    ckpt_metrics = model.get_save_metrics(metrics)
    model.agent.save(a.export_path, checkpoint_metrics=ckpt_metrics)

def main(a):
    # =============== Reset logger ==============
    ub.logger.Config.use(filename=a.ARGS.logging, level=a.ARGS.log_level,
                    colored=True, reset=False)
    LOG = ub.logger.getLogger('DQN')
    # ========== Print welcome message ==========
    LOG.add_row('')
    LOG.add_rows('DQN', fmt='{:@f:ANSI_Shadow}', align='center')
    LOG.add_line()
    LOG.add_rows(ub.__copyright__)
    LOG.flush('INFO')
    time.sleep(1)
    # ============ Print parameters =============
    print_args(LOG, a.ARGS,  'ARGS')
    print_args(LOG, a.ENV,   'ENV')
    print_args(LOG, a.MODEL, 'MODEL')
    print_args(LOG, a.LEARN, 'LEARN')
    print_args(LOG, a.EVAL,  'EVAL')
    LOG.flush('WARN')
    # ================ Make envs ================
    env = ub.envs.SubprocVecEnv([
        functools.partial(make_env, a.ENV, rank=rank, eval=False)
        for rank in range(a.ARGS.n_envs)
    ])
    eval_env = make_env(a.ENV, eval=True)
    env.seed(a.ARGS.seed) # seed ~ seed+n_envs
    eval_env.seed(a.ARGS.eval_seed)
    ub.utils.set_seed(a.ARGS.seed)
    # =============== Train model ===============
    try:
        # --- Setup model & train ---
        model = DQN(env, **a.MODEL).learn(eval_env=eval_env, **a.LEARN)
        LOG.info('DONE')
        # Save model
        saved_path = model.save(a.LEARN.save_path)
        LOG.info(f'Saving model to {saved_path}')
        del model
        # --- Load model from the latest checkpoint ---
        loaded_model = DQN.load(a.LEARN.save_path)
        # Evaluate model
        LOG.info('Evaluating the latest model ...')
        evaluate_and_export_final_model(loaded_model, eval_env, a.EVAL)
        # --- Load model from the best checkpoint ---
        loaded_model = DQN.load(a.LEARN.save_path, best=True)
        # Evaluate model
        LOG.info('Evaluating the best model ...')
        evaluate_and_export_final_model(loaded_model, eval_env, a.EVAL)
    except:
        LOG.exception('Exception occurred')
    env.close()
    eval_env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Double Deep Q-learning')
    parser.add_argument('--env_id', type=str, default='BeamRiderNoFrameskip-v4')
    parser.add_argument('--root',   type=str, default='log/ppo')
    args = parser.parse_args()
    main(parse_config(args.env_id, args.root))