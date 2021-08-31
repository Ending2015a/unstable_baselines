# --- built in ---
import os
import time
import argparse
import functools
# --- 3rd party ---
import ray
from ray import tune

# --- my module ---
import unstable_baselines as ub
from unstable_baselines.algo.ppo import PPO
from experiments import BaseExperiments
from experiments.ppo.config import default_config

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
        env = make_atari_env(a, eval=eval, **monitor_params)
    else:
        env = make_pybullet_env(a, eval=eval, **monitor_params)
    return env

def evaluate_and_export_final_model(model, eval_env, a):
    results = model.eval(eval_env, a.n_episodes, a.max_steps)
    metrics = model.get_eval_metrics(results)
    model.log_eval(a.n_episodes, results, metrics)
    # export PPO agents (only inference mode)
    ckpt_metrics = model.get_save_metrics(metrics)
    model.agent.save(a.export_path, checkpoint_metrics=ckpt_metrics)

def train(a):
    # =============== Reset logger ==============
    ub.logger.Config.use(filename=a.ARGS.logging, level=a.ARGS.log_level,
                    colored=True, reset=False)
    LOG = ub.logger.getLogger('PPO')
    # ============ Print parameters =============
    print_args(LOG, a.ARGS,  'ARGS')
    print_args(LOG, a.ENV,   'ENV')
    print_args(LOG, a.MODEL, 'MODEL')
    print_args(LOG, a.LEARN, 'LEARN')
    print_args(LOG, a.EVAL,  'EVAL')
    print_args(LOG, a.FINAL, 'FINAL')
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
        model = PPO(env, **a.MODEL).learn(eval_env=eval_env, **a.LEARN)
        LOG.info('DONE')
        # Save model
        saved_path = model.save(a.LEARN.save_path)
        LOG.info(f'Saving model to {saved_path}')
        del model
        # --- Load model from the latest checkpoint ---
        loaded_model = PPO.load(a.LEARN.save_path)
        # Evaluate model
        LOG.info('Evaluating the latest model ...')
        evaluate_and_export_final_model(loaded_model, eval_env, a.EVAL)
        # --- Load model from the best checkpoint ---
        loaded_model = PPO.load(a.LEARN.save_path, best=True)
        # Evaluate model
        LOG.info('Evaluating the best model ...')
        evaluate_and_export_final_model(loaded_model, eval_env, a.EVAL)
    except:
        LOG.exception('Exception occurred')
    env.close()
    eval_env.close()

# === ray ===
class PPOExperiments(BaseExperiments):
    def __init__(self):
        super().__init__(name='ppo')

    def search_config(self):
        return {
            'clipnorm': tune.grid_search([None, 0.5]),
            'dual_clip': tune.grid_search([None, 2.0]),
            'value_clip': tune.grid_search([None, 0.5]),
            'reg_coef': tune.grid_search([0.0, 1e-6]),
        }

    @staticmethod
    def run_experiment(config):
        # Create root logging path
        env_id = config['env_id']
        rank = config['rank']
        root_path = f'{rank}'
        # Get default configuration
        a = default_config(
            env_id    = env_id,
            root_path = root_path,
        )
        # Overwrite configuration
        a.MODEL.update({
            'clipnorm':   config['clipnorm'],
            'dual_clip':  config['dual_clip'],
            'value_clip': config['value_clip'],
            'reg_coef':   config['reg_coef'],
        })
        # Start experiment
        train(a)

if __name__ == '__main__':
    PPOExperiments().run()
