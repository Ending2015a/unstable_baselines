# --- built in ---
import os
import time
import argparse
import functools
from datetime import datetime
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

def evaluate_and_export_final_model(a, model, eval_env, stats_path):
    results = model.eval(eval_env, a.n_episodes, a.max_steps)
    metrics = model.get_eval_metrics(results)
    model.log_eval(results, metrics, verbose=3)
    # export PPO agents (only inference mode)
    ckpt_metrics = model.get_save_metrics(metrics)
    model.agent.save(a.export_path, checkpoint_metrics=ckpt_metrics)
    ub.utils.safe_json_dump(stats_path, metrics)

def main(a):
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
        evaluate_and_export_final_model(a.EVAL, loaded_model, eval_env, 'metrics.json')
        # --- Load model from the best checkpoint ---
        loaded_model = PPO.load(a.LEARN.save_path, best=True)
        # Evaluate model
        LOG.info('Evaluating the best model ...')
        evaluate_and_export_final_model(a.EVAL, loaded_model, eval_env, 'metrics_best.json')
    except:
        LOG.exception('Exception occurred')
    env.close()
    eval_env.close()

# === ray ===
class PPOExperiments(BaseExperiments):
    def __init__(self):
        super().__init__(name='ppo')

    def parse_args(self, parser=None):
        if parser is None:
            parser = self.add_args()
        parser = super().add_args(parser)
        return super().parse_args(parser)

    def add_args(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser('PPO experiments')
        parser.add_argument('--seed', type=int, default=1)
        parser.add_argument('--eval_seed', type=int, default=0)
        parser.add_argument('--n_envs', type=int, default=8)
        parser.add_argument('--video', action='store_true', default=False)
        parser.add_argument('--train_steps', type=int, default=int(1e7)) #10M
        return parser

    def search_config(self):
        return dict(
            ARGS = {
                'seed': self.args.seed,
                'eval_seed': self.args.eval_seed,
                'n_envs': self.args.n_envs
            },
            ENV = {
                'video': self.args.video
            },
            MODEL = {
                'learning_rate': 3e-4,
                'policy_clip': 0.05, #tune.grid_search([0.05, 0.1, 0.2, 0.4, 0.8]),
                'value_clip': 0.4, #tune.grid_search([None, 0.2, 0.4, 0.6, 0.8]),
                'dual_clip': None, #tune.grid_search([None, 2.0, 3.0, 4.0]),
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'reg_coef': 0.0, #tune.grid_search([0.0, 1e-6]),
                'clipnorm': None, #tune.grid_search([None, 0.5, 1.0, 5.0]),
            },
            LEARN = {
                'total_timesteps': int(self.args.train_steps),
            },
            EVAL = {
                'n_episodes': 100,
                'max_steps': 10000
            }
        )

    @staticmethod
    def run_experiment(config):
        # Create root logging path
        env_id = config['env_id']
        root_path = './'
        # Get default configuration
        a = default_config(
            env_id    = env_id,
            root_path = root_path,
        )
        # Overwrite configuration
        a.ARGS.update(config['ARGS'])
        a.ENV.update(config['ENV'])
        a.MODEL.update(config['MODEL'])
        a.LEARN.update(config['LEARN'])
        a.EVAL.update(config['EVAL'])
        # Start experiment
        main(a)

if __name__ == '__main__':
    PPOExperiments().run()
