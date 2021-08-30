# --- built in ---
import os
import time
import argparse
# --- 3rd party ---

# --- my module ---
import unstable_baselines as ub

Config = ub.utils.StateObject

def base_config():
    a = Config()
    a.ARGS  = Config()
    a.ENV   = Config()
    a.MODEL = Config()
    a.LEARN = Config()
    a.EVAL  = Config()
    a.FINAL = Config()
    return a

def default_atari_config(env_id, root_path):
    a = base_config()
    # Configurations
    a.ARGS.logging          = os.path.join(root_path, 'training.log')
    a.ARGS.log_level        = 'INFO'
    a.ARGS.seed             = 1
    a.ARGS.eval_seed        = 0
    a.ARGS.n_envs           = 8
    # Env/Monitor parameters
    a.ENV.env_id            = env_id
    a.ENV.monitor_dir       = os.path.join(root_path, 'monitor')
    a.ENV.video             = True
    # Hyper parameters
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
    a.MODEL.share_net       = True
    a.MODEL.force_mlp       = False
    a.MODEL.mlp_units       = [64, 64]
    a.MODEL.n_steps         = 125
    a.MODEL.n_subepochs     = 8
    a.MODEL.batch_size      = 256
    a.MODEL.verbose         = 2
    # Training parameters
    a.LEARN.total_timesteps = a.ARGS.n_envs * a.MODEL.n_steps * 10000 # ~10M
    a.LEARN.log_interval    = 1    # epoch
    a.LEARN.eval_interval   = 1000 # epoch
    a.LEARN.eval_episodes   = 10
    a.LEARN.eval_max_steps  = 5000
    a.LEARN.save_interval   = 1000 # epoch
    a.LEARN.save_path       = os.path.join(root_path, 'save')
    a.LEARN.tb_logdir       = root_path
    # Performance evaluations
    a.EVAL.n_episodes       = 100
    a.EVAL.max_steps        = 10000
    a.FINAL.export_path     = os.path.join(root_path, 'export')
    return a

def default_pybullet_config(env_id, root_path):
    a = base_config()
    # Configurations
    a.ARGS.logging          = os.path.join(root_path, 'training.log')
    a.ARGS.log_level        = 'INFO'
    a.ARGS.seed             = 1
    a.ARGS.eval_seed        = 0
    a.ARGS.n_envs           = 4
    # Env/Monitor parameters
    a.ENV.env_id            = env_id
    a.ENV.monitor_dir       = os.path.join(root_path, 'monitor')
    a.ENV.video             = True
    # Hyper parameters
    a.MODEL.learning_rate   = 3e-4
    a.MODEL.gamma           = 0.99
    a.MODEL.gae_lambda      = 0.95
    a.MODEL.policy_clip     = 0.2
    a.MODEL.value_clip      = None
    a.MODEL.dual_clip       = None
    a.MODEL.ent_coef        = 0.0
    a.MODEL.vf_coef         = 0.5
    a.MODEL.reg_coef        = 0.0
    a.MODEL.clipnorm        = 0.5
    a.MODEL.target_kl       = None
    a.MODEL.share_net       = True
    a.MODEL.force_mlp       = False
    a.MODEL.mlp_units       = [256, 256]
    a.MODEL.n_steps         = 250
    a.MODEL.n_subepochs     = 8
    a.MODEL.batch_size      = 256
    a.MODEL.verbose         = 2
    # Training parameters
    a.LEARN.total_timesteps = a.ARGS.n_envs * a.MODEL.n_steps * 2000 # ~2M
    a.LEARN.log_interval    = 1    # epoch
    a.LEARN.eval_interval   = 200 # epoch
    a.LEARN.eval_episodes   = 10
    a.LEARN.eval_max_steps  = 1000
    a.LEARN.save_interval   = 200 # epoch
    a.LEARN.save_path       = os.path.join(root_path, 'save')
    a.LEARN.tb_logdir       = root_path
    # Performance evaluations
    a.EVAL.n_episodes       = 100
    a.EVAL.max_steps        = 1000
    a.FINAL.export_path     = os.path.join(root_path, 'export')
    return a

def default_config(env_id, **kwargs):
    if 'NoFrameskip' in env_id:
        a = default_atari_config(env_id=env_id, **kwargs)
    else:
        a = default_pybullet_config(env_id=env_id, **kwargs)
    return a