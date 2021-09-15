# --- built in ---
import os
import time
import argparse
# --- 3rd party ---
import gym
import numpy as np
# --- my module ---
import unstable_baselines as ub
from unstable_baselines.algo.dqn import DQN

def main():
    ub.logger.Config.use(level='DEBUG', colored=True)
    LOG = ub.logger.getLogger('DQN')
    ub.utils.set_seed(1)
    # create envs
    env = ub.envs.VecEnv([gym.make('CartPole-v0') for i in range(10)])
    env.seed(1)
    eval_env = gym.make('CartPole-v0')
    eval_env.seed(0)
    try:
        model = DQN(
            env,
            buffer_size=20000,
            multi_step=3,
            learning_rate=1e-3,
            gamma=0.8,
            mlp_units=[128, 128],
            batch_size=64,
            n_steps=10,
            verbose=0
        ).learn(
            200000,
            log_interval=1000,
            target_update=100
        )
        # evaluate model
        results = model.eval(eval_env, 20, 200)
        metrics = model.get_eval_metrics(results)
        model.log_eval(20, results, metrics)
    except:
        LOG.exception('Exception occurred')
    env.close()
    eval_env.close()

if __name__ == '__main__':
    main()
