# This example shows how to train a built in DQN model to play CartPole-v0.
# Example usage:
#     python -m example.cartpole.train_dqn

# --- built in ---
import time
import json
# --- 3rd party ---
import gym
import tensorflow as tf
# run on cpu
tf.config.set_visible_devices([], 'GPU')
# --- my module ---
import unstable_baselines as ub
from unstable_baselines.algo.ppo import PPO

def main():
    ub.logger.Config.use(level='INFO', colored=True)
    ub.utils.set_seed(1)
    # create envs
    env = ub.envs.VecEnv([gym.make('CartPole-v0') for _ in range(10)])
    env.seed(1)
    eval_env = gym.make('CartPole-v0')
    eval_env.seed(0)
    start_time = time.time()
    # create and train model
    model = PPO(
        env,
        learning_rate=1e-3,
        gamma=0.8,
        batch_size=128,
        n_steps=500,
    ).learn(
        20000,
        verbose=1
    )
    # evaluate model
    results = model.eval(eval_env, 20, 200)
    metrics = model.get_eval_metrics(results)
    print(json.dumps(metrics))
    print('time spent:', time.time()-start_time)
    env.close()
    eval_env.close()

if __name__ == '__main__':
    main()
