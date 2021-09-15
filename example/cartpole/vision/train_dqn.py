# --- built in ---
import os
import time
import argparse
# --- 3rd party ---
import gym
# --- my module ---
import unstable_baselines as ub
from unstable_baselines.algo.dqn import DQN

class ImageCartPole(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                    shape=(400, 600, 3), dtype=np.uint8)
        self.action_space = env.action_space
    
    def reset(self, **kwargs):
        super().reset(**kwargs)
        return self.env.render('rgb_array')
    
    def step(self, act):
        obs, rew, done, info = super().step(act)
        return self.env.render('rgb_array'), rew, done, info

def make_cartpole_env():
    env = gym.make('CartPole-v0')
    env = ImageCartPole(env)
    env = ub.envs.WarpFrame(env)
    env = ub.envs.FrameStack(env)
    return env

def main():
    ub.logger.Config.use(level='DEBUG', colored=True)
    LOG = ub.logger.getLogger('DQN')
    # create envs
    env = ub.envs.VecEnv([make_cartpole_env()])
    eval_env = make_cartpole_env()
    try:
        model = DQN(env).learn(
            200000,
            log_interval=1000,
            eval_env=eval_env, 
            target_update=100
        )
        LOG.info('Training done, evaluate model')
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