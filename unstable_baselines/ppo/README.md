# PPO
* Original paper: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)


## How to use
```python
python -m unstable_baselines.ppo.discrete --rank 0 --seed 1 --logdir='./log/{env_id}/ppo_20m/{rank}' \
               --logging='training.log' --monitor_dir='monitor' --tb_logdir='' --model_dir='model' \
               --env_id="BreakoutNoFrameskip-v0" --num_envs=8 --num_episodes=20000 \
               --num_steps=128 --num_epochs=4 --batch_size=256 --verbose=2
```

Total timesteps (Samples) = num_envs * num_steps * num_episodes (~20M in this case)