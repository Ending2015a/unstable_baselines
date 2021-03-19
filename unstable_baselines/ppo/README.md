# Proximal Policy Optimization (PPO)

> [Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal Policy Optimization Algorithms. *arXiv:1707.06347*, 2017.](https://arxiv.org/abs/1707.06347)


## How to use
```python
python -m unstable_baselines.ppo.run --rank 0 --seed 1 --logdir='./log/{env_id}/ppo_20m/{rank}' \
               --logging='training.log' --monitor_dir='monitor' --tb_logdir='' --model_dir='model' \
               --env_id="BreakoutNoFrameskip-v0" --num_envs=8 --num_episodes=20000 \
               --num_steps=128 --num_epochs=4 --batch_size=256 --verbose=2
```

Total timesteps (Samples) = num_envs * num_steps * num_episodes (~20M in this case)


## Atari 2600

### Video

> Best video

### Learning Curve

> Learning curve


### Hyperparametrs
| | `num_envs` | `num_episodes` | `num_steps` | `num_epochs` | `batch_size` |
|-|-|-|-|-|-|
| `BeamRiderNoFrameskip-v0`| 1 | 1000 | 1000 | 1000 | 256 |
| `BreakoutNoFrameskip-v0` | 1 | 1000 | 1000 | 1000 | 256 |

## Architecture

|             | `Box` | `Discrete`         | `MultiDiscrete` | `MultiBinary` |
|-------------|-------|--------------------|-----------------|---------------|
| Observation |       | :heavy_check_mark: | :x:             | :x:           |
| Action      |       | :heavy_check_mark: | :x:             | :x:           |


Nature CNN
