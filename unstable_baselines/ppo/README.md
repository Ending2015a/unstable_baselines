# Proximal Policy Optimization (PPO)

> [Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal Policy Optimization Algorithms. *arXiv:1707.06347*, 2017.](https://arxiv.org/abs/1707.06347)


## How to use
```python
python -m unstable_baselines.ppo.run --rank 0 --seed 1 --logdir='./log/{env_id}/ppo/{rank}' \
               --logging='training.log' --monitor_dir='monitor' --tb_logdir='' --model_dir='model' \
               --env_id="BreakoutNoFrameskip-v0" --num_envs=8 --num_episodes=20000 \
               --num_steps=128 --num_epochs=4 --batch_size=256 --verbose=2
```

Total timesteps (Samples) = num_envs * num_steps * num_episodes (~20M in this case)


## Atari 2600

### Video

| `BeamRiderNoFrameskip-v0` | `BreakoutNoFrameskip-v0` |
|-|-|
|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.BeamRiderNoFrameskip-v0.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.BreakoutNoFrameskip-v0.eval.gif" height=300px>|

### Learning Curve

> Learning curve


### Hyperparametrs
| `env_id` | `num_envs` | `num_episodes` | `num_steps` | `num_epochs` | `batch_size` |
|-|:-:|:-:|:-:|:-:|:-:|
| `BeamRiderNoFrameskip-v0`| 1 | 1000 | 1000 | 1000 | 256 |
| `BreakoutNoFrameskip-v0` | 1 | 1000 | 1000 | 1000 | 256 |

## Architecture

|             | `Box`              | `Discrete`         | `MultiDiscrete` | `MultiBinary` |
|:-----------:|:------------------:|:------------------:|:---------------:|:-------------:|
| Observation | :heavy_check_mark: | :heavy_check_mark: | :x:             | :x:           |
| Action      | :heavy_check_mark: | :heavy_check_mark: | :x:             | :x:           |

<br/>
<br/>


![](https://g.gravizo.com/source/svg/ppo_discrete?https%3A%2F%2Fraw.githubusercontent.com%2FEnding2015a%2Funstable_baselines_assets%2Fmaster%2Fscripts%2Farch%2Fppo.arch.md)
