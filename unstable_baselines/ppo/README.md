# Proximal Policy Optimization (PPO)

> [Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal Policy Optimization Algorithms. *arXiv:1707.06347*, 2017.](https://arxiv.org/abs/1707.06347)


## How to use

### Run with default arguments
```python
./train.sh --rank 0 --seed 1 "BreakoutNoFrameskip-v4"
```

### Run multiple environments with default arguments
```python
./train.sh --rank 0 --seed 1 "BreakoutNoFrameskip-v4" "SeaquestNoFrameskip-v4"
```

### Atari-like environment (Image observation + discrete action)
```python
python -m unstable_baselines.ppo.run --rank 0 --seed 1 --logdir='./log/{env_id}/ppo/{rank}' \
               --logging='training.log' --monitor_dir='monitor' --tb_logdir='' --model_dir='model' \
               --env_id="BreakoutNoFrameskip-v4" --num_envs=8 --num_episodes=20000 \
               --num_steps=128 --num_epochs=4 --batch_size=256 --verbose=2 \
               --shared_net --record_video
```
<sup>Enable `shared_net` shares the CNN between policy and value function.</sup><br/>
<sup>Total timesteps (Samples) = num_envs * num_steps * num_episodes (~20M in this case)</sup><br>

### Continuous control environment
```python
python -m unstable_baselines.ppo.run --rank 0 --seed 1 --logdir='./log/{env_id}/ppo/{rank}' \
               --logging='training.log' --monitor_dir='monitor' --tb_logdir='' --model_dir='model' \
               --env_id="HalfCheetahBulletEnv-v0" --num_envs=1 --num_episodes=1000 \
               --num_steps=1024 --num_epochs=10 --batch_size=256 --verbose=2 \
               --ent_coef=0.0 --record_video
```
<sup>Total timesteps (Samples) = num_envs * num_steps * num_episodes (~1M in this case)</sup><br>

## Atari 2600

### Video

| `BeamRiderNoFrameskip-v0` | `BreakoutNoFrameskip-v0` |
|-|-|
|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.BeamRiderNoFrameskip-v0.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.BreakoutNoFrameskip-v0.eval.gif" height=300px>|

### Learning Curve

> Learning curve


### Hyperparametrs
| `env_id`                | `num_envs` | `num_episodes` | `num_steps` | `num_epochs` | `batch_size` | `ent_coef` | `vf_coef` | `shared_net`       |
| ----------------------- |:----------:|:--------------:|:-----------:|:------------:|:------------:|:----------:|:---------:|:------------------:|
|`BeamRiderNoFrameskip-v0`| 8          | 20000          | 128         | 4            | 256          | 0.01       | 0.5       | :heavy_check_mark: |
|`BreakoutNoFrameskip-v0` | 8          | 20000          | 128         | 4            | 256          | 0.01       | 0.5       | :heavy_check_mark: |


## Pybullet

### Video

| Environment | Video |
|-|-|
|`HalfCheetahBulletEnv-v0`|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.HalfCheetahBulletEnv-v0.eval.gif" width=300px/>|
| `AntBulletEnv-v0`|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.AntBulletEnv-v0.eval.gif" width=300px/>|
|`HopperBulletEnv-v0`  |<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.HopperBulletEnv-v0.eval.gif" width=300px/>|
|`Walker2DBulletEnv-v0`|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.Walker2DBulletEnv-v0.eval.gif" width=300px/>|
|`HumanoidBulletEnv-v0`|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.HumanoidBulletEnv-v0.eval.gif" width=300px/>|

### Learning Curve

> Learning curve


### Hyperparametrs
| `env_id`                | `num_envs` | `num_episodes` | `num_steps` | `num_epochs` | `batch_size` | `ent_coef` | `vf_coef` | `shared_net`  |
| ----------------------- |:----------:|:--------------:|:-----------:|:------------:|:------------:|:----------:|:---------:|:-------------:|
|`HalfCheetahBulletEnv-v0`| 1          | 1000           | 2000        | 10           | 200          | 0.0        | 0.5       | :x:           |
|`AntBulletEnv-v0`        | 1          | 1000           | 2000        | 10           | 200          | 0.0        | 0.5       | :x:           |
|`HopperBulletEnv-v0`     | 1          | 1000           | 2000        | 10           | 200          | 0.0        | 0.5       | :x:           |
|`Walker2DBulletEnv-v0`   | 32         | 1000           | 512         | 15           | 4096         | 0.0        | 0.5       | :x:           |
|`HumanoidBulletEnv-v0`   | 32         | 1000           | 512         | 15           | 4096         | 0.0        | 0.5       | :x:           |

## Architecture

|             | `Box`              | `Discrete`         | `MultiDiscrete` | `MultiBinary` |
|:-----------:|:------------------:|:------------------:|:---------------:|:-------------:|
| Observation | :heavy_check_mark: | :heavy_check_mark: | :x:             | :x:           |
| Action      | :heavy_check_mark: | :heavy_check_mark: | :x:             | :x:           |

<br/>
<br/>

### Atari-like environment
![](https://g.gravizo.com/source/svg/ppo_discrete?https%3A%2F%2Fraw.githubusercontent.com%2FEnding2015a%2Funstable_baselines_assets%2Fmaster%2Fscripts%2Farch%2Fppo.arch.md)
<br/>
<sup>`shared_net=True`</sup><br/>

### Continuous control environment
![](https://g.gravizo.com/source/svg/ppo_continuous?https%3A%2F%2Fraw.githubusercontent.com%2FEnding2015a%2Funstable_baselines_assets%2Fmaster%2Fscripts%2Farch%2Fppo.arch.md)
<br/>
<sup>`shared_net=False`</sup><br/>