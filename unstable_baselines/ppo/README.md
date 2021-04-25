# Proximal Policy Optimization (PPO)

> [John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, & Oleg Klimov. (2017). Proximal Policy Optimization Algorithms.](https://arxiv.org/abs/1707.06347)


## How to use

### Run with default arguments
```python
./unstable_baselines/ppo/train.sh --rank 0 --seed 1 "BreakoutNoFrameskip-v4"
```

### Run multiple environments with default arguments
```python
./unstable_baselines/ppo/train.sh --rank 0 --seed 1 "BreakoutNoFrameskip-v4" "SeaquestNoFrameskip-v4" "PongNoFrameskip-v4"
```

### Atari-like environment (Image observation + discrete action)
```python
python -m unstable_baselines.ppo.run --rank 0 --seed 1 --logdir='./log/{env_id}/ppo/{rank}' \
               --logging='training.log' --monitor_dir='monitor' --tb_logdir='' --model_dir='model' \
               --env_id="BreakoutNoFrameskip-v4" --num_envs=8 --num_epochs=10000 \
               --num_steps=125 --num_subepochs=8 --batch_size=256 --verbose=2 \
               --shared_net --record_video
```
<sup>Enable `shared_net` shares the CNN between policy and value function.</sup><br/>
<sup>Total timesteps (Samples) ≈ num_envs * num_steps * num_epochs (~10M in this case)</sup><br>
<sup>Number of times each sample reused ≈ num_subepochs (~8 in this case)</sup><br>

### Continuous control environment
<!-- ```python
python -m unstable_baselines.ppo.run --rank 0 --seed 1 --logdir='./log/{env_id}/ppo/{rank}' \
               --logging='training.log' --monitor_dir='monitor' --tb_logdir='' --model_dir='model' \
               --env_id="HalfCheetahBulletEnv-v0" --num_envs=1 --num_epochs=1000 \
               --num_steps=1024 --num_subepochs=10 --batch_size=256 --verbose=2 \
               --ent_coef=0.0 --record_video
```
<sup>Total timesteps (Samples) = num_envs * num_steps * num_epochs (~1M in this case)</sup><br>
<sup>Number of times each sample reused = num_subepochs (~10 in this case)</sup><br> -->

## Atari 2600

### Video


| `BeamRiderNoFrameskip-v4` | `BreakoutNoFrameskip-v4` | `PongNoFrameskip-v4`  | `SeaquestNoFrameskip-v4` |
|---------------------------|--------------------------|-----------------------|--------------------------|
|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.BeamRiderNoFrameskip-v4.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.BreakoutNoFrameskip-v4.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.PongNoFrameskip-v4.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.SeaquestNoFrameskip-v4.eval.gif" height=300px>|
| `AsteroidsNoFrameskip-v4` | `EnduroNoFrameskip-v4`   | `QbertNoFrameskip-v4` | `MsPacmanNoFrameskip-v4` |
|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.AsteroidsNoFrameskip-v4.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.EnduroNoFrameskip-v4.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.QbertNoFrameskip-v4.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.MsPacmanNoFrameskip-v4.eval.gif" height=300px>|


### Results

> Learning curve

| `env_id`                  | Max rewards | Mean rewards | Std rewards | Train samples | Train seed | Eval episodes | Eval seed |
|---------------------------|------------:|-------------:|------------:|--------------:|-----------:|--------------:|----------:|
| `AsteroidsNoFrameskip-v4` |        1570 |         1072 |      281.73 |           20M |        1~8 |            20 |         0 |
| `BeamRiderNoFrameskip-v4` |        2832 |       1513.4 |      647.36 |           20M |        1~8 |            20 |         0 |
| `BreakoutNoFrameskip-v4`  |         368 |       131.85 |      118.28 |           20M |        1~8 |            20 |         0 |
| `EnduroNoFrameskip-v4`    |         302 |        189.2 |       29.79 |           20M |        1~8 |            20 |         0 |
| `MsPacmanNoFrameskip-v4`  |        2650 |       2035.5 |       463.1 |           20M |        1~8 |            20 |         0 |
| `PongNoFrameskip-v4`      |          21 |           21 |           0 |           20M |        1~8 |            20 |         0 |
| `QbertNoFrameskip-v4`     |       16925 |     16441.25 |      259.23 |           20M |        1~8 |            20 |         0 |
| `SeaquestNoFrameskip-v4`  |        1760 |         1750 |       17.32 |           20M |        1~8 |            20 |         0 |

<sup>M = million (1e6)</sup><br>

### Hyperparametrs

| `env_id`        | `AsteroidsNoFrameskip-v4` | `BeamRiderNoFrameskip-v4` | `BreakoutNoFrameskip-v4` | `EnduroNoFrameskip-v4` | `MsPacmanNoFrameskip-v4` | `PongNoFrameskip-v4` | `QbertNoFrameskip-v4` | `SeaquestNoFrameskip-v4` |
|-----------------|:-------------------------:|:-------------------------:|:------------------------:|:----------------------:|:------------------------:|:--------------------:|:---------------------:|:------------------------:|
| `num_envs`      |             8             |             8             |             8            |            8           |             8            |           8          |           8           |             8            |
| `num_epochs`    |           10000           |           10000           |           10000          |          10000         |           10000          |         10000        |         10000         |           10000          |
| `num_steps`     |            125            |            125            |            125           |           125          |            125           |          125         |          125          |            125           |
| `num_subepochs` |             8             |             8             |             8            |            8           |             8            |           8          |           8           |             8            |
| `batch_size`    |            256            |            256            |            256           |           256          |            256           |          256         |          256          |            256           |
| `ent_coef`      |            0.01           |            0.01           |           0.01           |          0.01          |           0.01           |         0.01         |          0.01         |           0.01           |
| `vf_coef`       |            0.5            |            0.5            |            0.5           |           0.5          |            0.5           |          0.5         |          0.5          |            0.5           |
| `shared_net`    |     :heavy_check_mark:    |     :heavy_check_mark:    |    :heavy_check_mark:    |   :heavy_check_mark:   |    :heavy_check_mark:    |  :heavy_check_mark:  |   :heavy_check_mark:  |    :heavy_check_mark:    |


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


<!-- | `env_id`                | `num_envs` | `num_episodes` | `num_steps` | `num_epochs` | `batch_size` | `ent_coef` | `vf_coef` | `shared_net`  |
| ----------------------- |:----------:|:--------------:|:-----------:|:------------:|:------------:|:----------:|:---------:|:-------------:|
|`HalfCheetahBulletEnv-v0`| 1          | 1000           | 2000        | 10           | 200          | 0.0        | 0.5       | :x:           |
|`AntBulletEnv-v0`        | 1          | 1000           | 2000        | 10           | 200          | 0.0        | 0.5       | :x:           |
|`HopperBulletEnv-v0`     | 1          | 1000           | 2000        | 10           | 200          | 0.0        | 0.5       | :x:           |
|`Walker2DBulletEnv-v0`   | 32         | 1000           | 512         | 15           | 4096         | 0.0        | 0.5       | :x:           |
|`HumanoidBulletEnv-v0`   | 32         | 1000           | 512         | 15           | 4096         | 0.0        | 0.5       | :x:           | -->

## Architecture

|             | `Box`              | `Discrete`         | `MultiDiscrete` | `MultiBinary` |
|:-----------:|:------------------:|:------------------:|:---------------:|:-------------:|
| Observation | :heavy_check_mark: | :x:                | :x:             | :x:           |
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