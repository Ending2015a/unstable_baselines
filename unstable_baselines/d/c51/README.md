# Categorical DQN (C51)

> [Bellemare, M., Dabney, W., & Munos, R. (2017). A Distributional Perspective on Reinforcement Learning. *In Proceedings of the 34th International Conference on Machine Learning.*](https://arxiv.org/abs/1707.06887)


## How to use

### Run with default arguments
```python
./unstable_baselines/d/c51/train.sh --rank 0 --seed 1 "BreakoutNoFrameskip-v4"
```

### Run multiple environments with default arguments
```python
./unstable_baselines/d/c51/train.sh --rank 0 --seed 1 "BreakoutNoFrameskip-v4" "SeaquestNoFrameskip-v4"
```

### Atari-like environment (Image observation + discrete action)
```python
python -m unstable_baselines.d.c51.run --rank 0 --seed 1 --logdir='./log/{env_id}/dqn/{rank}' \
               --logging='training.log' --monitor_dir='monitor' --tb_logdir='' --model_dir='model' \
               --env_id="BreakoutNoFrameskip-v4" --num_envs=8 --num_epochs=312500 \
               --num_steps=4 --num_gradsteps=1 --batch_size=256 --target_update=625 \
               --explore_rate=1.0 --explore_final=0.05 --explore_progress=0.1 \
               --v_min=-10 --v_max=10 --num_atoms=51 --record_video
```

<sup>Total timesteps (Samples) ≈ num_envs * num_steps * num_epochs (~10M in this case)</sup><br>
<sup>Number of times each sample reused ≈ batch_size/num_steps * num_gradsteps/num_envs (~8 in this case)</sup><br>

## Atari 2600

### Video

| `BeamRiderNoFrameskip-v4` | `BreakoutNoFrameskip-v4` | `PongNoFrameskip-v4` | `SeaquestNoFrameskip-v4` |
|---------------------------|--------------------------|----------------------|--------------------------|
|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/c51.BeamRiderNoFrameskip-v4.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/c51.BreakoutNoFrameskip-v4.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/c51.PongNoFrameskip-v4.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/c51.SeaquestNoFrameskip-v4.eval.gif" height=300px>|
| `AsteroidsNoFrameskip-v4` | `EnduroNoFrameskip-v4`   | `QbertNoFrameskip-v4` | `MsPacmanNoFrameskip-v4` |
|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/c51.AsteroidsNoFrameskip-v4.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/c51.EnduroNoFrameskip-v4.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/c51.QbertNoFrameskip-v4.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/c51.MsPacmanNoFrameskip-v4.eval.gif" height=300px>|

### Results

> Learning curve

| `env_id`                  | Max rewards | Mean rewards | Std rewards | Train samples | Train seeds | Eval episodes | Eval seed |
|---------------------------|------------:|-------------:|------------:|--------------:|------------:|--------------:|----------:|
| `AsteroidsNoFrameskip-v4` |        1980 |          752 |      332.68 |           10M |         1~8 |            20 |         0 |
| `BeamRiderNoFrameskip-v4` |       11500 |       7791.1 |     2570.17 |           10M |         1~8 |            20 |         0 |
| `BreakoutNoFrameskip-v4`  |         424 |       393.05 |       25.72 |           10M |         1~8 |            20 |         0 |
| `EnduroNoFrameskip-v4`    |        2229 |       1726.1 |      310.23 |           10M |         1~8 |            20 |         0 |
| `MsPacmanNoFrameskip-v4`  |        3060 |       2796.5 |      316.31 |           10M |         1~8 |            20 |         0 |
| `PongNoFrameskip-v4`      |          21 |         20.8 |         0.6 |           10M |         1~8 |            20 |         0 |
| `QbertNoFrameskip-v4`     |       16550 |     15978.75 |      373.35 |           10M |         1~8 |            20 |         0 |
| `SeaquestNoFrameskip-v4`  |        9660 |         8307 |      665.29 |           10M |         1~8 |            20 |         0 |

<sup>M = million (1e6)</sup><br>

### Hyperparameters


| `env_id`           | `AsteroidsNoFrameskip-v4` | `BeamRiderNoFrameskip-v4` | `BreakoutNoFrameskip-v4` | `EnduroNoFrameskip-v4` | `MsPacmanNoFrameskip-v4` | `PongNoFrameskip-v4` | `QbertNoFrameskip-v4` | `SeaquestNoFrameskip-v4` |
|--------------------|:-------------------------:|:-------------------------:|:------------------------:|:----------------------:|:------------------------:|:--------------------:|:---------------------:|:------------------------:|
| `num_envs`         |             8             |             8             |             8            |            8           |             8            |           8          |           8           |             8            |
| `num_epochs`       |           312500          |           312500          |          312500          |         312500         |          312500          |        312500        |         312500        |          312500          |
| `num_steps`        |             4             |             4             |             4            |            4           |             4            |           4          |           4           |             4            |
| `num_gradsteps`    |             1             |             1             |             1            |            1           |             1            |           1          |           1           |             1            |
| `batch_size`       |            256            |            256            |            256           |           256          |            256           |          256         |          256          |            256           |
| `target_update`    |            625            |            625            |            625           |           625          |            625           |          625         |          625          |            625           |
| `exploration`      |     Linear(1.0, 0.05)     |     Linear(1.0, 0.05)     |     Linear(1.0, 0.05)    |    Linear(1.0, 0.05)   |     Linear(1.0, 0.05)    |   Linear(1.0, 0.05)  |   Linear(1.0, 0.05)   |     Linear(1.0, 0.05)    |
| `explore_progress` |            0.1            |            0.1            |            0.1           |           0.1          |            0.1           |          0.1         |          0.1          |            0.1           |
| `v_min`            |           -10.0           |           -10.0           |           -10.0          |          -10.0         |           -10.0          |         -10.0        |         -10.0         |           -10.0          |
| `v_max`            |            10.0           |            10.0           |           10.0           |          10.0          |           10.0           |         10.0         |          10.0         |           10.0           |
| `num_atoms`        |             51            |             51            |            51            |           51           |            51            |          51          |           51          |            51            |

## Architecture

|             | `Box`              | `Discrete`         | `MultiDiscrete` | `MultiBinary` |
|:-----------:|:------------------:|:------------------:|:---------------:|:-------------:|
| Observation | :heavy_check_mark: | :x:                | :x:             | :x:           |
| Action      | :x:                | :heavy_check_mark: | :x:             | :x:           |

<br/>
<br/>

### Atari-like environment
![](https://g.gravizo.com/source/svg/c51_discrete?https%3A%2F%2Fraw.githubusercontent.com%2FEnding2015a%2Funstable_baselines_assets%2Fmaster%2Fscripts%2Farch%2Fc51.arch.md)
<br/>
<sup>force_mlp=`False`</sup><br>


![](https://g.gravizo.com/source/svg/c51_continuous?https%3A%2F%2Fraw.githubusercontent.com%2FEnding2015a%2Funstable_baselines_assets%2Fmaster%2Fscripts%2Farch%2Fc51.arch.md)
<br/>
<sup>force_mlp=`True`</sup><br>
