# Implicit Quantile Networks (IQN) (Early Access)

> [Dabney, W., Ostrovski, G., Silver, D., & Munos, R. (2018). Implicit Quantile Networks for Distributional Reinforcement Learning. In *Proceedings of the 35th International Conference on Machine Learning*](https://arxiv.org/abs/1806.06923)


## How to use

### Run with default arguments
```python
./unstable_baselines/d/iqn/train.sh --rank 0 --seed 1 "BreakoutNoFrameskip-v4"
```

### Run multiple environments with default arguments
```python
./unstable_baselines/d/iqn/train.sh --rank 0 --seed 1 "BreakoutNoFrameskip-v4" "SeaquestNoFrameskip-v4"
```

### Atari-like environment (Image observation + discrete action)
```python
python -m unstable_baselines.d.iqn.run --rank 0 --seed 1 --logdir='./log/{env_id}/iqn/{rank}' \
               --logging='training.log' --monitor_dir='monitor' --tb_logdir='' --model_dir='model' \
               --env_id="BreakoutNoFrameskip-v4" --num_envs=8 --num_epochs=312500 \
               --num_steps=4 --num_gradsteps=1 --batch_size=256 --target_update=625 \
               --explore_rate=1.0 --explore_final=0.05 --explore_progress=0.1 \
               --num_taus=64 --num_target_taus=64 --num_ks=32 --embed_size=64 --record_video
```

<sup>Total timesteps (Samples) ≈ num_envs * num_steps * num_epochs (~10M in this case)</sup><br>
<sup>Number of times each sample reused ≈ batch_size/num_steps * num_gradsteps/num_envs (~8 in this case)</sup><br>

## Atari 2600

### Video

| `BeamRiderNoFrameskip-v4` | `BreakoutNoFrameskip-v4` | `PongNoFrameskip-v4` | `SeaquestNoFrameskip-v4` |
|---------------------------|--------------------------|----------------------|--------------------------|
|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/iqn.BeamRiderNoFrameskip-v4.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/iqn.BreakoutNoFrameskip-v4.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/iqn.PongNoFrameskip-v4.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/iqn.SeaquestNoFrameskip-v4.eval.gif" height=300px>|
| `AsteroidsNoFrameskip-v4` | `EnduroNoFrameskip-v4`   | `QbertNoFrameskip-v4` | `MsPacmanNoFrameskip-v4` |
|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/iqn.AsteroidsNoFrameskip-v4.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/iqn.EnduroNoFrameskip-v4.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/iqn.QbertNoFrameskip-v4.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/iqn.MsPacmanNoFrameskip-v4.eval.gif" height=300px>|

### Results

> Learning curve

| `env_id`                  | Max rewards | Mean rewards | Std rewards | Train samples | Train seeds | Eval episodes | Eval seed |
|---------------------------|------------:|-------------:|------------:|--------------:|------------:|--------------:|----------:|
| `AsteroidsNoFrameskip-v4` |             |              |             |           10M |         1~8 |            20 |         0 |
| `BeamRiderNoFrameskip-v4` |             |              |             |           10M |         1~8 |            20 |         0 |
| `BreakoutNoFrameskip-v4`  |             |              |             |           10M |         1~8 |            20 |         0 |
| `EnduroNoFrameskip-v4`    |             |              |             |           10M |         1~8 |            20 |         0 |
| `MsPacmanNoFrameskip-v4`  |             |              |             |           10M |         1~8 |            20 |         0 |
| `PongNoFrameskip-v4`      |             |              |             |           10M |         1~8 |            20 |         0 |
| `QbertNoFrameskip-v4`     |             |              |             |           10M |         1~8 |            20 |         0 |
| `SeaquestNoFrameskip-v4`  |             |              |             |           10M |         1~8 |            20 |         0 |

<sup>M = million (1e6)</sup><br>

### Hyperparameters


<!-- | `env_id`           | `AsteroidsNoFrameskip-v4` | `BeamRiderNoFrameskip-v4` | `BreakoutNoFrameskip-v4` | `EnduroNoFrameskip-v4` | `MsPacmanNoFrameskip-v4` | `PongNoFrameskip-v4` | `QbertNoFrameskip-v4` | `SeaquestNoFrameskip-v4` |
|--------------------|:-------------------------:|:-------------------------:|:------------------------:|:----------------------:|:------------------------:|:--------------------:|:---------------------:|:------------------------:|
| `num_envs`         |             8             |             8             |             8            |            8           |             8            |           8          |           8           |             8            |
| `num_epochs`       |           312500          |           312500          |          312500          |         312500         |          312500          |        312500        |         312500        |          312500          |
| `num_steps`        |             4             |             4             |             4            |            4           |             4            |           4          |           4           |             4            |
| `num_gradsteps`    |             1             |             1             |             1            |            1           |             1            |           1          |           1           |             1            |
| `batch_size`       |            256            |            256            |            256           |           256          |            256           |          256         |          256          |            256           |
| `target_update`    |            625            |            625            |            625           |           625          |            625           |          625         |          625          |            625           |
| `exploration`      |     Linear(1.0, 0.01)     |     Linear(1.0, 0.01)     |     Linear(1.0, 0.01)    |    Linear(1.0, 0.01)   |     Linear(1.0, 0.01)    |   Linear(1.0, 0.01)  |   Linear(1.0, 0.01)   |     Linear(1.0, 0.01)    |
| `explore_progress` |            0.1            |            0.1            |            0.1           |           0.1          |            0.1           |          0.1         |          0.1          |            0.1           |
| `num_quantiles`    |            200            |            200            |            200           |           200          |            200           |          200         |          200          |            200           | -->

## Architecture

|             |        `Box`       |     `Discrete`     |   `MultiDiscrete`  |    `MultiBinary`   |
|-------------|:------------------:|:------------------:|:------------------:|:------------------:|
| Observation | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Action      |         :x:        | :heavy_check_mark: |         :x:        |         :x:        |

<br/>
<br/>

### Atari-like environment
![](https://g.gravizo.com/source/svg/iqn_discrete?https%3A%2F%2Fraw.githubusercontent.com%2FEnding2015a%2Funstable_baselines_assets%2Fmaster%2Fscripts%2Farch%2Fiqn.arch.md)
<br/>
<sup>force_mlp=`False`</sup><br>


![](https://g.gravizo.com/source/svg/iqn_continuous?https%3A%2F%2Fraw.githubusercontent.com%2FEnding2015a%2Funstable_baselines_assets%2Fmaster%2Fscripts%2Farch%2Fiqn.arch.md)
<br/>
<sup>force_mlp=`True`</sup><br>