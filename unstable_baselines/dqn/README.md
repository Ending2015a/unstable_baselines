# Double Deep Q-Learning (DDQN)

> [Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning with Double Q-Learning. *In Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence.*](https://arxiv.org/abs/1509.06461)


## How to use

### Run with default arguments
```python
./unstable_baselines/dqn/train.sh --rank 0 --seed 1 "BreakoutNoFrameskip-v4"
```

### Run multiple environments with default arguments
```python
./unstable_baselines/dqn/train.sh --rank 0 --seed 1 "BreakoutNoFrameskip-v4" "SeaquestNoFrameskip-v4"
```

### Atari-like environment (Image observation + discrete action)
```python
python -m unstable_baselines.dqn.run --rank 0 --seed 1 --logdir='./log/{env_id}/dqn/{rank}' \
               --logging='training.log' --monitor_dir='monitor' --tb_logdir='' --model_path='model/weights' \
               --env_id="BreakoutNoFrameskip-v4" --num_envs=8 --num_epochs=312500 \
               --num_steps=4 --num_gradsteps=1 --batch_size=128 --target_update=625 \
               --explore_rate=1.0 --explore_final=0.05 --explore_progress=0.1 \
               --huber --record_video
```

<sup>Total timesteps (Samples) = num_envs * num_steps * num_epochs (~10M in this case)</sup><br>


## Atari 2600

### Video

| `BeamRiderNoFrameskip-v4` | `BreakoutNoFrameskip-v4` | `PongNoFrameskip-v4`  | `SeaquestNoFrameskip-v4` |
|---------------------------|--------------------------|-----------------------|--------------------------|
|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/dqn.BeamRiderNoFrameskip-v4.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/dqn.BreakoutNoFrameskip-v4.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/dqn.PongNoFrameskip-v4.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/dqn.SeaquestNoFrameskip-v4.eval.gif" height=300px>|
| `AsteroidsNoFrameskip-v4` | `EnduroNoFrameskip-v4`   | `QbertNoFrameskip-v4` | `MsPacmanNoFrameskip-v4` |
|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/dqn.AsteroidsNoFrameskip-v4.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/dqn.EnduroNoFrameskip-v4.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/dqn.QbertNoFrameskip-v4.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/dqn.MsPacmanNoFrameskip-v4.eval.gif" height=300px>|

### Results

> Learning curve

| `env_id`                  | Max rewards | Mean rewards | Std rewards | Total frames | Eval episodes |
|---------------------------|------------:|-------------:|------------:|-------------:|--------------:|
| `AsteroidsNoFrameskip-v4` |             |              |             | 10M          | 20            |
| `BeamRiderNoFrameskip-v4` | 10000       | 6011.5       | 1796.79     | 10M          | 20            |
| `BreakoutNoFrameskip-v4`  | 412         | 366          | 32.35       | 10M          | 20            |
| `EnduroNoFrameskip-v4`    |             |              |             | 10M          | 20            |
| `MsPacmanNoFrameskip-v4`  |             |              |             | 10M          | 20            |
| `PongNoFrameskip-v4`      | 21          | 20           | 0.64        | 10M          | 20            |
| `QbertNoFrameskip-v4`     |             |              |             | 10M          | 20            |
| `SeaquestNoFrameskip-v4`  | 5030        | 3555         | 1393.45     | 10M          | 20            |

<sup>M = million (1e6)</sup><br>

### Hyperparameters


| `env_id`           | `AsteroidsNoFrameskip-v4` | `BeamRiderNoFrameskip-v4` | `BreakoutNoFrameskip-v4` | `EnduroNoFrameskip-v4` | `MsPacmanNoFrameskip-v4` | `PongNoFrameskip-v4` | `QbertNoFrameskip-v4` | `SeaquestNoFrameskip-v4` |
|--------------------|:-------------------------:|:-------------------------:|:------------------------:|:----------------------:|:------------------------:|:--------------------:|:---------------------:|:------------------------:|
| `num_envs`         |             8             |             8             |             8            |            8           |             8            |           8          |           8           |             8            |
| `num_epochs`       |           312500          |           312500          |          312500          |         312500         |          312500          |        312500        |         312500        |          312500          |
| `num_steps`        |             4             |             4             |             4            |            4           |             4            |           4          |           4           |             4            |
| `num_gradsteps`    |             1             |             1             |             1            |            1           |             1            |           1          |           1           |             1            |
| `batch_size`       |            128            |            128            |            128           |           128          |            128           |          128         |          128          |            128           |
| `target_update`    |            625            |            625            |            625           |           625          |            625           |          625         |          625          |            625           |
| `exploration`      |     Linear(1.0, 0.05)     |     Linear(1.0, 0.05)     |     Linear(1.0, 0.05)    |    Linear(1.0, 0.05)   |     Linear(1.0, 0.05)    |   Linear(1.0, 0.05)  |   Linear(1.0, 0.05)   |     Linear(1.0, 0.05)    |
| `explore_progress` |            0.1            |            0.1            |            0.1           |           0.1          |            0.1           |          0.1         |          0.1          |            0.1           |

## Architecture

|             | `Box`              | `Discrete`         | `MultiDiscrete` | `MultiBinary` |
|:-----------:|:------------------:|:------------------:|:---------------:|:-------------:|
| Observation | :x:                | :heavy_check_mark: | :x:             | :x:           |
| Action      | :x:                | :heavy_check_mark: | :x:             | :x:           |

<br/>
<br/>

### Atari-like environment
![](https://g.gravizo.com/source/svg/dqn_discrete?https%3A%2F%2Fraw.githubusercontent.com%2FEnding2015a%2Funstable_baselines_assets%2Fmaster%2Fscripts%2Farch%2Fdqn.arch.md)
<br/>
<sup>force_mlp=`False`</sup><br>


![](https://g.gravizo.com/source/svg/dqn_continuous?https%3A%2F%2Fraw.githubusercontent.com%2FEnding2015a%2Funstable_baselines_assets%2Fmaster%2Fscripts%2Farch%2Fdqn.arch.md)
<br/>
<sup>force_mlp=`True`</sup><br>
