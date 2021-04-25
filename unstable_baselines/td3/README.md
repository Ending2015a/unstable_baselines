# Twin Deplayed Deep Deterministic Policy Gradient (TD3)

> [Fujimoto, S., Hoof, H., & Meger, D. (2018). Addressing Function Approximation Error in Actor-Critic Methods. *In Proceedings of the 35th International Conference on Machine Learning*](https://arxiv.org/abs/1802.09477)


## How to run

### Run with default arguments
```python
./unstable_baselines/td3/train.sh --rank 0 --seed 1 "HalfCheetahBulletEnv-v0"
```

### Run multiple environments with default arguments
```python
./unstable_baselines/td3/train.sh --rank 0 --seed 1 "HalfCheetahBulletEnv-v0" "AntBulletEnv-v0"
```

### Continuous control environment
```python
python -m unstable_baselines.td3.run  --rank 0 --seed 1 --logdir='./log/{env_id}/td3/{rank}' \
               --logging='training.log' --monitor_dir='monitor' --tb_logdir='' --model_dir='model' \
               --env_id="HalfCheetahBulletEnv-v0" --num_envs=1 --num_epochs=10000 --min_buffer=10000 \
               --num_steps=100 --num_gradsteps=1 --batch_size=100 --verbose=2
```

<sup>Total timesteps (Samples) ≈ num_envs * num_steps * num_epochs (~1M in this case)</sup><br>
<sup>Number of times each sample reused ≈ batch_size/num_steps * num_gradsteps/num_envs (~1 in this case)</sup><br>


## PyBullet

### Video

| Environment | Video | Environment | Video |
|:-:|:-:|:-:|:-:|
|`HalfCheetahBulletEnv-v0`|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/td3.HalfCheetahBulletEnv-v0.eval.gif" width=300px/>|`AntBulletEnv-v0`|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/td3.AntBulletEnv-v0.eval.gif" width=300px/>|
|`HopperBulletEnv-v0`  |<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/td3.HopperBulletEnv-v0.eval.gif" width=300px/>|`Walker2DBulletEnv-v0`|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/td3.Walker2DBulletEnv-v0.eval.gif" width=300px/>|
|`HumanoidBulletEnv-v0`|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/td3.HumanoidBulletEnv-v0.eval.gif" width=300px/>|||


### Results

> Learning curve

| `env_id`                  | Max rewards | Mean rewards | Std rewards | Train samples | Train seeds | Eval episodes | Eval seed |
|---------------------------|------------:|-------------:|------------:|--------------:|------------:|--------------:|----------:|
| `AntBulletEnv-v0`         |             |              |             |            1M |           1 |            20 |         0 |
| `HalfCheetahBulletEnv-v0` |             |              |             |            1M |           1 |            20 |         0 |
| `HopperBulletEnv-v0`      |             |              |             |            1M |           1 |            20 |         0 |
| `HumanoidBulletEnv-v0`    |             |              |             |            1M |           1 |            20 |         0 |
| `Walker2DBulletEnv-v0`    |             |              |             |            1M |           1 |            20 |         0 |

### Hyperparameters

| `env_id`        | `AntBulletEnv-v0` | `HalfCheetahBulletEnv-v0` | `HopperBulletEnv-v0` | `HumanoidBulletEnv-v0` | `Walker2DBulletEnv-v0` |
|-----------------|:-----------------:|:-------------------------:|:--------------------:|:----------------------:|:----------------------:|
| `num_envs`      |         1         |             1             |           1          |            4           |            1           |
| `num_epochs`    |       10000       |           10000           |         10000        |          10000         |          10000         |
| `num_steps`     |        100        |            100            |          100         |           100          |           100          |
| `num_gradsteps` |        100        |            100            |          100         |           100          |           100          |
| `batch_size`    |        100        |            100            |          100         |           100          |           100          |
| `policy_update` |         2         |             2             |           2          |            2           |            2           |
| `target_update` |         2         |             2             |           2          |            2           |            2           |
| `lr`            |        1e-3       |            1e-3           |         1e-3         |          1e-3          |          1e-3          |
| `tau`           |       0.005       |           0.005           |         0.005        |          0.005         |          0.005         |
| `explore_noise` |     N(0, 0.1)     |         N(0, 0.1)         |       N(0, 0.1)      |        N(0, 0.1)       |        N(0, 0.1)       |



## Architecture

|             |        `Box`       |     `Discrete`     |   `MultiDiscrete`  |    `MultiBinary`   |
|-------------|:------------------:|:------------------:|:------------------:|:------------------:|
| Observation | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Action      | :heavy_check_mark: |         :x:        |         :x:        |         :x:        |


<br/>
<br/>

![](https://g.gravizo.com/source/svg/td3_arch?https%3A%2F%2Fraw.githubusercontent.com%2FEnding2015a%2Funstable_baselines_assets%2Fmaster%2Fscripts%2Farch%2Ftd3.arch.md)


