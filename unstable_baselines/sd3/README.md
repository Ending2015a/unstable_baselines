# Softmax Deep Double Deterministic Policy Gradients (SD3)

> [Ling P., Qingpeng C., Longbo H. Softmax Deep Double Deterministic Policy Gradients. *NeurIPS 2020*.](https://arxiv.org/abs/2010.09177)



## How to run
```python
python -m unstable_baselines.sd3.run \
            --rank $rank --seed $seed --logdir='./log/{env_id}/sd3/{rank}' --logging='training.log' \
            --monitor_dir='monitor' --tb_logdir='' --model_dir='model' --env_id="HalfCheetahBulletEnv-v0" \
            --num_envs=1 --num_episodes=1000 --min_buffer=10000 --num_steps=1000 --gradient_steps=1000 \
            --batch_size=100 --verbose=2 --explore_noise --importance_sampling \
```

* Total timesteps (Samples) = num_envs * num_steps * num_episodes (~1M in this case)


## PyBullet

### Video

| Environment | Video |
|-|-|
|`HalfCheetahBulletEnv-v0`|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/sd3.HalfCheetahBulletEnv-v0.eval.gif" width=300px/>|
| `AntBulletEnv-v0`|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/sd3.AntBulletEnv-v0.eval.gif" width=300px/>|
|`HopperBulletEnv-v0`  |<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/sd3.HopperBulletEnv-v0.eval.gif" width=300px/>|
|`Walker2DBulletEnv-v0`|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/sd3.Walker2DBulletEnv-v0.eval.gif" width=300px/>|
|`HumanoidBulletEnv-v0`|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/sd3.HumanoidBulletEnv-v0.eval.gif" width=300px/>|

### Learning Curve

> Learning curve

### Hyperparameters

| `env_id`                  | `num_envs` | `num_episodes` | `num_steps` | `gradient_steps` | `batch_size` | `learing_rate` |`explore_noise` |
|---------------------------|------------|----------------|-------------|------------------|--------------|----------------|----------------|
| `HalfCheetahBulletEnv-v0` | 1          | 1000           | 2000        | 1000             | 200          | 1e-3           | N(0, 0.1)      |
| `AntBulletEnv-v0`         | 1          | 1000           | 2000        | 1000             | 200          | 1e-3           | N(0, 0.1)      |
| `HopperBulletEnv-v0`      | 1          | 1000           | 2000        | 1000             | 200          | 1e-3           | N(0, 0.1)      |
| `Walker2DBulletEnv-v0`    | 1          | 1000           | 2000        | 1000             | 200          | 1e-3           | N(0, 0.1)      |
| `HumanoidBulletEnv-v0`    | 4          | 1000           | 2000        | 1000             | 800          | 3e-4           | N(0, 0.1)      |



## Architecture

|             | `Box`              | `Discrete` | `MultiDiscrete` | `MultiBinary` |
|-------------|--------------------|------------|-----------------|---------------|
| Observation | :heavy_check_mark: | :x:        | :x:             | :x:           |
| Action      | :heavy_check_mark: | :x:        | :x:             | :x:           |


<br/>
<br/>

![](https://g.gravizo.com/source/svg/sd3_arch?https%3A%2F%2Fraw.githubusercontent.com%2FEnding2015a%2Funstable_baselines_assets%2Fmaster%2Fscripts%2Farch%2Fsd3.arch.md)



