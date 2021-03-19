# Twin Deplayed Deep Deterministic Policy Gradient (TD3)

> [Fujimoto, S., van Hoof, H., and Meger, D. Addressing Function Approximation Error in Actor-Critic Methods. *ICML 2018*.](https://arxiv.org/abs/1802.09477)


## How to run
```python
python -m unstable_baselines.td3.run  --rank 0 --seed 1 --logdir='./log/{env_id}/td3_1m/{rank}' \
               --logging='training.log' --monitor_dir='monitor' --tb_logdir='' --model_dir='model' \
               --env_id="HalfCheetahBulletEnv-v0" --num_envs=1 --num_episodes=1000 --min_buffer=10000 \
               --num_steps=1000 --gradient_steps=1000 --batch_size=100 --verbose=2

```

* Total timesteps (Samples) = num_envs * num_steps * num_episodes (~1M in this case)


## PyBullet

### Video

> Best video

### Learning Curve

> Learning curve

### Hyperparameters
| | `num_envs` | `num_episodes` | `num_steps` | `gradient_steps` | `batch_size` |
|-|-|-|-|-|-|
| `HalfCheetahBulletEnv-v0` | 1 | 1000 | 1000 | 1000 | 100 |
| `AntBulletEnv-v0` | 1 | 1000 | 1000 | 1000 | 100 |
| `Walker2DBulletEnv-v0` | 1 | 1000 | 1000 | 1000 | 100 |
| `HumanoidBulletEnv-v0` | 4 | 2500 | 1000 | 1000 | 256 | 



## Architecture

|             | `Box`              | `Discrete` | `MultiDiscrete` | `MultiBinary` |
|-------------|--------------------|------------|-----------------|---------------|
| Observation | :heavy_check_mark: | :x:        | :x:             | :x:           |
| Action      | :heavy_check_mark: | :x:        | :x:             | :x:           |


* Actor network
```
Dense(400)
ReLU
Dense(300)
ReLU
Dense(action_space)
Tanh
```
* Critic network
```
Concat(observation_space + action_space)
Dense(400)
ReLU
Dense(300)
ReLU
Dense(1)
```


