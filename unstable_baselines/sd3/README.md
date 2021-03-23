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

> Best video

### Learning Curve

> Learning curve

### Hyperparameters
|                           | `num_envs` | `num_episodes` | `num_steps` | `gradient_steps` | `batch_size` | `learing_rate` |`action_noise` |
|---------------------------|------------|----------------|-------------|------------------|--------------|----------------|----------------|
| `HalfCheetahBulletEnv-v0` | 1          | 1000           | 1000        | 1000             | 100          | 1e-3           | $\mathcal{N}(0, 0.1)$ |
| `AntBulletEnv-v0`         | 1          | 1000           | 1000        | 1000             | 100          | 1e-3           | $\mathcal{N}(0, 0.1)$ |
| `Walker2DBulletEnv-v0`    | 1          | 1000           | 1000        | 1000             | 100          | 1e-3           | $\mathcal{N}(0, 0.1)$ |
| `HumanoidBulletEnv-v0`    | 4          | 2500           | 1000        | 1000             | 256          | 3e-4           | $\mathcal{N}(0, 0.1)$ |



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


