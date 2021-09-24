# Proximal Policy Optimization (PPO)

> [John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, & Oleg Klimov. (2017). Proximal Policy Optimization Algorithms.](https://arxiv.org/abs/1707.06347)

## Features

|             |        `Box`       |     `Discrete`     |   `MultiDiscrete`  |    `MultiBinary`   |       `Dict`       |       `Tuple`      |
|-------------|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|
| Observation | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Action      | :heavy_check_mark: | :heavy_check_mark: |         :x:        |         :x:        |         :x:        |         :x:        |

## How to use

Training a `CartPole-v0` agent.

```python
import unstable_baselines as ub
from unstable_baselines.algo.ppo import PPO
# Create training/evaluating environments
env = ub.envs.VecEnv([gym.make('CartPole-v0') for _ in range(10)])
eval_env = gym.make('CartPole-v0')
# Training for 100000 timesteps
model = PPO(env, batch_size=128).learn(100000, verbose=1)
# Evaluate model
results = model.eval(eval_env, n_episodes=20, max_steps=200)
metrics = model.get_eval_metrics(results)
for k, v in metrics.items():
    print(f'{k}: {v}')
# Don't forget to close the environments!
env.close()
eval_env.close()
```

More examples:
* [Playing with Atari](example/atari)
* [Solving continuous control problems](example/pybullet)

<!---
### Continuous control environment
```python
python -m unstable_baselines.ppo.run --rank 0 --seed 1 --logdir='./log/{env_id}/ppo/{rank}' \
               --logging='training.log' --monitor_dir='monitor' --tb_logdir='' --model_dir='model' \
               --env_id="HalfCheetahBulletEnv-v0" --num_envs=1 --num_epochs=1000 \
               --num_steps=1000 --num_subepochs=10 --batch_size=100 --verbose=2 \
               --ent_coef=0.0 --record_video
```
<sup>Total timesteps (Samples) = num_envs * num_steps * num_epochs (~1M in this case)</sup><br>
<sup>Number of times each sample reused = num_subepochs (~10 in this case)</sup><br>

-->

## Atari 2600

| `BeamRiderNoFrameskip-v4` | `BreakoutNoFrameskip-v4` | `PongNoFrameskip-v4`  | `SeaquestNoFrameskip-v4` |
|:-------------------------:|:------------------------:|:---------------------:|:------------------------:|
|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.BeamRiderNoFrameskip-v4.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.BreakoutNoFrameskip-v4.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.PongNoFrameskip-v4.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.SeaquestNoFrameskip-v4.eval.gif" height=300px>|
| `AsteroidsNoFrameskip-v4` | `EnduroNoFrameskip-v4`   | `QbertNoFrameskip-v4` | `MsPacmanNoFrameskip-v4` |
|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.AsteroidsNoFrameskip-v4.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.EnduroNoFrameskip-v4.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.QbertNoFrameskip-v4.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.MsPacmanNoFrameskip-v4.eval.gif" height=300px>|


### Benchmarks

* [See the full benchmarks]

For atari domains, we run 8 trials for each environment. In each trial we trained an agent on 8 environments with seeds 1~8, and tested on seed 0 for 100 episodes after learning from totally 10M of training samples. The final evaluation is limited to 10000 steps per episode. The following is the 95% confidence intervals of the maximum and the mean rewards of the final scores for each trial. 

| `env_id`                  | Max rewards                           | Mean rewards                         | Train samples | Eval episodes |
|---------------------------|--------------------------------------:|-------------------------------------:|--------------:|--------------:|
| `AsteroidsNoFrameskip-v4` |   2088.750<br><small>±531.823</small> |   1092.788<br><small>±99.973</small> |           10M |           800 |
| `BeamRiderNoFrameskip-v4` |   7436.000<br><small>±944.819</small> |  3891.302<br><small>±476.673</small> |           10M |           800 |
| `BreakoutNoFrameskip-v4`  |    341.250<br><small>±110.693</small> |    198.714<br><small>±99.136</small> |           10M |           800 |
| `EnduroNoFrameskip-v4`    |     471.125<br><small>±84.151</small> |    344.624<br><small>±87.094</small> |           10M |           800 |
| `MsPacmanNoFrameskip-v4`  |   2536.250<br><small>±369.694</small> |  2434.113<br><small>±321.755</small> |           10M |           800 |
| `PongNoFrameskip-v4`      |       21.000<br><small>±0.000</small> |      20.192<br><small>±2.136</small> |           10M |           800 |
| `QbertNoFrameskip-v4`     | 17828.125<br><small>±1431.861</small> | 16485.531<br><small>±534.247</small> |           10M |           800 |
| `SeaquestNoFrameskip-v4`  |  2835.000<br><small>±1362.672</small> | 2647.625<br><small>±1154.382</small> |           10M |           800 |

<sup>M = million (1e6)</sup><br>

## Pybullet

|`HalfCheetahBulletEnv-v0`|`AntBulletEnv-v0`|`HopperBulletEnv-v0`
|:-:|:-:|:-:|
|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.HalfCheetahBulletEnv-v0.eval.gif" />|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.AntBulletEnv-v0.eval.gif" />|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.HopperBulletEnv-v0.eval.gif" />|
|`Walker2DBulletEnv-v0`|`HumanoidBulletEnv-v0`||
|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.Walker2DBulletEnv-v0.eval.gif" />|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.HumanoidBulletEnv-v0.eval.gif" />||

### Benchmarks

* [See the full benchmarks]

<!-- ### Learning Curve

> Learning curve

| `env_id`                  | Max rewards | Mean rewards | Std rewards | Train samples | Train seeds | Eval episodes | Eval seed |
|---------------------------|------------:|-------------:|------------:|--------------:|------------:|--------------:|----------:|
| `AntBulletEnv-v0`         |    2247.002 |     2157.180 |     107.803 |            2M |           1 |            20 |         0 |
| `HalfCheetahBulletEnv-v0` |    2696.556 |     2477.882 |     759.322 |            2M |           1 |            20 |         0 |
| `HopperBulletEnv-v0`      |    2689.504 |     2542.172 |     373.381 |            2M |           1 |            20 |         0 |
| `HumanoidBulletEnv-v0`    |    2447.299 |     1883.564 |     923.937 |            8M |           1 |            20 |         0 |
| `Walker2DBulletEnv-v0`    |    2108.727 |     2005.461 |     286.699 |            4M |           1 |            20 |         0 |


### Hyperparametrs

| `env_id`        | `AntBulletEnv-v0` | `HalfCheetahBulletEnv-v0` | `HopperBulletEnv-v0` | `HumanoidBulletEnv-v0` | `Walker2DBulletEnv-v0` |
|-----------------|:-----------------:|:-------------------------:|:--------------------:|:----------------------:|:----------------------:|
| `num_envs`      |         1         |             1             |           1          |           16           |            4           |
| `num_epochs`    |        2000       |            2000           |         2000         |          1000          |          2000          |
| `num_steps`     |        1000       |            1000           |         1000         |           500          |           500          |
| `num_subepochs` |         10        |             10            |          10          |           20           |           20           |
| `batch_size`    |        100        |            100            |          100         |          1000          |          1000          |
| `lr`            |        3e-4       |            3e-4           |         3e-4         |          3e-4          |          3e-4          |
| `ent_coef`      |        0.0        |            0.0            |          0.0         |           0.0          |           0.0          |
| `vf_coef`       |        0.5        |            0.5            |          0.5         |           0.5          |           0.5          |
| `shared_net`    |        :x:        |            :x:            |          :x:         |           :x:          |           :x:          |
| `MlpNet`        |     [256, 256]    |         [256, 256]        |      [256, 256]      |       [256, 256]       |       [256, 256]       | -->


