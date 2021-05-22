> Some cool video

# Unstable Baselines (Early Access)

A Deep Reinforcement Learning codebase in TensorFlow 2.0 with an unified, flexible and highly customizable structure for fast prototyping.



| **Features**                   | Unstable Baselines                | [**Stable-Baselines**](https://github.com/hill-a/stable-baselines) | [**OpenAI Baselines**](https://github.com/openai/baselines) |
| ------------------------------ | --------------------------------- | ------------------- | ------------------ |
| State of the art RL methods    | :heavy_minus_sign: <sup>(1)</sup> | :heavy_check_mark:  | :heavy_check_mark: |
| Documentation                  | :x:                               | :heavy_check_mark:  | :x:                |
| Custom callback <sup>(2)</sup> | :x:                               | :heavy_check_mark:  | :heavy_minus_sign: |
| **TensorFlow 2.0 support**     | :heavy_check_mark:                | :x:                 | :x:                |
| **Clean, elegant code**        | :heavy_check_mark:                | :x:                 | :x:                |
| **Easy to trace, customize**   | :heavy_check_mark:                | :x: <sup>(3)</sup>  | :x: <sup>(3)</sup> |
| **Standalone implementations for each algo** | :heavy_check_mark:  | :heavy_minus_sign:  | :x: <sup>(4)</sup> |

<sup>(1) Currently only support DQN, C51, PPO, TD3, ...etc. We are still working on other algorithms.</sup><br>
<sup>(2) Do we really need this? </sup><br>
<sup>(3) If you have traced Stable-baselines or OpenAI/baselines once, you'll never do that again.</sup><br>
<sup>(4) Many cross-dependencies across all algos make the code very hard to trace, for example [baselines/common/policies.py](https://github.com/openai/baselines/blob/master/baselines/common/policies.py#L3), [baselines/a2c/a2c.py](https://github.com/openai/baselines/blob/master/baselines/a2c/a2c.py#L14).... Great job! OpenAI!:cat:</sup><br>


## Documentation
We don't have any documentation yet.

## Installation
<sup>Worked on Linux, but not test on Windows or other operating systems.</sup><br>
* Python >= 3.6
* TensorFlow (CPU/[GPU](https://www.tensorflow.org/install/source#gpu)) >= 2.1.0

### Clone repository

```
git clone --recursive https://github.com/Ending2015a/unstable_baselines.git
cd unstable_baselines
```

### Install dependencies
```
pip install -r requirements.txt
```

### Done

* Modify the code to fit your needs
* or you can run the demo codes. (Please see README in each algo folder)

## Algorithms

### Model-free RL


|           Algorithm           |        `Box`       |     `Discrete`     | `MultiDiscrete` | `MultiBinary` |
|:-----------------------------:|:------------------:|:------------------:|:---------------:|:-------------:|
| [DQN](unstable_baselines/dqn) | :x:                | :heavy_check_mark: | :x:             | :x:           |
| [PPO](unstable_baselines/ppo) | :heavy_check_mark: | :heavy_check_mark: | :x:             | :x:           |
| [TD3](unstable_baselines/td3) | :heavy_check_mark: | :x:                | :x:             | :x:           |
| [SD3](unstable_baselines/sd3) | :heavy_check_mark: | :x:                | :x:             | :x:           |

* 2021.04.19: Implemented [DQN](unstable_baselines/dqn)
  * From paper: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
* 2021.03.27: [PPO](unstable_baselines/ppo) support continuous (Box) action space
* 2021.03.23: Implemented [SD3](unstable_baselines/sd3)
  * From paper: [Softmax Deep Double Deterministic Policy Gradients](https://arxiv.org/abs/2010.09177)
* 2021.03.20: Implemented [TD3](unstable_baselines/td3)
  * From paper: [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)
* 2021.03.10: Implemented [PPO](unstable_baselines/ppo)
  * From paper: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)


### Distributional RL

|              Algorithm              | `Box` |     `Discrete`     | `MultiDiscrete` | `MultiBinary` |
|:-----------------------------------:|:-----:|:------------------:|:---------------:|:-------------:|
| [C51](unstable_baselines/d/c51)     |  :x:  | :heavy_check_mark: |       :x:       |      :x:      |
| [QRDQN](unstable_baselines/d/qrdqn) |  :x:  | :heavy_check_mark: |       :x:       |      :x:      |
| [IQN](unstable_baselines/d/iqn)     |  :x:  | :heavy_check_mark: |       :x:       |      :x:      |

* 2021.04.28: Implemented [IQN](unstable_baselines/d/iqn)
  * From paper: [Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923)
* 2021.04.21: Implemented [QRDQN](unstable_baselines/d/qrdqn)
  * From paper: [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044)
* 2021.04.20: Implemented [C51](unstable_baselines/d/c51)
  * From paper: [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)

<!---
### Hierarchical RL

| Algorithm | `Box`              | `Discrete`         | `MultiDiscrete`    | `MultiBinary`      |
|:-----------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|


### Other RL

| Algorithm | `Box`              | `Discrete`         | `MultiDiscrete`    | `MultiBinary`      |
|:-----------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|

--->
## Examples

> some video


## Tutorials
> colab tutorials

## Update Logs
* 2021.05.22: Add benchmarks
* 2021.04.27: Update to framework v2: supports saving/loading the best performed checkpoints.

## Benchmarks

See [benchmark.md](benchmark.md).

### Atari 2600

| Environments              |  Algo | Max rewards | Mean rewards | Std rewards | Train samples | Train seeds | Eval episodes | Eval seed |
|---------------------------|-------|------------:|-------------:|------------:|--------------:|------------:|--------------:|----------:|
| `AsteroidsNoFrameskip-v4` | DQN   |        1530 |          667 |      265.68 |           10M |         1~8 |            20 |         0 |
|                           | C51   |        1610 |        851.5 |       302.2 |           10M |         1~8 |            20 |         0 |
|                           | QRDQN |        2030 |          985 |      363.23 |           10M |         1~8 |            20 |         0 |
|                           | IQN   |        1830 |        928.5 |      458.79 |           10M |         1~8 |            20 |         0 |
|                           | PPO   |        1570 |         1072 |      281.73 |           10M |         1~8 |            20 |         0 |
| `BeamRiderNoFrameskip-v4` | DQN   |       10408 |       6806.6 |     1689.98 |           10M |         1~8 |            20 |         0 |
|                           | C51   |       11500 |       7791.1 |     2570.17 |           10M |         1~8 |            20 |         0 |
|                           | QRDQN |       13578 |         7731 |     2588.28 |           10M |         1~8 |            20 |         0 |
|                           | IQN   |       12862 |       7434.1 |     2064.42 |           10M |         1~8 |            20 |         0 |
|                           | PPO   |        2832 |       1513.4 |      647.36 |           10M |         1~8 |            20 |         0 |
| `BreakoutNoFrameskip-v4`  | DQN   |         385 |       364.45 |       31.98 |           10M |         1~8 |            20 |         0 |
|                           | C51   |         424 |       393.05 |       25.72 |           10M |         1~8 |            20 |         0 |
|                           | QRDQN |         778 |          321 |      301.88 |           10M |         1~8 |            20 |         0 |
|                           | IQN   |         826 |       372.55 |      116.21 |           10M |         1~8 |            20 |         0 |
|                           | PPO   |         368 |       131.85 |      118.28 |           10M |         1~8 |            20 |         0 |
| `EnduroNoFrameskip-v4`    | DQN   |        1354 |       838.95 |      276.42 |           10M |         1~8 |            20 |         0 |
|                           | C51   |        2229 |       1726.1 |      310.23 |           10M |         1~8 |            20 |         0 |
|                           | QRDQN |        1092 |        944.5 |      137.87 |           10M |         1~8 |            20 |         0 |
|                           | IQN   |        1656 |      1206.85 |      247.98 |           10M |         1~8 |            20 |         0 |
|                           | PPO   |         302 |        189.2 |       29.79 |           10M |         1~8 |            20 |         0 |
| `MsPacmanNoFrameskip-v4`  | DQN   |        2700 |       2109.5 |      295.82 |           10M |         1~8 |            20 |         0 |
|                           | C51   |        3060 |       2796.5 |      316.31 |           10M |         1~8 |            20 |         0 |
|                           | QRDQN |        4990 |         3960 |     1033.51 |           10M |         1~8 |            20 |         0 |
|                           | IQN   |        2690 |       2347.5 |      230.84 |           10M |         1~8 |            20 |         0 |
|                           | PPO   |        2650 |       2035.5 |       463.1 |           10M |         1~8 |            20 |         0 |
| `PongNoFrameskip-v4`      | DQN   |          21 |         20.9 |         0.3 |           10M |         1~8 |            20 |         0 |
|                           | C51   |          21 |         20.8 |         0.6 |           10M |         1~8 |            20 |         0 |
|                           | QRDQN |          21 |           21 |           0 |           10M |         1~8 |            20 |         0 |
|                           | IQN   |          21 |           21 |           0 |           10M |         1~8 |            20 |         0 |
|                           | PPO   |          21 |           21 |           0 |           10M |         1~8 |            20 |         0 |
| `QbertNoFrameskip-v4`     | DQN   |       11450 |         9575 |     1633.19 |           10M |         1~8 |            20 |         0 |
|                           | C51   |       16550 |     15978.75 |      373.35 |           10M |         1~8 |            20 |         0 |
|                           | QRDQN |        7825 |      5246.25 |     1567.27 |           10M |         1~8 |            20 |         0 |
|                           | IQN   |       15925 |        14860 |     1073.99 |           10M |         1~8 |            20 |         0 |
|                           | PPO   |       16925 |     16441.25 |      259.23 |           10M |         1~8 |            20 |         0 |
| `SeaquestNoFrameskip-v4`  | DQN   |       11660 |         9434 |     2410.74 |           10M |         1~8 |            20 |         0 |
|                           | C51   |       15970 |         8951 |      2381.3 |           10M |         1~8 |            20 |         0 |
|                           | QRDQN |        8520 |         7764 |      638.80 |           10M |         1~8 |            20 |         0 |
|                           | IQN   |       19780 |      12193.5 |     4172.15 |           10M |         1~8 |            20 |         0 |
|                           | PPO   |        1760 |         1750 |       17.32 |           10M |         1~8 |            20 |         0 |


### PyBullet

| Environments              | Algo | Max rewards | Mean rewards | Std rewards | Train samples | Train seeds | Eval episodes | Eval seed |
|---------------------------|------|------------:|-------------:|------------:|--------------:|------------:|--------------:|----------:|
| `AntBulletEnv-v0`         | PPO  |    2247.002 |     2157.180 |     107.803 |            2M |           1 |            20 |         0 |
|                           | TD3  |    2910.995 |     2707.637 |     504.815 |            1M |           1 |            20 |         0 |
|                           | SD3  |    2659.900 |     2341.345 |     308.402 |            1M |           1 |            20 |         0 |
| `HalfCheetahBulletEnv-v0` | PPO  |    2696.556 |     2477.882 |     759.322 |            2M |           1 |            20 |         0 |
|                           | TD3  |    2756.600 |     2654.996 |      72.471 |            1M |           1 |            20 |         0 |
|                           | SD3  |    2692.213 |     2591.908 |      52.607 |            1M |           1 |            20 |         0 |
| `HopperBulletEnv-v0`      | PPO  |    2689.504 |     2542.172 |     373.381 |            2M |           1 |            20 |         0 |
|                           | TD3  |    2719.811 |     2689.082 |      18.047 |            1M |           1 |            20 |         0 |
|                           | SD3  |    2793.888 |     2775.304 |      11.150 |            1M |           1 |            20 |         0 |
| `HumanoidBulletEnv-v0`    | PPO  |    2447.299 |     1883.564 |     923.937 |            8M |           1 |            20 |         0 |
|                           | TD3  |    2643.223 |     2360.450 |     766.838 |            2M |           1 |            20 |         0 |
|                           | SD3  |    2661.585 |     2596.141 |      41.613 |            2M |           1 |            20 |         0 |
| `Walker2DBulletEnv-v0`    | PPO  |    2108.727 |     2005.461 |     286.699 |            4M |           1 |            20 |         0 |
|                           | TD3  |    2117.742 |     2104.583 |       9.381 |            1M |           1 |            20 |         0 |
|                           | SD3  |    2166.296 |     2016.048 |     457.554 |            1M |           1 |            20 |         0 |