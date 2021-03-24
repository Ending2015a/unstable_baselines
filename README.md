# Unstable Baselines

A TensorFlow 2.0 implementation of some Reinforcement Learning algorithms.

## Installation

### Install dependencies
```
gym==0.18.0
tensorflow==2.4.1
dill==0.3.3
pybullet==3.0.8
atari-py==0.2.6
```

### Clone repo

```
git clone --recursive https://github.com/Ending2015a/unstable_baselines.git
```

## Algorithms

### Basic RL



| Algorithm | `Box`              | `Discrete`         | `MultiDiscrete`    | `MultiBinary`      |
|:-----------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|
| PPO       |   | :heavy_check_mark: | :x: | :x: |
| TD3       | :heavy_check_mark: | :x: | :x: | :x: |
| SD3       | :heavy_check_mark: | :x: | :x: | :x: |


* 2021.03.23: Implement SD3
  * From paper: [Softmax Deep Double Deterministic Policy Gradients](https://arxiv.org/abs/2010.09177)
* 2021.03.20: Implement TD3
  * From paper: [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)
* 2021.03.10: Implement PPO
  * From paper: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

### Distributional RL

| Algorithm | `Box`              | `Discrete`         | `MultiDiscrete`    | `MultiBinary`      |
|:-----------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|
| C51 |   |   | :x: | :x: |
| IQN |   |   | :x: | :x: |


### Hierarchical RL

| Algorithm | `Box`              | `Discrete`         | `MultiDiscrete`    | `MultiBinary`      |
|:-----------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|


### Other RL

| Algorithm | `Box`              | `Discrete`         | `MultiDiscrete`    | `MultiBinary`      |
|:-----------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|


## Examples

> some video

## Benchmarks

> learning curves
