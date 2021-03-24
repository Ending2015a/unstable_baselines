# Unstable Baselines

A TensorFlow 2.0 implementation of some Reinforcement Learning algorithms.

## Installation

### Clone repo

```
git clone --recursive https://github.com/Ending2015a/unstable_baselines.git
cd unstable_baselines
```

### Install dependencies
* Python >= 3.6
* TensorFlow (CPU/GPU) >= 2.1.0

```
pip install -r requirements.txt
```

### Done

Modify the code to fit your needs.



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

<!---
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

--->
## Examples

> some video

## Benchmarks

> learning curves
