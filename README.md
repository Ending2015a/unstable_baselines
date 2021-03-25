# Unstable Baselines

A TensorFlow 2.0 implementation of some Reinforcement Learning algorithms.



| **Features**                   | Unstable Baselines                | [**Stable-Baselines**](https://github.com/hill-a/stable-baselines) | [**OpenAI Baselines**](https://github.com/openai/baselines) |
| ------------------------------ | --------------------------------- | ------------------- | ------------------ |
| State of the art RL methods    | :heavy_minus_sign: <sup>(1)</sup> | :heavy_check_mark:  | :heavy_check_mark: |
| Documentation                  | :x:                               | :heavy_check_mark:  | :x:                |
| Custom environments            | :heavy_check_mark:                | :heavy_check_mark:  | :heavy_check_mark: |
| Custom policies                | :heavy_minus_sign:                | :heavy_check_mark:  | :heavy_minus_sign: |
| Common interface               | :heavy_check_mark:                | :heavy_check_mark:  | :heavy_minus_sign: |
| Tensorboard support            | :heavy_check_mark:                | :heavy_check_mark:  | :heavy_minus_sign: |
| Ipython / Notebook friendly    | :x:                               | :heavy_check_mark:  | :x:                |
| PEP8 code style                | :x:                               | :heavy_check_mark:  | :heavy_check_mark: |
| Custom callback <sup>(2)</sup> | :x:                               | :heavy_check_mark:  | :heavy_minus_sign: |
| **TensorFlow 2.0 support**     | :heavy_check_mark:                | :x:                 | :x:                |
| **Clean, elegant code**        | :heavy_check_mark:                | :x:                 | :x:                |
| **Easy to trace, customize**   | :heavy_check_mark:                | :x: <sup>(3)</sup>  | :x: <sup>(3)</sup> |            
| **Standalone implementations for each algo** | :heavy_check_mark:  | :heavy_minus_sign:  | :x: <sup>(4)</sup> |


<sup>(1) Currently only support PPO, TD3. We are still working on other algorithms.</sup><br>
<sup>(2) Do we really need this?? </sup><br>
<sup>(3) If you have traced Stable-baselines or OpenAI/baselines once, you'll never do that again.</sup><br>
<sup>(4) Many cross-dependencies across all algos make the code very hard to trace, for example [baselines/common/policies.py](https://github.com/openai/baselines/blob/master/baselines/common/policies.py#L3), [baselines/a2c/a2c.py](https://github.com/openai/baselines/blob/master/baselines/a2c/a2c.py#L14).... Great job! OpenAI!:cat:</sup><br>


## Documentation
We don't have any documentation yet.

## Installation
<sup>Worked on Linux, not test on Windows or other operating systems.</sup><br>
* Python >= 3.6
* TensorFlow (CPU/[GPU](https://www.tensorflow.org/install/source#gpu)) >= 2.1.0

### Clone repo

```
git clone --recursive https://github.com/Ending2015a/unstable_baselines.git
cd unstable_baselines
```

### Install dependencies
```
pip install -r requirements.txt
```

### Done

Modify the code to fit your needs, or you can run demo code. (Please see README in each algo folder)

## Algorithms

### Model-free RL



| Algorithm                     | `Box`              | `Discrete`         | `MultiDiscrete`    | `MultiBinary`      |
|:-----------------------------:|:------------------:|:------------------:|:------------------:|:------------------:|
| [PPO](unstable_baselines/ppo) |                    | :heavy_check_mark: | :x:                | :x:                |
| [TD3](unstable_baselines/td3) | :heavy_check_mark: | :x:                | :x:                | :x:                |
| [SD3](unstable_baselines/sd3) | :heavy_check_mark: | :x:                | :x:                | :x:                |


* 2021.03.23: Implement [SD3](unstable_baselines/sd3)
  * From paper: [Softmax Deep Double Deterministic Policy Gradients](https://arxiv.org/abs/2010.09177)
* 2021.03.20: Implement [TD3](unstable_baselines/td3)
  * From paper: [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)
* 2021.03.10: Implement [PPO](unstable_baselines/ppo)
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
