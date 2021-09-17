# Unstable Baselines (Early Access)

A Deep Reinforcement Learning codebase in TensorFlow 2.0 with an unified, flexible and highly customizable structure for fast prototyping.



| **Features**                   | Unstable Baselines                | [**Stable-Baselines3**](https://github.com/DLR-RM/stable-baselines3) | [**OpenAI Baselines**](https://github.com/openai/baselines) |
| ------------------------------ | --------------------------------- | ------------------- | ------------------ |
| State of the art RL methods    | :heavy_minus_sign: <sup>(1)</sup> | :heavy_check_mark:  | :heavy_check_mark: |
| Documentation                  | :x:                               | :heavy_check_mark:  | :x:                |
| Custom callback <sup>(2)</sup> | :x:                               | :vomiting_face:     | :heavy_minus_sign: |
| **TensorFlow 2.0 support**     | :heavy_check_mark:                | :x:                 | :x:                |
| **Clean, elegant code**        | :heavy_check_mark:                | :x:                 | :x:                |
| **Easy to trace, customize**   | :heavy_check_mark:                | :x: <sup>(3)</sup>  | :x: <sup>(3)</sup> |
| **Standalone implementations** | :heavy_check_mark:  | :heavy_minus_sign:  | :x: <sup>(4)</sup> |

<sup>(1) Currently only support DQN, C51, PPO, TD3, ...etc. We are still working on other algorithms.</sup><br>
<sup>(2) For example, in Stable-Baselines, you need to [write this yucky custom callbacks](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#using-callback-monitoring-training) to save the best-performed model :vomiting_face:, while in Unstable Baselines, they are automatically saved.</sup><br>
<sup>(3) If you have traced Stable-baselines or OpenAI/baselines once, you'll never do that again.</sup><br>
<sup>(4) Many cross-dependencies across all algos make the code very hard to trace, for example [baselines/common/policies.py](https://github.com/openai/baselines/blob/master/baselines/common/policies.py#L3), [baselines/a2c/a2c.py](https://github.com/openai/baselines/blob/master/baselines/a2c/a2c.py#L14).... Great job! OpenAI!:cat:</sup><br>


## Documentation
We don't have any documentation yet.

## Installation
Basic requirements:
* Python >= 3.6
* TensorFlow (CPU/[GPU](https://www.tensorflow.org/install/source#gpu)) >= 2.3.0

You can install from PyPI
```
$ pip install unstable_baselines
```
Or you can also install the latest version from this repository
```
$ pip install git+https://github.com/Ending2015a/unstable_baselines.git@master
```

Done! Now, you can
* Go through the [Quick Start](#quick-start) section
* Or run the example codes in [example folder](example/).

## Algorithms

### Model-free RL


|           Algorithm           |        `Box`       |     `Discrete`     | `MultiDiscrete` | `MultiBinary` |
|:-----------------------------:|:------------------:|:------------------:|:---------------:|:-------------:|
| [DQN](unstable_baselines/algo/dqn) | :x:                | :heavy_check_mark: | :x:             | :x:           |
| [PPO](unstable_baselines/algo/ppo) | :heavy_check_mark: | :heavy_check_mark: | :x:             | :x:           |
| [TD3](unstable_baselines/algo/td3) | :heavy_check_mark: | :x:                | :x:             | :x:           |
| [SD3](unstable_baselines/algo/sd3) | :heavy_check_mark: | :x:                | :x:             | :x:           |
* 2021.09.17: [DQN](unstable_baselines/algo/dqn) supports 
  * Multi-step learning
  * Prioritized experience replay: [arXiv:1511.05952](https://arxiv.org/abs/1511.05952)
  * Dueling network: [arXiv:1511.06581](https://arxiv.org/abs/1511.06581)
* 2021.04.19: Implemented [DQN](unstable_baselines/dqn)
  * From paper: [arXiv:1509.06461](https://arxiv.org/abs/1509.06461)
* 2021.03.27: [PPO](unstable_baselines/ppo) support continuous (Box) action space
* 2021.03.23: Implemented [SD3](unstable_baselines/algo/sd3)
  * From paper: [arXiv:2010.09177](https://arxiv.org/abs/2010.09177)
* 2021.03.20: Implemented [TD3](unstable_baselines/algo/td3)
  * From paper: [arXiv:1802.09477](https://arxiv.org/abs/1802.09477)
* 2021.03.10: Implemented [PPO](unstable_baselines/algo/ppo)
  * From paper: [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)


### Distributional RL

|              Algorithm              | `Box` |     `Discrete`     | `MultiDiscrete` | `MultiBinary` |
|:-----------------------------------:|:-----:|:------------------:|:---------------:|:-------------:|
| [C51](unstable_baselines/d/c51)     |  :x:  | :heavy_check_mark: |       :x:       |      :x:      |
| [QRDQN](unstable_baselines/d/qrdqn) |  :x:  | :heavy_check_mark: |       :x:       |      :x:      |
| [IQN](unstable_baselines/d/iqn)     |  :x:  | :heavy_check_mark: |       :x:       |      :x:      |

* 2021.04.28: Implemented [IQN](unstable_baselines/algo/d/iqn)
  * From paper: [arXiv:1806.06923](https://arxiv.org/abs/1806.06923)
* 2021.04.21: Implemented [QRDQN](unstable_baselines/algo/d/qrdqn)
  * From paper: [arXiv:1710.10044](https://arxiv.org/abs/1710.10044)
* 2021.04.20: Implemented [C51](unstable_baselines/algo/d/c51)
  * From paper: [arXiv:1707.06887](https://arxiv.org/abs/1707.06887)

<!---
### Hierarchical RL

| Algorithm | `Box`              | `Discrete`         | `MultiDiscrete`    | `MultiBinary`      |
|:-----------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|


### Other RL

| Algorithm | `Box`              | `Discrete`         | `MultiDiscrete`    | `MultiBinary`      |
|:-----------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|

--->

## Quick Start
This example shows how to train a PPO agent to play `CartPole-v0`. You can find the full scripts in [example/cartpole/train_ppo.py](example/cartpole/train_ppo.py).

First, import dependencies
```python
import gym
import unstable_baselines as ub
from unstable_baselines.algo.ppo import PPO
```
Create environments for training and evaluation
```python
# create environments
env = ub.envs.VecEnv([gym.make('CartPole-v0') for _ in range(10)])
eval_env = gym.make('CartPole-v0')
```
Create a PPO model and train it
```python
model = PPO(
    env,
    learning_rate=1e-3,
    gamma=0.8,
    batch_size=128,
    n_steps=500
).learn(  # train for 20000 steps
    20000,
    verbose=1
)
```
Save and load the trained model
```python
model.save('./my_ppo_model')
model = PPO.load('./my_ppo_model')
```
Evaluate the training results
```python
model.eval(eval_env, 20, 200, render=True)
# don't forget to close the environments!
env.close()
eval_env.close()
```

More examples:
* [Playing with Atari](example/atari)
* [Solving continuous control problems](example/pybullet)

## Update Logs
* 2021.05.22: Add benchmarks
* 2021.04.27: Update to framework v2: supports saving/loading the best performed checkpoints.
