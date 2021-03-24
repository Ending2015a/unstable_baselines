# Proximal Policy Optimization (PPO)

> [Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal Policy Optimization Algorithms. *arXiv:1707.06347*, 2017.](https://arxiv.org/abs/1707.06347)


## How to use
```python
python -m unstable_baselines.ppo.run --rank 0 --seed 1 --logdir='./log/{env_id}/ppo/{rank}' \
               --logging='training.log' --monitor_dir='monitor' --tb_logdir='' --model_dir='model' \
               --env_id="BreakoutNoFrameskip-v0" --num_envs=8 --num_episodes=20000 \
               --num_steps=128 --num_epochs=4 --batch_size=256 --verbose=2
```

Total timesteps (Samples) = num_envs * num_steps * num_episodes (~20M in this case)


## Atari 2600

### Video

| `BeamRider` | `Breakout` |
|-|-|
|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.BeamRiderNoFrameskip-v0.eval.gif" height=300px>|<img src="https://github.com/Ending2015a/unstable_baselines_assets/blob/master/images/ppo.BreakoutNoFrameskip-v0.eval.gif" height=300px>|



### Learning Curve

> Learning curve


### Hyperparametrs
| `env_id` | `num_envs` | `num_episodes` | `num_steps` | `num_epochs` | `batch_size` |
|-|:-:|:-:|:-:|:-:|:-:|
| `BeamRiderNoFrameskip-v0`| 1 | 1000 | 1000 | 1000 | 256 |
| `BreakoutNoFrameskip-v0` | 1 | 1000 | 1000 | 1000 | 256 |

## Architecture

|             | `Box` | `Discrete`         | `MultiDiscrete` | `MultiBinary` |
|:-----------:|:-----:|:------------------:|:---------------:|:-------------:|
| Observation |       | :heavy_check_mark: | :x:             | :x:           |
| Action      |       | :heavy_check_mark: | :x:             | :x:           |

<br/>
<br/>


![](https://g.gravizo.com/source/svg/ppo_discrete?https%3A%2F%2Fraw.githubusercontent.com%2FEnding2015a%2Funstable_baselines%2Fmaster%2Funstable_baselines%2Fppo%2FREADME.md)


<details>
<summary></summary>
ppo_discrete
digraph D {
    splines=false;
    bgcolor=white;
    node [shape=box, color=black, fontsize=12, height=0.1, width=0.1];
    obs[label="Observation"];
    subgraph cluster_cnn{
        label="Nature CNN";
        labeljust="l";
        graph[style=dotted];
        nature_cnn [shape=record, label="{Conv2D(32, 8, 4)|ReLU|Conv2D(64, 4, 2)|ReLU|Conv2D(32, 3, 1)|ReLU|Dense(512)|ReLU}"]
    }
    subgraph cluster_policy{
        label="Policy";
        labeljust="l";
        graph[style=dashed];
        policy_net [shape=record, label="Dense(Action space)"];
    }
    subgraph cluster_value{
        label="Value";
        labeljust="l";
        graph[style=dashed];
        value_net [shape=record, label="Dense(1)"];
    }
    obs -> nature_cnn;
    nature_cnn:s->{policy_net, value_net};
    policy_net -> pi;
    value_net -> v;
    pi[label="Action"];
    v[label="Value"]
}
ppo_discrete
</details>
