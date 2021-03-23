# Proximal Policy Optimization (PPO)

> [Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal Policy Optimization Algorithms. *arXiv:1707.06347*, 2017.](https://arxiv.org/abs/1707.06347)


## How to use
```python
python -m unstable_baselines.ppo.run --rank 0 --seed 1 --logdir='./log/{env_id}/ppo_20m/{rank}' \
               --logging='training.log' --monitor_dir='monitor' --tb_logdir='' --model_dir='model' \
               --env_id="BreakoutNoFrameskip-v0" --num_envs=8 --num_episodes=20000 \
               --num_steps=128 --num_epochs=4 --batch_size=256 --verbose=2
```

Total timesteps (Samples) = num_envs * num_steps * num_episodes (~20M in this case)


## Atari 2600

### Video

> Best video

### Learning Curve

> Learning curve


### Hyperparametrs
| | `num_envs` | `num_episodes` | `num_steps` | `num_epochs` | `batch_size` |
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


<img src='https://g.gravizo.com/svg?digraph%20D%20%7B%0A%20%20%20%20splines%3Dfalse%3B%0A%20%20%20%20bgcolor%3D%22white%22%3B%0A%20%20%20%20node%20%5Bshape%3Dbox%2C%20color%3Dblack%2C%20fontsize%3D12%2C%20height%3D0.1%2C%20width%3D0.1%5D%3B%0A%20%20%20%20obs%5Blabel%3D%22Observation%22%5D%3B%0A%20%20%20%20subgraph%20cluster_cnn%7B%0A%20%20%20%20%20%20%20%20label%3D%22Nature%20CNN%22%3B%0A%20%20%20%20%20%20%20%20labeljust%3D%22l%22%3B%0A%20%20%20%20%20%20%20%20graph%5Bstyle%3Ddotted%5D%3B%0A%20%20%20%20%20%20%20%20nature_cnn%20%5Bshape%3Drecord%2C%20label%3D%22%7BConv2D%2832%2C%208%2C%204%29%7CReLU%7CConv2D%2864%2C%204%2C%202%29%7CReLU%7CConv2D%2832%2C%203%2C%201%29%7CReLU%7CDense%28512%29%7CReLU%7D%22%5D%0A%20%20%20%20%7D%0A%20%20%20%20subgraph%20cluster_policy%7B%0A%20%20%20%20%20%20%20%20label%3D%22Policy%22%3B%0A%20%20%20%20%20%20%20%20labeljust%3D%22l%22%3B%0A%20%20%20%20%20%20%20%20graph%5Bstyle%3Ddashed%5D%3B%0A%20%20%20%20%20%20%20%20policy_net%20%5Bshape%3Drecord%2C%20label%3D%22Dense%28Action%20space%29%22%5D%3B%0A%20%20%20%20%7D%0A%20%20%20%20subgraph%20cluster_value%7B%0A%20%20%20%20%20%20%20%20label%3D%22Value%22%3B%0A%20%20%20%20%20%20%20%20labeljust%3D%22l%22%3B%0A%20%20%20%20%20%20%20%20graph%5Bstyle%3Ddashed%5D%3B%0A%20%20%20%20%20%20%20%20value_net%20%5Bshape%3Drecord%2C%20label%3D%22Dense%281%29%22%5D%3B%0A%20%20%20%20%7D%0A%20%20%20%20obs%20-%3E%20nature_cnn%3B%0A%20%20%20%20nature_cnn%3As-%3E%7Bpolicy_net%2C%20value_net%7D%3B%0A%20%20%20%20policy_net%20-%3E%20pi%3B%0A%20%20%20%20value_net%20-%3E%20v%3B%0A%20%20%20%20pi%5Blabel%3D%22Action%22%5D%3B%0A%20%20%20%20v%5Blabel%3D%22Value%22%5D%0A%7D'>

<img src='https://g.gravizo.com/svg?
digraph D {
    splines=false;
    bgcolor="white";
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
}'/>
