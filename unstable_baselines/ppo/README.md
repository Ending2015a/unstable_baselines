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

![PPO discrete](https://g.gravizo.com/source/svg/ppo_discrete?https://raw.githubusercontent.com/Ending2015a/unstable_baselines/master/unstable_baselines/ppo/README.md)

<details>
<summary></summary>
ppo_discrete
    digraph D {
        compound=true;
        splines=false;
        bgcolor="transparent";
        node [shape=box, color=black, fontsize=12, height=0.1, width=0.1];
        
        obs[label="Observation"];
        
        subgraph cluster_cnn{
            label="Nature CNN";
            graph[style=dotted];
            nature_cnn [label=<
            <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="2">
                <TR><TD PORT="conv1">Conv2D(32, 8, 4)</TD></TR>
                <TR><TD PORT="relu1">ReLU</TD></TR>
                <TR><TD PORT="conv2">Conv2D(64, 4, 2)</TD></TR>
                <TR><TD PORT="relu2">ReLU</TD></TR>
                <TR><TD PORT="conv3">Conv2D(32, 3, 1)</TD></TR>
                <TR><TD PORT="relu3">ReLU</TD></TR>
                <TR><TD PORT="dense">Dense(512)</TD></TR>
                <TR><TD PORT="relu4">ReLU</TD></TR>
            </TABLE>>, shape=plaintext];
        }    
        
        policy_net [label=<
            <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="2">
                <TR><TD PORT="dense">Dense(action_space)</TD></TR>
            </TABLE>>, shape=plaintext];
        
        value_net [label=<
            <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="2">
                <TR><TD PORT="dense">Dense(1)</TD></TR>
            </TABLE>>, shape=plaintext];
            
        
        obs -> nature_cnn[ltail=obs, lhead=cluster_cnn, label=" "];
        
        nature_cnn:s->{policy_net, value_net}[ltail=cluster_cnn];
        policy_net -> pi;
        value_net -> v;
        
        pi[label="Action"];
        v[label="Value"]
    }
ppo_discrete
</details>
