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

<img src='https://g.gravizo.com/svg?
digraph D {
    splines=false;
    node [shape=box, color=black, fontsize=12, height=0.1, width=0.1];
    input1[label="Observation"];
    input2[shape=record, label="Observation|Action"];
    subgraph cluster_actor{
        label="Actor";
        labeljust="l";
        graph[style=dotted];
        actor [shape=record, label="{Dense(400)|ReLU|Dense(300)|ReLU|Dense(Action space)|Tanh}"]
    }
    subgraph cluster_critic{
        label="Critic";
        labeljust="l";
        graph[style=dotted];
        critic [shape=record, label="{Dense(400)|ReLU|Dense(300)|ReLU|Dense(1)}"]
    }    
    input1 -> actor:n;
    input2 -> critic:n;
    actor:s -> pi;
    critic:s -> v;
    pi[label="Action"];
    v[label="Value"]
}'/>

