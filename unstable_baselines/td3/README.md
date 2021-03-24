# Twin Deplayed Deep Deterministic Policy Gradient (TD3)

> [Fujimoto, S., van Hoof, H., and Meger, D. Addressing Function Approximation Error in Actor-Critic Methods. *ICML 2018*.](https://arxiv.org/abs/1802.09477)


## How to run
```python
python -m unstable_baselines.td3.run  --rank 0 --seed 1 --logdir='./log/{env_id}/td3/{rank}' \
               --logging='training.log' --monitor_dir='monitor' --tb_logdir='' --model_dir='model' \
               --env_id="HalfCheetahBulletEnv-v0" --num_envs=1 --num_episodes=1000 --min_buffer=10000 \
               --num_steps=1000 --gradient_steps=1000 --batch_size=100 --verbose=2
```

* Total timesteps (Samples) = num_envs * num_steps * num_episodes (~1M in this case)


## PyBullet

### Video

| Environment | Video |
|-|-|
|`HalfCheetahBulletEnv-v0`|
| `AntBulletEnv-v0`|
|`Walker2DBulletEnv-v0`|
|`HumanoidBulletEnv-v0`|


### Learning Curve

> Learning curve

### Hyperparameters
| Environment               | `num_envs` | `num_episodes` | `num_steps` | `gradient_steps` | `batch_size` | `learing_rate` |`action_noise` |
|---------------------------|------------|----------------|-------------|------------------|--------------|----------------|----------------|
| `HalfCheetahBulletEnv-v0` | 1          | 1000           | 1000        | 1000             | 100          | 1e-3           | `None`         |
| `AntBulletEnv-v0`         | 1          | 1000           | 1000        | 1000             | 100          | 1e-3           | `None`         |
| `Walker2DBulletEnv-v0`    | 1          | 1000           | 1000        | 1000             | 100          | 1e-3           | `None`         |
| `HumanoidBulletEnv-v0`    | 4          | 2500           | 1000        | 1000             | 256          | 3e-4           | `None`         |



## Architecture

|             | `Box`              | `Discrete` | `MultiDiscrete` | `MultiBinary` |
|-------------|--------------------|------------|-----------------|---------------|
| Observation | :heavy_check_mark: | :x:        | :x:             | :x:           |
| Action      | :heavy_check_mark: | :x:        | :x:             | :x:           |


<br/>
<br/>

![](https://g.gravizo.com/source/svg/td3_arch?https%3A%2F%2Fraw.githubusercontent.com%2FEnding2015a%2Funstable_baselines%2Fmaster%2Funstable_baselines%2Ftd3%2FREADME.md)

<details>
<summary></summary>
td3_arch
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
}
td3_arch
</details>
