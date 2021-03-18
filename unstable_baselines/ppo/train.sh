#!/bin/bash

rank=$1
seed=$2

ppo=unstable_baselines.ppo.discrete

python -m $ppo --rank $rank --seed $seed --logdir='./log/{env_id}/ppo_20m/{rank}' --logging='training.log' \
               --monitor_dir='monitor' --tb_logdir='' --model_dir='model' \
               --env_id="BreakoutNoFrameskip-v0" --num_envs=8 --num_episodes=20000 \
               --num_steps=128 --num_epochs=4 --batch_size=256 --verbose=2
