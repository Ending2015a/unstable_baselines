#!/bin/bash
# example: ./train.sh 0 1
rank=$1
seed=$2


python td3.py  --rank $rank --seed $seed --logdir='./log/{env_id}/td3_1m/{rank}' --logging='training.log' \
               --monitor_dir='monitor' --tb_logdir='' --model_dir='model' \
               --env_id="HalfCheetahBulletEnv-v0" --num_envs=1 --num_episodes=1000 --min_buffer=10000 \
               --num_steps=1000 --gradient_steps=1000 --batch_size=100 --verbose=2