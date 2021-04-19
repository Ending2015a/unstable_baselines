#!/bin/bash

rank=$1
seed=$2
env_id=$3

dqn=unstable_baselines.dqn.run

python -m $dqn --rank $rank --seed $seed \
               --logdir='./log/{env_id}/dqn/{rank}' \
               --logging='training.log' \
               --monitor_dir='monitor' \
               --tb_logdir='' \
               --model_path='model/weights' \
               --env_id="${env_id}" \
               --log_interval=1000 \
               --eval_interval=10000 \
               --save_interval=10000 \
               --num_envs=8 \
               --num_epochs=312500 \
               --num_steps=4 \
               --num_gradsteps=1 \
               --batch_size=128 \
               --buffer_size=1000000 \
               --min_buffer=50000 \
               --lr=3e-4 \
               --target_update=625 \
               --explore_rate=1.0 \
               --explore_final=0.05 \
               --explore_progress=0.1 \
               --verbose=2 \
               --huber \
               --record_video
