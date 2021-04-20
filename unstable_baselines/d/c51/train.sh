#!/bin/bash

rank=$1
seed=$2
env_id=$3

c51=unstable_baselines.d.c51.run

python -m $c51 --rank $rank --seed $seed \
               --logdir='./log/{env_id}/c51/{rank}' \
               --logging='training.log' \
               --monitor_dir='monitor' \
               --tb_logdir='' \
               --model_path='model/weights' \
               --env_id="${env_id}" \
               --log_interval=1000 \
               --eval_interval=10000 \
               --save_interval=10000 \
               --num_atoms=51 \
               --v_min=-10.0 \
               --v_max=10.0 \
               --num_envs=8 \
               --num_epochs=312500 \
               --num_steps=4 \
               --num_gradsteps=1 \
               --batch_size=128 \
               --buffer_size=1000000 \
               --min_buffer=50000 \
               --target_update=625 \
               --explore_rate=1.0 \
               --explore_final=0.05 \
               --explore_progress=0.1 \
               --verbose=2 \
               --record_video
