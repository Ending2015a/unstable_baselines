#!/bin/bash
# usage:
#   ./train.sh --rank 0 --seed 1 "BeamRiderNoFrameskip-v4"
# batch training on multiple environments:
#   ./train.sh --rank 0 --seed 1 "SeaquestNoFrameskip-v4" \
#                                "BeamRiderNoFrameskip-v4" \
#                                "PongNoFrameskip-v4"


rank=0
seed=1
env_ids=('BeamRiderNoFrameskip-v4')

exe=unstable_baselines.dqn.run

function train() {
    # train ${rank} ${seed} ${end_id}
    echo "Start training, rank=$1, seed=$2, env_id=$3"
    
    python -m $exe --rank $1 --seed $2 \
                   --logdir='./log/{env_id}/dqn/{rank}' \
                   --logging='training.log' \
                   --monitor_dir='monitor' \
                   --tb_logdir='' \
                   --model_dir='model' \
                   --env_id="$3" \
                   --log_interval=1000 \
                   --eval_interval=10000 \
                   --save_interval=10000 \
                   --num_envs=8 \
                   --num_epochs=312500 \
                   --num_steps=4 \
                   --num_gradsteps=1 \
                   --batch_size=256 \
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
}

# Formalize arguments
ARGS=`getopt -o r:s: -l rank:,seed: -n "$0" -- "$@"`

if [ $? -ne 0 ]; then
    echo "Terminating..." >&2
    exit 1
fi

eval set -- "$ARGS"

# Parse arguments
while true; do
    case "$1" in
    -r|--rank) rank="$2"; shift;;
    -s|--seed) seed="$2"; shift;;
    --) shift; break;;
    *)
        echo "Unknown args: $@"
        exit 1
    esac
    shift
done

if [[ $# -gt 0 ]]; then
    env_ids=("$@")
fi

# Start training
for env_id in "${env_ids[@]}"; do
    train "$rank" "$seed" "$env_id"
done
