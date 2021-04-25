#!/bin/bash
# usage:
#   ./train.sh --rank 0 --seed 1 "BeamRiderNoFrameskip-v4"
# batch training on multiple environments:
#   ./train.sh --rank 0 --seed 1 "SeaquestNoFrameskip-v4" \
#                                "BeamRiderNoFrameskip-v4" \
#                                "PongNoFrameskip-v4"

rank=0
seed=1
env_ids=('HalfCheetahBulletEnv-v0')

exe=unstable_baselines.sd3.run

function train() {
    # train ${rank} ${seed} ${end_id}
    echo "Start training, rank=$1, seed=$2, env_id=$3"
    
    python -m $exe --rank $1 --seed $2 \
                    --logdir='./log/{env_id}/sd3/{rank}' \
                    --logging='training.log' \
                    --monitor_dir='monitor' \
                    --tb_logdir='' \
                    --model_dir='model' \
                    --env_id=$3 \
                    --num_envs=1 \
                    --num_epochs=10000 \
                    --num_steps=100 \
                    --num_gradsteps=100 \
                    --batch_size=100 \
                    --buffer_size=1000000 \
                    --min_buffer=10000 \
                    --policy_update=2 \
                    --target_update=2 \
                    --lr=1e-3 \
                    --log_interval=1 \
                    --eval_interval=1000 \
                    --eval_episodes=5 \
                    --eval_max_steps=1000 \
                    --save_interval=1000 \
                    --verbose=1 \
                    --action_samples=50 \
                    --action_noise=0.2 \
                    --action_noise_clip=0.5 \
                    --explore_noise_mean=0 \
                    --explore_noise_scale=0.1 \
                    --explore_noise \
                    --importance_sampling \
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

