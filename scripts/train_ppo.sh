#!/bin/bash
# usage:
#   ./train.sh --rank 0 --seed 1 --trace "ppo-test-1" "BeamRiderNoFrameskip-v4"
# batch training on multiple environments:
#   ./train.sh --rank 0 --seed 1 --trace "ppo-test-1" \
#                                "SeaquestNoFrameskip-v4" \
#                                "BeamRiderNoFrameskip-v4" \
#                                "PongNoFrameskip-v4"

rank=0
seed=1
trace="ppo"
env_ids=('BeamRiderNoFrameskip-v4')

exe=unstable_baselines.algo.ppo.run

function train() {
    # train ${rank} ${seed} ${end_id}
    echo "Start training, rank=$1, seed=$2, trace=$3 env_id=$4"
    
    python -m $exe --root "./log/$4/$3/$1" \
                   --env_id $4 \
                   --seed $2 \
                   --eval_seed 0 \
                   --n_envs 8
}

# Formalize arguments
ARGS=`getopt -o r:s:t: -l rank:,seed:,trace: -n "$0" -- "$@"`

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
    -t|--trace) trace="$2"; shift;;
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
    train "$rank" "$seed" "$trace" "$env_id"
done

