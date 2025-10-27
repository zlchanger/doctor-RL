#!/bin/bash

# Function to set environment variables based on environment type
set_env_vars() {
    local env_type=$1
    if [ "$env_type" = "sokoban" ]; then
        export DIM_X=6
        export DIM_Y=6
        export NUM_BOXES=1
        export MAX_STEPS=5
        export SEARCH_DEPTH=30
    elif [ "$env_type" = "frozenlake" ]; then
        export SIZE=6
        export P=0.8
        export IS_SLIPPERY=True
    fi
}

# Get environment type from command line argument
ENV_TYPE=${1:-sokoban}  # Default to sokoban if no argument provided

# Set environment variables
set_env_vars $ENV_TYPE

# Run the appropriate script based on environment
if [ "$ENV_TYPE" = "sokoban" ]; then
    python sft/utils/generate_sft_verl_sokoban.py \
        --env sokoban \
        --algo bfs \
        --seed 100000 \
        --output data/sft/sokoban \
        --train_size 1000 \
        --test_size 100 \
        --bfs_max_depths 100 \
        --prefix message \
        --num_processes 16
elif [ "$ENV_TYPE" = "frozenlake" ]; then
    python sft/utils/generate_sft_verl_frozenlake.py \
        --env frozenlake \
        --algo bfs \
        --seed 100000 \
        --output data/sft/frozenlake \
        --train_size 1000 \
        --test_size 100 \
        --bfs_max_depths 100 \
        --prefix message \
        --num_processes 16
fi