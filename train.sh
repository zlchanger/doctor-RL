#!/bin/bash
# RAGEN Training Script Wrapper
# Usage: bash train.sh <env_name> [key=value overrides...]
# Example: bash train.sh sokoban model.experiment_name=my_exp training.train_batch_size=128

set -e

# Check if at least environment name is provided
if [ $# -lt 1 ]; then
    echo "Usage: bash train.sh <env_name> [overrides...]"
    echo "Example: bash train.sh sokoban model.experiment_name=my_exp"
    exit 1
fi

# Run the Python training script with all arguments
python ragen/train.py "$@"