#!/bin/bash

# Function to run a single experiment
run_experiment() {
    local exp_name=$1
    local data_dir=$2
    local low_risk=$3
    local high_risk=$4
    local gpu_id=$5
    local no_think_rl=${6:-false}  # Optional parameter, defaults to false
    local low_risk_val=${7:-""}    # Optional validation set low risk arm
    local high_risk_val=${8:-""}   # Optional validation set high risk arm

    # Export environment variables
    export LOW_RISK_NAME=$low_risk
    export HIGH_RISK_NAME=$high_risk
    if [ -n "$low_risk_val" ]; then
        export LOW_RISK_VAL_NAME=$low_risk_val
        export HIGH_RISK_VAL_NAME=$high_risk_val
    fi

    # Create dataset
    python ragen/env/bandit/create_dataset_two_armed.py \
        --output $data_dir \
        --seed 100000 \
        --train_size 10000 \
        --test_size 500 \
        --prefix qwen-instruct

    # Common training parameters
    local common_params="
        env.data_dir=$data_dir \
        env.env_kwargs.low_risk_name=$LOW_RISK_NAME \
        env.env_kwargs.high_risk_name=$HIGH_RISK_NAME \
        system.cuda_visible_devices=$gpu_id \
        training.micro_batch_size=1 \
        training.train_batch_size=128 \
        training.ppo_batch_size=128 \
        training.max_turns=1 \
        training.n_rollout=1 \
        training.total_training_steps=50 \
        trainer.test_freq=10 \
        optimization.adv_estimator=brpo"

    # Add validation set arms if provided
    if [ -n "$low_risk_val" ]; then
        common_params="$common_params \
        env.env_kwargs.low_risk_val_name=$LOW_RISK_VAL_NAME \
        env.env_kwargs.high_risk_val_name=$HIGH_RISK_VAL_NAME"
    fi

    # Run main experiment
    local current_exp_name=$exp_name
    if [ "$no_think_rl" = true ]; then
        current_exp_name="${exp_name}_no_think_rl"
        common_params="$common_params training.no_think_rl=True"
    fi

    bash train.sh two_armed_bandit \
        model.experiment_name=$current_exp_name \
        $common_params >> "./log/terminal/${current_exp_name}.log" &
}

mkdir -p log/terminal

# Run original experiments (without validation arms)
# run_experiment "bandit_main" "data/two_armed_bandit" "phoenix" "dragon" 0
# run_experiment "bandit_main" "data/two_armed_bandit" "phoenix" "dragon" 1 true

# Run reverse experiments (without validation arms)
# run_experiment "bandit_reverse" "data/two_armed_bandit_reverse" "dragon" "phoenix" 2
# run_experiment "bandit_reverse" "data/two_armed_bandit_reverse" "dragon" "phoenix" 3 true

# Run generalization experiments (without validation arms)
# run_experiment "bandit_genea_regular" "data/two_armed_bandit_genea_regular" "teacher" "engineer" 0
# run_experiment "bandit_genea_regular" "data/two_armed_bandit_genea_regular" "teacher" "engineer" 1 true

# Run generalization experiments with different validation arms
# run_experiment "bandit_genea_reverse" "data/two_armed_bandit_genea_reverse" "engineer" "teacher" 2
# run_experiment "bandit_genea_reverse" "data/two_armed_bandit_genea_reverse" "engineer" "teacher" 3 true

run_experiment "bandit_genea_reverse_testdiff" "data/two_armed_bandit_genea_reverse" "engineer" "teacher" 0 false "trader"   "librarian"
# run_experiment "bandit_genea_reverse_testdiff" "data/two_armed_bandit_genea_reverse" "engineer" "teacher" 1 true "trader"   "librarian"
