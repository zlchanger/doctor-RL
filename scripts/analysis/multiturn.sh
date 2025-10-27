mkdir -p log/terminal

# bash train.sh sokoban \
#     model.experiment_name=sokoban_main \
#     system.cuda_visible_devices=0 \
#     training.micro_batch_size=2 \
#     training.total_training_steps=100 \
#     trainer.test_freq=10  >> ./log/terminal/sokoban_main.log &

# bash train.sh sokoban \
#     model.experiment_name=sokoban_main_no_think_rl \
#     system.cuda_visible_devices=1 \
#     training.micro_batch_size=2 \
#     training.total_training_steps=100 \
#     training.no_think_rl=True \
#     trainer.test_freq=10  >> ./log/terminal/sokoban_main_no_think_rl.log &

    
# bash train.sh frozenlake \
#     model.experiment_name=frozenlake_main \
#     system.cuda_visible_devices=2 \
#     training.micro_batch_size=2 \
#     training.total_training_steps=100 \
#     trainer.test_freq=10  >> ./log/terminal/frozenlake_main.log &


# bash train.sh frozenlake \
#     model.experiment_name=frozenlake_main_no_think_rl \
#     system.cuda_visible_devices=3 \
#     training.micro_batch_size=2 \
#     training.total_training_steps=100 \
#     training.no_think_rl=True \
#     trainer.test_freq=10  >> ./log/terminal/frozenlake_main_no_think_rl.log &





# bash train.sh sokoban \
#     model.experiment_name=sokoban_arpo \
#     system.cuda_visible_devices=0 \
#     training.micro_batch_size=2 \
#     training.total_training_steps=100 \
#     optimization.adv_estimator=arpo \
#     trainer.test_freq=10  >> ./log/terminal/sokoban_arpo.log &

# bash train.sh sokoban \
#     model.experiment_name=sokoban_brpo \
#     system.cuda_visible_devices=1 \
#     training.micro_batch_size=2 \
#     training.total_training_steps=100 \
#     optimization.adv_estimator=brpo \
#     trainer.test_freq=10  >> ./log/terminal/sokoban_brpo.log &

    
# bash train.sh frozenlake \
#     model.experiment_name=frozenlake_arpo \
#     system.cuda_visible_devices=2 \
#     training.micro_batch_size=2 \
#     training.total_training_steps=100 \
#     optimization.adv_estimator=arpo \
#     trainer.test_freq=10  >> ./log/terminal/frozenlake_arpo.log &


# bash train.sh frozenlake \
#     model.experiment_name=frozenlake_brpo \
#     system.cuda_visible_devices=3 \
#     training.micro_batch_size=2 \
#     training.total_training_steps=100 \
#     optimization.adv_estimator=brpo \
#     trainer.test_freq=10  >> ./log/terminal/frozenlake_brpo.log &


######## BELOW IS UNDER CONSTRUCTION ########
export PYTHONHASHSEED=10000
export DIM_X=6
export DIM_Y=6
export NUM_BOXES=1

# generate normal difficulty dataset
export MAX_STEPS=10
export SEARCH_DEPTH=5
export data_dir="data/sokoban_single_step"

python ragen/env/sokoban/create_dataset.py \
    --output $data_dir \
    --seed 10000 \
    --train_size 100 \
    --test_size 50 \
    --prefix qwen-instruct


bash train.sh sokoban \
    model.experiment_name=sokoban_onestep_debug \
    system.cuda_visible_devices=0 \
    training.micro_batch_size=1 \
    training.total_training_steps=100 \
    optimization.adv_estimator=gae \
    env.data_dir=$data_dir \
    training.max_turns=1 \
    trainer.test_freq=10  >> ./log/terminal/sokoban_onestep_debug.log