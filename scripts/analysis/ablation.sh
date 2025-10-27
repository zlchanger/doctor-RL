######################### Section 5.3.1 

# bash train.sh sokoban \
#     model.experiment_name=sokoban_ablation_p2n64 \
#     system.cuda_visible_devices=0 \
#     training.micro_batch_size=2 \
#     training.total_training_steps=100 \
#     training.train_batch_size=2 \
#     training.n_rollout=64 \
#     trainer.test_freq=10  >> ./log/terminal/sokoban_ablation_p2n64.log &

# bash train.sh sokoban \
#     model.experiment_name=sokoban_ablation_p4n32 \
#     system.cuda_visible_devices=1 \
#     training.micro_batch_size=2 \
#     training.total_training_steps=100 \
#     training.train_batch_size=4 \
#     training.n_rollout=32 \
#     trainer.test_freq=10  >> ./log/terminal/sokoban_ablation_p4n32.log &

# bash train.sh sokoban \
#     model.experiment_name=sokoban_ablation_p16n8 \
#     system.cuda_visible_devices=2 \
#     training.micro_batch_size=2 \
#     training.total_training_steps=100 \
#     training.train_batch_size=16 \
#     training.n_rollout=8 \
#     trainer.test_freq=10  >> ./log/terminal/sokoban_ablation_p16n8.log &

# bash train.sh sokoban \
#     model.experiment_name=sokoban_ablation_p32n4 \
#     system.cuda_visible_devices=3 \
#     training.micro_batch_size=2 \
#     training.total_training_steps=100 \
#     training.train_batch_size=32 \
#     training.n_rollout=4 \
#     trainer.test_freq=10  >> ./log/terminal/sokoban_ablation_p32n4.log &


######################### Section 5.3.3 

# bash train.sh sokoban \
#     model.experiment_name=sokoban_ablation_offline2 \
#     system.cuda_visible_devices=0 \
#     training.micro_batch_size=2 \
#     training.ppo_batch_size=128 \
#     training.n_rollout=16 \
#     training.train_batch_size=16 \
#     training.total_training_steps=50 \
#     trainer.test_freq=5  >> ./log/terminal/sokoban_ablation_offline2.log &


# bash train.sh sokoban \
#     model.experiment_name=sokoban_ablation_offline5 \
#     system.cuda_visible_devices=1 \
#     training.micro_batch_size=2 \
#     training.ppo_batch_size=128 \
#     training.n_rollout=16 \
#     training.train_batch_size=40 \
#     training.total_training_steps=20 \
#     trainer.test_freq=2  >> ./log/terminal/sokoban_ablation_offline5.log &


# bash train.sh sokoban \
#     model.experiment_name=sokoban_ablation_offline10 \
#     system.cuda_visible_devices=2 \
#     training.micro_batch_size=2 \
#     training.ppo_batch_size=128 \
#     training.n_rollout=16 \
#     training.train_batch_size=80 \
#     training.total_training_steps=10 \
#     trainer.test_freq=1 >> ./log/terminal/sokoban_ablation_offline10.log &


# bash train.sh sokoban \
#     model.experiment_name=sokoban_ablation_offline20 \
#     system.cuda_visible_devices=3 \
#     training.micro_batch_size=2 \
#     training.ppo_batch_size=128 \
#     training.n_rollout=16 \
#     training.train_batch_size=160 \
#     training.total_training_steps=5 \
#     trainer.test_freq=1 >> ./log/terminal/sokoban_ablation_offline20.log &

# bash train.sh sokoban \
#     model.experiment_name=sokoban_ablation_p64n2 \
#     system.cuda_visible_devices=0 \
#     training.micro_batch_size=2 \
#     training.total_training_steps=100 \
#     training.train_batch_size=64 \
#     training.n_rollout=2 \
#     trainer.test_freq=10  >> ./log/terminal/sokoban_ablation_p64n2.log &



######################### Section 5.3.4

bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-0.5B \
    model.experiment_name=sokoban_abl_base \
    system.cuda_visible_devices=1 \
    training.micro_batch_size=2 \
    training.total_training_steps=100 \
    trainer.test_freq=10  >> ./log/terminal/sokoban_abl_base.log &

bash train.sh frozenlake \
    model.base_model=Qwen/Qwen2.5-0.5B \
    model.experiment_name=frozenlake_abl_base \
    system.cuda_visible_devices=3 \
    training.micro_batch_size=2 \
    training.total_training_steps=100 \
    trainer.test_freq=10  >> ./log/terminal/frozenlake_abl_base.log &

