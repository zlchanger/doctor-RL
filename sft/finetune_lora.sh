# NOTE only tested with 1 GPU

set -x

# Set default environment type
env_type=${1:-sokoban}

shift 1

if [ "$#" -lt 2 ]; then
    echo "Usage: finetune_lora.sh [env_type] <nproc_per_node> <save_path> [other_configs...]"
    echo "env_type defaults to 'sokoban' if not specified"
    exit 1
fi

nproc_per_node=$1
save_path=$2

if [ ! -d $save_path ]; then
    mkdir -p $save_path
fi

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m ragen.trainer.fsdp_sft_trainer \
    data.train_files=data/sft/${env_type}/train.parquet \
    data.val_files=data/sft/${env_type}/test.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=2048 \
    optim.lr=1e-4 \
    data.train_batch_size=128 \
    data.micro_batch_size=4 \
    model.partial_pretrain=Qwen/Qwen2.5-0.5B \
    trainer.default_local_dir=$save_path \
    trainer.experiment_name=test_zpy_${env_type}-sft-lora-qwen-2.5-0.5b-base \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=5 \
    trainer.default_hdfs_dir=null $@ \
    trainer.validate_before_training=True \
    model.lora_rank=64 \
    model.lora_alpha=32 \
    model.target_modules=all-linear \
    model.enable_gradient_checkpointing=False \
    2>&1 | tee  $save_path/train.log

    # Or you can do this:
    # model.target_modules=all-linear \
