# NOTE only tested with 1 GPU

set -x

nproc_per_node=$1
save_path=$2

if [ ! -d $save_path ]; then
    mkdir -p $save_path
fi

shift 2
torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m ragen.trainer.fsdp_sft_trainer \
    data.train_files=data/MTMedDialog_sft_train.parquet \
    data.val_files=data/MTMedDialog_sft_val.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=6784 \
    optim.lr=1e-4 \
    data.train_batch_size=128 \
    data.micro_batch_size=8 \
    +data.with_thinking=False \
    model.partial_pretrain=Qwen/Qwen2.5-7B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.experiment_name=med_dialogue-sft-thinking-lora-Qwen2.5-7B-Instruct \
    trainer.project_name=Medical-Dialogue \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=3 \
    trainer.default_hdfs_dir=null $@ \
    trainer.validate_before_training=True \
    model.lora_rank=64 \
    model.lora_alpha=32 \
    model.target_modules=all-linear \
    model.enable_gradient_checkpointing=True \
    2>&1 | tee  $save_path/train.log

# Find the latest checkpoint directory
latest_checkpoint=$(ls -dt $save_path/global_step_* 2>/dev/null | head -n 1)

if [ -z "$latest_checkpoint" ]; then
    echo "Error: No checkpoint found in $save_path"
    exit 1
fi

echo "Using checkpoint: $latest_checkpoint"

python sft/utils/merge_lora.py \
    --base_model_name Qwen/Qwen2.5-7B-Instruct \
    --lora_model_path $latest_checkpoint \
    --output_path DoctorLLM-7B-SFT-1000-thinking