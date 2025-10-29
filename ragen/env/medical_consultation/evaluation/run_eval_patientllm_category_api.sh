#!/bin/bash

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# 强制使用IPv4
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=en,eth,em,bond
export NCCL_DEBUG=INFO
export API_KEY="YOUR_API_KEY"
# Inference

# 设置模型路径
# MODEL_PATH="Qwen2.5-7B-Instruct"
MODEL_PATH=$1
OUTPUT_PREFIX=${2:-test_en_output_multi_gpu_patientllm_iter}

# 迭代次数遍历 (4,6,8)
for MAX_ITERATIONS in 10; do
    OUTPUT_PREFIX_ITER="${OUTPUT_PREFIX}_${MAX_ITERATIONS}"
    # 设置输入和输出文件路径
    INPUT_FILE="data/MTMedDialog_test.json"
    OUTPUT_DIR="outputs/$MODEL_PATH/"

    # 创建输出目录
    mkdir -p "$OUTPUT_DIR"

    # 设置批处理大小和其他参数
    BATCH_SIZE=32
    TEMPERATURE=0.7
    TOP_P=0.9

    # 清理可能存在的旧进程
    pkill -f "python.*inference_fast_v2.py"

    # 生成随机端口号以避免冲突
    RANDOM_PORT=$((40000 + RANDOM % 500))

    # 使用torchrun启动分布式训练
    torchrun \
        --nproc_per_node=8 \
        --master_port=$RANDOM_PORT \
        --master_addr=127.0.0.1 \
        ragen/env/medical_consultation/evaluation/inference_fast_for_patientllm_with_api.py \
        --doctor_model_name $MODEL_PATH \
        --patient_model_path Qwen2.5-7B-Instruct \
        --input_file $INPUT_FILE \
        --output_dir $OUTPUT_DIR \
        --output_prefix $OUTPUT_PREFIX_ITER \
        --max_iterations $MAX_ITERATIONS \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --batch_size $BATCH_SIZE \
        --verbose \
        2>&1 | tee $OUTPUT_DIR/output.log

    # Eval

    # 设置输入输出路径
    SIMULATION_DATA="$OUTPUT_DIR/$OUTPUT_PREFIX_ITER.json"
    REFERENCE_DATA="data/MTMedDialog_test.json"
    OUTPUT_FILE="$OUTPUT_DIR/eval_scores_patientllm_category_iter$MAX_ITERATIONS.json"

    # 设置批处理大小和其他参数
    BATCH_SIZE=6
    ALPHA=1.0
    BETA=0.01

    # 生成随机端口号以避免冲突
    RANDOM_PORT=$((35000 + RANDOM % 500))

    # 使用torchrun启动分布式训练
    # --nproc_per_node=8 表示使用8个GPU
    # --master_port 设置主进程端口
    torchrun --nproc_per_node=8 \
        --master_addr=127.0.0.1 \
        --master_port=$RANDOM_PORT \
        ragen/env/medical_consultation/evaluation/evaluation_for_patientllm_category.py \
        --simulation_data $SIMULATION_DATA \
        --reference_data $REFERENCE_DATA \
        --output $OUTPUT_FILE \
        --batch_size $BATCH_SIZE \
        --alpha $ALPHA \
        --beta $BETA 
done