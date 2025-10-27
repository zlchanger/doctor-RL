#!/bin/bash
#
# DoctorAgent-RL 快速测试脚本
# 用于快速验证训练流程是否正常工作
# 使用小规模参数以加快测试速度
#

set -e  # 遇到错误立即退出

echo "========================================="
echo "DoctorAgent-RL 快速测试训练"
echo "========================================="
echo ""
echo "⚠️  注意: 这是一个测试脚本,使用极小的训练规模"
echo "   仅用于验证环境配置和训练流程"
echo "   不会产生有用的模型!"
echo ""

# ============================================
# 配置区域 - 根据你的环境修改
# ============================================

# GPU配置 (使用2个GPU进行测试)
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_ATTENTION_BACKEND=XFORMERS

# 模型路径 (需要修改为你的实际路径)
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"  # 或本地路径如 "./models/Qwen2.5-7B-Instruct"

# 项目名称
PROJECT_NAME="doctor-agent-quick-test"
EXP_NAME="test-$(date +%Y%m%d-%H%M%S)"

# 数据路径
TRAIN_DATA="data/MTMedDialog_RL.parquet"

# ============================================
# 训练参数 - 测试用的小规模配置
# ============================================

# 检查模型是否存在
if [ ! -d "$BASE_MODEL" ] && [ ! -f "${BASE_MODEL}/config.json" ]; then
    echo "❌ 错误: 模型路径不存在: $BASE_MODEL"
    echo ""
    echo "请执行以下操作之一:"
    echo "  1. 修改脚本中的 BASE_MODEL 变量为你的模型路径"
    echo "  2. 下载模型:"
    echo "     pip install huggingface_hub"
    echo "     huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/Qwen2.5-7B-Instruct"
    exit 1
fi

# 检查数据文件
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ 错误: 训练数据不存在: $TRAIN_DATA"
    exit 1
fi

echo "✅ 配置检查通过"
echo "   - 模型: $BASE_MODEL"
echo "   - 数据: $TRAIN_DATA"
echo "   - GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# ============================================
# 启动训练
# ============================================

echo "🚀 开始训练..."
echo "   项目: $PROJECT_NAME"
echo "   实验: $EXP_NAME"
echo ""

python -m ragen.trainer.main_ppo \
    data.train_files="$TRAIN_DATA" \
    data.train_batch_size=2 \
    actor_rollout_ref.rollout.n_agent=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.05 \
    env=medical_consultation \
    env.max_turns=3 \
    training.total_training_steps=5 \
    training.save_freq=3 \
    training.test_freq=5 \
    project_name="$PROJECT_NAME" \
    experiment_name="$EXP_NAME"

# ============================================
# 训练完成
# ============================================

echo ""
echo "========================================="
echo "✅ 测试训练完成!"
echo "========================================="
echo ""
echo "📁 Checkpoint位置:"
echo "   checkpoints/$PROJECT_NAME/$EXP_NAME/"
echo ""
echo "💡 下一步:"
echo "   1. 检查训练日志,确认没有错误"
echo "   2. 如果测试成功,可以使用完整训练脚本:"
echo "      bash scripts_exp/doctor-agent-rl-dynamic.sh"
echo ""
echo "📖 查看完整学习指南: cat LEARNING_GUIDE.md"
echo ""