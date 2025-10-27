#!/bin/bash
#
# Sokoban 快速测试脚本
# 用于测试 RAGEN 框架在 Sokoban 环境上的训练
#

set -e  # 遇到错误立即退出

echo "========================================="
echo "RAGEN - Sokoban 训练测试"
echo "========================================="
echo ""

# ============================================
# 配置区域 - 根据你的环境修改
# ============================================

# GPU配置
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ATTENTION_BACKEND=XFORMERS

# 模型路径 - 修改为你的实际路径
BASE_MODEL="/opt/tiger/Qwen2.5-7B-Instruct"
# BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"  # 或使用 HuggingFace 自动下载

# 项目名称
PROJECT_NAME="RAGEN-Sokoban"
EXP_NAME="sokoban-test-$(date +%Y%m%d-%H%M%S)"

echo "配置信息:"
echo "  - 模型: $BASE_MODEL"
echo "  - GPU: $CUDA_VISIBLE_DEVICES"
echo "  - 项目: $PROJECT_NAME"
echo "  - 实验: $EXP_NAME"
echo ""

# ============================================
# 选择训练模式
# ============================================

echo "请选择训练模式:"
echo "  1) 快速测试 (5 steps, 小 batch size) - 推荐用于验证环境"
echo "  2) 小规模训练 (50 steps, 适合调试)"
echo "  3) 完整训练 (200 steps, 参考论文设置)"
echo ""
read -p "请输入选项 [1-3, 默认为1]: " mode

mode=${mode:-1}

case $mode in
    1)
        echo ""
        echo "🧪 模式 1: 快速测试 (约5分钟)"
        TRAIN_BATCH_SIZE=4
        PPO_BATCH_SIZE=4
        N_ROLLOUT=2
        TOTAL_STEPS=5
        MICRO_BATCH_SIZE=1
        ;;
    2)
        echo ""
        echo "🔧 模式 2: 小规模训练 (约30分钟)"
        TRAIN_BATCH_SIZE=8
        PPO_BATCH_SIZE=16
        N_ROLLOUT=4
        TOTAL_STEPS=50
        MICRO_BATCH_SIZE=2
        ;;
    3)
        echo ""
        echo "🚀 模式 3: 完整训练 (数小时)"
        TRAIN_BATCH_SIZE=16
        PPO_BATCH_SIZE=64
        N_ROLLOUT=16
        TOTAL_STEPS=200
        MICRO_BATCH_SIZE=4
        ;;
    *)
        echo "无效选项，使用默认模式 1"
        TRAIN_BATCH_SIZE=4
        PPO_BATCH_SIZE=4
        N_ROLLOUT=2
        TOTAL_STEPS=5
        MICRO_BATCH_SIZE=1
        ;;
esac

echo ""
echo "训练参数:"
echo "  - train_batch_size: $TRAIN_BATCH_SIZE (每次更新的 rollout 数量)"
echo "  - ppo_batch_size: $PPO_BATCH_SIZE (PPO 更新批量)"
echo "  - n_rollout: $N_ROLLOUT (每个 prompt 的响应数, GRPO)"
echo "  - total_steps: $TOTAL_STEPS (总训练步数)"
echo "  - micro_batch_size: $MICRO_BATCH_SIZE (梯度累积)"
echo ""

read -p "按 Enter 开始训练，或 Ctrl+C 取消..."

# ============================================
# 启动训练
# ============================================

echo ""
echo "🚀 开始训练..."
echo ""

# 使用 train.sh 脚本调用
bash train.sh sokoban \
    model.base_model="$BASE_MODEL" \
    model.experiment_name="$EXP_NAME" \
    training.train_batch_size=$TRAIN_BATCH_SIZE \
    training.ppo_batch_size=$PPO_BATCH_SIZE \
    training.micro_batch_size=$MICRO_BATCH_SIZE \
    training.n_rollout=$N_ROLLOUT \
    training.max_turns=5 \
    training.temperature=0.7 \
    training.total_training_steps=$TOTAL_STEPS \
    training.val_data_num=10 \
    optimization.adv_estimator=grpo \
    optimization.actor_lr=1e-6 \
    optimization.kl_coef=0.01 \
    trainer.project_name="$PROJECT_NAME" \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    system.cuda_visible_devices="$CUDA_VISIBLE_DEVICES" \
    system.n_gpus=4

# ============================================
# 训练完成
# ============================================

echo ""
echo "========================================="
echo "✅ 训练完成!"
echo "========================================="
echo ""
echo "📁 Checkpoint 位置:"
echo "   checkpoints/$PROJECT_NAME/$EXP_NAME/"
echo ""
echo "📊 查看日志:"
echo "   - WandB: 如果配置了 WANDB_API_KEY"
echo "   - 本地日志: outputs/exp_configs/logs/"
echo ""
echo "🎮 Sokoban 环境说明:"
echo "   - 任务: 推箱子游戏 (6x6 网格, 1个箱子)"
echo "   - 目标: 学习将箱子推到目标位置"
echo "   - 动作: up/down/left/right"
echo "   - 最大步数: 100"
echo ""
echo "💡 下一步:"
echo "   - 查看训练曲线 (reward, success rate)"
echo "   - 尝试模式 2 或 3 进行更完整的训练"
echo "   - 参考 ragen_cmd.md 中的超参数搜索实验"
echo ""
