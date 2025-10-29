#!/usr/bin/env python3
"""
DoctorAgent-RL 快速入门脚本
用于探索数据集和理解项目结构
"""

import pandas as pd
import json
from pathlib import Path
import os

# STORAGE_OPTIONS = {
#     'key': os.environ['ALI_KEY'],
#     'secret': os.environ['ALI_SECRET'],
#     'endpoint': 'oss-cn-hangzhou.aliyuncs.com'
# }

def print_header(title):
    """打印格式化的标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def explore_rl_data():
    """探索RL训练数据"""
    print_header("1. RL训练数据探索 (MTMedDialog_RL.parquet)")

    # rl_df = pd.read_parquet('oss://buguk12/datasets/doctor_rl_data/MTMedDialog_RL.parquet',
    #                         storage_options=STORAGE_OPTIONS)
    rl_df = pd.read_parquet('/mnt/workspace/datasets/doctor_rl_data/MTMedDialog_RL.parquet')

    print(f"\n📊 数据集统计:")
    print(f"  总样本数: {len(rl_df):,}")
    print(f"  列名: {rl_df.columns.tolist()}")

    # 分析第一个样本
    sample = rl_df.iloc[0]
    reward_model = sample['reward_model']
    extra_info = sample['extra_info']

    print(f"\n📋 样本结构:")
    print(f"  - reward_model 字段:")
    print(f"    ├─ ground_truth (诊断标签): {reward_model['ground_truth']}")
    print(f"    ├─ patient_information (有效问答对数): {len(reward_model['patient_information'])}")
    print(f"    └─ enhanced_description (患者描述长度): {len(reward_model['enhanced_description'])} 字符")
    print(f"  - extra_info 字段:")
    print(f"    └─ index (样本ID/Seed): {extra_info['index']}")

    # 显示患者描述示例
    print(f"\n💬 患者描述示例:")
    description = reward_model['enhanced_description']
    print(f"  {description[:200]}...")

    # 显示问答对示例
    if len(reward_model['patient_information']) > 0:
        qa_pair = reward_model['patient_information'][0]
        print(f"\n🗣️  有效问答对示例:")
        print(f"  医生问题: {qa_pair.get('doctor_question', 'N/A')}")
        print(f"  患者回答: {qa_pair.get('patient_response', 'N/A')}")

    # 统计诊断标签分布
    print(f"\n📈 诊断标签分布 (Top 10):")
    diagnosis_counts = {}
    for item in rl_df['reward_model']:
        diagnosis = item['ground_truth']
        # ground_truth可能是字典或字符串
        if isinstance(diagnosis, dict):
            diagnosis_text = diagnosis.get('diagnosis', str(diagnosis))
        else:
            diagnosis_text = str(diagnosis)

        # 截取前100个字符用于统计（避免太长）
        diagnosis_key = diagnosis_text[:100]
        diagnosis_counts[diagnosis_key] = diagnosis_counts.get(diagnosis_key, 0) + 1

    sorted_diagnoses = sorted(diagnosis_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for diagnosis, count in sorted_diagnoses:
        print(f"  {diagnosis}... ({count} 样本)")

    return rl_df


def explore_sft_data():
    """探索SFT训练数据"""
    print_header("2. SFT训练数据探索 (MTMedDialog_sft_train.parquet)")

    sft_df = pd.read_parquet('/mnt/workspace/datasets/doctor_rl_data/MTMedDialog_sft_train.parquet')

    print(f"\n📊 数据集统计:")
    print(f"  总样本数: {len(sft_df):,}")
    print(f"  列名: {sft_df.columns.tolist()}")

    # 分析第一个样本
    sample = sft_df.iloc[0]

    print(f"\n📋 对话格式示例:")
    print(f"\n  [Prompt (患者初始描述)]")
    print(f"  {sample['prompt'][:200]}...")

    print(f"\n  [Response (医生推理+回答)]")
    response = sample['response']

    # 尝试提取<think>和<answer>部分
    if '<think>' in response and '</think>' in response:
        think_start = response.find('<think>') + 7
        think_end = response.find('</think>')
        think_content = response[think_start:think_end]
        print(f"  <think> (内部推理):")
        print(f"    {think_content[:150]}...")

    if '<answer>' in response and '</answer>' in response:
        answer_start = response.find('<answer>') + 8
        answer_end = response.find('</answer>')
        answer_content = response[answer_start:answer_end]
        print(f"\n  <answer> (实际输出):")
        print(f"    {answer_content[:150]}...")

    # 统计平均长度
    avg_prompt_len = sft_df['prompt'].str.len().mean()
    avg_response_len = sft_df['response'].str.len().mean()

    print(f"\n📏 文本长度统计:")
    print(f"  平均Prompt长度: {avg_prompt_len:.0f} 字符")
    print(f"  平均Response长度: {avg_response_len:.0f} 字符")

    return sft_df


def explore_test_data():
    """探索测试数据"""
    print_header("3. 测试数据探索 (MTMedDialog_test.json)")

    with open('/mnt/workspace/datasets/doctor_rl_data/MTMedDialog_test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    print(f"\n📊 数据集统计:")
    print(f"  测试样本数: {len(test_data):,}")

    if len(test_data) > 0:
        sample = test_data[0]
        print(f"\n📋 样本结构:")
        print(f"  键名: {list(sample.keys())}")

        print(f"\n💡 第一个测试样本:")
        for key, value in sample.items():
            if isinstance(value, str):
                display_value = value[:100] + "..." if len(value) > 100 else value
            else:
                display_value = str(value)
            print(f"  {key}: {display_value}")

    return test_data


def show_workflow():
    """展示训练工作流程"""
    print_header("4. 训练工作流程概览")

    workflow = """
    阶段1: 监督微调 (SFT)
    ────────────────────────────────────────────────
    输入: MTMedDialog_sft_train.parquet
    模型: Qwen2.5-7B-Instruct + LoRA
    目标: 学习医疗对话格式和基本推理能力
    输出: SFT checkpoint (用于RL冷启动)

    命令示例:
      bash sft/finetune_lora_med.sh 8 ./sft_checkpoints
      python sft/utils/merge_lora.py --base_model_name Qwen/Qwen2.5-7B-Instruct \\
                                      --lora_model_path ./sft_checkpoints \\
                                      --output_path ./DoctorLLM-7B-SFT

    ────────────────────────────────────────────────
    阶段2: 强化学习训练 (RL)
    ────────────────────────────────────────────────
    输入: MTMedDialog_RL.parquet
    环境: MedicalConsultationEnvWithPatientLLM
      - Doctor Agent: 你的策略模型 (从SFT初始化)
      - Patient Agent: 固定的Qwen2.5-7B (vLLM推理)

    算法: GRPO (默认) / PPO / BRPO / APO

    训练循环:
      1. Doctor生成问题 (rollout采样)
      2. Patient回答 (环境LLM worker)
      3. 多轮对话 (2-10轮,动态或固定)
      4. 计算奖励:
         - 诊断正确: +1.0
         - 有效问题: +0.1/轮
         - 无效问题: -0.05/轮
         - 轮次惩罚: -0.01/轮
      5. 策略更新 (PPO/GRPO优化)

    命令示例:
      bash scripts_exp/doctor-agent-rl-dynamic.sh

    ────────────────────────────────────────────────
    阶段3: 模型评估
    ────────────────────────────────────────────────
    输入: MTMedDialog_test.json
    评估指标:
      - 诊断准确率
      - 平均对话轮次
      - 有效问题比例
      - 各疾病类别的表现

    命令示例:
      bash ragen/env/medical_consultation/evaluation/run_eval_patientllm_category.sh \\
           ./checkpoints/your_model
    """

    print(workflow)


def show_key_files():
    """展示关键文件说明"""
    print_header("5. 关键文件导航")

    key_files = {
        "环境实现": [
            ("ragen/env/medical_consultation/env_patient_llm.py", "核心环境实现,定义reset/step/reward"),
            ("ragen/env/medical_consultation/env_patient_llm_rm.py", "带奖励模型的环境版本"),
            ("ragen/env/base.py", "环境基类定义"),
        ],
        "训练脚本": [
            ("scripts_exp/doctor-agent-rl-dynamic.sh", "推荐:动态轮次+SFT冷启动"),
            ("scripts_exp/doctor-agent-rl-rm-dynamic.sh", "带奖励模型+动态轮次"),
            ("sft/finetune_lora_med.sh", "SFT LoRA训练脚本"),
        ],
        "训练器": [
            ("ragen/trainer/main_ppo.py", "训练主入口,Hydra配置"),
            ("ragen/trainer/ppo/ray_trainer.py", "Ray分布式训练逻辑"),
        ],
        "Workers": [
            ("ragen/workers/env_llm_worker.py", "患者LLM推理worker (vLLM)"),
            ("ragen/workers/actor_worker.py", "策略模型worker (FSDP)"),
        ],
        "评估": [
            ("ragen/env/medical_consultation/evaluation/run_eval_patientllm_category.sh", "评估脚本"),
            ("ragen/env/medical_consultation/evaluation/inference_fast_for_patientllm.py", "推理生成对话"),
            ("ragen/env/medical_consultation/evaluation/evaluation_for_patientllm_category.py", "计算评估指标"),
        ],
        "配置": [
            ("config/base.yaml", "基础配置模板"),
            ("CLAUDE.md", "项目说明文档"),
        ],
    }

    for category, files in key_files.items():
        print(f"\n📂 {category}:")
        for filepath, description in files:
            exists = "✓" if Path(filepath).exists() else "✗"
            print(f"  {exists} {filepath}")
            print(f"     └─ {description}")


def show_next_steps():
    """展示下一步行动建议"""
    print_header("6. 推荐的学习路径")

    steps = """
    ✅ 已完成: 环境安装 (setup_ragen.sh)

    📚 建议按以下顺序学习:

    第1步: 理解数据 (今天)
      □ 运行本脚本: python quick_start.py
      □ 手动查看数据文件,理解格式
      □ 阅读 LEARNING_GUIDE.md 中的"数据探索"章节

    第2步: 阅读核心代码 (1-2天)
      □ ragen/env/medical_consultation/env_patient_llm.py (最重要!)
      □ scripts_exp/doctor-agent-rl-dynamic.sh (了解训练参数)
      □ 理解 reset(), step(), _calculate_reward() 方法

    第3步: 小规模SFT训练 (1天)
      □ 准备Qwen2.5-7B-Instruct模型
      □ 创建测试配置 (减少训练步数到100)
      □ 运行: bash sft/finetune_lora_test.sh 1 ./test_sft
      □ 观察训练loss下降曲线

    第4步: 小规模RL训练 (1-2天)
      □ 创建测试脚本 (减少数据量和迭代次数)
      □ 理解GRPO算法原理
      □ 运行测试训练并监控指标
      □ 重点观察: reward, diagnosis_rate, valid_action_rate

    第5步: 完整训练与评估 (2-3天)
      □ 使用完整数据集训练SFT
      □ 使用SFT checkpoint进行RL训练
      □ 运行评估并对比基线模型
      □ 分析各疾病类别的表现差异

    第6步: 深入研究 (持续)
      □ 修改奖励函数,观察影响
      □ 对比不同RL算法 (GRPO/PPO/BRPO)
      □ 调整超参数优化性能
      □ 阅读论文理解理论基础

    💡 提示:
      - 每个阶段都做好实验记录
      - 遇到问题先查看日志和源码注释
      - 使用WandB监控训练过程
      - 从小规模实验开始,逐步扩大规模
    """

    print(steps)


def main():
    """主函数"""
    print("\n" + "🚀" * 35)
    print("  DoctorAgent-RL 快速入门指南")
    print("  Multi-Agent RL for Medical Consultation")
    print("🚀" * 35)

    # # 检查数据文件是否存在
    # required_files = [
    #     '/mnt/workspace/datasets/doctor_rl_data/MTMedDialog_RL.parquet',
    #     '/mnt/workspace/datasets/doctor_rl_data/MTMedDialog_sft_train.parquet',
    #     '/mnt/workspace/datasets/doctor_rl_data/MTMedDialog_test.json'
    # ]
    #
    # missing_files = [f for f in required_files if not Path(f).exists()]
    # if missing_files:
    #     print(f"\n⚠️  警告: 以下数据文件不存在:")
    #     for f in missing_files:
    #         print(f"  - {f}")
    #     print("\n请确保已下载数据集并放置在正确位置。")
    #     return
    #
    # # 执行各个探索函数
    # try:
    #     explore_rl_data()
    #     explore_sft_data()
    #     explore_test_data()
    #     show_workflow()
    #     show_key_files()
    #     show_next_steps()
    #
    #     print_header("完成!")
    #     print("""
    #     📖 更多详细信息请查看:
    #       - LEARNING_GUIDE.md (完整学习路线)
    #       - CLAUDE.md (项目技术文档)
    #       - README.md (项目说明)
    #
    #     🎯 建议下一步:
    #       1. 阅读 ragen/env/medical_consultation/env_patient_llm.py
    #       2. 查看 scripts_exp/doctor-agent-rl-dynamic.sh 了解训练参数
    #       3. 准备模型并开始小规模SFT实验
    #
    #     祝学习顺利! 🎓
    #     """)
    #
    # except Exception as e:
    #     print(f"\n❌ 错误: {e}")
    #     import traceback
    #     traceback.print_exc()


if __name__ == "__main__":
    main()
