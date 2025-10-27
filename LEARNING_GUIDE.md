# DoctorAgent-RL 学习指南

> 环境安装完成后的循序渐进学习路线

---

## 📋 学习路线图概览

```
第一阶段 → 第二阶段 → 第三阶段 → 第四阶段 → 第五阶段
理解架构    数据探索    SFT实验    RL训练      深入源码
(1-2天)     (0.5天)    (1-2天)    (2-3天)     (持续)
```

---

## 🎯 第一阶段：理解项目架构（1-2天）

### 核心概念

这是一个**多智能体强化学习医疗对话系统**：

| 组件 | 角色 | 实现 |
|------|------|------|
| **Doctor Agent** | 策略模型（待训练） | Qwen2.5-7B-Instruct + LoRA/FSDP |
| **Patient Agent** | 环境模拟器（固定） | Qwen2.5-7B-Instruct + vLLM |
| **RL算法** | 训练方法 | GRPO (默认) / PPO / BRPO / APO |
| **奖励函数** | 优化目标 | 诊断准确性 + 问诊效率 + 问题质量 |

### 系统架构

```
                    ┌──────────────────────┐
                    │   Ray 分布式协调器    │
                    └──────────┬───────────┘
                              │
        ┌──────────┬──────────┼──────────┬──────────┐
        │          │          │          │          │
   ┌────▼────┐ ┌──▼────┐ ┌───▼───┐ ┌────▼─────┐  │
   │Actor    │ │Critic │ │Rollout│ │Env LLM   │  │
   │Worker   │ │Worker │ │Worker │ │Worker    │  │
   │(FSDP)   │ │(PPO)  │ │       │ │(vLLM)    │  │
   └────┬────┘ └──┬────┘ └───┬───┘ └────┬─────┘  │
        │         │          │          │          │
        └─────────┴──────────┴──────────┴──────────┘
                         │
                    训练循环流程:
        1. Rollout: 医生问诊（采样动作）
        2. Patient: 患者回答（环境反馈）
        3. Reward: 计算奖励（准确性+效率）
        4. Update: 更新策略（PPO/GRPO）
```

### 目录结构速查

```bash
DoctorAgent-RL/
├── ragen/                      # 主框架代码
│   ├── env/
│   │   └── medical_consultation/  # 医疗对话环境
│   │       ├── env_patient_llm.py      # 核心环境实现
│   │       ├── env_patient_llm_rm.py   # 带奖励模型版本
│   │       └── evaluation/             # 评估脚本
│   ├── trainer/
│   │   ├── main_ppo.py         # 训练入口
│   │   └── ppo/ray_trainer.py  # Ray分布式训练器
│   ├── workers/                # 分布式Worker
│   └── utils/                  # 工具函数
├── verl/                       # veRL子模块（RL基础设施）
├── config/                     # Hydra配置文件
├── scripts_exp/                # 实验训练脚本
│   ├── doctor-agent-rl-dynamic.sh           # 推荐：动态轮次+SFT
│   ├── doctor-agent-rl-rm-dynamic.sh        # 带奖励模型
│   └── ...
├── sft/                        # 监督微调
│   ├── finetune_lora_med.sh    # LoRA训练脚本
│   └── utils/merge_lora.py     # 权重合并
└── data/                       # 数据集
    ├── MTMedDialog_RL.parquet          # RL训练数据
    ├── MTMedDialog_sft_train.parquet   # SFT训练数据
    └── MTMedDialog_test.json           # 测试数据
```

### 必读文件清单

```bash
# 1. 项目说明
cat README.md
cat README_RAGEN.md
cat CLAUDE.md

# 2. 核心环境实现（重点）
less ragen/env/medical_consultation/env_patient_llm.py

# 3. 训练脚本（查看参数配置）
cat scripts_exp/doctor-agent-rl-dynamic.sh

# 4. 配置文件
cat config/base.yaml
```

---

## 🔍 第二阶段：数据探索（0.5天）

### 快速查看数据

```bash
# 创建数据探索脚本
cat > explore_data.py << 'EOF'
import pandas as pd
import json

print("=" * 60)
print("1. RL训练数据 (MTMedDialog_RL.parquet)")
print("=" * 60)
rl_df = pd.read_parquet('data/MTMedDialog_RL.parquet')
print(f"样本数量: {len(rl_df)}")
print(f"列名: {rl_df.columns.tolist()}\n")

# 查看第一条样本
sample = rl_df.iloc[0]
print("示例样本结构:")
print(f"  - reward_model: {type(sample['reward_model'])}")
print(f"    - ground_truth (诊断标签): {sample['reward_model']['ground_truth']}")
print(f"    - patient_information (有效问答): {len(sample['reward_model']['patient_information'])} 条")
print(f"    - enhanced_description (患者描述):")
print(f"      {sample['reward_model']['enhanced_description'][:200]}...\n")

print("=" * 60)
print("2. SFT训练数据 (MTMedDialog_sft_train.parquet)")
print("=" * 60)
sft_df = pd.read_parquet('data/MTMedDialog_sft_train.parquet')
print(f"样本数量: {len(sft_df)}")
print(f"列名: {sft_df.columns.tolist()}\n")

sample_sft = sft_df.iloc[0]
print("示例对话格式:")
print(f"Prompt:\n{sample_sft['prompt'][:150]}...\n")
print(f"Response (带<think>标签):\n{sample_sft['response'][:300]}...\n")

print("=" * 60)
print("3. 测试数据 (MTMedDialog_test.json)")
print("=" * 60)
with open('data/MTMedDialog_test.json') as f:
    test_data = json.load(f)
print(f"测试样本数: {len(test_data)}")
print(f"第一条键名: {list(test_data[0].keys())}")
EOF

python explore_data.py
```

### 数据格式说明

#### RL训练数据格式

```python
{
    'reward_model': {
        'ground_truth': '糖尿病',  # 正确诊断标签
        'patient_information': [   # 有效的问答对
            {
                'doctor_question': '您多久前出现这些症状的？',
                'patient_response': '大约三个月前开始的'
            },
            # ... 更多问答
        ],
        'enhanced_description': '患者主诉：我最近总是感到口渴...'  # 用于LLM模拟
    },
    'extra_info': {
        'index': 12345  # 用作环境seed
    }
}
```

#### SFT数据格式

```python
{
    'prompt': '患者主诉：我最近总是感到口渴，还经常上厕所...',
    'response': '<think>患者出现多饮多尿症状，需要询问...</think><answer>您好，请问您这些症状持续多久了？</answer>'
}
```

**关键点**：
- `<think>...</think>` 标签：内部推理过程
- `<answer>...</answer>` 标签：实际输出（问题/诊断）

---

## 🧪 第三阶段：SFT训练实验（1-2天）

### 为什么先做SFT？

SFT（监督微调）提供了**冷启动**，让模型学会基本的医疗对话格式和推理模式，避免RL从零开始训练。

### 准备工作

```bash
# 1. 下载基础模型（如果还没有）
# 方式1：使用 HuggingFace CLI
pip install huggingface_hub
huggingface-cli login  # 输入你的token
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/Qwen2.5-7B-Instruct

# 方式2：在Python中自动下载（首次训练时会自动下载）
```

### 快速测试训练（小规模）

```bash
# 1. 创建测试配置（修改参数以加快训练）
cp sft/finetune_lora_med.sh sft/finetune_lora_test.sh

# 2. 编辑 sft/finetune_lora_test.sh，修改以下参数：
#    --max_steps 100                    # 从1000改为100
#    --learning_rate 5e-5               # 可以提高学习率加快收敛
#    --per_device_train_batch_size 4    # 根据显存调整
#    --gradient_accumulation_steps 2

# 3. 运行测试训练（单GPU）
bash sft/finetune_lora_test.sh 1 ./test_sft_checkpoint

# 4. 观察训练日志
# 注意：
#   - 训练loss应逐步下降
#   - 关注 'train/loss' 指标
#   - 如果使用 wandb，可以在网页查看曲线
```

### 完整SFT训练

```bash
# 使用8个GPU进行完整训练（需要约2小时）
bash sft/finetune_lora_med.sh 8 ./sft_checkpoints

# 训练完成后合并LoRA权重
python sft/utils/merge_lora.py \
    --base_model_name Qwen/Qwen2.5-7B-Instruct \
    --lora_model_path ./sft_checkpoints \
    --output_path ./DoctorLLM-7B-SFT-Custom
```

### SFT阶段的学习重点

1. **LoRA技术**：理解低秩适应如何减少训练参数
2. **Prompt格式**：学习 `<think><answer>` 的使用
3. **评估指标**：观察训练loss和验证loss
4. **超参数调整**：学习率、batch size的影响

---

## 🚀 第四阶段：RL训练实验（2-3天）

### 理解RL训练流程

```
循环迭代:
  for 每个训练批次:
    1. [Rollout] Doctor发起问诊对话
       - 输入: 患者初始描述
       - 输出: 生成问题（采样多个响应用于GRPO）

    2. [Environment] Patient LLM响应
       - 使用vLLM进行快速推理
       - 返回患者回答

    3. [Multi-turn] 循环问答（2-10轮）
       - Doctor继续提问或给出诊断
       - Patient继续回答

    4. [Reward] 计算奖励
       - 诊断准确性（主要奖励）
       - 问题有效性（是否在patient_information中）
       - 对话效率（轮次惩罚）

    5. [Policy Update] 更新策略
       - GRPO: 组内归一化优势函数
       - PPO: 使用Critic估计value
       - 优化策略以最大化累积奖励
```

### 小规模测试（推荐首次运行）

```bash
# 1. 创建测试脚本（减少数据量和迭代次数）
cat > scripts_exp/test_rl_training.sh << 'EOF'
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用4个GPU
export VLLM_ATTENTION_BACKEND=XFORMERS

python -m ragen.trainer.main_ppo \
    data.train_files=data/MTMedDialog_RL.parquet \
    data.train_batch_size=4 \
    actor_rollout_ref.rollout.n_agent=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    algorithm.adv_estimator=grpo \
    env=medical_consultation \
    env.max_turns=-1 \
    training.total_training_steps=10 \
    training.save_freq=5 \
    project_name=doctor-agent-test \
    experiment_name=quick-test
EOF

bash scripts_exp/test_rl_training.sh
```

### 完整训练（使用SFT冷启动）

```bash
# 推荐：动态轮次 + SFT初始化
bash scripts_exp/doctor-agent-rl-dynamic.sh

# 关键参数说明：
# - actor_rollout_ref.model.path: 使用你的SFT模型路径
# - env.max_turns=-1: 每个episode随机选择2-10轮对话
# - algorithm.adv_estimator=grpo: 使用Group Relative Policy Optimization
# - training.total_training_steps=500: 总训练步数
```

### 监控训练进度

```bash
# 1. 使用 WandB 在线监控（推荐）
export WANDB_API_KEY=your_key_here
# 训练时会自动上传指标到 wandb.ai

# 2. 查看本地日志
tail -f logs/train.log  # 如果有日志文件

# 3. 检查checkpoint
ls -lh checkpoints/doctor-agent/exp_name/
```

### 重要指标

| 指标名称 | 含义 | 期望趋势 |
|---------|------|---------|
| `reward/mean` | 平均奖励 | 上升 |
| `reward/diagnosis_correct_rate` | 诊断准确率 | 上升 |
| `metrics/valid_action_rate` | 有效问题占比 | 上升 |
| `metrics/avg_turns` | 平均对话轮次 | 下降（更高效） |
| `policy/kl_divergence` | KL散度 | 稳定在合理范围 |
| `policy/loss` | 策略损失 | 下降后稳定 |

### RL训练的学习重点

1. **GRPO vs PPO**：理解不同算法的优势
2. **奖励设计**：诊断准确性如何平衡对话效率
3. **探索与利用**：temperature参数的作用
4. **分布式训练**：Ray如何协调多个worker

---

## 🎯 第五阶段：模型评估（1天）

### 评估流程

```bash
# 1. 确保模型已合并（训练脚本通常会自动完成）
ls ./checkpoints/your_model/

# 2. 运行评估（本地模型）
bash ragen/env/medical_consultation/evaluation/run_eval_patientllm_category.sh \
    ./checkpoints/your_model

# 3. 查看结果
cat results_patientllm_category_*.json
```

### 评估指标解读

```json
{
    "overall": {
        "accuracy": 0.75,           // 诊断准确率
        "avg_turns": 4.2,           // 平均对话轮次
        "invalid_rate": 0.05        // 无效问题比例
    },
    "by_category": {
        "糖尿病": {
            "accuracy": 0.82,
            "samples": 50
        },
        // ... 各疾病类别的表现
    }
}
```

### 对比基线

```bash
# 评估基础模型（未经RL训练）
bash ragen/env/medical_consultation/evaluation/run_eval_patientllm_category.sh \
    Qwen/Qwen2.5-7B-Instruct

# 评估SFT模型
bash ragen/env/medical_consultation/evaluation/run_eval_patientllm_category.sh \
    ./DoctorLLM-7B-SFT-Custom

# 对比三者性能提升
```

---

## 📚 第六阶段：深入源码理解（持续学习）

### 核心代码阅读路径

#### 1. 环境实现 (最重要)

```bash
# 阅读顺序：
# 1. 基类定义
ragen/env/base.py

# 2. 医疗对话环境
ragen/env/medical_consultation/env_patient_llm.py
#   重点方法：
#   - reset(): 初始化episode
#   - step(): 处理动作，计算奖励
#   - _extract_answer(): 解析<answer>标签
#   - _calculate_reward(): 奖励计算逻辑

# 3. 带奖励模型的版本
ragen/env/medical_consultation/env_patient_llm_rm.py
```

#### 2. 训练流程

```bash
# 1. 训练入口
ragen/trainer/main_ppo.py
#   - Hydra配置加载
#   - Ray集群初始化
#   - 调用 ray_trainer

# 2. Ray分布式训练器
ragen/trainer/ppo/ray_trainer.py
#   - Actor/Critic/Rollout/EnvLLM worker管理
#   - 训练循环orchestration
```

#### 3. Worker实现

```bash
# 环境LLM Worker（患者模拟）
ragen/workers/env_llm_worker.py
#   - vLLM推理封装
#   - 批量生成患者回答

# Actor Worker（策略模型）
ragen/workers/actor_worker.py
#   - FSDP模型管理
#   - PPO更新逻辑
```

### 调试技巧

```bash
# 1. 打印调试信息
export RAY_DEBUG_POST_MORTEM=1  # Ray详细错误信息
export CUDA_LAUNCH_BLOCKING=1   # CUDA同步执行

# 2. 单步调试
# 在代码中添加：
import pdb; pdb.set_trace()

# 3. 日志级别
export LOGLEVEL=DEBUG
```

### 代码修改实验建议

1. **修改奖励函数**
   - 文件：`ragen/env/medical_consultation/env_patient_llm.py`
   - 方法：`_calculate_reward()`
   - 实验：尝试不同的奖励权重组合

2. **调整max_turns策略**
   - 实验：固定轮次 vs 动态轮次的影响

3. **改变GRPO实现**
   - 文件：相关算法实现
   - 实验：对比GRPO/PPO/BRPO的收敛速度

---

## 🛠️ 常见问题与解决

### Q1: CUDA内存不足

```bash
# 解决方案：
# 1. 减小batch size
data.train_batch_size=2
actor_rollout_ref.actor.ppo_mini_batch_size=2

# 2. 使用梯度累积
actor_rollout_ref.actor.gradient_accumulation_steps=4

# 3. 启用FSDP
actor_rollout_ref.actor.fsdp_config.enable=true
```

### Q2: Ray初始化失败

```bash
# 清理Ray进程
ray stop --force

# 重启训练
```

### Q3: vLLM加载失败

```bash
# 检查模型路径
ls -l path/to/model/

# 确认环境变量
export VLLM_ATTENTION_BACKEND=XFORMERS
```

### Q4: 训练不收敛

```bash
# 调试步骤：
# 1. 检查奖励是否合理（不应为常数）
# 2. 降低学习率
actor_rollout_ref.actor.optim.lr=1e-7

# 3. 增加KL系数（减少偏离参考模型）
algorithm.kl_ctrl.kl_coef=0.1

# 4. 使用SFT冷启动（而非随机初始化）
```

---

## 📈 学习进度检查清单

### 基础理解 (完成后可进入实践)
- [ ] 理解Doctor/Patient两个Agent的角色
- [ ] 理解GRPO/PPO的基本原理
- [ ] 熟悉项目目录结构
- [ ] 理解数据格式（RL和SFT）

### 实践操作 (完成后可深入研究)
- [ ] 成功运行数据探索脚本
- [ ] 完成一次SFT训练（小规模即可）
- [ ] 完成一次RL训练（小规模即可）
- [ ] 运行模型评估并理解指标

### 深入研究 (进阶目标)
- [ ] 阅读并理解环境实现代码
- [ ] 修改奖励函数并观察影响
- [ ] 对比不同RL算法的效果
- [ ] 尝试在新数据集上训练

---

## 🎓 推荐学习资源

### 论文阅读

1. **DoctorAgent-RL原论文**
   - https://arxiv.org/pdf/2505.19630

2. **GRPO算法**
   - Group Relative Policy Optimization (GRPO) 相关论文

3. **RLHF基础**
   - "Training language models to follow instructions with human feedback" (InstructGPT)

### 相关文档

```bash
# RAGEN框架文档
https://ragen-tutorial.readthedocs.io/

# veRL文档
https://verl.readthedocs.io/

# Qwen2.5模型文档
https://github.com/QwenLM/Qwen2.5
```

### 社区资源

- GitHub Issues: 查看其他用户的问题
- Discussions: 参与技术讨论

---

## 💡 学习建议

1. **循序渐进**：不要跳过前面的阶段，确保理解每个概念
2. **实践为主**：先运行代码，再深入理解原理
3. **记录实验**：每次训练记录配置和结果，便于对比
4. **修改尝试**：不要怕改代码，通过修改加深理解
5. **查阅文档**：遇到问题先看README和源码注释

---

## 📞 获取帮助

1. **查看日志**：大多数问题可从错误日志找到答案
2. **阅读源码**：代码注释详细，可直接阅读
3. **GitHub Issues**：搜索或提交新问题
4. **论文附录**：查看实验细节和超参数设置

---

祝学习顺利！ 🚀