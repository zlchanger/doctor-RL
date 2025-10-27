# DoctorAgent-RL 🚀: 面向临床对话的多智能体协作强化学习

[![Dataset](https://img.shields.io/badge/Dataset-MTMedDialog-orange)](DATASET.md) [![arXiv](https://img.shields.io/badge/arXiv-2505.19630-b31b1b.svg)](https://arxiv.org/pdf/2505.19630) [![Hugging Face Collection](https://img.shields.io/badge/Hugging%20Face%20Collection-doctoragent--rl-blue)](https://huggingface.co/collections/Jarvis1111/doctoragent-rl-684ffbcade52305ba0e3e97f)

<div align="center">

  <img width="1231" alt="image" src="https://github.com/user-attachments/assets/bd9f676e-01f9-406c-881d-c2b9f45e62f3" />
</div>



## 目录
- [新闻](#新闻)
- [简介](#简介)
- [核心特性](#核心特性)
- [方法论](#方法论)
- [实验](#实验)
- [环境配置](#环境配置)
- [实验脚本](#实验脚本)
- [引用](#引用)

## 新闻
* **[2025.10.15]** 我们在 [arXiv](https://arxiv.org/pdf/2505.19630) 上发布了论文的重大更新！新版本包含了更实用和严谨的实验，展示了 DoctorAgent-RL 模型在真实场景中的能力。我们对模型在真实患者诊断上进行了全面评估，并通过专家反馈验证了其性能。
* **[2025.6.16]** 我们在 [GitHub](https://github.com/JarvisUSTC/DoctorAgent-RL) 上发布了源代码，并在 [Huggingface](https://huggingface.co/collections/Jarvis1111/doctoragent-rl-684ffbcade52305ba0e3e97f) 上发布了模型！
* **[2025.5.26]** 我们在 arXiv 上发布了我们的[论文](https://arxiv.org/pdf/2505.19630)！

## 简介

介绍 **DoctorAgent-RL**：一个革命性的多智能体协作强化学习框架，用于临床对话。通过将医疗咨询建模为不确定性下的动态决策过程，DoctorAgent-RL 直接解决了静态临床对话系统的关键局限性，实现了：

1.  **自适应信息收集**：基于患者反馈智能调整对话路径。
2.  **临床推理对齐**：自主发展与医学逻辑一致的交互策略。
3.  **突破静态范式**：超越现有对话数据集中的表面模式模仿。

通过医生和患者智能体之间的持续多轮交互，并经由强化学习优化，DoctorAgent-RL 在诊断准确性和交互效率方面取得了显著提升。

## 核心特性

-   🧠 **多智能体协作**：具有不同角色和目标的医生和患者智能体。
-   📈 **动态策略优化**：基于强化学习的策略更新实现自适应行为。
-   🎯 **综合奖励设计**：多维度咨询评估指标引导最优策略。
-   📊 **医学知识整合**：决策过程中嵌入临床推理逻辑。
-   📄 **MTMedDialog 数据集**：首个专为模拟能力设计的英文多轮医疗咨询数据集。

## 方法论

<div align="center">
  <img src="Figures/framework.png" alt="系统架构" width="600">
</div>

我们的框架由三个核心组件构成，它们在持续学习循环中相互作用：

1.  **医生智能体**：负责诊断推理和制定适当的问题。
2.  **患者智能体**：基于给定的病史和症状进展模拟患者反应。
3.  **咨询评估器**：通过多维度奖励信号向智能体提供全面反馈，评估咨询质量。

强化学习过程包括：

1.  交互的医生和患者智能体之间的多轮对话模拟。
2.  基于实时咨询质量和目标的动态奖励计算。
3.  使用先进强化学习算法（如分组相对策略优化 GRPO）进行策略更新。
4.  通过迭代交互持续完善策略，推动智能体朝着最优诊断和沟通策略发展。

## 实验

我们的实验展示了 DoctorAgent-RL 在各种指标上的有效性。

### 患者智能体评估

我们选择 **Qwen2.5-7B-Instruct** 作为这些实验中患者智能体的基础模型，评估其模拟真实患者行为的保真度。

<div align="center">
  <img src="Figures/Patient_Performance.png" alt="系统架构" width="600">
</div>

### 医生智能体评估

对医生智能体的诊断准确性和信息收集效率进行了严格评估。

<div align="center">
  <img src="Figures/Doctor_Performance.png" alt="系统架构" width="600">
</div>

### 消融研究

进行了消融研究以了解 DoctorAgent-RL 每个核心组件对其整体性能的贡献。

<div align="center">
  <img src="Figures/Ablation_Study_1.png" alt="系统架构" width="600">
</div>

我们还研究了框架在不同轮次预算下的适应性，突出显示了其在不同交互长度下的鲁棒性能。

<div align="center">
  <img src="Figures/Dynamic_Turn_Budget.png" alt="系统架构" width="600">
</div>

---

## 环境配置

要配置环境并运行 DoctorAgent-RL，请按照以下步骤操作：

### 1. 克隆仓库
```bash
git clone https://github.com/JarvisUSTC/DoctorAgent-RL.git
cd DoctorAgent-RL
```

### 2. 配置环境

按照 RAGEN 的配置脚本：
```bash
bash scripts/setup_ragen.sh
```

### 3. 下载必要的模型

- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/main)
- [DoctorAgent-RL-SFT-1k-Thinking](https://huggingface.co/Jarvis1111/DoctorAgent-RL-SFT-1k-Thinking)（我们的 SFT 模型）
- [DoctorAgent-RL](https://huggingface.co/Jarvis1111/DoctorAgent-RL)（我们的 RL 模型）

## 实验脚本

环境配置完成后，您可以运行实验：

### 1. 数据预处理

我们预处理的训练数据位于 `data/` 目录中。对于**监督微调（SFT）**冷启动，我们使用 `MTMedDialog_sft_train.parquet` 数据集。该数据集通过提示 DeepSeek-V3 为每个样本生成思考过程而创建。

对于**强化学习（RL）**训练，我们使用 `MTMedDialog_RL.parquet` 数据集。该数据集包含详细的患者描述，这些描述是通过提示 Qwen2.5-7B-Instruct 生成的。值得注意的是，Qwen2.5-7B-Instruct 在 RL 设置中也作为我们的患者智能体。

### 2. 训练医生智能体

```bash
# 示例：
# 动态轮次 + SFT 冷启动
bash scripts_exp/doctor-agent-rl-dynamic.sh
# 奖励模型 + 动态轮次 + SFT 冷启动
bash scripts_exp/doctor-agent-rl-rm-dynamic.sh
# 奖励模型 + SFT 冷启动
bash scripts_exp/doctor-agent-rl-rm.sh
# 奖励模型 + 动态轮次
bash doctor-agent-rl-dynamic-wo-sft.sh
```

对于 SFT 冷启动，您可以使用 "sft/finetune_lora_med.sh" 或 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)。

### 3. 运行评估

评估脚本位于 `ragen/env/medical_consultation/evaluation/` 目录中。

```bash
# 示例：
bash ragen/env/medical_consultation/evaluation/run_eval_patientllm_category.sh ${MODEL_PATH}
# 如果您想使用 API 运行，请记住配置 API 密钥和请求命令
bash ragen/env/medical_consultation/evaluation/run_eval_patientllm_category_api.sh ${MODEL_NAME}
```

有关更详细的命令行参数和配置选项，请参考将与代码一起发布的各个脚本文件。

## 引用

如果 DoctorAgent-RL 对您的研究有所贡献，请考虑引用我们的工作：

```latex
@article{feng2025doctoragent,
  title={DoctorAgent-RL: A Multi-Agent Collaborative Reinforcement Learning System for Multi-Turn Clinical Dialogue},
  author={Feng, Yichun and Wang, Jiawei and Zhou, Lu and Li, Yixue},
  journal={arXiv preprint arXiv:2505.19630},
  year={2025}
}
```