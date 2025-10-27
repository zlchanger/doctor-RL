<h1 align="center"> RAGEN：通过强化推理训练智能体 </h1>


<p align="center"><img src="./public/ragen.png" width="800px" alt="RICO 框架" /></p>
<p align="center" style="font-size: 18px;">
  <strong>RAGEN</strong> 利用强化学习在交互式、随机环境中训练<strong>大语言模型推理智能体</strong>。<br>
  <em>我们坚信 RL + LLM + Agents 的未来。此次发布是向前迈出的最小可行跨越。</em>
</p>


<p align="center">
  <a href="https://ragen-tutorial.readthedocs.io/"><img src="https://img.shields.io/badge/📚_Documentation-4285F4?style=for-the-badge&logoColor=white" alt="文档"></a>
  <a href="#"><img src="https://img.shields.io/badge/📝_Blog-FF5722?style=for-the-badge&logoColor=white" alt="博客"></a>
  <a href="#"><img src="https://img.shields.io/badge/📄_Paper-EA4335?style=for-the-badge&logoColor=white" alt="论文"></a>
  <a href="#"><img src="https://img.shields.io/badge/🔍_Post-34A853?style=for-the-badge&logoColor=white" alt="帖子"></a>
</p>

**2025.3.8 更新：**

1. 在之前的 veRL 实现中，存在一个 [KL 项问题](https://github.com/volcengine/verl/pull/179/files)，已在最近版本中修复。
2. 我们从多个来源发现证据表明 PPO 在训练中可能比 GRPO 更稳定，包括 [Open-Reasoner-Zero](https://x.com/rosstaylor90/status/1892664646890312125)、[TinyZero](https://github.com/Jiayi-Pan/TinyZero) 和 [知乎](https://www.zhihu.com/search?type=content&q=%E6%97%A0%E5%81%8FGRPO)。我们已将默认优势估计器更改为 GAE（使用 PPO），并致力于在后续版本中找到更稳定且高效的 RL 优化方法。

## 概述

基于规则奖励的强化学习（RL）在增强大语言模型（LLM）的推理能力方面显示出潜力。然而，现有方法主要集中在静态的单轮任务上，如数学推理和编码。将这些方法扩展到智能体场景引入了两个根本性挑战：

1. **多轮交互**：智能体必须执行顺序决策并对环境反馈做出反应
2. **随机环境**：存在不确定性，相同的动作可能导致不同的结果

RAGEN 通过以下方式应对这些挑战：
- 针对智能体任务的马尔可夫决策过程（MDP）建模
- 推理-交互链优化（RICO）算法，优化整个轨迹分布
- 渐进式奖励归一化策略，处理多样化、复杂的环境

## 算法

RAGEN 引入了一个强化学习框架，用于训练能够在交互式、随机环境中运行的具有推理能力的大语言模型智能体。

<p align="center"><img src="./public/rico.png" width="800px" alt="RICO 框架" /></p>
<p align="center" style="font-size: 16px; max-width: 800px; margin: 0 auto;">
推理-交互链优化（RICO）框架包含两个交错阶段：<b>展开阶段</b>和<b>更新阶段</b>。大语言模型迭代生成推理引导的动作与环境交互以获得轨迹级奖励，归一化后用于大语言模型更新，以共同优化推理和动作策略。
</p>


## 算法

RAGEN 引入了一个强化学习框架，用于训练能够在交互式、随机环境中运行的具有推理能力的大语言模型智能体。该框架由两个关键组件组成：

### > MDP 建模
我们将智能体-环境交互建模为马尔可夫决策过程（MDP），其中状态和动作是词元序列，允许大语言模型对环境动态进行推理。在时间 t，状态 $s_t$ 通过动作 $a_t$ 根据转移函数转换到下一个状态。策略根据轨迹历史生成动作。目标是最大化多个交互轮次的预期累积奖励。

### > 推理-交互链优化
RICO 使大语言模型能够在整个轨迹上共同优化推理和动作策略。该算法在两个阶段之间交替：

#### 展开阶段：推理-交互链生成
给定初始状态，大语言模型生成多个轨迹。在每个步骤，模型接收轨迹历史并生成推理引导的动作：`<think>...</think><ans> 动作 </ans>`。环境接收动作并返回反馈（奖励和下一个状态）。

#### 更新阶段：多轮轨迹优化
生成轨迹后，我们训练大语言模型以优化预期奖励。RICO 不是逐步优化，而是使用重要性采样优化整个轨迹。这种方法在保持计算效率的同时实现长期推理。

### > 奖励归一化策略
我们实现了三种渐进归一化策略以稳定训练：
1. **ARPO**：直接保留原始奖励
2. **BRPO**：使用批次统计在每个训练批次中归一化奖励
3. **GRPO**：在提示组内归一化以平衡不同任务难度的学习

## 环境配置
有关详细的配置说明，请查看我们的[文档](https://ragen-tutorial.readthedocs.io/)。这里是快速入门指南：

```bash
# 配置环境并下载数据（2.7MB）
bash scripts/setup_ragen.sh
python scripts/download_data.py
```

如果失败，您可以按照 `scripts/setup_ragen.md` 中的手动配置说明操作。

## 训练模型
以下是如何使用 RAGEN 训练模型：

### 导出变量并训练
我们在 `verl/trainer/config/ppo_trainer.yaml` 中提供了默认配置。进行训练：

```bash
bash train.sh sokoban \
    model.experiment_name=new_test

# 根据需要覆盖配置参数
bash train.sh sokoban \
    model.experiment_name=new_test_debug \
    training.train_batch_size=128 \
    training.ppo_batch_size=64
```

## 监督微调（可选）
使用 LoRA 进行监督微调：

1. 创建监督微调数据：
```bash
bash sft/generate_data.sh <env_type>
```

2. 微调模型：
```bash
bash sft/finetune_lora.sh <env_type> <num_gpus> <save_path>
```

3. 将 LoRA 权重与基础模型合并：
```bash
python sft/utils/merge_lora.py \
    --base_model_name <base_model_name> \
    --lora_model_path <lora_model_path> \
    --output_path <output_path>
```

## 可视化
要可视化智能体轨迹：

1. 在 `train.sh` 中设置可视化参数：
```bash
logging.log_images=True
logging.log_image_dir=log/trajectory
logging.log_image_step_size=4
logging.log_n_image_per_batch=32
```

2. 查看可视化结果：
```bash
cd log/trajectory
python -m http.server 8000
# 访问 http://localhost:8000/[EXP_NAME]/step_[STEP_NUM]/trajectory_data_[ID].html
```

3. 为了正确渲染字体：
```bash
sudo apt-get install fonts-noto-cjk
```


## 性能

我们在多个模型大小和配置上评估了 RAGEN。以下是使用 Qwen-2.5-{0.5B, 3B}-{Instruct, None} 和 DeepSeek-R1-Distill-Qwen-1.5B 进行的 Sokoban 实验结果。

<img src="./public/loss_curve.png" width="800px" alt="不同模型的损失曲线" />

**注意：损失显示了奖励曲线，其中考虑了 KL 项。**

主要观察：
- 指令微调模型显示出早期优势，但随着训练进展，差距缩小
- 较大模型（3B）通常优于较小模型（0.5B），尽管优势不明显
- R1 蒸馏的 1.5B 模型最初表现不如 0.5B 模型
- 这些实验中的训练尚未收敛

我们的分析揭示了使用 RL 训练大语言模型智能体的两个关键方面：
1. **提示多样性**：平衡观察多样性和有效响应比较
2. **在线展开频率**：协调训练稳定性和数据新鲜度

## 示例轨迹

Sokoban 任务上的智能体推理可视化：

<p align="center" style="display: flex; justify-content: center; gap: 10px;">
    <img src="./public/step_1.png" width="200px" alt="步骤 1" />
    <img src="./public/step_2.png" width="200px" alt="步骤 2" />
</p>

可视化显示了智能体如何通过顺序步骤进行推理以解决谜题。

## 案例研究
我们提供了几个展示模型行为的案例研究：
- [奖励黑客](https://github.com/ZihanWang314/agent-r1/blob/main/cases/reward_hacking.txt)
- [挑战时刻](https://github.com/ZihanWang314/agent-r1/blob/main/cases/suck_moment.txt)

将添加更多案例研究以展示成功的推理模式和失败模式。

## 反馈
我们欢迎所有形式的反馈！请就错误、问题或建议提出 issue。这有助于我们的团队高效解决常见问题并建立更有生产力的社区。

## 贡献者

[**Zihan Wang**\*](https://zihanwang314.github.io/)、[**Kangrui Wang**\*](https://jameskrw.github.io/)、[**Qineng Wang**\*](https://qinengwang-aiden.github.io/)、[**Pingyue Zhang**\*](https://williamzhangsjtu.github.io/)、[**Linjie Li**\*](https://scholar.google.com/citations?user=WR875gYAAAAJ&hl=en)、[**Zhengyuan Yang**](https://zyang-ur.github.io/)、[**Kefan Yu**](https://www.linkedin.com/in/kefan-yu-22723a25b/en/)、[**Minh Nhat Nguyen**](https://www.linkedin.com/in/menhguin/?originalSubdomain=sg)、[**Monica Lam**](https://suif.stanford.edu/~lam/)、[**Yiping Lu**](https://2prime.github.io/)、[**Kyunghyun Cho**](https://kyunghyuncho.me/)、[**Jiajun Wu**](https://jiajunwu.com/)、[**Li Fei-Fei**](https://profiles.stanford.edu/fei-fei-li)、[**Lijuan Wang**](https://www.microsoft.com/en-us/research/people/lijuanw/)、[**Yejin Choi**](https://homes.cs.washington.edu/~yejin/)、[**Manling Li**](https://limanling.github.io/)

*：同等贡献。

## 致谢
我们感谢 [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1) 提供 DeepSeek-R1 模型和思路。我们感谢 [veRL](https://github.com/volcengine/verl) 团队提供的基础设施。我们感谢 [TinyZero](https://github.com/Jiayi-Pan/TinyZero) 团队的发现，激发了我们的早期探索。我们感谢 Han Liu、Xinyu Xing、Li Erran Li、Akari Asai、Eiso Kant、Lu Lu、Runxin Xu、Huajian Xin、Zijun Liu、Weiyi Liu、Weimin Wu、Yibo Wen、Jiarui Liu、Lorenzo Xiao、Ishan Mukherjee、Anabella Isaro、Haosen Sun、How-Yeh Wan、Lester Xue、Weiyi Liu 的深刻讨论。

## 引用
```md
@misc{RAGEN,
  author       = {Zihan Wang* and Kangrui Wang* and Qineng Wang* and Pingyue Zhang* and Linjie Li* and Zhengyuan Yang and Kefan Yu and Minh Nhat Nguyen and Monica Lam and Yiping Lu and Kyunghyun Cho and Jiajun Wu and Li Fei-Fei and Lijuan Wang and Yejin Choi and Manling Li},
  title        = {Training Agents by Reinforcing Reasoning},
  year         = {2025},
  organization = {GitHub},
  url          = {https://github.com/ZihanWang314/ragen},
}
```