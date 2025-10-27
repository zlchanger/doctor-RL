# DoctorAgent-RL 🚀: Multi-Agent Collaborative Reinforcement Learning for Clinical Dialogue

[![Dataset](https://img.shields.io/badge/Dataset-MTMedDialog-orange)](DATASET.md) [![arXiv](https://img.shields.io/badge/arXiv-2505.19630-b31b1b.svg)](https://arxiv.org/pdf/2505.19630) [![Hugging Face Collection](https://img.shields.io/badge/Hugging%20Face%20Collection-doctoragent--rl-blue)](https://huggingface.co/collections/Jarvis1111/doctoragent-rl-684ffbcade52305ba0e3e97f)

<div align="center">

  <img width="1231" alt="image" src="https://github.com/user-attachments/assets/bd9f676e-01f9-406c-881d-c2b9f45e62f3" />
</div>



## Table of Contents
- [News](#news)
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Methodology](#methodology)
- [Experiments](#experiments)
- [Setup](#setup)
- [Experiment Scripts](#experiment-scripts)
- [Citation](#citation)

## News
* **[2025.10.15]** We've released a significant update to our paper, now available on [arXiv](https://arxiv.org/pdf/2505.19630)! This new version includes more practical and rigorous experiments, showcasing the real-world capabilities of our DoctorAgent-RL model. We've conducted a thorough evaluation of the model on real patient diagnoses and validated its performance with expert feedback.
* **[2025.6.16]** We released the source code in [GitHub](https://github.com/JarvisUSTC/DoctorAgent-RL) and Models in [Huggingface](https://huggingface.co/collections/Jarvis1111/doctoragent-rl-684ffbcade52305ba0e3e97f)!
* **[2025.5.26]** We released our [paper](https://arxiv.org/pdf/2505.19630) in arXiv!

## Introduction

Introducing **DoctorAgent-RL**: A multi-agent collaborative reinforcement learning framework revolutionizing clinical dialogue. By modeling medical consultations as dynamic decision-making processes under uncertainty, DoctorAgent-RL directly addresses the critical limitations of static clinical dialogue systems, enabling:

1.  **Adaptive Information Gathering**: Intelligent adjustment of dialogue paths based on patient responses.
2.  **Clinical Reasoning Alignment**: Autonomous development of interaction strategies consistent with medical logic.
3.  **Overcoming Static Paradigms**: Moving beyond superficial pattern imitation in existing dialogue datasets.

Through continuous multi-turn interactions between doctor and patient agents, optimized via reinforcement learning, DoctorAgent-RL achieves significant improvements in diagnostic accuracy and interaction efficiency.

## Key Features

-   🧠 **Multi-Agent Collaboration**: Doctor and patient agents with distinct roles and objectives.
-   📈 **Dynamic Strategy Optimization**: Reinforcement learning-based policy updates for adaptive behavior.
-   🎯 **Comprehensive Reward Design**: Multi-dimensional consultation evaluation metrics guiding optimal strategies.
-   📊 **Medical Knowledge Integration**: Clinical reasoning logic embedded in decision-making processes.
-   📄 **MTMedDialog Dataset**: The first English multi-turn medical consultation dataset designed with simulation capabilities.

## Methodology

<div align="center">
  <img src="Figures/framework.png" alt="System Architecture" width="600">
</div>

Our framework consists of three core components that interact in a continuous learning loop:

1.  **Doctor Agent**: Responsible for diagnostic reasoning and formulating appropriate questions.
2.  **Patient Agent**: Simulates patient responses based on a given medical history and symptom progression.
3.  **Consultation Evaluator**: Provides comprehensive feedback to the agents through multi-dimensional reward signals, assessing the quality of the consultation.

The reinforcement learning process involves:

1.  Multi-turn dialogue simulation between the interacting doctor and patient agents.
2.  Dynamic reward calculation based on real-time consultation quality and objectives.
3.  Policy updates using advanced reinforcement learning algorithms, such as Group Relative Policy Optimization (GRPO).
4.  Continuous strategy refinement through iterative interactions, driving agents towards optimal diagnostic and communication strategies.

## Experiments

Our experiments demonstrate the effectiveness of DoctorAgent-RL across various metrics.

### Patient Agent Evaluation

We selected **Qwen2.5-7B-Instruct** as the foundation for our Patient Agent in these experiments, evaluating its fidelity in simulating realistic patient behaviors.

<div align="center">
  <img src="Figures/Patient_Performance.png" alt="System Architecture" width="600">
</div>

### Doctor Agent Evaluation

The Doctor Agent's performance was rigorously evaluated for its diagnostic accuracy and efficiency in information gathering.

<div align="center">
  <img src="Figures/Doctor_Performance.png" alt="System Architecture" width="600">
</div>

### Ablation Study

An ablation study was conducted to understand the contribution of each core component of DoctorAgent-RL to its overall performance.

<div align="center">
  <img src="Figures/Ablation_Study_1.png" alt="System Architecture" width="600">
</div>

We also investigated the framework's adaptability under varying turn budgets, highlighting its robust performance across different interaction lengths.

<div align="center">
  <img src="Figures/Dynamic_Turn_Budget.png" alt="System Architecture" width="600">
</div>

---

## Setup

To set up your environment and run DoctorAgent-RL, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/JarvisUSTC/DoctorAgent-RL.git
cd DoctorAgent-RL
```

### 2. Setup the environment

Follow RAGEN's setup script:
```bash
bash scripts/setup_ragen.sh
```

### 3. Download the neccessary models

- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/main)
- [DoctorAgent-RL-SFT-1k-Thinking](https://huggingface.co/Jarvis1111/DoctorAgent-RL-SFT-1k-Thinking) (Our SFT Model)
- [DoctorAgent-RL](https://huggingface.co/Jarvis1111/DoctorAgent-RL) (Our RL Model)

## Experiment Scripts

Once your environment is set up, you can run the experiments:

### 1. Data Preprocessing

Our preprocessed training data is located in the `data/` directory. For **Supervised Fine-Tuning (SFT)** cold start, we use the `MTMedDialog_sft_train.parquet` dataset. This dataset was created by prompting DeepSeek-V3 to generate the thinking process for each sample.

For **Reinforcement Learning (RL)** training, we utilize the `MTMedDialog_RL.parquet` dataset. This dataset includes detailed patient descriptions, which were generated by prompting Qwen2.5-7B-Instruct. Notably, Qwen2.5-7B-Instruct also serves as our patient agent in the RL setup.

### 2. Train the Doctor Agent

```bash
# Example:
# Dynamic Turns + SFT Cold Start
bash scripts_exp/doctor-agent-rl-dynamic.sh
# Reward Model + Dynamic Turns + SFT Cold Start
bash scripts_exp/doctor-agent-rl-rm-dynamic.sh
# Reward Model + SFT Cold Start
bash scripts_exp/doctor-agent-rl-rm.sh
# Reward Model + Dynamic Turns
bash doctor-agent-rl-dynamic-wo-sft.sh
```

For SFT cold start, you can use the "sft/finetune_lora_med.sh" or [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

### 3. Run Evaluation

The evaluation scripts are located in the `ragen/env/medical_consultation/evaluation/` directory.

```bash
# Example:
bash ragen/env/medical_consultation/evaluation/run_eval_patientllm_category.sh ${MODEL_PATH}
# If you want to run with api, remember to config the api key and request command
bash ragen/env/medical_consultation/evaluation/run_eval_patientllm_category_api.sh ${MODEL_NAME}
```

For more detailed command-line arguments and configuration options, please refer to the individual script files that will be released with the code.

## Citation

If DoctorAgent-RL contributes to your research, please consider citing our work:

```latex
@article{feng2025doctoragent,
  title={DoctorAgent-RL: A Multi-Agent Collaborative Reinforcement Learning System for Multi-Turn Clinical Dialogue},
  author={Feng, Yichun and Wang, Jiawei and Zhou, Lu and Li, Yixue},
  journal={arXiv preprint arXiv:2505.19630},
  year={2025}
}
```
