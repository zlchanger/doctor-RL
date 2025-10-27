# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DoctorAgent-RL** is a multi-agent collaborative reinforcement learning framework for clinical dialogue systems. It models medical consultations as dynamic decision-making processes where a Doctor Agent interacts with a Patient Agent through multi-turn conversations to reach accurate diagnoses. The system is trained using reinforcement learning (GRPO/PPO) on the MTMedDialog dataset.

The framework is built on top of **RAGEN** (Reinforcement Learning for Agents by Reinforcing Reasoning) and **veRL** (Volcano Engine RL), which provide the underlying RL training infrastructure.

## Repository Structure

### Core Components

- **`ragen/`** - Main framework code
  - **`ragen/env/medical_consultation/`** - Medical consultation environment implementations
    - `env_patient_llm.py` - Environment with LLM-based patient simulation
    - `env_patient_llm_rm.py` - Environment with reward model integration
    - `evaluation/` - Evaluation scripts for trained models
  - **`ragen/env/`** - Other demonstration environments (sokoban, frozen_lake, bandit, countdown)
  - **`ragen/trainer/`** - Training orchestration
    - `main_ppo.py` - Main PPO training entry point
    - `ppo/ray_trainer.py` - Ray-based distributed training
  - **`ragen/workers/`** - Distributed worker implementations (actor, critic, env_llm_worker)
  - **`ragen/utils/`** - Utilities for datasets, FSDP, reward scoring, chat templates
  - **`ragen/train.py`** - Configuration loader and command generator

- **`verl/`** - veRL submodule (RLHF infrastructure from ByteDance)
  - Provides distributed RL training capabilities, hybrid-controller programming model
  - Integrates with vLLM, FSDP, Megatron-LM

- **`config/`** - YAML configuration files
  - `base.yaml` - Base configuration template
  - `env/` - Environment-specific configs (not present for medical consultation, defined via command-line args)

- **`scripts_exp/`** - Experiment training scripts for DoctorAgent-RL
  - `doctor-agent-rl-dynamic.sh` - Dynamic turns + SFT cold start
  - `doctor-agent-rl-rm-dynamic.sh` - With reward model + dynamic turns
  - `doctor-agent-rl-rm.sh` - With reward model
  - `doctor-agent-rl-dynamic-wo-sft.sh` - Dynamic turns without SFT

- **`sft/`** - Supervised fine-tuning utilities
  - `finetune_lora_med.sh` - LoRA fine-tuning for medical dialogue
  - `utils/merge_lora.py` - Merge LoRA weights with base model

- **`data/`** - Training and evaluation datasets
  - `MTMedDialog_RL.parquet` - RL training data with patient descriptions
  - `MTMedDialog_sft_train.parquet` - SFT training data with thinking process
  - `MTMedDialog_test.json` - Test data for evaluation

## Environment Setup

### Initial Setup

```bash
# Clone repository
git clone https://github.com/JarvisUSTC/DoctorAgent-RL.git
cd DoctorAgent-RL

# Setup environment (includes verl installation)
bash scripts/setup_ragen.sh
```

If the setup script fails, follow manual instructions in `scripts/setup_ragen.md`.

### Required Models

Download these models to your local machine or ensure HuggingFace access:
- **Qwen2.5-7B-Instruct** - Base model and patient agent
- **DoctorAgent-RL-SFT-1k-Thinking** - Pre-trained SFT model (optional for cold start)
- **DoctorAgent-RL** - Final RL-trained model (for evaluation)

## Training Commands

### Supervised Fine-Tuning (SFT Cold Start)

SFT provides a warm start before RL training:

```bash
# Run LoRA fine-tuning on SFT dataset
bash sft/finetune_lora_med.sh <num_gpus> <save_path>

# Example:
bash sft/finetune_lora_med.sh 8 ./sft_checkpoints

# Merge LoRA weights
python sft/utils/merge_lora.py \
    --base_model_name Qwen2.5-7B-Instruct \
    --lora_model_path ./sft_checkpoints \
    --output_path DoctorLLM-7B-SFT-1000-thinking
```

### Reinforcement Learning Training

Main training scripts are in `scripts_exp/`. All use the Python module interface via `ragen.trainer.main_ppo`:

```bash
# Dynamic turns with SFT cold start (recommended)
bash scripts_exp/doctor-agent-rl-dynamic.sh

# With reward model + dynamic turns
bash scripts_exp/doctor-agent-rl-rm-dynamic.sh

# With reward model (fixed turns)
bash scripts_exp/doctor-agent-rl-rm.sh

# Dynamic turns without SFT
bash scripts_exp/doctor-agent-rl-dynamic-wo-sft.sh
```

**Key training parameters** (set in scripts via Hydra override syntax):
- `data.train_batch_size` - Number of rollout prompts per update
- `actor_rollout_ref.rollout.n_agent` - Responses per prompt (for GRPO)
- `actor_rollout_ref.actor.ppo_mini_batch_size` - PPO update batch size
- `actor_rollout_ref.actor.ppo_micro_batch_size` - Micro batch for gradient accumulation
- `algorithm.adv_estimator` - Advantage estimator: `grpo` (default), `brpo`, `apo`, or `gae` (PPO)
- `env.max_turns` - Max conversation turns (`-1` for random 2-10 turns)
- `actor_rollout_ref.rollout.temperature` - Sampling temperature for rollouts
- `algorithm.kl_ctrl.kl_coef` - KL divergence coefficient

### Model Merging After Training

After RL training completes:

```bash
# Scripts automatically call merger.sh at the end
bash merger.sh <base_model> <checkpoint_path> <target_model_name>
```

## Evaluation

Evaluate trained models on the test set:

```bash
# Local model evaluation
bash ragen/env/medical_consultation/evaluation/run_eval_patientllm_category.sh <MODEL_PATH>

# API-based evaluation (requires API key configuration)
bash ragen/env/medical_consultation/evaluation/run_eval_patientllm_category_api.sh <MODEL_NAME>
```

**Evaluation process:**
1. **Inference**: Generates doctor-patient dialogues via `inference_fast_for_patientllm.py`
2. **Scoring**: Evaluates dialogues with `evaluation_for_patientllm_category.py`
   - Measures diagnostic accuracy, question quality, conversation efficiency
   - Outputs JSON file with scores by disease category

## Architecture Overview

### Multi-Agent RL System

**Training Loop:**
1. **Doctor Agent** (policy model): Generates questions and diagnoses
   - Uses `<think>...</think><answer>...</answer>` format for reasoning + action
   - Trained with FSDP/LoRA on 7B+ models
2. **Patient Agent** (environment): Simulates patient responses
   - Fixed LLM (Qwen2.5-7B-Instruct) via vLLM inference
   - Responds based on patient description and dialogue history
3. **Reward Calculation**: Multi-dimensional consultation evaluation
   - Diagnosis correctness (primary reward)
   - Information gathering efficiency
   - Question relevance and quality
4. **Policy Update**: GRPO/PPO optimization over full trajectories

### Distributed Training Architecture

- **Ray-based orchestration** - Manages distributed workers
- **Actor workers** - Policy model rollout and training (FSDP)
- **Critic workers** - Value network training (for PPO)
- **Environment LLM workers** - Patient agent inference (vLLM)
- **Hybrid placement** - Flexible GPU allocation for actors/rollout/environment

### Key Design Patterns

**Environment Interface (`ragen/env/base.py`):**
- `reset(seed)` - Initialize episode with patient case
- `step(response)` - Parse doctor's response, update state, calculate reward
- `_extract_answer(text)` - Parse `<answer>...</answer>` tags
- Tracks valid/invalid/effective actions for analysis

**Multi-turn State Management:**
- State = concatenation of all previous turns (observations + actions)
- Tokenized and passed to policy with proper masking
- Patient LLM worker maintains dialogue history across turns

**Reward Normalization:**
- GRPO (default) - Normalize within prompt groups
- BRPO - Batch-level normalization
- APO - No normalization (raw rewards)

## Data Format

**RL Training Data (`MTMedDialog_RL.parquet`):**
- `reward_model` column contains:
  - `ground_truth`: Correct diagnosis
  - `patient_information`: List of valid Q&A pairs
  - `enhanced_description`: Patient description for LLM simulation
- `extra_info.index`: Original case ID (used as seed)

**SFT Data (`MTMedDialog_sft_train.parquet`):**
- `prompt`: Initial patient description
- `response`: Example dialogue with `<think>` tags (generated by DeepSeek-V3)

## Configuration System

Uses **Hydra** for hierarchical configuration:

```bash
# Override any config parameter via command line
python -m ragen.trainer.main_ppo \
    data.train_files=data/custom.parquet \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    training.max_turns=8 \
    +custom_key=value  # Add new key with +
```

Configuration hierarchy: `base.yaml` → environment YAML → command-line overrides

## Development Tips

- **WandB integration**: Set `WANDB_API_KEY` environment variable before training
- **GPU visibility**: Scripts set `CUDA_VISIBLE_DEVICES` - adjust as needed
- **Checkpoint location**: `checkpoints/<project_name>/<exp_name>/global_step_<N>/actor`
- **Ray debugging**: Set `RAY_DEBUG_POST_MORTEM=1` for detailed error traces
- **vLLM backend**: Uses `VLLM_ATTENTION_BACKEND=XFORMERS` by default

## Related Documentation

- Full RAGEN docs: https://ragen-tutorial.readthedocs.io/
- veRL documentation: https://verl.readthedocs.io/
- Paper: https://arxiv.org/pdf/2505.19630 (DoctorAgent-RL)
- MTMedDialog dataset: See `DATASET.md`