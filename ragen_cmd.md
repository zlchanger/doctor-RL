## Commands to run
***NOTE! Before you run, please make sure you are under `/RAGEN` dir***
### *EXP SET 1*: Hyperparameter search.
We first do hyperparameter search, hoping to find a good combination to guide later experiment settings.

**[Note]** Current multi-GPUs strategy is **FSDP**. We are running with **3B** models.

#### [BUDGET]: In total: 49 runs, Qwen2.5-3B-Instruct, Sokoban
- Search group 1: 5 runs
    - ppo_batch_size: [16, 32, 64, 128, 256]
- Search group 2: 25 runs
    - train_batch_size: [8, 32, 64, 128, 256]
    - n_rollout: [1, 2, 4, 8, 16]
- Search group 3: 5 runs
    - kl_coef: [0.001, 0.005, 0.01, 0.04, 0.1, 0.5]
- Search group 4: 9 runs
    - max_turns: [2, 5, 8]
    - temperature: [0.1, 0.5, 1]
- Search group 5: 5 runs
    - actor_lr: [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
#### [EXP 1]: Search group 1 [ppo_batch_size] (Tested)
```bash
bash scripts/hyperparam_search.sh \
    --env_name=sokoban \
    --exp_base_name="hyperparam_searching" \
    --search_group 1 \
    --n_gpus 1 \
    --micro_batch_size 1
```
#### [EXP 2]: Search group 2 [train_batch_size, n_rollout] (Tested)
```bash
bash scripts/hyperparam_search.sh \
    --env_name=sokoban \
    --exp_base_name="hyperparam_searching" \
    --search_group 2 \
    --n_gpus 1 \
    --micro_batch_size 1
```
#### [EXP 3]: Search group 3 [kl_coef]  (Tested)
```bash
bash scripts/hyperparam_search.sh \
    --env_name=sokoban \
    --exp_base_name="hyperparam_searching" \
    --search_group 3 \
    --n_gpus 1 \
    --micro_batch_size 1
```
#### [EXP 4]: Search group 4 [max_turns, temperature] (Tested)
```bash
bash scripts/hyperparam_search.sh \
    --env_name=sokoban \
    --exp_base_name="hyperparam_searching" \
    --search_group 4 \
    --n_gpus 1 \
    --micro_batch_size 1
```
#### [EXP 5]: Search group 5 [actor_lr] (Tested)
```bash
bash scripts/hyperparam_search.sh \
    --env_name=sokoban \
    --exp_base_name="hyperparam_searching" \
    --search_group 5 \
    --n_gpus 1 \
    --micro_batch_size 1
```

***Searched results will be saved to `./log/searched_hyper_params/searched_params_group_5.json`***

> **NOTE**: Normally, we need to get all the best searched params for the following exps. As we need to test current exp settings, we will use default value for now. But below is the template to insert. *micro_batch_size can be as large as possible*
```bash
    ...
    training.ppo_batch_size=<best searched ppo_batch_size> \
    training.train_batch_size=<best searched train_batch_size> \
    training.n_rollout=<best searched n_rollout> \
    optimization.kl_coef=<best searched kl_coef> \
    training.max_turns=<best searched max_turns> \
    training.temperature=<best searched temperature> \
    optimization.actor_lr=<best searched actor_lr>
    ...
```

> **NOTE**: Normally, we need to get all the best searched params for the following exps. As we need to test current exp settings, we will use default value for now. But below is the template to insert. *micro_batch_size can be as large as possible*
```bash
    ...
    training.ppo_batch_size=<best searched ppo_batch_size> \
    training.train_batch_size=<best searched train_batch_size> \
    training.n_rollout=<best searched n_rollout> \
    optimization.kl_coef=<best searched kl_coef> \
    training.max_turns=<best searched max_turns> \
    training.temperature=<best searched temperature> \
    optimization.actor_lr=<best searched actor_lr>
    ...
```

### *EXP SET 2*: Main results for task Bandits.
The first main results we want to show is on task Bandits. This task aims to show that RAGEN can understand what 'golden bandit' and 'silver bandit' mean, even through one-turn interaction.
**Waiting for data generation**
#### [EXP 6]: RAGEN for bandits
```bash
bash train.sh frozenlake \
    model.base_model=Qwen/Qwen2.5-7B-Instruct \
    model.experiment_name=two_armed_bandit_qwen_2.5_7b_instruct_ragen \
    training.micro_batch_size=2 \
    training.use_kl_loss=True \
    training.max_turns=1 \
    optimization.adv_estimator=grpo \
    training.n_rollout=16 \
    training.train_batch_size=8 \
    training.ppo_batch_size=128
```
#### [EXP 7]: RAGEN w/o thinking for bandits
```bash
bash train.sh frozenlake \
    model.base_model=Qwen/Qwen2.5-7B-Instruct \
    model.experiment_name=two_armed_bandit_qwen_2.5_7b_instruct_ragen_no_think \
    training.micro_batch_size=2 \
    training.use_kl_loss=True \
    training.max_turns=1 \
    optimization.adv_estimator=grpo \
    training.no_think_rl=True \
    training.n_rollout=16 \
    training.train_batch_size=8 \
    training.ppo_batch_size=128
```

#### Both golden and silver cases can be calculated by math expectation values.

### *EXP SET 3*: Main results for task Sokoban.
We test Sokoban with RAGEN, RAGEN w/o thinking, SFT, and prompt. This task aims to show that RAGEN can interact with the environment and learn from it, without any human supervision.
#### [EXP 8]: RAGEN for sokoban
```bash
bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-7B-Instruct \
    model.experiment_name=sokoban_qwen_2.5_7b_instruct_ragen \
    training.micro_batch_size=2 \
    training.use_kl_loss=True \
    optimization.adv_estimator=grpo \
    training.max_turns=5 \
    training.n_rollout=16 \
    training.train_batch_size=8 \
    training.ppo_batch_size=128
```
#### [EXP 9]: RAGEN w/o thinking for sokoban
```bash
bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-7B-Instruct \
    model.experiment_name=sokoban_qwen_2.5_7b_instruct_ragen_no_think \
    training.micro_batch_size=2 \
    training.use_kl_loss=True \
    optimization.adv_estimator=grpo \
    training.no_think_rl=True \
    training.max_turns=5 \
    training.n_rollout=16 \
    training.train_batch_size=8 \
    training.ppo_batch_size=128
```
#### [EXP 10]: SFT for sokoban
```bash
bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-7B-Instruct \
    model.experiment_name=sokoban_qwen_2.5_7b_instruct_sft \
    training.use_sft=True \
    ... Waiting to be done
```
#### [EXP 11]: Prompt for sokoban
```bash
waiting to be done
```

### *EXP SET 4*: Main results for task Frozen Lake.
We test Frozen Lake with RAGEN, RAGEN w/o thinking, SFT, and prompt. This task aims to show that RAGEN can interact with the more complex and non-deterministic environment and learn from it, without any human supervision.
#### [EXP 12]: RAGEN for frozen lake
```bash
bash train.sh frozenlake \
    model.base_model=Qwen/Qwen2.5-7B-Instruct \
    model.experiment_name=frozenlake_qwen_2.5_7b_instruct_ragen \
    training.micro_batch_size=2 \
    training.use_kl_loss=True \
    optimization.adv_estimator=grpo \
    training.max_turns=5 \
    training.n_rollout=16 \
    training.train_batch_size=8 \
    training.ppo_batch_size=128
```
#### [EXP 13]: RAGEN w/o thinking for frozen lake
```bash
bash train.sh frozenlake \
    model.base_model=Qwen/Qwen2.5-7B-Instruct \
    model.experiment_name=frozenlake_qwen_2.5_7b_instruct_ragen_no_think \
    training.micro_batch_size=2 \
    training.use_kl_loss=True \
    optimization.adv_estimator=grpo \
    training.no_think_rl=True \
    training.max_turns=5 \
    training.n_rollout=16 \
    training.train_batch_size=8 \
    training.ppo_batch_size=128
```
#### [EXP 14]: SFT for frozen lake
```bash
bash train.sh frozenlake \
    model.base_model=Qwen/Qwen2.5-7B-Instruct \
    model.experiment_name=frozenlake_qwen_2.5_7b_instruct_sft \
    training.use_sft=True \
    ... Waiting to be done
```
#### [EXP 15]: Prompt for frozen lake
```bash
waiting to be done
```

### *EXP SET 5*: Generalization.
Question: how to do inference?

### *EXP SET 6*: ICL.
Question: how to import prompt to the environment?

### *EXP SET 7*: Model Scaling.
This analysis aims to investigate how model scale affects the performance of RAGEN. We use Sokoban as the base environment and test the model scaling from 0.5B to 7B.
#### [EXP ]: 0.5B Model.
```bash
bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-0.5B-Instruct \
    model.experiment_name=model_scaling_sokoban_qwen_2.5_0.5b_instruct \
    training.micro_batch_size=2 \
    training.use_kl_loss=True \
    optimization.adv_estimator=grpo \
    training.max_turns=5 \
    training.n_rollout=16 \
    training.train_batch_size=8 \
    training.ppo_batch_size=128
```
#### [EXP ]: 1.5B Model.
```bash
bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-1.5B-Instruct \
    model.experiment_name=model_scaling_sokoban_qwen_2.5_1.5b_instruct \
    training.micro_batch_size=2 \
    training.use_kl_loss=True \
    optimization.adv_estimator=grpo \
    training.max_turns=5 \
    training.n_rollout=16 \
    training.train_batch_size=8 \
    training.ppo_batch_size=128
```
#### [EXP xxx]: 3B Model.
```bash
bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-3B-Instruct \
    model.experiment_name=model_scaling_sokoban_qwen_2.5_3b_instruct \
    training.micro_batch_size=2 \
    training.use_kl_loss=True \
    optimization.adv_estimator=grpo \
    training.max_turns=5 \
    training.n_rollout=16 \
    training.train_batch_size=8 \
    training.ppo_batch_size=128
```
#### [EXP ]: 7B Model.
**Same as [EXP 8]**

### *EXP SET 8*: Base vs Instruct.
In this analysis, we aim to investigate how instruction tuning affects the performance of RAGEN. We use Sokoban as the base environment.
#### [EXP ]: Base Model.
```bash
bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-3B \
    model.experiment_name=base_vs_instruct_sokoban_qwen_2.5_3b_base \
    training.micro_batch_size=2 \
    training.use_kl_loss=True \
    optimization.adv_estimator=grpo \
    training.max_turns=5 \
    training.n_rollout=16 \
    training.train_batch_size=8 \
    training.ppo_batch_size=128
```
#### [EXP ]: Instruct Model.
**Same as [EXP xxx]**

### *EXP SET 9*: RL algorithm.
In this analysis, we aim to investigate how RL algorithm affects the performance of RAGEN. We use Sokoban as the base environment.
#### [EXP ]: APO.
```bash
bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-3B-Instruct \
    model.experiment_name=rl_algorithm_sokoban_qwen_2.5_3b_instruct_apo \
    training.micro_batch_size=2 \
    training.use_kl_loss=True \
    optimization.adv_estimator=apo \
    training.max_turns=5 \
    training.n_rollout=1 \
    training.train_batch_size=8 \
    training.ppo_batch_size=128
```

#### [EXP ]: BRPO.
```bash
bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-3B-Instruct \
    model.experiment_name=rl_algorithm_sokoban_qwen_2.5_3b_instruct_brpo \
    training.micro_batch_size=2 \
    training.use_kl_loss=True \
    optimization.adv_estimator=brpo \
    training.max_turns=5 \
    training.n_rollout=16 \
    training.train_batch_size=8 \
    training.ppo_batch_size=128
```

#### [EXP ]: GRPO.
**Same as [EXP xxx]**

### *EXP SET 10*: Context length extrapolation.
In this analysis, we aim to investigate how context length affects the performance of RAGEN. We use Sokoban as the base environment.
#### [EXP ]: 4000.
```bash
waiting to be done
```
#### [EXP ]: 8000.
```bash
waiting to be done
```
#### [EXP ]: 16000.
```bash
waiting to be done
```

### *EXP SET 11*: Do turns and temperature work?
In this analysis, we aim to investigate how turns and temperature affect the performance of RAGEN. We use Sokoban as the base environment.
#### [EXP yyy-zzz]: Max turns in [2, 5, 8], temperature in [0, 0.5, 1].
**Same as [EXP 4]**

### *EXP SET 12*: Binary and non-binary reward.
In this ablation study, we aim to investigate how binary and non-binary reward affects the performance of RAGEN. We use Sokoban as the base environment.
#### [EXP ]: Binary reward.
```bash
bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-3B-Instruct \
    model.experiment_name=binary_reward_sokoban_qwen_2.5_3b_instruct \
    training.micro_batch_size=2 \
    training.use_kl_loss=True \
    optimization.adv_estimator=grpo \
    training.max_turns=5 \
    training.n_rollout=16 \
    training.train_batch_size=8 \
    training.ppo_batch_size=128 \
    training.binary_reward=True
```
#### [EXP ]: Non-binary reward.
**Same as [EXP xxx]**

### *EXP SET 13*: Do we need to mask state?
In this ablation study, we aim to investigate how masking state affects the performance of RAGEN. We use Sokoban as the base environment.
#### [EXP ]: Mask state.
```bash
bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-3B-Instruct \
    model.experiment_name=mask_state_sokoban_qwen_2.5_3b_instruct \
    training.micro_batch_size=2 \
    training.use_kl_loss=True \
    optimization.adv_estimator=grpo \
    training.max_turns=5 \
    training.n_rollout=16 \
    training.train_batch_size=8 \
    training.ppo_batch_size=128 \
    training.mask_state=True
```

#### [EXP ]: Non-mask state.
**Same as [EXP xxx]**

### *EXP SET 14*: Output length penalty.
In this ablation study, we aim to investigate how output length penalty affects the performance of RAGEN. We use Sokoban as the base environment.
#### [EXP ]: Output length penalty.
```bash
bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-3B-Instruct \
    model.experiment_name=output_length_penalty_sokoban_qwen_2.5_3b_instruct \
    training.micro_batch_size=2 \
    training.use_kl_loss=True \
    optimization.adv_estimator=grpo \
    training.max_turns=5 \
    training.n_rollout=16 \
    training.train_batch_size=8 \
    training.ppo_batch_size=128 \
    training.length_penalty=True
```
#### [EXP ]: Non-output length penalty.
**Same as [EXP xxx]**

---
## Below are some examples.

```bash
bash train.sh sokoban \
    model.experiment_name=new_test

# override config
bash train.sh sokoban \
    model.experiment_name=test_zihan \
    training.train_batch_size=8 \
    training.ppo_batch_size=4

# For developers, if you want to add your own config keys, please check [ base.yaml | train.sh | ragen/train.py | verl/trainer/config/ppo_trainer.yaml | and the main_ppo.py in verl/trainer/ppo ] to make sure the changes are reflected coherently.
```

Below:Base experiment -> figure X in paper, aiming to xxx

```bash
bash train.sh sokoban \
    model.experiment_name=XXXX \
    argument ...
```


Below:Base experiment

```bash
bash train.sh sokoban \
    model.experiment_name=test_base
```

Below:GRPO

```bash
bash train.sh sokoban \
    model.experiment_name=test_zihan_brpo_p8r16 \
    training.n_rollout=16 \
    training.train_batch_size=8 \
    training.ppo_batch_size=128 \
    training.micro_batch_size=2 \
    optimization.adv_estimator=brpo \
    training.use_kl_loss=True
    # train_batch_size: rollout prompts
    # n_rollout: responses for each prompt
    # ppo_batch_size: update things
    # consider making the "epoch X step X" as "Rollout step X, update step X*X"?
    # grpo | brpo | apo
    # effective batch size: training.train_batch_size * training.n_rollout
```
