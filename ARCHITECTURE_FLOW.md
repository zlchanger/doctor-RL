# DoctorAgent-RL 分布式训练架构详解

> 本文档详细解释你提到的架构图在代码中的具体实现位置

---

## 架构图回顾

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
                    训练循环流程
```

---

## 一、架构组件代码定位

### 1. Ray 分布式协调器

**核心文件**: `ragen/trainer/main_ppo.py`

#### 1.1 入口函数 (main_ppo.py:167-173)

```python
@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # 初始化本地Ray集群
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    # 通过Ray远程执行主任务
    ray.get(main_task.remote(config))
```

**说明**:
- 使用 Hydra 加载配置
- 初始化 Ray 集群
- 启动远程任务 `main_task`

#### 1.2 主任务函数 (main_ppo.py:177-270)

```python
@ray.remote
def main_task(config):
    # 1. 加载tokenizer (main_ppo.py:194)
    tokenizer = hf_tokenizer(local_path)

    # 2. 定义Worker类型映射 (main_ppo.py:215-233)
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),  # Actor + Rollout 合并
        Role.Critic: ray.remote(CriticWorker),                 # Critic Worker
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),     # Reference Policy
        Role.EnvLLM: ray.remote(EnvironmentLLMWorker),         # 环境LLM (Patient)
    }

    # 3. 定义资源池 (main_ppo.py:221-229)
    resource_pool_spec = {
        'global_pool': [n_gpus_per_node] * nnodes,  # 例如: [8, 8] 表示2个节点,每个8个GPU
    }
    mapping = {
        Role.ActorRollout: 'global_pool',
        Role.Critic: 'global_pool',
        Role.RefPolicy: 'global_pool',
        Role.EnvLLM: 'global_pool',
    }

    # 4. 创建训练器 (main_ppo.py:259-268)
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=ResourcePoolManager(resource_pool_spec, mapping),
        reward_fn=reward_fn,
        env=train_env,
    )

    # 5. 初始化Workers并开始训练 (main_ppo.py:269-270)
    trainer.init_workers()  # 分配GPU资源,创建Worker进程
    trainer.fit()           # 开始训练循环
```

---

### 2. Worker 初始化流程

**核心文件**: `ragen/trainer/ppo/ray_trainer.py`

#### 2.1 RayPPOTrainer 类定义 (ray_trainer.py:399-478)

```python
class RayPPOTrainer(object):
    """
    在Driver进程(单个CPU/GPU节点)上运行的训练器
    """

    def __init__(self, config, tokenizer, role_worker_mapping, resource_pool_manager, ...):
        self.config = config
        self.tokenizer = tokenizer
        self.role_worker_mapping = role_worker_mapping  # Role -> Worker类的映射
        self.resource_pool_manager = resource_pool_manager

        # 根据算法选择是否使用Critic
        if config.algorithm.adv_estimator == 'gae':
            self.use_critic = True   # PPO使用Critic
        elif config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False  # GRPO不需要Critic
```

#### 2.2 init_workers() 方法 (ray_trainer.py:686-773)

这是**核心方法**,创建所有Worker实例:

```python
def init_workers(self):
    """
    创建所有Worker组,分配GPU资源
    """
    # 1. 创建资源池
    self.resource_pool_manager.create_resource_pool()

    # 2. 为每个Role创建Worker类配置
    self.resource_pool_to_cls = defaultdict(dict)

    # 2.1 创建 ActorRollout Worker (ray_trainer.py:695-703)
    resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
    actor_rollout_cls = RayClassWithInitArgs(
        cls=self.role_worker_mapping[Role.ActorRollout],
        config=self.config.actor_rollout_ref,
        role='actor_rollout'
    )
    self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls

    # 2.2 创建 Critic Worker (如果需要) (ray_trainer.py:705-709)
    if self.use_critic:
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
        critic_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Critic],
            config=self.config.critic
        )
        self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

    # 2.3 创建 Reference Policy Worker (ray_trainer.py:711-717)
    if self.use_reference_policy:
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
        ref_policy_cls = RayClassWithInitArgs(
            self.role_worker_mapping[Role.RefPolicy],
            config=self.config.actor_rollout_ref,
            role='ref'
        )
        self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

    # 2.4 创建 Env LLM Worker (Patient模拟) (ray_trainer.py:726-732)
    if self.config.env.use_env_llm:
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.EnvLLM)
        env_llm_cls = RayClassWithInitArgs(
            self.role_worker_mapping[Role.EnvLLM],
            config=self.config.env.env_llm,
            role='env_llm'
        )
        self.resource_pool_to_cls[resource_pool]['env_llm'] = env_llm_cls

    # 3. 初始化 WorkerGroup (ray_trainer.py:738-773)
    all_wg = {}
    for resource_pool, class_dict in self.resource_pool_to_cls.items():
        # 创建联合Worker类 (多个Worker共享GPU)
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)

        # 创建 RayWorkerGroup
        wg_dict = self.ray_worker_group_cls(
            resource_pool=resource_pool,
            ray_cls_with_init=worker_dict_cls
        )

        # 生成Worker实例
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        all_wg.update(spawn_wg)

    # 4. 初始化各个Worker的模型
    if self.use_critic:
        self.critic_wg = all_wg['critic']
        self.critic_wg.init_model()  # 加载Critic模型

    if self.use_reference_policy:
        self.ref_policy_wg = all_wg['ref']
        self.ref_policy_wg.init_model()  # 加载Reference模型

    if self.config.env.use_env_llm:
        self.env_llm_wg = all_wg['env_llm']
        self.env_llm_wg.init_model()  # 加载Patient LLM (vLLM)
        self.env.env_llm_worker = self.env_llm_wg  # 注入到环境

    # Actor + Rollout Worker最后初始化 (让vLLM先占用显存)
    self.actor_rollout_wg = all_wg['actor_rollout']
    self.actor_rollout_wg.init_model()  # 加载Actor模型 (FSDP)
```

**关键点**:
- `create_colocated_worker_cls`: 允许多个Worker共享同一组GPU资源
- 初始化顺序很重要: EnvLLM先初始化(vLLM需要预留显存), ActorRollout最后初始化

---

### 3. 训练循环流程

**核心文件**: `ragen/trainer/ppo/ray_trainer.py`

#### 3.1 fit() 方法 - 主训练循环 (ray_trainer.py:904-1199)

```python
def fit(self):
    """
    PPO训练循环
    Driver进程通过RPC调用Worker的计算函数来构建PPO数据流
    轻量级的advantage计算在Driver进程完成
    """

    # 准备Generation配置
    gen_config = GenerationConfig(
        max_turns=self.config.max_turns,  # 最大对话轮次
        max_response_length=self.config.data.max_response_length,
        # ...
    )

    # 创建LLM生成管理器
    generation_manager = LLMGenerationManager(
        tokenizer=self.tokenizer,
        actor_rollout_wg=self.actor_rollout_wg,  # Actor Worker
        env_class=self.env_class,
        config=gen_config,
    )

    # 为每个prompt创建独立的环境实例
    envs = [self.env.copy() for _ in range(batch_size * n_agent)]

    # =========================================
    # 开始训练循环
    # =========================================
    for epoch in range(total_epochs):
        for batch_dict in self.train_dataloader:

            # ====================================
            # 步骤1: 环境重置 (ray_trainer.py:976-982)
            # ====================================
            batch = DataProto.from_single_dict(batch_dict)
            batch = batch.repeat(repeat_times=n_agent, interleave=True)

            env_seeds = [item['index'] for item in batch.non_tensor_batch['extra_info']]
            for env, seed in zip(envs, env_seeds):
                env.reset(seed=seed)  # 重置环境,加载患者案例

            # ====================================
            # 步骤2: Rollout - 多轮对话生成 (ray_trainer.py:1007-1031)
            # ====================================
            gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

            # 运行多轮对话循环
            final_gen_batch_output = generation_manager.run_llm_loop(
                gen_batch=gen_batch,
                envs=envs,                    # 环境列表 (Patient模拟)
                initial_input_ids=first_input_ids,
                output_dir=output_dir,
                global_steps=self.global_steps,
            )
            # 这个循环内部:
            # for turn in range(max_turns):
            #     1. Actor生成问题/诊断
            #     2. 提取<answer>标签内容
            #     3. 调用env.step(action) -> 计算reward
            #     4. 如果未结束, EnvLLM生成患者回答
            #     5. 拼接对话历史,继续下一轮

            # ====================================
            # 步骤3: 收集Reward和Metrics (ray_trainer.py:1037-1084)
            # ====================================
            # 设置uid用于GRPO分组归一化
            if adv_estimator == 'grpo':
                batch.non_tensor_batch['uid'] = np.array([str(i) for i in env_seeds])

            # 收集每个episode的reward
            batch.non_tensor_batch['reward'] = np.array([env.reward for env in envs])

            # 收集诊断得分 (如果环境支持)
            if hasattr(envs[0], 'diagnosis_score'):
                batch.non_tensor_batch['diagnosis_score'] = [env.diagnosis_score for env in envs]

            # Reward归一化 (GRPO/BRPO/ARPO)
            if reward_norm_type is not None:
                batch.non_tensor_batch['reward'] = normalize_reward(
                    batch.non_tensor_batch['reward'],
                    batch.non_tensor_batch['uid'],
                    reward_norm_type
                )

            # 收集action统计
            for idx, env in enumerate(envs):
                tracking_vars = env.get_tracking_variables()
                batch.non_tensor_batch['valid_action'][idx] = sum(1 for x in tracking_vars['actions_valid'] if x is not None)
                batch.non_tensor_batch['effective_action'][idx] = sum(1 for x in tracking_vars['actions_effective'] if x is not None)

            # ====================================
            # 步骤4: 计算Log Probs (ray_trainer.py:1104-1116)
            # ====================================
            # 使用当前策略重新计算log_prob (用于PPO更新)
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            batch = batch.union(old_log_prob)

            # ====================================
            # 步骤5: 计算Reference Log Probs (ray_trainer.py:1118-1122)
            # ====================================
            if self.use_reference_policy:
                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

            # ====================================
            # 步骤6: 计算Values (仅PPO需要) (ray_trainer.py:1124-1128)
            # ====================================
            if self.use_critic:
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)

            # ====================================
            # 步骤7: 计算Advantages (ray_trainer.py:1130-1157)
            # ====================================
            # 应用KL惩罚 (如果使用)
            if use_kl_in_reward:
                batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward)
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            # 计算优势函数
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,  # 'grpo' or 'gae'
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=n_agent
            )

            # ====================================
            # 步骤8: 更新Critic (仅PPO) (ray_trainer.py:1159-1164)
            # ====================================
            if self.use_critic:
                critic_output = self.critic_wg.update_critic(batch)
                metrics.update(critic_output.meta_info['metrics'])

            # ====================================
            # 步骤9: 更新Actor (ray_trainer.py:1167-1174)
            # ====================================
            if self.global_steps > critic_warmup:
                actor_output = self.actor_rollout_wg.update_actor(batch)
                metrics.update(actor_output.meta_info['metrics'])

            # ====================================
            # 步骤10: 验证和保存 (ray_trainer.py:1176-1186)
            # ====================================
            if self.global_steps % test_freq == 0:
                val_metrics = self._validate()
                metrics.update(val_metrics)

            if self.global_steps % save_freq == 0:
                self._save_checkpoint()

            # 记录指标
            logger.log(data=metrics, step=self.global_steps)
            self.global_steps += 1
```

---

## 二、核心数据流

### 1. Rollout阶段的多轮对话流程

**关键文件**: `ragen/llm_agent/generation.py` (LLMGenerationManager)

```python
class LLMGenerationManager:
    def run_llm_loop(self, gen_batch, envs, initial_input_ids, ...):
        """
        运行多轮对话循环
        """
        for turn_idx in range(max_turns):
            # 1. Actor生成回答 (问题或诊断)
            gen_output = self.actor_rollout_wg.generate_sequences(current_gen_batch)

            # 2. 解码生成的文本
            responses = self.tokenizer.batch_decode(gen_output.batch['responses'])

            # 3. 提取<answer>标签内容
            actions = [extract_answer(resp) for resp in responses]

            # 4. 环境执行action,计算reward
            observations = []
            for env, action in zip(envs, actions):
                obs, reward, done, info = env.step(action)
                observations.append(obs)

            # 5. 如果episode未结束,生成患者回答
            if not all_done:
                # 调用 EnvLLM Worker (vLLM)
                patient_responses = self.env_llm_wg.generate_patient_response(
                    patient_descriptions=descriptions,
                    dialogue_histories=histories
                )

                # 6. 拼接对话历史
                for idx, patient_resp in enumerate(patient_responses):
                    histories[idx] += f"\nDoctor: {actions[idx]}\nPatient: {patient_resp}"

            # 7. 如果所有episode都结束,退出循环
            if all([env.finished() for env in envs]):
                break

        return final_gen_batch_output
```

### 2. 环境 (Environment) 的执行流程

**关键文件**: `ragen/env/medical_consultation/env_patient_llm.py`

```python
class MedicalConsultationEnvWithPatientLLM:
    def reset(self, seed):
        """
        初始化episode
        """
        self.index = self._shared_seed_to_index[seed]
        self.description = self._shared_data[self.index]['description']  # 患者描述
        self.target_diagnosis = self._shared_data[self.index]['target']  # 正确诊断
        self.patient_information = self._shared_data[self.index]['patient_information']  # 有效问答对

        self.current_turn = 0
        self.diagnosis_made = False
        self.reward = 0.0

        # 返回初始观察
        return self.description

    def step(self, action):
        """
        执行doctor的action (问题或诊断)
        """
        self.current_turn += 1

        # 1. 检查是否是诊断
        if self._is_diagnosis(action):
            self.diagnosis_made = True

            # 计算诊断奖励
            if self._check_diagnosis_correct(action, self.target_diagnosis):
                self.reward += 1.0  # 诊断正确
            else:
                self.reward += 0.0  # 诊断错误

            done = True
            obs = "Diagnosis submitted."

        # 2. 否则是提问
        else:
            # 检查问题是否有效 (在patient_information中)
            is_valid = self._check_question_valid(action, self.patient_information)

            if is_valid:
                self.reward += 0.1  # 有效问题奖励
                self._actions_valid.append(True)
            else:
                self.reward += -0.05  # 无效问题惩罚
                self._actions_valid.append(False)

            # 对话轮次惩罚
            self.reward += -0.01

            # 检查是否达到最大轮次
            done = (self.current_turn >= self.max_turns)

            # 观察将由EnvLLM生成 (患者回答)
            obs = None  # 将在外部通过env_llm_worker填充

        return obs, self.reward, done, {}

    def _check_question_valid(self, question, patient_information):
        """
        检查问题是否在有效问答对列表中
        """
        for qa_pair in patient_information:
            doctor_q = qa_pair['doctor_question']
            # 使用ROUGE分数或语义相似度判断
            if similarity(question, doctor_q) > threshold:
                return True
        return False
```

---

## 三、Worker 内部实现

### 1. ActorRollout Worker (FSDP策略模型)

**关键文件**: `ragen/workers/fsdp_workers.py` (或 `verl/workers/fsdp_workers.py`)

```python
class ActorRolloutRefWorker:
    def init_model(self):
        """
        初始化FSDP模型
        """
        # 加载HuggingFace模型
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

        # 包装为FSDP
        self.model = FSDP(
            self.model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            # ...
        )

    def generate_sequences(self, batch):
        """
        Rollout生成 (采样多个回答用于GRPO)
        """
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=batch.batch['input_ids'],
                attention_mask=batch.batch['attention_mask'],
                max_new_tokens=max_response_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=n_agent,  # GRPO: 每个prompt生成多个回答
            )

        return DataProto(batch={'responses': outputs})

    def compute_log_prob(self, batch):
        """
        计算当前策略的log_prob
        """
        with torch.no_grad():
            logits = self.model(
                input_ids=batch.batch['input_ids'],
                attention_mask=batch.batch['attention_mask']
            ).logits

            log_probs = compute_log_probs_from_logits(logits, batch.batch['responses'])

        return DataProto(batch={'old_log_probs': log_probs})

    def update_actor(self, batch):
        """
        PPO/GRPO策略更新
        """
        for micro_batch in split_into_micro_batches(batch):
            # 前向传播
            logits = self.model(
                input_ids=micro_batch.batch['input_ids'],
                attention_mask=micro_batch.batch['attention_mask']
            ).logits

            # 计算PPO loss
            loss = compute_ppo_loss(
                logits=logits,
                old_log_probs=micro_batch.batch['old_log_probs'],
                advantages=micro_batch.batch['advantages'],
                clip_ratio=self.config.clip_ratio
            )

            # 反向传播
            loss.backward()

            # 梯度累积后更新
            if (step + 1) % gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        return DataProto(meta_info={'metrics': {'actor/loss': loss.item()}})
```

### 2. Env LLM Worker (vLLM患者模拟)

**关键文件**: `ragen/workers/env_llm_worker.py`

```python
class EnvironmentLLMWorker:
    def init_model(self):
        """
        初始化vLLM推理引擎
        """
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=self.config.model_path,
            tensor_parallel_size=self.config.tp_size,
            gpu_memory_utilization=0.85,
            # vLLM高性能推理配置
        )

        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=self.config.max_tokens
        )

    def generate_patient_response(self, prompts):
        """
        批量生成患者回答

        Args:
            prompts: List[str], 每个prompt包含:
                - 患者描述
                - 对话历史
                - 医生最新问题

        Returns:
            List[str]: 患者回答
        """
        # 使用vLLM批量推理
        outputs = self.llm.generate(prompts, self.sampling_params)

        # 提取生成的文本
        responses = [output.outputs[0].text for output in outputs]

        return responses
```

**Prompt格式示例**:
```
你是一位患者,根据以下描述回答医生的问题:

患者描述:
{patient_description}

对话历史:
Doctor: 您好,请问哪里不舒服?
Patient: 我最近总是感到口渴,还经常上厕所。

Doctor: {latest_question}
Patient:
```

### 3. Critic Worker (仅PPO需要)

**关键文件**: `ragen/workers/fsdp_workers.py`

```python
class CriticWorker:
    def init_model(self):
        """
        初始化Value网络
        """
        # 通常基于Actor模型,替换最后一层为单输出
        self.value_model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=1  # 输出单个value
        )

        # FSDP包装
        self.value_model = FSDP(self.value_model, ...)

    def compute_values(self, batch):
        """
        计算state value
        """
        with torch.no_grad():
            values = self.value_model(
                input_ids=batch.batch['input_ids'],
                attention_mask=batch.batch['attention_mask']
            ).logits.squeeze(-1)

        return DataProto(batch={'values': values})

    def update_critic(self, batch):
        """
        更新Value网络
        """
        # 计算target value (使用GAE)
        target_values = batch.batch['advantages'] + batch.batch['values']

        # 前向传播
        predicted_values = self.value_model(...).logits.squeeze(-1)

        # MSE loss
        loss = F.mse_loss(predicted_values, target_values)

        loss.backward()
        self.optimizer.step()

        return DataProto(meta_info={'metrics': {'critic/loss': loss.item()}})
```

---

## 四、关键算法实现

### 1. GRPO优势函数计算

**文件**: `ragen/trainer/ppo/ray_trainer.py:145-200`

```python
def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    """
    计算优势函数

    Args:
        data: 包含rewards, uid (用于分组)
        adv_estimator: 'grpo' | 'gae' | 'brpo' | ...
        num_repeat: 每个prompt的响应数 (GRPO用)
    """

    if adv_estimator == 'grpo':
        # Group Relative Policy Optimization
        # 1. 按uid分组
        grouped_rewards = defaultdict(list)
        for uid, reward in zip(data.non_tensor_batch['uid'], data.non_tensor_batch['reward']):
            grouped_rewards[uid].append(reward)

        # 2. 组内归一化
        advantages = []
        for uid in data.non_tensor_batch['uid']:
            group_rewards = grouped_rewards[uid]
            # 减去组内均值,除以组内标准差
            mean = np.mean(group_rewards)
            std = np.std(group_rewards) + 1e-8
            adv = (data.non_tensor_batch['reward'] - mean) / std
            advantages.append(adv)

        data.batch['advantages'] = torch.tensor(advantages)

    elif adv_estimator == 'gae':
        # Generalized Advantage Estimation (PPO)
        # A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2 δ_{t+2} + ...
        # 其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)

        values = data.batch['values']
        rewards = data.batch['token_level_rewards']

        advantages = []
        last_gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t+1] - values[t]
            last_gae = delta + gamma * lam * last_gae
            advantages.insert(0, last_gae)

        data.batch['advantages'] = torch.tensor(advantages)

    return data
```

### 2. PPO Loss计算

**文件**: `verl/trainer/ppo/core_algos.py`

```python
def compute_ppo_loss(logits, old_log_probs, advantages, clip_ratio=0.2):
    """
    PPO Clipped Objective

    L^CLIP(θ) = E[ min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A) ]
    其中 r(θ) = π_θ(a|s) / π_old(a|s)
    """
    # 计算当前策略的log_prob
    log_probs = compute_log_probs_from_logits(logits, actions)

    # 计算ratio
    ratio = torch.exp(log_probs - old_log_probs)

    # 计算两个目标
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages

    # 取最小值并求负 (因为我们最大化目标)
    loss = -torch.min(surr1, surr2).mean()

    return loss
```

---

## 五、完整训练流程总结

### 训练一个Batch的完整步骤:

```
1. 数据加载 (Driver)
   ├─ 从DataLoader获取batch (患者案例)
   └─ 重复n_agent次 (GRPO需要每个prompt多个响应)

2. 环境重置 (Driver)
   ├─ 为每个样本创建环境实例
   └─ env.reset(seed) - 加载患者描述、正确诊断等

3. Rollout - 多轮对话生成 (Actor Worker + Env LLM Worker)
   ├─ for turn in range(max_turns):
   │   ├─ Actor生成: 调用 actor_rollout_wg.generate_sequences()
   │   ├─ 提取action: 解析 <answer>标签
   │   ├─ 环境执行: env.step(action) -> reward, done
   │   ├─ 如果未结束:
   │   │   ├─ 构造Patient Prompt (描述 + 历史 + 问题)
   │   │   ├─ Env LLM生成患者回答: env_llm_wg.generate()
   │   │   └─ 拼接对话历史
   │   └─ 如果所有episode结束 -> break
   └─ 返回完整对话历史 + rewards

4. 收集Metrics (Driver)
   ├─ 从环境收集: reward, diagnosis_score, valid_actions
   └─ Reward归一化 (GRPO/BRPO)

5. 计算Log Probs (Actor Worker)
   ├─ 当前策略: actor_rollout_wg.compute_log_prob()
   └─ 参考策略: ref_policy_wg.compute_ref_log_prob() [可选]

6. 计算Values (Critic Worker) [仅PPO]
   └─ critic_wg.compute_values()

7. 计算Advantages (Driver)
   ├─ 应用KL惩罚 [可选]
   └─ 根据算法计算:
       ├─ GRPO: 组内归一化
       └─ GAE: 时序差分估计

8. 更新Critic (Critic Worker) [仅PPO]
   └─ critic_wg.update_critic() - MSE loss

9. 更新Actor (Actor Worker)
   └─ actor_rollout_wg.update_actor() - PPO/GRPO loss

10. 验证和保存 (Driver + Actor Worker)
    ├─ 定期验证: _validate()
    └─ 定期保存: _save_checkpoint()
```

---

## 六、配置文件对应关系

### 关键配置参数 -> 代码位置

| 配置参数 | 代码位置 | 说明 |
|---------|---------|------|
| `data.train_batch_size` | `ray_trainer.py:976` | 每次rollout的prompt数 |
| `actor_rollout_ref.rollout.n_agent` | `ray_trainer.py:977` | 每个prompt的响应数(GRPO) |
| `algorithm.adv_estimator` | `ray_trainer.py:451-462` | 'grpo' 或 'gae' |
| `env.max_turns` | `generation.py` | 最大对话轮次 (-1为随机2-10) |
| `env.use_env_llm` | `ray_trainer.py:727-732` | 是否使用vLLM模拟患者 |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | `fsdp_workers.py` | PPO更新batch大小 |
| `algorithm.kl_ctrl.kl_coef` | `ray_trainer.py:114-143` | KL散度惩罚系数 |
| `trainer.n_gpus_per_node` | `main_ppo.py:223` | 每个节点GPU数 |
| `trainer.nnodes` | `main_ppo.py:223` | 节点数 |

---

## 七、调试和监控

### 1. 查看Worker分配

```bash
# 启动训练后,可以在Ray Dashboard查看
# http://localhost:8265

# 或使用命令行
ray status
```

### 2. 关键日志位置

- **训练指标**: WandB (如果配置了WANDB_API_KEY)
- **Worker日志**: Ray会自动重定向到 `/tmp/ray/session_*/logs/`
- **Checkpoint**: `checkpoints/{project_name}/{exp_name}/global_step_{N}/`

### 3. 常见调试点

```python
# 在 ray_trainer.py:1007 添加断点,查看rollout输入
# 在 env_patient_llm.py:84 查看环境重置
# 在 generation.py 查看多轮对话生成过程
```

---

## 八、性能优化技巧

### 1. GPU显存分配策略

```python
# ray_trainer.py:771-773
# Actor在最后初始化,让vLLM先占用显存
self.env_llm_wg.init_model()  # vLLM先初始化
self.actor_rollout_wg.init_model()  # Actor后初始化
```

### 2. 批量大小调优

```yaml
# 总有效batch大小 = train_batch_size * n_agent * gradient_accumulation_steps
data.train_batch_size: 8       # Rollout批量
actor_rollout_ref.rollout.n_agent: 4  # GRPO响应数
actor_rollout_ref.actor.gradient_accumulation_steps: 2

# 实际每次更新的样本数 = 8 * 4 * 2 = 64
```

### 3. vLLM并行度

```yaml
env.env_llm.tensor_parallel_size: 2  # vLLM使用2个GPU并行
env.env_llm.gpu_memory_utilization: 0.85
```

---

## 总结

这个架构图在代码中的对应关系:

1. **Ray分布式协调器** → `main_ppo.py` + `ray_trainer.py:RayPPOTrainer`
2. **Actor Worker** → `fsdp_workers.py:ActorRolloutRefWorker`
3. **Critic Worker** → `fsdp_workers.py:CriticWorker`
4. **Rollout Worker** → 与Actor合并在 `ActorRolloutRefWorker`
5. **Env LLM Worker** → `env_llm_worker.py:EnvironmentLLMWorker`
6. **训练循环流程** → `ray_trainer.py:fit()` (第904-1199行)

**核心训练流程**: 环境重置 → 多轮Rollout → 收集Reward → 计算Advantage → 更新策略

希望这个详细的文档能帮助你理解整个分布式训练架构! 🚀