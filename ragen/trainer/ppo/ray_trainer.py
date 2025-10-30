# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
import copy

import re
import json
from collections import defaultdict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.trainer.ppo.core_algos import agg_loss

import re
from ragen.llm_agent.generation import LLMGenerationManager, GenerationConfig

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6
    EnvLLM = 7


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def compute_response_mask(data: DataProto):
    """
    计算响应部分的mask
    从完整的attention_mask中提取出仅对应于模型生成响应（response）部分的mask
    用于在后续计算中只关注响应token，忽略提示（prompt）部分
    """
    # 获取响应tensor，shape: (batch_size, response_length)
    responses = data.batch["responses"]
    # 获取响应序列的长度（第二维度大小）
    response_length = responses.size(1)
    # 获取完整的attention_mask，shape: (batch_size, prompt_length + response_length)
    attention_mask = data.batch["attention_mask"]
    # 切片提取最后response_length个位置的mask，即只保留响应部分的mask
    # 返回shape: (batch_size, response_length)
    return attention_mask[:, -response_length:]


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """
    应用KL散度惩罚到奖励中
    主要功能：
    1. 计算当前策略与参考策略之间的KL散度，防止策略更新过快偏离初始策略
    2. 从原始token级别奖励中减去加权的KL散度，形成最终的token级别奖励
    3. 使用自适应KL控制器动态调整KL惩罚系数beta
    4. 返回更新后的数据和相关指标（KL散度值和系数）
    """
    # 获取响应tensor和相关尺寸信息
    responses = data.batch["responses"]
    response_length = responses.size(1)  # 响应序列长度
    # 获取环境/奖励模型计算的原始token级别分数
    token_level_scores = data.batch["token_level_scores"]
    # 获取批次大小（用于更新KL控制器）
    batch_size = data.batch.batch_size[0]

    # 如果response_mask未提前计算，则现在计算
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    response_mask = data.batch["response_mask"]

    # 计算参考策略和当前策略之间的KL散度
    # 用于衡量策略更新的幅度，防止策略突变
    # old_log_probs: 当前策略在rollout时的log概率
    # ref_log_prob: 参考策略（固定）的log概率
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # shape: (batch_size, response_length)

    # 将KL散度乘以response_mask，屏蔽掉padding部分的KL散度
    kld = kld * response_mask

    # 获取当前的KL惩罚系数beta（自适应调整）
    beta = kl_ctrl.value

    # 计算最终的token级别奖励：原始分数 - beta * KL散度
    # 这样可以在优化奖励的同时，惩罚过大的策略偏移
    token_level_rewards = token_level_scores - beta * kld

    # 计算当前批次的平均KL散度（用于监控和自适应调整）
    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # 先在序列维度求平均
    current_kl = torch.mean(current_kl, dim=0).item()  # 再在批次维度求平均，得到标量

    # 根据当前KL散度更新KL控制器
    # KL控制器会自适应地调整beta值，使得KL散度保持在目标范围内
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)

    # 将计算好的token级别奖励存回data中
    data.batch["token_level_rewards"] = token_level_rewards

    # 记录指标：当前KL散度值和惩罚系数
    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    """
    计算优势函数（Advantage）和回报（Return）
    主要功能：
    1. 根据指定的优势估计器算法（adv_estimator）计算优势值
    2. 支持多种算法：GAE（广义优势估计）、GRPO（分组相对策略优化）、REINFORCE++、REMAX、RLOO等
    3. 优势值用于指导策略梯度更新，衡量某个动作相比平均水平的好坏程度
    4. 返回值用于计算策略损失

    参数：
    - gamma: 折扣因子，用于计算未来奖励的现值
    - lam: GAE中的lambda参数，用于平衡偏差和方差
    """
    # 向后兼容：如果response_mask未提前计算，则现在计算
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)

    # 根据不同的优势估计器类型，调用相应的计算函数
    # TODO: add other ways to estimate advantages

    # ===== GAE（Generalized Advantage Estimation）广义优势估计 =====
    # 基于时序差分（TD）的方法，需要critic提供value函数
    # 优点：平衡偏差和方差，训练更稳定
    if adv_estimator == AdvantageEstimator.GAE:
        # 使用GAE算法计算优势和回报
        # 需要输入：token级别奖励、value函数预测值、响应mask、折扣因子gamma、lambda参数
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    # ===== GRPO（Group Relative Policy Optimization）分组相对策略优化 =====
    # 基于outcome的方法，不需要critic
    # 在同一个prompt的多个响应中进行相对比较，计算优势
    elif adv_estimator == AdvantageEstimator.GRPO:
        # index参数（uid）用于标识哪些样本属于同一个prompt组
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    # ===== REINFORCE++ with Baseline =====
    # REINFORCE算法的改进版本，使用baseline减少方差
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    # ===== REINFORCE++ =====
    # REINFORCE算法的增强版本
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    # ===== REMAX（Reward Maximization）奖励最大化 =====
    # 使用reward baseline进行归一化
    elif adv_estimator == AdvantageEstimator.REMAX:
        # reward_baselines: 预先计算的奖励基线，用于归一化
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    # ===== RLOO（Reinforcement Learning with Leave-One-Out）留一法强化学习 =====
    # 使用leave-one-out方法计算baseline，减少方差
    elif adv_estimator == AdvantageEstimator.RLOO:
        # 对每个样本，使用同组其他样本的平均奖励作为baseline
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    else:
        # 不支持的优势估计器类型
        raise NotImplementedError(f"Advantage estimator {adv_estimator} is not implemented")

    # 返回包含advantages和returns的data对象
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def normalize_reward(reward, uid, reward_norm_type):
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    for r, u in zip(reward, uid):
        id2score[u].append(r)
    for u in id2score:
        if len(id2score[u]) == 1:
            id2mean[u] = torch.tensor(0.0)
            id2std[u] = torch.tensor(1.0)
        elif len(id2score[u]) > 1:
            id2mean[u] = torch.mean(torch.tensor(id2score[u], dtype=torch.float32))
            id2std[u] = torch.std(torch.tensor([id2score[u]], dtype=torch.float32))
        else:
            raise ValueError(f"no score in prompt index: {u}")
    normalized_reward = [(r - id2mean[u]) / (id2std[u] + 1e-6) for r, u in zip(reward, uid)]  # NOTE: +1e-6, maybe +1!
    # transform to the same dtype as reward
    return np.array(normalized_reward, dtype=reward.dtype)


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    if "response_mask" not in batch.batch:
        response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()
    else:
        response_mask = batch.batch['response_mask'].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
               # values
               'critic/values/mean': torch.mean(valid_values).detach().item(),
               'critic/values/max': torch.max(valid_values).detach().item(),
               'critic/values/min': torch.min(valid_values).detach().item(),
               # vf explained var
               'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
           } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),

        # metrics for actions
        'metric/total_env':
            int(np.array(batch.non_tensor_batch['total_env'], dtype=np.int16).sum()),
        'metric/finished_env':
            int(np.array(batch.non_tensor_batch['finished_env'], dtype=np.int16).sum()),
        'metric/success_env':
            int(np.array(batch.non_tensor_batch['success_env'], dtype=np.int16).sum()),
        'metric/traj_length':
            float(np.array(batch.non_tensor_batch['traj_length'], dtype=np.int16).mean()),
        'metric/valid_action':
            float(np.array(batch.non_tensor_batch['valid_action'], dtype=np.int16).mean()),
        'metric/effective_action':
            float(np.array(batch.non_tensor_batch['effective_action'], dtype=np.int16).mean()),
        'metric/effective_action_ratio':
            float(np.array(batch.non_tensor_batch['effective_action_ratio'], dtype=np.float32).mean()),
        'metric/reward':
            float(np.array(batch.non_tensor_batch['ori_reward'], dtype=np.float32).mean()),
    }

    if 'diagnosis_score' in batch.non_tensor_batch:
        metrics['metric/diagnosis_score'] = float(
            np.array(batch.non_tensor_batch['diagnosis_score'], dtype=np.float32).mean())
        metrics['metric/recommandation_score'] = float(
            np.array(batch.non_tensor_batch['recommandation_score'], dtype=np.float32).mean())

    # metric for two-armed bandit
    if batch.non_tensor_batch['data_source'][0] == 'two_armed_bandit':
        batch_action = np.array(batch.non_tensor_batch['bandit_metrics'], dtype=np.int16)
        metrics['metric/n_low_arm'] = int(np.sum(batch_action == 1))
        metrics['metric/n_high_arm'] = int(np.sum(batch_action == 2))
        metrics['metric/n_invalid'] = int(np.sum(batch_action == 0))

    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor', 'rollout']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in
            set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    基于Ray的分布式PPO训练器。

    该训练器运行在驱动进程上（单个CPU/GPU节点），通过Ray管理分布式worker进行训练。
    支持FSDP、Megatron等多种并行策略，适用于大规模语言模型的强化学习训练。

    主要功能：
    - 分布式训练编排（Actor、Critic、RefPolicy、RewardModel等角色）
    - 支持多种优势估计算法（GAE、GRPO、RLOO等）
    - 支持混合引擎（Actor + Rollout融合）
    - 支持参考策略KL惩罚
    - 支持checkpoint保存和恢复
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None,
                 env=None,
                 val_env=None,
                 env_class=None):
        """
        初始化PPO训练器。

        Args:
            config: 训练配置对象（Hydra配置）
            tokenizer: 分词器实例
            role_worker_mapping: 角色到Worker类的映射字典
            resource_pool_manager: 资源池管理器，负责分配GPU资源
            ray_worker_group_cls: Ray worker group类，默认为RayWorkerGroup
            reward_fn: 奖励函数，用于计算token级别的奖励
            val_reward_fn: 验证时使用的奖励函数
            env: 训练环境实例
            val_env: 验证环境实例，默认与训练环境相同
            env_class: 环境类，用于创建环境副本
        """

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        # 基本配置
        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.env = env
        self.val_env = env if val_env is None else val_env
        self.env_class = env_class

        if val_env is not None:
            print(
                "[INFO] val env is different from train env, it means you are evaluating the model's generalization capabilities.")

        # 检查混合引擎配置
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        # Worker管理配置
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping and not config.algorithm.no_ref_policy
        print(f"use_reference_policy: {self.use_reference_policy}")
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        self.val_num = 0  # 验证次数计数器

        # 定义奖励中的KL控制器
        # 注意：KL损失控制当前不支持
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        # 根据优势估计器类型决定是否使用critic
        # GAE需要value function，因此需要critic
        # GRPO/RLOO等outcome-based方法不需要critic
        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        # 初始化流程
        self._validate_config()  # 验证配置参数的正确性
        self._create_dataloader()  # 创建训练和验证数据加载器
        self._init_logger()  # 初始化日志记录器

    def _validate_config(self):
        """
        验证训练配置的正确性。

        检查各项配置参数是否合理，包括：
        - 批次大小是否能被GPU数量整除
        - micro_batch_size和micro_batch_size_per_gpu参数是否冲突
        - 序列并行配置是否正确
        - 验证配置是否合理

        如果发现配置错误，会抛出AssertionError或ValueError。
        """
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. "
                        f"Please remove '{name}.{param}' because only '*_{param_per_gpu}'"
                        + "is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic"
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                        config.actor_rollout_ref.actor.ppo_mini_batch_size
                        % config.actor_rollout_ref.actor.ppo_micro_batch_size
                        == 0
                )
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (
                config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
                or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
        ):
            assert config.actor_rollout_ref.model.use_remove_padding, (
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, (
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                )

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _init_logger(self):
        """
        初始化日志记录器。

        创建Tracking实例用于记录训练指标到WandB、TensorBoard等后端。
        所有训练过程中的指标（loss、reward、奖励等）都通过此logger记录。
        """
        from verl.utils.tracking import Tracking
        self.logger = Tracking(
            project_name=self.config.trainer.project_name,  # 项目名称
            experiment_name=self.config.trainer.experiment_name,  # 实验名称
            default_backend=self.config.trainer.logger,  # 日志后端（wandb/tensorboard）
            config=OmegaConf.to_container(self.config, resolve=True)  # 配置转为字典
        )

    def _create_dataloader(self):
        """
        创建训练和验证数据加载器。

        该方法负责：
        1. 从parquet文件加载训练和验证数据集
        2. 过滤掉超长的prompt（根据max_prompt_length）
        3. 创建StatefulDataLoader以支持checkpoint恢复
        4. 计算总训练步数并注入到优化器配置中

        数据集使用RLHFDataset类加载，支持：
        - 自动tokenization
        - Prompt长度过滤
        - 可选的数据采样
        """
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from ragen.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        # 创建训练数据集
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        if self.config.data.train_data_num is not None:
            if self.config.data.train_data_num > len(self.train_dataset.dataframe):
                print(
                    f"[WARNING] training dataset size is smaller than desired size. Using the dataset as the original size {len(self.train_dataset.dataframe)}")
            else:
                self.train_dataset.dataframe = self.train_dataset.dataframe.sample(self.config.data.train_data_num,
                                                                                   random_state=42)
        print(f"filtered training dataset size: {len(self.train_dataset.dataframe)}")

        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get("seed", 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=8,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler,
        )

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        if self.config.data.val_data_num is not None:
            if self.config.data.val_data_num > len(self.val_dataset.dataframe):
                print(
                    f"[WARNING] validation dataset size is smaller than desired size. Using the dataset as the original size {len(self.val_dataset.dataframe)}")
            else:
                self.val_dataset.dataframe = self.val_dataset.dataframe.sample(self.config.data.val_data_num,
                                                                               random_state=42)
        print(f"filtered validation dataset size: {len(self.val_dataset.dataframe)}")

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.data.val_batch_size,
            num_workers=8,
            shuffle=False,
            drop_last=True,
            collate_fn=collate_fn,
        )

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def init_workers(self):
        """
        初始化Ray分布式workers。

        该方法负责创建和初始化所有分布式训练所需的worker组：
        1. 创建资源池（GPU资源分配）
        2. 为每个角色创建对应的Worker类（Actor/Critic/RefPolicy/RewardModel/EnvLLM）
        3. 使用Ray的worker group机制管理分布式进程
        4. 初始化所有模型（加载权重到GPU显存）

        Worker组包括：
        - actor_rollout_wg: Actor和Rollout融合的worker（策略网络训练和推理）
        - critic_wg: Critic网络worker（价值函数估计，仅GAE算法使用）
        - ref_policy_wg: 参考策略worker（计算KL散度，可选）
        - rm_wg: 奖励模型worker（基于模型的奖励打分，可选）
        - env_llm_wg: 环境LLM worker（模拟环境响应，用于对话任务）
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # create env_llm worker
        if self.config.env.use_env_llm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.EnvLLM)
            env_llm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.EnvLLM],
                                               config=self.config.env.env_llm,
                                               role='env_llm')
            self.resource_pool_to_cls[resource_pool]['env_llm'] = env_llm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # initialize env_llm worker
        if self.config.env.use_env_llm:
            self.env_llm_wg = all_wg['env_llm']
            self.env_llm_wg.init_model()
            self.env.env_llm_worker = self.env_llm_wg
            self.val_env.env_llm_worker = self.env_llm_wg

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        """
        保存训练checkpoint。

        该方法负责保存完整的训练状态,包括:
        1. Actor模型参数和优化器状态
        2. Critic模型参数和优化器状态(如果使用GAE)
        3. DataLoader状态(用于断点续训时精确恢复数据批次)
        4. 最新checkpoint标记文件

        Checkpoint路径结构:
        - 本地路径: {default_local_dir}/global_step_{N}/actor/
        - 远程路径: {default_hdfs_dir}/global_step_{N}/actor/ (可选)

        支持自动清理旧checkpoint,保留最新的N个checkpoint(由max_*_ckpt_to_keep控制)。
        """
        # 步骤1: 构建checkpoint保存路径
        # 路径格式: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        # Actor模型的本地保存路径
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        # 步骤2: 构建可选的远程路径(如HDFS)
        # 如果配置了HDFS路径,则将checkpoint同步到远程存储
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        # 步骤3: 配置checkpoint保留策略
        # 旧版参数remove_previous_ckpt_in_save已废弃,建议使用max_*_ckpt_to_keep
        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        # 确定Actor和Critic分别保留多少个checkpoint
        # None表示保留所有checkpoint,1表示只保留最新的
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        # 步骤4: 保存Actor模型checkpoint
        # 包括模型参数、优化器状态、学习率调度器状态等
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        # 步骤5: 保存Critic模型checkpoint(如果使用GAE算法)
        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # 步骤6: 保存DataLoader状态
        # 保存当前的数据迭代器状态,确保恢复训练时从正确的数据批次开始
        # 这对于可重现性和断点续训非常重要
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # 步骤7: 更新最新checkpoint标记文件
        # 用于自动恢复训练时快速定位最新的checkpoint
        # 使用独立文件确保原子性操作,避免并发写入问题
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        """
        加载训练checkpoint恢复训练状态。

        该方法负责从磁盘加载保存的模型参数和训练状态,实现断点续训。

        支持三种恢复模式:
        1. "disable": 禁用恢复,从头开始训练
        2. "auto": 自动查找最新的checkpoint并恢复（如果没有则从头开始）
        3. "resume_path": 从指定路径恢复

        加载内容包括:
        - Actor模型参数和优化器状态
        - Critic模型参数和优化器状态(如果使用GAE)
        - DataLoader状态(确保从正确的批次继续)
        - 全局训练步数

        Returns:
            int: 恢复的global_steps数值，如果从头开始则返回0
        """
        # 步骤1: 检查是否禁用恢复
        if self.config.trainer.resume_mode == "disable":
            return 0

        # 步骤2: 定位checkpoint文件夹
        # 从HDFS加载（当前未实现）
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            # 从本地文件系统加载
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            # 将相对路径转换为绝对路径
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            # 在checkpoint目录下查找最新的global_step文件夹
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # 步骤3: 根据resume_mode确定要加载的checkpoint
        if self.config.trainer.resume_mode == "auto":
            # 自动模式：如果找不到checkpoint，从头开始训练
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                # 手动指定路径模式：从指定的checkpoint恢复
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                # 将相对路径转换为绝对路径
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")

        # 步骤4: 从checkpoint路径提取全局步数
        # 路径格式：.../global_step_{N}/...
        # 从路径中解析出N作为当前的global_steps
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        # 步骤5: 构建Actor和Critic的checkpoint路径
        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")

        # 步骤6: 加载Actor模型
        # 包括模型参数、优化器状态、学习率调度器状态等
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )

        # 步骤7: 加载Critic模型（如果使用GAE算法）
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # 步骤8: 加载DataLoader状态
        # 这确保训练从正确的数据批次继续，避免数据重复或遗漏
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            # 加载DataLoader的随机数生成器状态和采样器状态
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            # 如果DataLoader状态文件不存在，发出警告
            # 模型权重会加载，但数据迭代会从头开始
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """
        在单个控制器上重新排序数据,使每个数据并行（DP）rank获得相似数量的token。

        批次平衡的目的:
        1. 均衡分配token数量到各个GPU上,减少负载不均
        2. 提高训练效率,避免某些GPU等待其他GPU完成计算
        3. 在分布式训练中实现更好的资源利用

        重要提示:
        - 此方法会改变batch内数据的顺序
        - 使用基于组的优势估计（如GRPO、RLOO）时需要谨慎
        - 数据重排序后,相同prompt的样本可能被分散到不同GPU

        Args:
            batch: DataProto对象,包含待平衡的批次数据
            metrics: 指标字典,用于记录平衡统计信息
            logging_prefix: 日志前缀,用于区分不同的平衡操作
        """
        # 步骤1: 提取attention mask并计算每个序列的有效token数量
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        # global_seqlen_lst: 每个序列的有效token数量列表
        # 通过sum attention_mask得到,pad token的mask为0不计入
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)

        # 步骤2: 获取数据并行的world size（总GPU数量）
        world_size = self.actor_rollout_wg.world_size

        # 步骤3: 计算平衡的分区策略
        # 使用贪心算法将序列分配到K个分区,使每个分区的总token数尽可能接近
        # equal_size=True: 确保每个分区的序列数量相等（某些序列可能被复制）
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)

        # 步骤4: 根据分区索引重新排序batch
        # 将分区列表展平,得到新的排序索引
        # 数据会被自动平均分配到各个GPU（通过dispatch函数）
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)

        # 步骤5: 计算并记录平衡统计信息
        # 包括：各分区的token数量、不平衡比率、最大/最小分区大小等
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        # 将平衡统计信息添加到metrics中
        metrics.update(global_balance_stats)

    def fit(self):
        """
        PPO训练主循环。

        驱动进程通过RPC调用worker group的计算函数来构建PPO数据流。
        轻量级的优势函数计算在驱动进程上完成。

        训练流程：
        1. 初始化和checkpoint加载
        2. 训练前验证（可选）
        3. 准备生成配置和环境
        4. 主训练循环：
           - 更新参考策略（按指定频率）
           - LLM与环境交互生成轨迹
           - 计算奖励和优势函数
           - 更新critic网络（如果使用GAE）
           - 更新actor策略网络
           - 定期验证和保存checkpoint
        5. 记录训练指标
        """

        logger = self.logger
        self.global_steps = 0

        # ============================================================
        # 第一步：加载checkpoint（如果存在）
        # ============================================================
        self._load_checkpoint()

        # ============================================================
        # 第二步：训练前验证（可选）
        # ============================================================
        # 目前仅支持使用reward_function进行验证
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            if self.config.trainer.get('val_only', False):
                return

        # 添加进度条
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # 从步骤1开始
        self.global_steps += 1

        # ============================================================
        # 第三步：准备生成配置和环境
        # ============================================================
        # Agent配置准备：设置LLM生成的各项参数
        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,  # 最大对话轮数（动态设置）
            max_start_length=self.config.data.max_start_length,  # 起始输入最大长度
            max_prompt_length=self.config.data.max_prompt_length,  # 提示文本最大长度
            max_response_length=self.config.data.max_response_length,  # 响应文本最大长度
            max_obs_length=self.config.data.max_obs_length,  # 观察文本最大长度
            logging=self.config.logging,  # 日志配置
            num_gpus=self.config.trainer.n_gpus_per_node,  # 每个节点的GPU数量
            no_think_rl=self.config.algorithm.no_think_rl,  # 是否禁用思考过程的RL
            state_masking=self.config.actor_rollout_ref.actor.state_masking,  # 是否启用状态掩码
            start_state_marker=self.config.algorithm.state_masking.start_state_marker,  # 状态开始标记
            end_state_marker=self.config.algorithm.state_masking.end_state_marker,  # 状态结束标记
        )

        # 创建LLM生成管理器：负责管理LLM与环境的多轮交互
        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,  # 分词器
            actor_rollout_wg=self.actor_rollout_wg,  # Actor-Rollout worker group
            env_class=self.env_class,  # 环境类
            config=gen_config,  # 生成配置
            logger=logger,  # 日志记录器
        )

        # 创建环境实例列表：每个prompt * n_agent都有一个独立的环境副本
        # train_batch_size: 训练批次中的prompt数量
        # n_agent: 每个prompt生成的agent响应数量（用于GRPO等算法）
        envs = [self.env.copy() for _ in
                range(self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n_agent)]

        # ============================================================
        # 第四步：主训练循环
        # ============================================================
        # 外层循环：遍历所有训练轮次（epochs）
        for epoch in range(self.config.trainer.total_epochs):
            # 内层循环：遍历当前epoch中的所有训练批次
            for batch_dict in self.train_dataloader:
                print(f'epoch {epoch}, step {self.global_steps}')

                # ------------------------------------------------------------
                # 步骤4.1：定期更新参考策略（Reference Policy）
                # ------------------------------------------------------------
                # 参考策略用于计算KL散度，防止策略更新偏离初始策略太远
                # 通常每隔一定步数，将当前actor的参数同步到reference policy
                if self.config.trainer.ref_update_steps is not None and self.global_steps % self.config.trainer.ref_update_steps == 0:
                    # 保存当前actor-rollout worker的checkpoint到临时目录
                    self.actor_rollout_wg.save_checkpoint(
                        local_path=f'./log/temp/actor_rollout_wg_global_step_{self.global_steps}',
                        hdfs_path=None
                    )
                    # 将保存的参数加载到参考策略worker中
                    self.ref_policy_wg.load_model_parameters(
                        source_model_path=f'./log/temp/actor_rollout_wg_global_step_{self.global_steps}',
                        strict=True
                    )
                    print(
                        f"load parameters from ./log/temp/actor_rollout_wg_global_step_{self.global_steps} to ref_policy_wg")

                # 初始化当前step的指标和计时字典
                metrics = {}
                timing_raw = {}

                # ------------------------------------------------------------
                # 步骤4.2：准备训练批次数据
                # ------------------------------------------------------------
                # 将字典格式的batch转换为DataProto对象（统一的数据格式）
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                # 重复batch：每个prompt生成n_agent个响应（用于GRPO等算法的多样本估计）
                # interleave=True表示交错排列，保持相同prompt的响应在一起
                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)

                # ------------------------------------------------------------
                # 步骤4.3：初始化环境
                # ------------------------------------------------------------
                # 提取环境的种子（通常使用数据的索引作为种子，保证可重复性）
                env_seeds = [i['index'] for i in batch.non_tensor_batch['extra_info']]
                print("env_seeds:", env_seeds)
                # 为每个环境实例设置对应的种子并重置状态
                # 每个(prompt, agent_id)对应一个独立的环境副本
                for env, seed in zip(envs, env_seeds):
                    env.reset(seed=seed)

                # ------------------------------------------------------------
                # 步骤4.4：提取生成所需的批次数据
                # ------------------------------------------------------------
                # 从batch中弹出生成所需的键（input_ids, attention_mask, position_ids）
                # input_ids, attention_mask, position_ids LLM 推理的标准输入格式
                # input_ids：token ID 序列 → 模型输入
                # attention_mask：标记哪些位置是真实token，哪些是padding → 防止模型关注填充部分
                # position_ids：每个token的位置索引 → 用于位置编码
                # 这些数据将用于LLM生成响应
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                ####################
                # 原始代码（已废弃）：直接生成序列的方式
                # 现在改用多轮交互的LLM循环生成方式（见下文）
                ####################

                # with _timer('gen', timing_raw):
                #     gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                #     batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                #                                              dtype=object)
                #     # repeat to align with repeated responses in rollout
                #     batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                #     batch = batch.union(gen_batch_output)

                #     # output batch to file
                #     self._record_batch(batch, path=f'.log/{self.config.trainer.experiment_name}/gen_batch.txt')

                ####################
                # 新的多轮交互生成方式：LLM + 环境循环
                ####################

                with _timer('step', timing_raw):
                    """
                    多轮交互生成流程：
                    1. 持续生成K轮响应（K可以是固定值或动态值）
                    2. 在生成过程中，每生成一轮新响应，就更新"右侧部分"（新生成的内容）
                    3. 最终，将"左侧部分"（原始prompt）和"右侧部分"（所有生成的响应）拼接
                    4. 得到完整的训练数据，送入模型进行训练

                    技术细节：
                    - Left-pad prompts（左填充提示）: 保持批次对齐
                    - Right-gen flow（右生成流程）: 响应从右侧生成
                    - Tensors dance like stardust glow: 张量在GPU间流动
                    - Errors swarm? Stay calm, don't fret: 遇到错误不要慌
                    - Code with coffee, debug sunset: 用咖啡编码，用日落调试
                    """

                    # ------------------------------------------------------------
                    # 步骤4.4.1：准备初始输入
                    # ------------------------------------------------------------
                    # 提取第一轮输入：取最后max_start_length个token作为初始输入
                    # 这是多轮对话的起点
                    first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()

                    # 设置输出目录：用于保存生成的对话日志和可视化
                    output_dir = (f"{self.config.logging.log_image_dir}/"
                                  f"{self.config.trainer.experiment_name}/"
                                  f"train/"
                                  f"step_{self.global_steps}")

                    # ------------------------------------------------------------
                    # 步骤4.4.2：运行LLM多轮交互循环
                    # ------------------------------------------------------------
                    with _timer('gen', timing_raw):
                        # 设置计时器：记录生成时间
                        generation_manager.timing_raw = timing_raw
                        # 运行LLM循环：生成完整的多轮对话轨迹
                        # 输入：gen_batch（提示数据）、envs（环境列表）、first_input_ids（初始输入）
                        # 输出：final_gen_batch_output（完整的生成结果，包括responses、log_probs等）
                        final_gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=gen_batch,
                            envs=envs,
                            initial_input_ids=first_input_ids,
                            output_dir=output_dir,
                            global_steps=self.global_steps,
                        )

                    # with torch.no_grad():
                    #     output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                    #     final_gen_batch_output = final_gen_batch_output.union(output)

                    # ------------------------------------------------------------
                    # 步骤4.5：设置唯一标识符（UID）用于分组
                    # ------------------------------------------------------------
                    # UID用于不同的优势估计算法中进行样本分组和奖励归一化
                    # 不同算法的UID设置策略：
                    # - GRPO：使用环境种子作为UID，相同prompt的多个响应共享UID（用于组内归一化）
                    # - BRPO：使用空字符串，所有样本共享一个UID（批次级归一化）
                    # - ARPO：使用随机UUID，每个样本独立UID（无相对归一化）
                    if self.config.algorithm.adv_estimator == 'grpo' or self.config.algorithm.reward_norm_type == 'grpo':
                        # GRPO：使用seed分组（注意：最好使用prompt的hash，但目前用seed）
                        batch.non_tensor_batch['uid'] = np.array([str(i) for i in env_seeds], dtype=object)
                    elif self.config.algorithm.adv_estimator == 'brpo' or self.config.algorithm.reward_norm_type == 'brpo':
                        # BRPO：批次级归一化，所有样本共享相同UID
                        batch.non_tensor_batch['uid'] = np.array(["" for _ in range(len(batch.batch))], dtype=object)
                    elif self.config.algorithm.adv_estimator == 'arpo' or self.config.algorithm.reward_norm_type == 'arpo':
                        # ARPO：无相对归一化，每个样本独立的UUID
                        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                                 dtype=object)

                    # ------------------------------------------------------------
                    # 步骤4.6：收集环境奖励
                    # ------------------------------------------------------------
                    # 从每个环境实例中收集最终奖励
                    # 每个环境在多轮交互后计算出一个总奖励值
                    batch.non_tensor_batch['reward'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    for idx, env in enumerate(envs):
                        batch.non_tensor_batch['reward'][idx] = env.reward
                    # 保存原始奖励的副本（用于记录和对比）
                    batch.non_tensor_batch['ori_reward'] = copy.deepcopy(batch.non_tensor_batch['reward'])

                    # ------------------------------------------------------------
                    # 步骤4.7：收集特定任务的额外奖励分数（医疗对话场景）
                    # ------------------------------------------------------------
                    # 判断当前环境是否为医疗对话环境（包含诊断分数）
                    if hasattr(envs[0], 'diagnosis_score'):
                        # 初始化诊断分数和推荐分数数组
                        batch.non_tensor_batch['diagnosis_score'] = np.array([0 for _ in range(len(envs))],
                                                                             dtype=object)
                        batch.non_tensor_batch['recommandation_score'] = np.array([0 for _ in range(len(envs))],
                                                                                  dtype=object)
                        # 从每个环境收集诊断和推荐的子分数
                        for idx, env in enumerate(envs):
                            batch.non_tensor_batch['diagnosis_score'][idx] = env.diagnosis_score
                            batch.non_tensor_batch['recommandation_score'][idx] = env.recommandation_score

                    # ------------------------------------------------------------
                    # 步骤4.8：奖励归一化
                    # ------------------------------------------------------------
                    # 对奖励进行归一化处理，减少方差，提高训练稳定性
                    # 归一化方式取决于reward_norm_type（grpo/brpo/arpo）
                    if self.config.algorithm.reward_norm_type is not None:
                        batch.non_tensor_batch['reward'] = normalize_reward(batch.non_tensor_batch['reward'],
                                                                            batch.non_tensor_batch['uid'],
                                                                            self.config.algorithm.reward_norm_type)

                    # ------------------------------------------------------------
                    # 步骤4.9：收集特定任务的指标（双臂老虎机场景）
                    # ------------------------------------------------------------
                    # 如果是双臂老虎机任务，记录每个环境选择的动作
                    # 注意：假设无效动作=0，低臂=1，高臂=2
                    if batch.non_tensor_batch['data_source'][0] == 'two_armed_bandit':
                        batch.non_tensor_batch['bandit_metrics'] = np.array([0 for _ in range(len(envs))], dtype=object)
                        for idx, env in enumerate(envs):
                            batch.non_tensor_batch['bandit_metrics'][idx] = env.get_last_action()

                    # ------------------------------------------------------------
                    # 步骤4.10：收集环境统计指标
                    # ------------------------------------------------------------
                    # 初始化环境的各项统计指标数组
                    batch.non_tensor_batch['total_env'] = np.array([1 for _ in range(len(envs))], dtype=object)
                    batch.non_tensor_batch['finished_env'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    batch.non_tensor_batch['success_env'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    batch.non_tensor_batch['traj_length'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    batch.non_tensor_batch['valid_action'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    batch.non_tensor_batch['effective_action'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    batch.non_tensor_batch['effective_action_ratio'] = np.array([0 for _ in range(len(envs))],
                                                                                dtype=object)
                    # 从每个环境收集指标
                    for idx, env in enumerate(envs):
                        # finished_env: 环境是否结束（达到终止条件）
                        batch.non_tensor_batch['finished_env'][idx] = int(env.finished())
                        # success_env: 环境是否成功完成任务
                        batch.non_tensor_batch['success_env'][idx] = int(env.success())
                        # 获取环境的追踪变量（记录每一步的动作和状态）
                        tracking_vars = env.get_tracking_variables()
                        # traj_length: 轨迹长度（执行的总步数）
                        batch.non_tensor_batch['traj_length'][idx] = len(tracking_vars['actions'])
                        # valid_action: 有效动作数量（格式正确且可解析的动作）
                        batch.non_tensor_batch['valid_action'][idx] = sum(
                            1 for x in tracking_vars['actions_valid'] if x is not None)
                        # effective_action: 有效动作数量（对环境状态有实际影响的动作）
                        batch.non_tensor_batch['effective_action'][idx] = sum(
                            1 for x in tracking_vars['actions_effective'] if x is not None)
                        # effective_action_ratio: 有效动作比率
                        batch.non_tensor_batch['effective_action_ratio'][idx] = sum(
                            1 for x in tracking_vars['actions_effective'] if x is not None) / len(
                            tracking_vars['actions'])

                    # ------------------------------------------------------------
                    # 步骤4.11：合并生成结果到batch
                    # ------------------------------------------------------------
                    # 重复batch以匹配PPO的n次采样（用于多次采样的PPO变体）
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # 将生成的结果合并到batch中（包括responses、old_log_probs等）
                    batch = batch.union(final_gen_batch_output)

                    # ------------------------------------------------------------
                    # 步骤4.12：创建响应掩码
                    # ------------------------------------------------------------
                    # 响应掩码用于标识哪些token需要参与loss计算
                    if self.config.actor_rollout_ref.actor.state_masking:
                        # 如果启用状态掩码，使用自定义的loss mask创建方法
                        # 这会屏蔽掉特定标记（如<think>标签）内的token
                        batch, metrics = self._create_loss_mask(batch, metrics)
                    else:
                        # 否则，使用标准的响应掩码（仅包含生成的response部分）
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    # ------------------------------------------------------------
                    # 步骤4.13：批次平衡
                    # ------------------------------------------------------------
                    # 平衡每个数据并行（DP）rank上的有效token数量
                    # 注意：这会打乱batch内数据的顺序
                    # 使用基于组的优势计算（如GRPO、RLOO）时需要特别小心
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # ------------------------------------------------------------
                    # 步骤4.14：记录全局token数量
                    # ------------------------------------------------------------
                    # 计算每个序列的总token数（用于后续的统计和调试）
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # ------------------------------------------------------------
                    # 步骤4.15：重新计算旧策略的log概率（old_log_probs）
                    # ------------------------------------------------------------
                    # 在环境交互后，需要重新计算当前策略下的log_probs
                    # 这些log_probs将作为"旧策略"用于PPO的clip目标计算
                    with _timer("old_log_prob", timing_raw):
                        # 调用actor_rollout worker计算log_probs和熵
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        # 提取熵值（用于熵正则化，鼓励策略探索）
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        # 聚合熵损失（根据loss_agg_mode：token-mean、seq-mean等）
                        entropy_loss = agg_loss(
                            loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                        )
                        # 记录熵损失指标
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        # 移除熵值，只保留log_probs
                        old_log_prob.batch.pop("entropys")
                        # 将old_log_probs合并到batch中
                        batch = batch.union(old_log_prob)

                    # ------------------------------------------------------------
                    # 步骤4.16：计算参考策略的log概率（ref_log_prob）
                    # ------------------------------------------------------------
                    # 参考策略是一个固定的模型，用于计算KL散度惩罚
                    # 防止策略更新偏离初始策略太远（保持训练稳定性）
                    if self.use_reference_policy:
                        with _timer('ref', timing_raw):
                            # 调用参考策略worker计算ref_log_prob
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            # 将ref_log_prob合并到batch中
                            batch = batch.union(ref_log_prob)

                    # ------------------------------------------------------------
                    # 步骤4.17：计算价值函数（values）
                    # ------------------------------------------------------------
                    # 价值函数V(s)用于GAE（Generalized Advantage Estimation）
                    # 仅在使用GAE优势估计器时需要critic网络
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            # 调用critic worker计算values
                            values = self.critic_wg.compute_values(batch)
                            # 将values合并到batch中
                            batch = batch.union(values)

                    # ------------------------------------------------------------
                    # 步骤4.18：计算token级别的奖励和优势函数
                    # ------------------------------------------------------------
                    with _timer('adv', timing_raw):
                        # 步骤4.18.1：计算token级别的分数（支持模型和函数两种方式）
                        # 首先使用奖励模型计算分数，然后调用reward_fn结合基于规则的结果
                        if self.use_rm:
                            # 如果使用奖励模型，先计算模型分数
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # 步骤4.18.2：调用奖励函数合并基于规则的奖励
                        # reward_fn将环境奖励分配到每个token上
                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor

                        # 步骤4.18.3：应用KL惩罚（如果启用）
                        # KL惩罚 = token_level_scores - beta * KL(π||π_ref)
                        # 防止策略偏离参考策略太远
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            # 如果不使用KL惩罚，token_level_rewards = token_level_scores
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # 步骤4.18.4：计算优势函数（advantages）
                        # 在驱动进程上执行（轻量级计算）
                        # 根据adv_estimator类型选择算法：GAE、GRPO、RLOO等
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # ------------------------------------------------------------
                    # 步骤4.19：更新Critic网络（如果使用GAE）
                    # ------------------------------------------------------------
                    # Critic学习价值函数V(s)，用于减少优势估计的方差
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            # 调用critic worker进行梯度更新
                            critic_output = self.critic_wg.update_critic(batch)
                        # 聚合并记录critic的训练指标
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # ------------------------------------------------------------
                    # 步骤4.20：更新Actor策略网络
                    # ------------------------------------------------------------
                    # Critic预热机制：只有在训练一定步数后才开始更新actor
                    # 这确保critic先学习到一个合理的价值函数
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer('update_actor', timing_raw):
                            # 调用actor_rollout worker进行PPO策略梯度更新
                            # 使用clip目标：L^CLIP = min(r_t(θ)·A, clip(r_t(θ), 1-ε, 1+ε)·A)
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        # 聚合并记录actor的训练指标
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # ------------------------------------------------------------
                    # 步骤4.21：定期验证
                    # ------------------------------------------------------------
                    # 按照test_freq频率在验证集上评估模型性能
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                            self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    # ------------------------------------------------------------
                    # 步骤4.22：定期保存checkpoint
                    # ------------------------------------------------------------
                    # 按照save_freq频率保存模型权重和训练状态
                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # ------------------------------------------------------------
                # 步骤4.23：收集和记录训练指标
                # ------------------------------------------------------------
                # 计算数据相关指标（奖励、优势、响应长度等）
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                # 计算计时指标（每个阶段的时间消耗）
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # 记录所有指标到日志系统（WandB/TensorBoard）
                logger.log(data=metrics, step=self.global_steps)

                # ------------------------------------------------------------
                # 步骤4.24：更新全局步数
                # ------------------------------------------------------------
                self.global_steps += 1

                # ------------------------------------------------------------
                # 步骤4.25：检查训练是否完成
                # ------------------------------------------------------------
                if self.global_steps >= self.total_training_steps:
                    # 训练结束后进行最终验证
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        metrics.update(val_metrics)
                    # 结束训练，退出主循环
                    return

    def _create_loss_mask(self, batch, metrics):
        """Create loss mask for state tokens."""
        response_length = batch.batch['responses'].shape[-1]
        if "response_mask" in batch.batch.keys():
            response_mask = batch.batch['response_mask']
        else:
            response_mask = batch.batch['attention_mask'][:, -response_length:]

        # Initialize state mask
        state_mask = torch.ones_like(response_mask)

        responses = [self.tokenizer.decode(resp, skip_special_tokens=False) for resp in batch.batch['responses']]

        for i, response in enumerate(responses):
            # Find all pairs of start and end marker positions
            start_marker = self.config.algorithm.state_masking.start_state_marker
            end_marker = self.config.algorithm.state_masking.end_state_marker

            # Get all start and end positions
            start_positions = [m.start() for m in re.finditer(re.escape(start_marker), response)]
            end_positions = [m.start() + len(end_marker) for m in re.finditer(re.escape(end_marker), response)]

            prev_end = 0
            prev_end_token_pos = 0
            # Convert character positions to token positions
            for start, end in zip(start_positions, end_positions):
                prefix_to_start = response[prev_end:start]
                state_section = response[start:end]
                prev_end = end

                start_tokens = self.tokenizer.encode(prefix_to_start, add_special_tokens=False)
                state_tokens = self.tokenizer.encode(state_section, add_special_tokens=False)

                start_token_pos = len(start_tokens) + prev_end_token_pos
                end_token_pos = start_token_pos + len(state_tokens)
                prev_end_token_pos = end_token_pos

                state_mask[i, start_token_pos:end_token_pos] = 0

        loss_mask = state_mask * response_mask  # 1 for valid tokens, 0 for masked tokens
        batch.batch['loss_mask'] = loss_mask
        batch.batch['response_mask'] = loss_mask
        batch.batch['critic_response_mask'] = state_mask * batch.batch['attention_mask'][:, -response_length - 1:-1]

        # Debug print
        print("\nRaw batch[0] (before masking):\n", self.tokenizer.decode(batch.batch['responses'][0]))
        response_ids = batch.batch['responses'][0]
        unmasked_ids = response_ids[loss_mask[0] == 1]
        print("\nUnmasked batch[0] (after masking):\n", self.tokenizer.decode(unmasked_ids))

        masked_ids = response_ids[loss_mask[0] == 0]
        print("\nMasked batch[0] (masked parts):\n", self.tokenizer.decode(masked_ids))

        metrics.update({
            'state_tokens/total': loss_mask.sum().item(),
            'state_tokens/coverage': (loss_mask.sum() / response_mask.sum()).item(),
        })

        return batch, metrics

    def _validate(self):
        """
        The training loop of PPO with global metric computation.
        Accumulates metrics across all batches before computing final statistics.
        """
        import torch
        # Initialize global metric storage
        global_token_scores = []
        global_metrics = {}
        metrics = defaultdict(list)

        self.val_num += 1

        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            logging=self.config.logging,
            num_gpus=self.config.trainer.n_gpus_per_node,
            no_think_rl=self.config.algorithm.no_think_rl,
            state_masking=self.config.actor_rollout_ref.actor.state_masking,
            start_state_marker=self.config.algorithm.state_masking.start_state_marker,
            end_state_marker=self.config.algorithm.state_masking.end_state_marker,
        )

        # Agent config preparation
        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            env_class=self.env_class,
            config=gen_config,
            logger=self.logger,
            is_validation=True,
        )

        envs = [self.val_env.copy() for _ in range(self.config.data.val_batch_size)]  # do not repeat
        # envs = [self.val_env.copy() for _ in range(self.config.data.val_batch_size * self.config.actor_rollout_ref.rollout.n_agent)]
        val_global_steps = 1

        for batch_dict in self.val_dataloader:
            timing_raw = {}
            test_batch: DataProto = DataProto.from_single_dict(batch_dict)
            # test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)

            env_seeds = [i['index'] for i in test_batch.non_tensor_batch['extra_info']]
            print("env_seeds:", env_seeds)
            for env, seed in zip(envs, env_seeds):
                env.reset(seed=seed)

            test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }
            with _timer('step', timing_raw):
                first_input_ids = test_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
                output_dir = (f"{self.config.logging.log_image_dir}/"
                              f"{self.config.trainer.experiment_name}/"
                              f"validation_{self.val_num}/"
                              f"step_{val_global_steps}")
                with _timer('gen', timing_raw):
                    generation_manager.timing_raw = timing_raw
                    final_gen_batch_output = generation_manager.run_llm_loop(
                        gen_batch=test_gen_batch,
                        envs=envs,
                        initial_input_ids=first_input_ids,
                        output_dir=output_dir,
                        global_steps=val_global_steps,
                    )
                with torch.no_grad():
                    output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                    final_gen_batch_output = final_gen_batch_output.union(output)

                test_batch.non_tensor_batch['reward'] = np.array([0 for _ in range(len(envs))], dtype=object)
                for idx, env in enumerate(envs):
                    test_batch.non_tensor_batch['reward'][idx] = env.reward

                if test_batch.non_tensor_batch['data_source'][0] == 'two_armed_bandit':
                    # metric for two-armed bandit
                    # NOTE here we assume invalid action is 0, low arm is 1, high arm is 2
                    test_batch.non_tensor_batch['bandit_metrics'] = np.array([0 for _ in range(len(envs))],
                                                                             dtype=object)
                    for idx, env in enumerate(envs):
                        test_batch.non_tensor_batch['bandit_metrics'][idx] = env.get_last_action()
                    metrics['bandit_metrics'].append(test_batch.non_tensor_batch['bandit_metrics'])

                test_batch.non_tensor_batch['total_env'] = np.array([1 for _ in range(len(envs))], dtype=object)
                test_batch.non_tensor_batch['finished_env'] = np.array([0 for _ in range(len(envs))], dtype=object)
                test_batch.non_tensor_batch['success_env'] = np.array([0 for _ in range(len(envs))], dtype=object)
                test_batch.non_tensor_batch['traj_length'] = np.array([0 for _ in range(len(envs))], dtype=object)
                test_batch.non_tensor_batch['valid_action'] = np.array([0 for _ in range(len(envs))], dtype=object)
                test_batch.non_tensor_batch['effective_action'] = np.array([0 for _ in range(len(envs))], dtype=object)
                test_batch.non_tensor_batch['effective_action_ratio'] = np.array([0 for _ in range(len(envs))],
                                                                                 dtype=object)
                for idx, env in enumerate(envs):
                    test_batch.non_tensor_batch['finished_env'][idx] = int(env.finished())
                    test_batch.non_tensor_batch['success_env'][idx] = int(env.success())
                    tracking_vars = env.get_tracking_variables()
                    test_batch.non_tensor_batch['traj_length'][idx] = len(tracking_vars['actions'])
                    test_batch.non_tensor_batch['valid_action'][idx] = sum(
                        1 for x in tracking_vars['actions_valid'] if x is not None)
                    test_batch.non_tensor_batch['effective_action'][idx] = sum(
                        1 for x in tracking_vars['actions_effective'] if x is not None)
                    test_batch.non_tensor_batch['effective_action_ratio'][idx] = sum(
                        1 for x in tracking_vars['actions_effective'] if x is not None) / len(tracking_vars['actions'])

                # action metrics
                metrics['total_env'].append(test_batch.non_tensor_batch['total_env'])
                metrics['finished_env'].append(test_batch.non_tensor_batch['finished_env'])
                metrics['success_env'].append(test_batch.non_tensor_batch['success_env'])
                metrics['traj_length'].append(test_batch.non_tensor_batch['traj_length'])
                metrics['valid_action'].append(test_batch.non_tensor_batch['valid_action'])
                metrics['effective_action'].append(test_batch.non_tensor_batch['effective_action'])
                metrics['effective_action_ratio'].append(test_batch.non_tensor_batch['effective_action_ratio'])

                # Accumulate batch metrics into global storage
                global_token_scores.append(test_batch.non_tensor_batch['reward'])

        global_scores = np.concatenate(global_token_scores, axis=0)
        global_metrics = {
            'global_score/mean': float(global_scores.mean()),
            'global_score/max': float(global_scores.max()),
            'global_score/min': float(global_scores.min()),
            'global_score/std': float(global_scores.std()),
            'validate_metric/total_env': int(np.array(metrics['total_env'], dtype=np.int16).sum()),
            'validate_metric/finished_env': int(np.array(metrics['finished_env'], dtype=np.int16).sum()),
            'validate_metric/success_env': int(np.array(metrics['success_env'], dtype=np.int16).sum()),
            'validate_metric/traj_length': float(np.array(metrics['traj_length'], dtype=np.int16).mean()),
            'validate_metric/valid_action': float(np.array(metrics['valid_action'], dtype=np.int16).mean()),
            'validate_metric/effective_action': float(np.array(metrics['effective_action'], dtype=np.int16).mean()),
            'validate_metric/effective_action_ratio': float(
                np.array(metrics['effective_action_ratio'], dtype=np.float32).mean()),
        }
        if 'bandit_metrics' in metrics:  # NOTE hard code for two-armed bandit
            batch_action = np.array(metrics['bandit_metrics'], dtype=np.int16)
            global_metrics['validate_metric/n_low_arm'] = int(np.sum(batch_action == 1))
            global_metrics['validate_metric/n_high_arm'] = int(np.sum(batch_action == 2))
            global_metrics['validate_metric/n_invalid'] = int(np.sum(batch_action == 0))
        print("global_metrics", global_metrics)
        self.logger.log(data=global_metrics, step=self.val_num)
        return global_metrics
