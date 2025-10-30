import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from ragen.utils import set_seed
from ragen.utils.plot import (
    save_trajectory_to_output,
    parse_llm_output
)
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil


# 生成配置数据类：用于控制生成过程的各种参数
@dataclass
class GenerationConfig:
    max_turns: int  # 最大交互轮数（即 LLM 与环境交互的最大步数）
    max_start_length: int  # 初始提示（prompt）的最大长度
    max_prompt_length: int  # 整个上下文（prompt + 历史）的最大长度
    max_response_length: int  # LLM 单次响应的最大长度
    max_obs_length: int  # 环境观测（observation）的最大长度
    logging: dict  # 日志配置（如是否记录轨迹图像、每批记录几张等）
    num_gpus: int  # 使用的 GPU 数量（用于多 GPU 批处理对齐）
    no_think_rl: bool = False  # 是否禁用“思考”过程，仅输出动作（用于简化 RL 训练）
    state_masking: bool = False  # 是否启用状态遮蔽（防止模型“作弊”看到内部状态）
    start_state_marker: str = "<start-state>"  # 状态开始标记
    end_state_marker: str = "<end-state>"  # 状态结束标记
    use_env_llm: bool = False  # 是否使用环境专用的 LLM（如用于模拟环境）
    batch_size: int = 1  # 批处理大小


class LLMGenerationManager:
    def __init__(
            self,
            tokenizer,
            actor_rollout_wg,  # 用于生成响应的 LLM 工作者组（worker group）
            env_class,  # 环境类（提供 execute_predictions、postprocess_predictions 等方法）
            config: GenerationConfig,  # 生成配置
            logger: Tracking,  # 日志记录器（如 WandB）
            env_llm_wg=None,  # 可选：环境专用的 LLM（用于复杂环境模拟）
            is_validation: bool = False,  # 是否为验证模式（影响日志保存行为）
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.env_class = env_class
        self.config = config
        self.logger = logger
        self.is_validation = is_validation
        self.env_llm_wg = env_llm_wg

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """对一批响应字符串进行分词，返回 token ID 张量"""
        return self.tokenizer(
            responses,
            add_special_tokens=False,  # 不添加特殊 token（如 bos/eos）
            return_tensors='pt',  # 返回 PyTorch 张量
            padding="longest"  # 按最长序列填充
        )['input_ids']

    @staticmethod
    def _process_answer_tag(responses_str):
        """
        处理响应字符串，只保留第一个 <answer>...</answer> 标签对，
        后续的 <answer> 标签仅保留内容（去除标签），防止模型多次输出答案。
        """

        def process_single_response(resp):
            # 如果没有 answer 标签，直接返回原字符串
            if '<answer>' not in resp or '</answer>' not in resp:
                return resp

            # 查找第一个完整的 <answer>...</answer> 匹配
            pattern = r'<answer>.*?</answer>'
            match = re.search(pattern, resp, re.DOTALL)  # DOTALL 使 . 匹配换行符

            if not match:
                return resp

            answer_content = match.group(0)  # 保留第一个完整标签对

            # 处理后续内容：移除后续 <answer> 标签，只保留内部文本
            rest_of_string = resp[match.end():]
            cleaned_rest = re.sub(r'<answer>(.*?)</answer>', r'\1', rest_of_string, flags=re.DOTALL)

            return resp[:match.start()] + answer_content + cleaned_rest

        return [process_single_response(resp) for resp in responses_str]

    def _postprocess_responses(self, responses: torch.Tensor, envs: List[Any]) -> Tuple[torch.Tensor, List[str]]:
        """
        对 LLM 生成的响应进行后处理：
        1. 解码为字符串
        2. 清理多余的 <answer> 标签
        3. （可选）移除状态标记（防止 reward hacking）
        4. （可选）仅保留动作（no_think_rl 模式）
        """
        responses_str = self.tokenizer.batch_decode(
            responses,
            skip_special_tokens=True  # 跳过特殊 token（如 <pad>）
        )

        # 清理 answer 标签
        responses_str = self._process_answer_tag(responses_str)

        # 如果启用了状态遮蔽，移除 <start-state>...<end-state> 之间的内容（防止模型“作弊”）
        if self.config.state_masking:
            start_marker = re.escape(self.config.start_state_marker)
            end_marker = re.escape(self.config.end_state_marker)
            hack_pattern = f'{start_marker}[\\s\\S]*?{end_marker}'

            # 检测是否有“作弊”响应并警告
            hacked = [resp for resp in responses_str if re.search(hack_pattern, resp, re.DOTALL)]
            if hacked:
                print(f"[WARNING] HACKED RESPONSES: {hacked}")
            # 移除状态标记及其内容
            responses_str = [re.sub(hack_pattern, '', resp, re.DOTALL) for resp in responses_str]

        # 如果启用 no_think_rl 模式，强制只输出动作（如 <answer>UP</answer>）
        if self.config.no_think_rl:
            actions, _ = self.env_class.postprocess_predictions(envs, responses_str)
            responses_str = [f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in
                             enumerate(actions)]
            print("RESPONSES:", responses_str)

        # 重新分词为 token ID
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """
        处理环境返回的下一观测（observation）：
        - 如果启用了 state_masking，则将原始状态包裹在 <start-state> 和 <end-state> 中
        - 同时将原始标记替换为 <inner_start-state> 等，防止模型混淆
        - 最后进行分词和长度截断
        """
        if self.config.state_masking:
            start_marker = self.config.start_state_marker
            end_marker = self.config.end_state_marker

            # Create inner versions by adding 'inner_' prefix
            inner_start = f"<inner_{start_marker[1:]}"
            inner_end = f"<inner_{end_marker[1:]}"

            # Replace any existing markers with inner versions
            next_obs = [re.sub(re.escape(start_marker), inner_start, obs) for obs in next_obs]
            next_obs = [re.sub(re.escape(end_marker), inner_end, obs) for obs in next_obs]

            # Wrap with state markers
            def wrap_obs(obs):
                marker = "<|im_end|>\n<|im_start|>assistant\n<think>"
                b_marker = "<|im_start|>user\n"
                if marker in obs and b_marker in obs:
                    start_index = obs.index(b_marker) + len(b_marker)
                    end_index = obs.index(marker)
                    return f"{obs[:start_index]}{start_marker}{obs[start_index:end_index]}{end_marker}{obs[end_index:]}"
                else:
                    return f"{start_marker}{obs}{end_marker}"

            next_obs = [wrap_obs(obs) for obs in next_obs]

        next_obs_ids = self.tokenizer(
            next_obs,
            padding='longest',
            return_tensors='pt'
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print("[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG")
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings, cur_responses: torch.Tensor,
                              next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])

        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        return DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })

    def _update_right_side(self, right_side: Dict,
                           cur_responses: torch.Tensor,
                           next_obs_ids: torch.Tensor) -> Dict:
        """Update right side state."""
        responses = self.tensor_fn.concatenate_with_padding([
            right_side['responses'],
            cur_responses,
            next_obs_ids
        ], pad_to_left=False)

        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        return {'responses': responses[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus

        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}

        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}

        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta

        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, envs: List[Any],
                     initial_input_ids: torch.Tensor,
                     output_dir: str,
                     global_steps: int) -> Tuple[Dict, Dict]:
        """
        # 这个循环内部:
        # for turn in range(max_turns):
        #     1. Actor生成问题/诊断
        #     2. 提取<answer>标签内容
        #     3. 调用env.step(action) -> 计算reward
        #     4. 如果未结束, EnvLLM生成患者回答
        #     5. 拼接对话历史,继续下一轮
        """
        # Setup visualization and Initialize states
        trajectory = self._setup_visualization()

        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []]}

        # 布尔掩码，标记当前哪些环境实例仍在活跃（未终止）
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # breakpoint()
        # Main generation loop
        for step in range(self.config.max_turns):
            # 如果所有环境都已终止（active_mask 全为 False），提前结束循环
            if not active_mask.sum():
                break
            # 保留最新上下文，优化显存
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            rollings.batch['input_ids'] = rollings.batch['input_ids'].long()

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            # 筛选出活跃的样本
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            # 处理多 GPU 对齐问题，调用actor_rollout_wg.generate_sequences 获取DataProto 对象
            # responses: LLM 生成的 token ID 序列
            # meta_info: 如 logprobs、scores 等（用于 RL 训练）
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'], envs=envs)
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Update visualization
            self._update_trajectory(trajectory, envs, responses_str, active_mask)

            # Execute in environment and process observations
            next_obs, dones = self.env_class.execute_predictions(
                envs, responses_str, responses_ids, self.tokenizer
            )

            # 更新活跃掩码和统计
            active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_num_list.append(active_mask.sum().item())
            next_obs_ids = self._process_next_obs(next_obs)

            # Update states
            responses_ids = responses_ids.long()
            next_obs_ids = next_obs_ids.long()
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
        print("ACTIVE_TRAJ_NUM:", active_num_list)

        # Save trajectory and return final output
        self._save_trajectory(trajectory, output_dir, global_steps)
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _setup_visualization(self) -> List[Dict]:
        """Setup visualization tracking if enabled."""
        if not self.config.logging.log_images:
            return None
        return [defaultdict(list) for _ in range(self.config.logging.log_n_image_per_batch)]

    def _update_trajectory(self, trajectory: List[Dict],
                           envs: List[Any], responses: List[str], active_mask: torch.Tensor):
        """Update visualization trajectory if enabled."""
        if not trajectory:
            return
        n_visualize = self.config.logging.log_n_image_per_batch
        for idx, (env, active) in enumerate(zip(envs[:n_visualize], active_mask[:n_visualize])):
            if active:
                trajectory[idx]['state'].append(env.render('rgb_array'))

        for idx, (response, env, active) in enumerate(zip(responses[:n_visualize],
                                                          envs[:n_visualize],
                                                          active_mask[:n_visualize])):
            if active:
                parsed = parse_llm_output(response, strategy="raw")

                trajectory[idx]['answer'].append(response)
                trajectory[idx]['parsed_response'].append(parsed)

    def _save_trajectory(self, trajectory: List[Dict],
                         output_dir: str, global_steps: int):
        """Save trajectory visualization if enabled."""
        if not trajectory:
            return

        save_step_size = self.config.logging.log_image_step_size
        if not global_steps % save_step_size or self.is_validation:
            os.makedirs(output_dir, exist_ok=True)
            filenames = save_trajectory_to_output(trajectory, save_dir=output_dir)
            if 'wandb' in self.logger.logger:
                for filename in filenames:
                    self.logger.logger['wandb'].save(filename)

    def _compose_final_output(self, left_side: Dict,
                              right_side: Dict,
                              meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']

        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)

        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)

        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )

        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)

        return final_output
