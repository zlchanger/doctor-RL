import torch
from typing import Dict, Tuple, List
from dataclasses import dataclass

@dataclass
class TensorConfig:
    """
    张量处理配置类。

    Attributes:
        pad_token_id: 填充token的ID
        max_prompt_length: 提示文本的最大长度
        max_obs_length: 观察文本的最大长度
        max_start_length: 起始文本的最大长度
    """
    pad_token_id: int
    max_prompt_length: int
    max_obs_length: int
    max_start_length: int


class TensorHelper:
    """
    张量处理辅助类，用于LLM agent工作流中的常见张量操作。

    提供填充、掩码、截断和张量操作的工具函数，用于医疗对话强化学习环境中的批量序列处理。
    """

    def __init__(self, config: TensorConfig):
        """
        使用配置初始化TensorHelper。

        Args:
            config: TensorConfig对象，包含pad_token_id和长度约束参数
        """
        self.config = config

    def cut_to_effective_len(self, tensor_dict: Dict[str, torch.Tensor],
                            keys: List[str], cut_left: bool = True) -> Dict[str, torch.Tensor]:
        """
        根据注意力掩码将张量截断到有效长度。

        通过计算批次中的最大有效序列长度（基于attention_mask），移除填充token并截断所有指定的张量。

        Args:
            tensor_dict: 包含待处理张量的字典，必须包含'attention_mask'键
            keys: tensor_dict中需要截断的键列表
            cut_left: 如果为True，保留最右侧的token（从左侧截断）；如果为False，保留最左侧的token

        Returns:
            包含截断后张量的字典，指定键的张量被截断，其他键保持不变
        """
        effective_len = tensor_dict['attention_mask'].sum(dim=1).max()
        result = tensor_dict.copy()

        for key in keys:
            if cut_left:
                result[key] = tensor_dict[key][:, -effective_len:]
            else:
                result[key] = tensor_dict[key][:, :effective_len]
        return result

    def convert_pad_structure(self, tensor: torch.Tensor, pad_to_left: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        转换填充结构并返回排序后的张量和索引。

        将张量的填充位置从左侧移到右侧，或从右侧移到左侧。通过创建掩码并对索引排序来重新排列张量。

        Args:
            tensor: 输入张量，包含pad_token_id作为填充
            pad_to_left: 如果为True，将填充移到左侧；如果为False，将填充移到右侧

        Returns:
            元组包含：
                - 重新排列后的张量
                - 用于排序的索引张量
        """
        mask = tensor != self.config.pad_token_id if pad_to_left else tensor == self.config.pad_token_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        return tensor.gather(1, sorted_indices), sorted_indices

    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        从输入ID创建注意力掩码。

        将非填充token的位置标记为1，填充token的位置标记为0。

        Args:
            input_ids: 输入token ID张量

        Returns:
            注意力掩码张量，形状与input_ids相同，非填充位置为1，填充位置为0
        """
        return torch.where(input_ids != self.config.pad_token_id, 1, 0)

    def create_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        从注意力掩码创建位置ID。

        基于注意力掩码为每个有效token分配连续的位置编号，填充位置的位置ID为0。

        Args:
            attention_mask: 注意力掩码张量，非填充位置为1，填充位置为0

        Returns:
            位置ID张量，有效token从0开始递增编号，填充位置为0
        """
        return (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

    def concatenate_with_padding(self, tensors: List[torch.Tensor],
                               pad_to_left: bool = True) -> torch.Tensor:
        """
        拼接多个张量并处理填充。

        将多个张量沿序列维度拼接，然后调整填充结构使填充token位于指定的一侧。

        Args:
            tensors: 待拼接的张量列表
            pad_to_left: 如果为True，将填充移到左侧；如果为False，将填充移到右侧

        Returns:
            拼接后的张量，填充位于指定的一侧
        """
        concatenated = torch.cat(tensors, dim=1)
        padded_tensor, _ = self.convert_pad_structure(concatenated, pad_to_left)
        return padded_tensor

    def _example_level_pad(self, responses: torch.Tensor,
                          responses_str: List[str],
                          active_mask: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """
        为非活动样本的响应填充pad token。

        在批处理中，某些样本可能是非活动的（例如已完成的对话）。此方法将活动样本的响应
        映射回完整批次，为非活动样本填充pad token和空字符串。

        Args:
            responses: 活动样本的响应张量，形状为 (num_active, seq_len)
            responses_str: 活动样本的响应字符串列表，长度为 num_active
            active_mask: 布尔掩码，标记批次中哪些样本是活动的，形状为 (batch_size,)

        Returns:
            元组包含：
                - padded_responses: 完整批次的响应张量，非活动位置填充pad_token_id
                - padded_responses_str: 完整批次的响应字符串列表，非活动位置为空字符串
        """
        assert active_mask.sum() == responses.shape[0]
        # 创建填充后的响应张量
        batch_size = active_mask.shape[0]
        seq_len = responses.shape[1]
        padded_responses = torch.full(
            (batch_size, seq_len), self.config.pad_token_id,
            dtype=responses.dtype, device=responses.device
        )
        padded_responses[active_mask] = responses

        # 创建填充后的响应字符串列表
        padded_responses_str = [""] * batch_size

        s = 0
        for i, is_active in enumerate(active_mask):
            if is_active:
                padded_responses_str[i] = responses_str[s]
                s += 1

        return padded_responses, padded_responses_str