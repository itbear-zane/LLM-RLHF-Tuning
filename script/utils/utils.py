# 工具模块 - 包含各种通用的工具函数和配置
#
# 本模块提供：
# 1. 提示词模板定义
# 2. 类型转换工具类

import os,sys
import torch
import torch.nn as nn

# 提示词模板定义 - 支持不同的对话格式
PROMPT_TEMPLATE = dict(
    # 中文LLaMA-Alpaca格式
    chinese_llama_alpaca=(
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response: "
        ),
    # 中文LLaMA2-Alpaca格式（使用对话模板）
    chinese_llama2_alpaca=(
        "[INST] <<SYS>>\n"
        "You are a helpful assistant. 你是一个乐于助人的助手。\n"
        "<</SYS>>\n\n{instruction} [/INST]"
    ),
    # 默认的Human-Assistant格式
    default=(
        "Human: {instruction}\nAssistant: "
    )
)


class CastOutputToFloat(nn.Sequential):
    """
    类型转换工具类 - 将输出转换为float32类型

    这个类用于在模型训练过程中确保输出的数值精度一致性。
    """
    def forward(self, x):
        """
        前向传播，将输出转换为float32

        Args:
            x: 输入张量

        Returns:
            转换为float32的张量
        """
        return super().forward(x).to(torch.float32)
























