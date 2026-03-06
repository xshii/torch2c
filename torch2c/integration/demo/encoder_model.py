"""2 层 Encoder Transformer 模型定义（demo 用）。

使用单头注意力 + LayerNorm + GELU FFN，所有算子均在已支持的 NPU 算子集合内。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torch2c.common import torch2c_config  # noqa: F401 — 供外部 demo 使用


class SelfAttention(nn.Module):
    """单头自注意力。"""

    def __init__(self, d_model: int):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.bmm(q, k.transpose(1, 2))
        if mask is not None:
            scores = scores + mask
        attn = torch.softmax(scores, dim=-1)
        out = torch.bmm(attn, v)
        return self.o_proj(out)


class EncoderLayer(nn.Module):
    """单层 Encoder：SelfAttention + FFN，各带残差连接和 LayerNorm。"""

    def __init__(self, d_model: int, dim_ff: int):
        super().__init__()
        self.self_attn = SelfAttention(d_model)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.self_attn(x, mask)
        x = self.norm1(x + attn_out)
        ff_out = self.linear2(self.activation(self.linear1(x)))
        x = self.norm2(x + ff_out)
        return x


# TODO: 启用 fp16 混合精度后 C golden 比对失败（fp16 pipeline 有 bug），
# 待修复后取消注释。
# @torch2c_config(
#     target_dtype="fp16",
#     compute_dtype_rules={
#         "default": "fp16",
#         "by_compute_unit": {"cube": "fp32"},
#         "by_op": {
#             "vector_softmax_part1": "fp32",
#             "vector_layernorm_part1": "fp32",
#         },
#     },
# )
class EncoderModel(nn.Module):
    """2 层 Encoder Transformer 模型。"""

    def __init__(self, d_model: int = 256, dim_ff: int = 512, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, dim_ff) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x
