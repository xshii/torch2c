"""Encoder Transformer 模型定义（demo 用）。

支持多头注意力 + LayerNorm + GELU FFN，所有算子均在已支持的 NPU 算子集合内。

支持两种精度模式：
- mixed：linear1/norm1 为 fp32 全精度，其余 fp16 IO + fp32 计算
- fp16：全部 fp16 IO + fp16 权重 + fp32 计算
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torch2c.common import NpuSpec, npu

# 常用 spec 缩写
_FP32 = NpuSpec("fp32", "nd")
_FP16 = NpuSpec("fp16", "nd")
_FP16_NZ = NpuSpec("fp16", "nz")


def _npu_linear(d_in: int, d_out: int, *, precision: str) -> nn.Linear:
    """按精度模式包装 nn.Linear。"""
    m = nn.Linear(d_in, d_out)
    if precision == "fp16":
        return npu(m, input=_FP16, output=_FP16, weight=_FP16_NZ, compute_dtype="fp32")
    return m  # mixed 模式由调用者单独标注


class MultiHeadAttention(nn.Module):
    """多头自注意力。

    forward 拆解为显式 reshape + transpose + bmm，确保图捕获产出已支持的算子。
    """

    def __init__(self, d_model: int, num_heads: int, *, precision: str = "mixed"):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model({d_model}) 不能被 num_heads({num_heads}) 整除"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model

        self.q_proj = _npu_linear(d_model, d_model, precision=precision)
        self.k_proj = _npu_linear(d_model, d_model, precision=precision)
        self.v_proj = _npu_linear(d_model, d_model, precision=precision)
        self.o_proj = _npu_linear(d_model, d_model, precision=precision)
        if precision == "mixed":
            for proj in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
                npu(proj, input=_FP16, output=_FP16, weight=_FP16_NZ, compute_dtype="fp32")

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, S, _ = x.shape
        H, Dh = self.num_heads, self.head_dim

        # [B, S, D] → proj → [B, S, D] → [B, S, H, Dh] → [B, H, S, Dh] → [B*H, S, Dh]
        q = self.q_proj(x).view(B, S, H, Dh).transpose(1, 2).reshape(B * H, S, Dh)
        k = self.k_proj(x).view(B, S, H, Dh).transpose(1, 2).reshape(B * H, S, Dh)
        v = self.v_proj(x).view(B, S, H, Dh).transpose(1, 2).reshape(B * H, S, Dh)

        # [B*H, S, Dh] @ [B*H, Dh, S] → [B*H, S, S]
        scores = torch.bmm(q, k.transpose(1, 2))

        # mask: [B, S, S] → [B, 1, S, S] → expand → [B*H, S, S]
        if mask is not None:
            scores = scores.view(B, H, S, S)
            scores = scores + mask.unsqueeze(1)
            scores = scores.view(B * H, S, S)

        attn = torch.softmax(scores, dim=-1)

        # [B*H, S, S] @ [B*H, S, Dh] → [B*H, S, Dh]
        out = torch.bmm(attn, v)

        # [B*H, S, Dh] → [B, H, S, Dh] → [B, S, H, Dh] → [B, S, D]
        out = out.view(B, H, S, Dh).transpose(1, 2).reshape(B, S, self.d_model)

        return self.o_proj(out)


class EncoderLayer(nn.Module):
    """单层 Encoder：MultiHeadAttention + FFN，各带残差连接和 LayerNorm。"""

    def __init__(
        self, d_model: int, dim_ff: int, num_heads: int, *, precision: str = "mixed",
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, precision=precision)
        if precision == "mixed":
            # linear1 + norm1: fp32 全精度（高精度路径）
            self.linear1 = npu(nn.Linear(d_model, dim_ff),
                               input=_FP32, output=_FP32, weight=_FP32, compute_dtype="fp32")
            self.norm1 = npu(nn.LayerNorm(d_model),
                             input=_FP32, output=_FP32, weight=_FP32, compute_dtype="fp32")
        else:
            # fp16: 全部 fp16
            self.linear1 = npu(nn.Linear(d_model, dim_ff),
                               input=_FP16, output=_FP16, weight=_FP16_NZ, compute_dtype="fp32")
            self.norm1 = npu(nn.LayerNorm(d_model),
                             input=_FP16, output=_FP16, weight=_FP16, compute_dtype="fp32")
        # linear2 + norm2 + activation: 两种模式都用 fp16
        self.linear2 = npu(nn.Linear(dim_ff, d_model),
                           input=_FP16, output=_FP16, weight=_FP16_NZ, compute_dtype="fp32")
        self.norm2 = npu(nn.LayerNorm(d_model),
                         input=_FP16, output=_FP16, weight=_FP16, compute_dtype="fp32")
        self.activation = npu(nn.GELU(),
                              input=_FP16, output=_FP16, compute_dtype="fp16")

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.self_attn(x, mask)
        x = self.norm1(x + attn_out)
        ff_out = self.linear2(self.activation(self.linear1(x)))
        x = self.norm2(x + ff_out)
        return x


class EncoderModel(nn.Module):
    """多层 Encoder Transformer 模型。

    Args:
        d_model: 模型维度（须能被 num_heads 整除）。
        dim_ff: FFN 中间维度。
        num_layers: Encoder 层数。
        num_heads: 注意力头数。
        precision: "mixed"（混合精度）或 "fp16"（全 fp16）。
    """

    def __init__(
        self, d_model: int = 192, dim_ff: int = 384,
        num_layers: int = 3, num_heads: int = 3,
        *, precision: str = "mixed",
    ):
        super().__init__()
        self.precision = precision
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, dim_ff, num_heads, precision=precision)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x
