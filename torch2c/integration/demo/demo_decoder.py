"""L5: 单层 Decoder — causal mask + 因果注意力

覆盖: cube_matmul, cube_matmul_bias, vector_softmax, vector_layernorm, vector_gelu,
      idma_reshape, idma_transpose, vector_add, vector_mul_scalar
验证: 因果注意力掩码、Decoder 结构

用法:
    python scripts/compile_model.py torch2c/integration/demo/demo_decoder.py --mode both
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class CausalSelfAttention(nn.Module):
    """单头因果自注意力。"""

    def __init__(self, d: int = 64):
        super().__init__()
        self.d = d
        self.scale = 1.0 / math.sqrt(d)
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        scores = torch.bmm(q, k.transpose(1, 2))  # [B, S, S]
        scores = scores * self.scale
        scores = scores + causal_mask  # 因果掩码（上三角 -inf）
        attn = F.softmax(scores, dim=-1)

        out = torch.bmm(attn, v)  # [B, S, D]
        return self.out_proj(out)


class DecoderBlock(nn.Module):
    """单层 Decoder: CausalAttention + FFN + 残差 + LayerNorm。"""

    def __init__(self, d: int = 64, d_ff: int = 256):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d_ff)
        self.fc2 = nn.Linear(d_ff, d)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        h = self.attn(h, causal_mask)
        x = x + h
        h = self.ln2(x)
        h = F.gelu(self.fc1(h))
        h = self.fc2(h)
        x = x + h
        return x


model = DecoderBlock()
dummy_input = torch.randn(1, 32, 64)
# 因果掩码: 上三角为 -10000（近似 -inf，避免 fp16 溢出）
mask = torch.triu(torch.full((1, 32, 32), -10000.0), diagonal=1)
