"""L7: GPT-2 style block — 全 op 覆盖压力测试

覆盖: 几乎全部 18 个 NPU op
  cube: matmul, matmul_bias
  vector: add, sub, mul, div, mul_scalar, gelu, relu, softmax, layernorm
  idma: reshape, transpose, embedding, broadcast

验证: 接近真实 GPT-2 block 的完整计算图

用法:
    python scripts/compile_model.py torch2c/integration/demo/demo_gpt_block.py --mode both
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class GPT2Attention(nn.Module):
    """GPT-2 多头因果自注意力。"""

    def __init__(self, d: int = 128, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.c_proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        q = self.wq(x).view(B, S, H, Dh).transpose(1, 2).reshape(B * H, S, Dh)
        k = self.wk(x).view(B, S, H, Dh).transpose(1, 2).reshape(B * H, S, Dh)
        v = self.wv(x).view(B, S, H, Dh).transpose(1, 2).reshape(B * H, S, Dh)

        scores = torch.bmm(q, k.transpose(1, 2)).view(B, H, S, S)
        scores = scores * self.scale
        scores = scores + causal_mask
        attn = F.softmax(scores, dim=-1)

        out = torch.bmm(attn.view(B * H, S, S), v)
        out = out.view(B, H, S, Dh).transpose(1, 2).reshape(B, S, D)
        return self.c_proj(out)


class GPT2MLP(nn.Module):
    """GPT-2 FFN: Linear → GELU → Linear。"""

    def __init__(self, d: int = 128, d_ff: int = 512):
        super().__init__()
        self.c_fc = nn.Linear(d, d_ff)
        self.c_proj = nn.Linear(d_ff, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x)))


class GPT2Block(nn.Module):
    """完整 GPT-2 block: LN → Attn → residual → LN → MLP → residual。"""

    def __init__(self, d: int = 128, num_heads: int = 4, d_ff: int = 512):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d)
        self.attn = GPT2Attention(d, num_heads)
        self.ln_2 = nn.LayerNorm(d)
        self.mlp = GPT2MLP(d, d_ff)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), causal_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Mini(nn.Module):
    """2 层 GPT-2（mini 版）。"""

    def __init__(self, d: int = 128, num_heads: int = 4, d_ff: int = 512):
        super().__init__()
        self.block1 = GPT2Block(d, num_heads, d_ff)
        self.block2 = GPT2Block(d, num_heads, d_ff)
        self.ln_f = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        x = self.block1(x, causal_mask)
        x = self.block2(x, causal_mask)
        x = self.ln_f(x)
        return x


model = GPT2Mini()
dummy_input = torch.randn(1, 32, 128)
mask = torch.triu(torch.full((1, 1, 32, 32), -10000.0), diagonal=1)
