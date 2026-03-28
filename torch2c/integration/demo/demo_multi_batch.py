"""L6: 多 batch Encoder — batch=4

覆盖: 同 demo_model，但 batch=4
验证: batch 维度对 memory planner / tiling / DMA 的影响

用法:
    python scripts/compile_model.py torch2c/integration/demo/demo_multi_batch.py --mode both
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

import math


class MultiBatchEncoder(nn.Module):
    """单层 Transformer Encoder，batch=4。"""

    def __init__(self, d: int = 64, num_heads: int = 4, d_ff: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.ln1 = nn.LayerNorm(d)
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.wo = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, d_ff)
        self.ff2 = nn.Linear(d_ff, d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        H, Dh = self.num_heads, self.head_dim

        h = self.ln1(x)
        q = self.wq(h).view(B, S, H, Dh).transpose(1, 2).reshape(B * H, S, Dh)
        k = self.wk(h).view(B, S, H, Dh).transpose(1, 2).reshape(B * H, S, Dh)
        v = self.wv(h).view(B, S, H, Dh).transpose(1, 2).reshape(B * H, S, Dh)

        scores = torch.bmm(q, k.transpose(1, 2)).view(B, H, S, S)
        scores = scores * self.scale + mask
        attn = F.softmax(scores, dim=-1)
        out = torch.bmm(
            attn.view(B * H, S, S), v
        ).view(B, H, S, Dh).transpose(1, 2).reshape(B, S, H * Dh)
        h = self.wo(out)
        x = x + h

        h = self.ln2(x)
        h = F.gelu(self.ff1(h))
        h = self.ff2(h)
        x = x + h
        return x


model = MultiBatchEncoder()
dummy_input = torch.randn(4, 16, 64)   # batch=4, seq=16
mask = torch.zeros(4, 1, 16, 16)       # broadcast mask
