"""单层 Attention 编译 + 可视化。

1 层 MHA（2 头）：LayerNorm → Q/K/V 投影 → bmm + softmax 注意力 → 输出投影 + 残差。
hidden=64, heads=2, seq=16, batch=1
"""

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch2c.common.paths import INTEGRATION_CONFIG_DIR
from torch2c.integration.pipeline import compile
from torch2c.viz.pipeline_viz import emit_pipeline_html


class SingleAttention(nn.Module):
    """单层多头注意力：LayerNorm + MHA + 残差。"""

    def __init__(self, hidden: int = 64, heads: int = 2):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.ln = nn.LayerNorm(hidden)
        self.wq = nn.Linear(hidden, hidden)
        self.wk = nn.Linear(hidden, hidden)
        self.wv = nn.Linear(hidden, hidden)
        self.wo = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        H, Dh = self.heads, self.head_dim

        h = self.ln(x)

        q = self.wq(h).view(B, S, H, Dh).transpose(1, 2).reshape(B * H, S, Dh)
        k = self.wk(h).view(B, S, H, Dh).transpose(1, 2).reshape(B * H, S, Dh)
        v = self.wv(h).view(B, S, H, Dh).transpose(1, 2).reshape(B * H, S, Dh)

        scores = torch.bmm(q, k.transpose(-2, -1)).view(B, H, S, S)
        scores = scores * self.scale + mask
        attn = F.softmax(scores, dim=-1)

        out = torch.bmm(
            attn.view(B * H, S, S), v
        ).view(B, H, S, Dh).transpose(1, 2).reshape(B, S, H * Dh)

        return x + self.wo(out)


OUTPUT_DIR = "output/single_attn"

model = SingleAttention()
x = torch.randn(1, 16, 64)
mask = torch.zeros(1, 1, 16, 16)

compile(
    model, x,
    config_dir=str(INTEGRATION_CONFIG_DIR),
    output_dir=OUTPUT_DIR,
    mask=mask,
    debug_dump=True,
)

timing_path = os.path.join(OUTPUT_DIR, "debug", "pass_timing.json")
timing = json.load(open(timing_path)) if os.path.isfile(timing_path) else None
path = emit_pipeline_html(OUTPUT_DIR, pass_timing=timing,
                           debug_dir=f"{OUTPUT_DIR}/debug")
print(f"\nPipeline: {path}")
