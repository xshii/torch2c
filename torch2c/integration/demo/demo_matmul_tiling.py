"""L1b: 大序列 matmul — 触发自动 tiling + ping-pong

y = x @ W + b，其中 x=[1,1024,256], W=[256,256]
总 tensor ~1.5MB 远小于 L1(16MB)，但 M 维足够大，global_tiler 评估
tiling+双 buffer ping-pong 可重叠 DMA 与计算，主动分为 16 块。

用法:
    python scripts/compile_model.py torch2c/integration/demo/demo_matmul_tiling.py --mode full
"""

import torch
import torch.nn as nn


class LargeMatmul(nn.Module):
    def __init__(self, d: int = 256):
        super().__init__()
        self.fc = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


model = LargeMatmul()
dummy_input = torch.randn(1, 1024, 256)
