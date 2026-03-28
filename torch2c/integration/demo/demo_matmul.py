"""L1: 最简模型 — y = x @ W + b

覆盖: cube_matmul_bias, vector_add
验证: 最小编译链路可走通

用法:
    python scripts/compile_model.py torch2c/integration/demo/demo_matmul.py --mode both
"""

import torch
import torch.nn as nn


class SimpleMatmul(nn.Module):
    def __init__(self, d_in: int = 64, d_out: int = 128):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


model = SimpleMatmul()
dummy_input = torch.randn(1, 32, 64)
