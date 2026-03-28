"""L2: 3 层 MLP — Linear + ReLU + GELU + LayerNorm

覆盖: cube_matmul_bias, vector_relu, vector_gelu, vector_layernorm, vector_add
验证: 多激活函数、长链依赖

用法:
    python scripts/compile_model.py torch2c/integration/demo/demo_mlp.py --mode both
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class MLP3(nn.Module):
    def __init__(self, d: int = 64, d_ff: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d_ff)
        self.fc2 = nn.Linear(d_ff, d_ff)
        self.fc3 = nn.Linear(d_ff, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = torch.relu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)
        return x


model = MLP3()
dummy_input = torch.randn(1, 32, 64)
