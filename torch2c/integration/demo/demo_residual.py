"""L3: 残差块 — 分支汇聚 + 多种算术 op

覆盖: cube_matmul_bias, vector_add, vector_sub, vector_div, vector_relu, vector_layernorm
验证: DAG 分支/汇聚（不再是线性链），sub/div 首次出现

用法:
    python scripts/compile_model.py torch2c/integration/demo/demo_residual.py --mode both
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """类 ResNet 残差块（用 Linear 替代 Conv）。

    x ──→ LN → Linear → ReLU → Linear ──→ add → div(2) → out
    │                                        ↑
    └──────────── shortcut ─────────────────┘
    """

    def __init__(self, d: int = 64, d_ff: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d_ff)
        self.fc2 = nn.Linear(d_ff, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        h = torch.relu(self.fc1(h))
        h = self.fc2(h)
        out = (residual + h) / 2.0  # add + div
        return out


class StackedResidual(nn.Module):
    """3 个残差块堆叠，测试多次分支汇聚。"""

    def __init__(self, d: int = 64):
        super().__init__()
        self.block1 = ResidualBlock(d)
        self.block2 = ResidualBlock(d)
        self.block3 = ResidualBlock(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


model = StackedResidual()
dummy_input = torch.randn(1, 32, 64)
