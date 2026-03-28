"""L4: Embedding + LayerNorm + Linear + scale

覆盖: idma_embedding, vector_layernorm, vector_mul_scalar, cube_matmul_bias
验证: 嵌入表查找（idma_embedding）首次出现、scalar 乘法

用法:
    python scripts/compile_model.py torch2c/integration/demo/demo_embedding.py --mode both
"""

import torch
import torch.nn as nn


class EmbeddingModel(nn.Module):
    """Embedding → LayerNorm → Linear → scale。

    模拟 LLM 的 token embedding 层 + norm + projection。
    """

    def __init__(self, vocab_size: int = 256, d: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, d)
        self.scale = d ** 0.5

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(token_ids)
        x = self.norm(x)
        x = self.proj(x)
        x = x * self.scale  # mul_scalar
        return x


model = EmbeddingModel()
dummy_input = torch.randint(0, 256, (1, 32))
