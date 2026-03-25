"""Embedding 模型：Embedding → Linear → ReLU。"""

import json, os, torch, torch.nn as nn
from torch2c.common.paths import INTEGRATION_CONFIG_DIR
from torch2c.integration.pipeline import compile
from torch2c.viz.pipeline_viz import emit_pipeline_html

class EmbedModel(nn.Module):
    def __init__(self, vocab: int = 128, dim: int = 32, out: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.proj = nn.Linear(dim, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        h = self.proj(h)
        return torch.relu(h)

OUTPUT_DIR = "output/demo_embedding"
model = EmbedModel()
x = torch.randint(0, 128, (1, 16))  # [batch=1, seq=16] 整数 token

compile(model, x, config_dir=str(INTEGRATION_CONFIG_DIR),
        output_dir=OUTPUT_DIR, debug_dump=True)

tp = os.path.join(OUTPUT_DIR, "debug", "pass_timing.json")
timing = json.load(open(tp)) if os.path.isfile(tp) else None
path = emit_pipeline_html(OUTPUT_DIR, pass_timing=timing,
                           debug_dir=f"{OUTPUT_DIR}/debug")
print(f"\nEmbedding pipeline: {path}")
