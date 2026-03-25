"""最简模型：Y = AX + B（单个线性层）。"""

import json, os, torch, torch.nn as nn
from torch2c.common.paths import INTEGRATION_CONFIG_DIR
from torch2c.integration.pipeline import compile
from torch2c.viz.pipeline_viz import emit_pipeline_html

class AXB(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

OUTPUT_DIR = "output/demo_axb"
model = AXB()
x = torch.randn(1, 8, 32)

compile(model, x, config_dir=str(INTEGRATION_CONFIG_DIR),
        output_dir=OUTPUT_DIR, debug_dump=True)

tp = os.path.join(OUTPUT_DIR, "debug", "pass_timing.json")
timing = json.load(open(tp)) if os.path.isfile(tp) else None
path = emit_pipeline_html(OUTPUT_DIR, pass_timing=timing,
                           debug_dir=f"{OUTPUT_DIR}/debug")
print(f"\nY=AX+B pipeline: {path}")
