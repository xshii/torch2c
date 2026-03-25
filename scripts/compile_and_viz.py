"""编译 DemoEncoder 模型 + 生成 pipeline 可视化。"""

import json
import os

import torch

from torch2c.a_capture.graph_capture.demo.demo_model import DemoEncoder
from torch2c.common.paths import INTEGRATION_CONFIG_DIR
from torch2c.integration.pipeline import compile
from torch2c.viz.pipeline_viz import emit_pipeline_html

OUTPUT_DIR = "output/compile_viz"

model = DemoEncoder()
x = torch.randn(1, 32, 64)
mask = torch.zeros(1, 1, 32, 32)

compile(
    model, x,
    config_dir=str(INTEGRATION_CONFIG_DIR),
    output_dir=OUTPUT_DIR,
    mask=mask,
    debug_dump=True,
)

# 读取 timing 数据
timing_path = os.path.join(OUTPUT_DIR, "debug", "pass_timing.json")
timing = json.load(open(timing_path)) if os.path.isfile(timing_path) else None

path = emit_pipeline_html(OUTPUT_DIR, pass_timing=timing,
                           debug_dir=f"{OUTPUT_DIR}/debug")
print(f"\nPipeline: {path}")
