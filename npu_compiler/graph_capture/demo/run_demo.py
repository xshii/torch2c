"""图捕获演示：捕获 DemoEncoder 并保存为 JSON。"""

from __future__ import annotations

import json
import pathlib

import torch

from ..graph_capture import capture
from .demo_model import DemoEncoder


def main() -> None:
    model = DemoEncoder(hidden_size=64, num_heads=4, ffn_dim=256)
    model.eval()

    dummy_input = torch.randn(1, 32, 64)
    mask = torch.zeros(1, 1, 32, 32)

    graph = capture(model, dummy_input, mask=mask)

    out_path = pathlib.Path(__file__).parent / "captured_graph.json"
    out_path.write_text(json.dumps(graph.to_dict(), indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"已保存到 {out_path}")
    print(graph.summary())


if __name__ == "__main__":
    main()
