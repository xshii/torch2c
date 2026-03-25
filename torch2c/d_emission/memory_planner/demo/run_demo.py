"""memory_planner demo：运行 5 算子线性链的内存编排。"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from torch2c.common import INTEGRATION_CONFIG_DIR, Graph, load_config
from torch2c.d_emission.memory_planner import run

DEMO_DIR = os.path.dirname(__file__)


def main() -> None:
    with open(os.path.join(DEMO_DIR, "demo_input_graph.json")) as f:
        data = json.load(f)
    graph = Graph.from_dict(data)
    config = load_config(str(INTEGRATION_CONFIG_DIR / "hardware_config.yaml"))

    graph = run(graph, config)

    print("=== HBM 分配结果 ===")
    for tid, t in sorted(graph.tensors.items()):
        print(f"  {tid}: hbm_offset={t.hbm_offset}, hbm_size={t.hbm_size}")

    print(f"\n=== DMA 计划 ({len(graph.dma_plans)} 个算子) ===")
    for plan in graph.dma_plans:
        print(f"  {plan.node_id}: {len(plan.loads)} loads, {len(plan.stores)} stores")

    # 验证
    for t in graph.tensors.values():
        if t.hbm_offset is not None:
            assert t.hbm_offset >= 0, f"{t.id} offset < 0"
            assert t.hbm_offset % 512 == 0, f"{t.id} 未对齐 512B"
    print("\n验证通过!")


if __name__ == "__main__":
    main()
