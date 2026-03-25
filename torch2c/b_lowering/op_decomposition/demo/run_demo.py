"""算子裂解演示：加载 demo 图，执行裂解，打印结果。"""

from __future__ import annotations

import json
import pathlib

from torch2c.common import INTEGRATION_CONFIG_DIR, Graph, load_config
from torch2c.b_lowering.op_decomposition import run

_DIR = pathlib.Path(__file__).parent
_CONFIG = INTEGRATION_CONFIG_DIR / "decompositions.yaml"


def main() -> None:
    graph_data = json.loads((_DIR / "demo_input_graph.json").read_text("utf-8"))
    graph = Graph.from_dict(graph_data)

    config = load_config(str(_CONFIG), required_keys=["decompositions"])
    run(graph, config)

    print("裂解结果:")
    for nid in graph.execution_order:
        node = graph.get_node(nid)
        if node:
            status = f"{node.npu_op} ({node.compute_unit})"
            print(f"  {node.id}: {status}")

    print(f"\n节点数: {len(graph.nodes)}, tensor数: {len(graph.tensors)}")


if __name__ == "__main__":
    main()
