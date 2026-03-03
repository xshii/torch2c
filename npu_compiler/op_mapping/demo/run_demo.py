"""算子映射演示：加载 demo 图，执行映射，打印结果。"""

from __future__ import annotations

import json
import pathlib

from ...common.config_loader import load_config
from ...common.graph_ir import Graph
from ..op_mapping import run

_DIR = pathlib.Path(__file__).parent
_CONFIG = pathlib.Path(__file__).parent.parent / "config" / "direct_mappings.yaml"


def main() -> None:
    graph_data = json.loads((_DIR / "demo_input_graph.json").read_text("utf-8"))
    graph = Graph.from_dict(graph_data)

    config = load_config(str(_CONFIG), required_keys=["mappings"])
    run(graph, config)

    print("映射结果:")
    for node in graph.nodes.values():
        status = f"{node.npu_op} ({node.compute_unit})" if node.is_mapped else "未映射"
        print(f"  {node.id}: {node.op_type} -> {status}")


if __name__ == "__main__":
    main()
