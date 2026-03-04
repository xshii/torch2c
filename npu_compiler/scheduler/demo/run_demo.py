"""scheduler demo：运行 4 算子线性链的调度。"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from npu_compiler.common import Graph
from npu_compiler.scheduler import run

DEMO_DIR = os.path.dirname(__file__)


def main() -> None:
    with open(os.path.join(DEMO_DIR, "demo_input_graph.json")) as f:
        data = json.load(f)
    graph = Graph.from_dict(data)

    graph = run(graph)

    print("=== 调度结果 ===")
    for nid in graph.execution_order:
        node = graph.nodes[nid]
        print(
            f"  {nid}: order={node.schedule_order}, "
            f"unit={node.compute_unit}, deps={node.dependencies}"
        )

    # 验证线性链：每个后续节点依赖前一个
    for i in range(1, len(graph.execution_order)):
        curr = graph.execution_order[i]
        prev = graph.execution_order[i - 1]
        assert prev in graph.nodes[curr].dependencies, f"{curr} 应依赖 {prev}"
    print("\n验证通过!")


if __name__ == "__main__":
    main()
