"""op_absorption demo：演示 add(scores, mask) 被吸收进 softmax_part1。"""

from __future__ import annotations

import json
from pathlib import Path

from npu_compiler.common import Graph, get_logger, setup_logging
from npu_compiler.op_absorption import load_absorption_config, run

logger = get_logger(__name__)


def main() -> None:
    setup_logging("DEBUG")
    demo_dir = Path(__file__).parent
    config_path = demo_dir.parent / "config" / "absorptions.yaml"

    # 加载输入图
    with open(demo_dir / "demo_input_graph.json", encoding="utf-8") as f:
        graph = Graph.from_dict(json.load(f))

    logger.info("输入图: %d 节点, %d tensor", len(graph.nodes), len(graph.tensors))

    # 加载配置并运行
    config = load_absorption_config(str(config_path))
    result = run(graph, config)

    logger.info("输出图: %d 节点, %d tensor", len(result.nodes), len(result.tensors))

    # 验证
    assert "node_add" not in result.nodes, "add 节点应被删除"
    assert "tensor_add_out" not in result.tensors, "add 输出 tensor 应被删除"
    sp1 = result.get_node("node_softmax_p1")
    assert sp1 is not None
    assert sp1.absorbed_inputs.get("mask") == "tensor_mask"
    assert sp1.inputs[0] == "tensor_scores"

    # 输出结果
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    logger.info("Demo 通过！")


if __name__ == "__main__":
    main()
