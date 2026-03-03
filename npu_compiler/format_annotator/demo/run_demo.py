"""format_annotator demo：演示 matmul(cube) → add(vector) 的 format 标注。"""

from __future__ import annotations

import json
from pathlib import Path

from npu_compiler.common import Graph, get_logger, setup_logging

from npu_compiler.format_annotator import load_format_config, run

logger = get_logger(__name__)


def main() -> None:
    setup_logging("DEBUG")
    demo_dir = Path(__file__).parent
    config_path = demo_dir.parent / "config" / "type_format_config.yaml"

    # 加载输入图
    with open(demo_dir / "demo_input_graph.json", encoding="utf-8") as f:
        graph = Graph.from_dict(json.load(f))

    logger.info("输入图: %d 节点, %d tensor", len(graph.nodes), len(graph.tensors))

    # 加载配置并运行
    config = load_format_config(str(config_path))
    result = run(graph, config)

    # 验证 matmul 输出 tensor format 变为 nz
    mm_out = result.get_tensor("tensor_mm_out")
    assert mm_out is not None and mm_out.format == "nz"

    # 验证 add 输出 tensor format 保持 nd
    add_out = result.get_tensor("tensor_add_out")
    assert add_out is not None and add_out.format == "nd"

    # 验证 annotation 结构
    mm_node = result.get_node("node_matmul")
    assert mm_node is not None
    ann = mm_node.format_annotation
    assert ann is not None
    assert ann["inputs"][0]["format"] == "nz"
    assert ann["compute_dtype"] == "fp16"

    # 输出结果
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    logger.info("Demo 通过！")


if __name__ == "__main__":
    main()
