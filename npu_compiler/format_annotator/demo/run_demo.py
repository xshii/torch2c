"""format_annotator demo：演示从模型继承 dtype/format 以及用户覆盖。"""

from __future__ import annotations

import json
from pathlib import Path

from npu_compiler.common import Graph, get_logger, setup_logging
from npu_compiler.format_annotator import run

logger = get_logger(__name__)


def main() -> None:
    setup_logging("DEBUG")
    demo_dir = Path(__file__).parent

    # 加载输入图
    with open(demo_dir / "demo_input_graph.json", encoding="utf-8") as f:
        graph = Graph.from_dict(json.load(f))

    logger.info("输入图: %d 节点, %d tensor", len(graph.nodes), len(graph.tensors))

    # 场景1：继承模型原始 dtype/format（不覆盖）
    result1 = run(Graph.from_dict(graph.to_dict()), {})
    mm_out = result1.get_tensor("tensor_mm_out")
    assert mm_out is not None
    assert mm_out.dtype == "fp16"  # 继承自模型
    assert mm_out.format == "nd"  # 继承自模型
    logger.info("场景1通过：继承模型 dtype/format")

    # 场景2：用户指定 target_format
    result2 = run(Graph.from_dict(graph.to_dict()), {"target_format": "nz"})
    mm_out2 = result2.get_tensor("tensor_mm_out")
    assert mm_out2 is not None
    assert mm_out2.format == "nz"
    logger.info("场景2通过：用户覆盖 format=nz")

    # 场景3：用户指定 target_dtype
    result3 = run(Graph.from_dict(graph.to_dict()), {"target_dtype": "fp32"})
    mm_out3 = result3.get_tensor("tensor_mm_out")
    assert mm_out3 is not None
    assert mm_out3.dtype == "fp32"
    logger.info("场景3通过：用户覆盖 dtype=fp32")

    # 输出最后一次结果
    print(json.dumps(result3.to_dict(), indent=2, ensure_ascii=False))
    logger.info("Demo 通过！")


if __name__ == "__main__":
    main()
