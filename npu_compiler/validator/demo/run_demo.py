"""validator demo：演示合法图通过、非法图报错。"""

from __future__ import annotations

import json
from pathlib import Path

from npu_compiler.common import ValidationError, get_logger, setup_logging

from npu_compiler.validator import load_validator_config, run

logger = get_logger(__name__)


def main() -> None:
    setup_logging("DEBUG")
    demo_dir = Path(__file__).parent
    config_path = demo_dir.parent / "config" / "supported_ops.yaml"
    config = load_validator_config(str(config_path))

    # 测试合法图
    with open(demo_dir / "demo_valid_graph.json", encoding="utf-8") as f:
        from npu_compiler.common import Graph
        valid_graph = Graph.from_dict(json.load(f))

    result = run(valid_graph, config)
    logger.info("合法图校验通过，节点数: %d", len(result.nodes))

    # 测试非法图
    with open(demo_dir / "demo_invalid_graph.json", encoding="utf-8") as f:
        invalid_graph = Graph.from_dict(json.load(f))

    try:
        run(invalid_graph, config)
        assert False, "应该抛出 ValidationError"
    except ValidationError as e:
        logger.info("非法图校验失败（预期）: %s", e)
        assert "npu_unknown" in str(e)

    logger.info("Demo 通过！")


if __name__ == "__main__":
    main()
