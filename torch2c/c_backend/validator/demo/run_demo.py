"""validator demo：演示合法图通过、非法图报错。"""

from __future__ import annotations

import json
from pathlib import Path

from torch2c.common import (
    INTEGRATION_CONFIG_DIR, Graph, ValidationError, get_logger,
    load_config, setup_logging,
)
from torch2c.c_backend.validator import run

logger = get_logger(__name__)


def _build_validator_config() -> dict:
    """从 c_api_signatures.yaml 构建 supported_ops 列表。"""
    sigs = load_config(
        str(INTEGRATION_CONFIG_DIR / "c_api_signatures.yaml"),
        required_keys=["compute_ops"],
    )
    ops: list[str] = []
    for section in ("compute_ops", "dma_ops", "idma_ops"):
        ops.extend(sigs.get(section, {}).keys())
    return {"supported_ops": ops}


def main() -> None:
    setup_logging("DEBUG")
    demo_dir = Path(__file__).parent
    config = _build_validator_config()

    # 测试合法图
    with open(demo_dir / "demo_valid_graph.json", encoding="utf-8") as f:
        valid_graph = Graph.from_dict(json.load(f))

    result = run(valid_graph, config)
    logger.info("合法图校验通过，节点数: %d", len(result.nodes))

    # 测试非法图
    with open(demo_dir / "demo_invalid_graph.json", encoding="utf-8") as f:
        invalid_graph = Graph.from_dict(json.load(f))

    try:
        run(invalid_graph, config)
        raise AssertionError("应该抛出 ValidationError")
    except ValidationError as e:
        logger.info("非法图校验失败（预期）: %s", e)
        if "npu_unknown" not in str(e):
            raise AssertionError("错误消息应包含 'npu_unknown'") from e

    logger.info("Demo 通过！")


if __name__ == "__main__":
    main()
