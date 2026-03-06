"""validator — Pass⑥：合法性校验。"""

from __future__ import annotations

from torch2c.common import Graph, ValidationError, get_logger, load_config

logger = get_logger(__name__)

_CONFIG_REQUIRED_KEYS = ["supported_ops"]


def load_validator_config(path: str) -> dict:
    """加载校验配置。"""
    return load_config(path, required_keys=_CONFIG_REQUIRED_KEYS)


def run(graph: Graph, config: dict) -> Graph:
    """校验所有节点是否都已映射到支持的 NPU 算子。"""
    supported = set(config["supported_ops"])
    unsupported: list[str] = []

    for node in graph.nodes.values():
        if node.npu_op not in supported:
            unsupported.append(f"{node.id}: {node.op_type} (npu_op={node.npu_op})")

    if unsupported:
        raise ValidationError(f"以下算子未映射: {unsupported}")

    logger.info("校验通过。%d个节点全部合法", len(graph.nodes))
    return graph
