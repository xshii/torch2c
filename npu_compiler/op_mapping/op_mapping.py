"""算子直接映射：将 ATen 算子名替换为 NPU 算子名（1 对 1）。

遍历 Graph 中所有未映射节点，根据 config 中的映射表填充
npu_op、compute_unit 并标记 is_mapped=True。
未在映射表中的节点保持原样，留给 op_decomposition 或 validator 处理。
"""

from __future__ import annotations

from ..common.graph_ir import Graph
from ..common.logger import get_logger

logger = get_logger(__name__)


def run(graph: Graph, config: dict) -> Graph:
    """对图执行算子直接映射。

    Args:
        graph: 输入 Graph IR（节点 op_type 为 ATen 全称）。
        config: 映射配置字典，需包含 ``mappings`` 键。

    Returns:
        同一 Graph 对象（原地修改），已映射节点标记完成。
    """
    mappings = config.get("mappings", {})
    mapped_count = 0
    unmapped_count = 0

    for node in graph.nodes.values():
        if node.is_mapped:
            continue

        entry = mappings.get(node.op_type)
        if entry:
            node.npu_op = entry["npu_op"]
            node.compute_unit = entry["compute_unit"]
            node.is_mapped = True
            mapped_count += 1
            logger.debug("映射 %s: %s → %s (%s)",
                         node.id, node.op_type, node.npu_op, node.compute_unit)
        else:
            unmapped_count += 1
            logger.debug("未映射 %s: %s", node.id, node.op_type)

    logger.info("映射完成。已映射: %d, 未映射: %d", mapped_count, unmapped_count)
    return graph
