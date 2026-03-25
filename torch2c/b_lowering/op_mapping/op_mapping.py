"""算子命名映射：将 ATen 算子名翻译为 NPU 算子名（纯命名，不设 is_mapped）。

遍历 Graph 中所有节点，根据 mappings 表填充 npu_op 和 compute_unit。
不设 is_mapped=True — 由 op_decomposition 最终决定是否需要进一步裂解。
"""

from __future__ import annotations

from torch2c.common.graph_ir import Graph
from torch2c.common.logger import get_logger
from torch2c.common.opt_log import log_opt

logger = get_logger(__name__)

_CU_WHY = {
    "cube": "Cube 单元含 16×16×16 MAC 阵列，适合矩阵乘/卷积等计算密集型算子",
    "vector": "Vector 单元含 SIMD 流水线，适合逐元素/归约等访存密集型算子",
    "idma": "IDMA 单元执行片上数据搬运，零计算开销完成 reshape/transpose/broadcast",
}


def run(graph: Graph, config: dict) -> Graph:
    """对图执行算子命名映射（不设 is_mapped）。"""
    mappings = config.get("mappings", {})
    mapped_count = 0
    unmapped_count = 0

    for node in graph.nodes.values():
        if node.npu_op:  # 已有 npu_op 的跳过
            continue
        entry = mappings.get(node.op_type)
        if entry:
            node.npu_op = entry["npu_op"]
            node.compute_unit = entry["compute_unit"]
            mapped_count += 1
            log_opt(node, "op_mapping", "命名映射",
                    f"{node.op_type} → {node.npu_op}。{_CU_WHY.get(node.compute_unit, '')}")
        else:
            unmapped_count += 1
            logger.debug("未映射 %s: %s", node.id, node.op_type)

    logger.info("命名映射完成: %d 已映射, %d 未映射", mapped_count, unmapped_count)
    return graph


def post_validate(graph: Graph) -> list[str]:
    """op_mapping 后的校验：所有节点应有 npu_op。"""
    errors: list[str] = []
    for n in graph.nodes.values():
        if not n.npu_op:
            errors.append(f"节点 {n.id} ({n.op_type}) 缺少 npu_op，映射表中未覆盖")
    return errors
