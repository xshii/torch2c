"""scheduler — Pass⑧：计算单元调度与依赖生成。"""

from __future__ import annotations

from torch2c.common import Graph, get_logger

logger = get_logger("scheduler")


def _has_data_dependency(graph: Graph, pred_id: str, succ_id: str) -> bool:
    """检查 succ 是否数据依赖于 pred（pred 的输出是 succ 的输入）。"""
    pred = graph.nodes[pred_id]
    succ = graph.nodes[succ_id]
    pred_outputs = set(pred.outputs)
    return any(tid in pred_outputs for tid in succ.inputs)


def run(graph: Graph, config: dict | None = None) -> Graph:
    """调度主函数：确定执行顺序，生成依赖关系。

    Args:
        graph: 已完成内存编排的 Graph IR。
        config: 可选配置（当前未使用）。

    Returns:
        填充了 schedule_order 和 dependencies 的 Graph。
    """
    logger.info("Pass 开始，输入图: %d 个节点", len(graph.nodes))

    # 1. 拓扑排序确定基本执行顺序
    topo_order = graph.topo_sort()

    # 更新 execution_order 和 schedule_order
    graph.execution_order = topo_order
    for idx, nid in enumerate(topo_order):
        graph.nodes[nid].schedule_order = idx
        graph.nodes[nid].dependencies = []

    # 2. 遍历相邻算子对，确定依赖关系
    dep_count = 0
    parallel_count = 0

    for i in range(len(topo_order) - 1):
        nid_i = topo_order[i]
        node_i = graph.nodes[nid_i]
        nid_j = topo_order[i + 1]
        node_j = graph.nodes[nid_j]

        if _has_data_dependency(graph, nid_i, nid_j):
            # 数据依赖 → 插入依赖
            node_j.dependencies.append(nid_i)
            dep_count += 1
        elif node_i.compute_unit == node_j.compute_unit:
            # 无数据依赖 + 相同 compute_unit → 串行
            node_j.dependencies.append(nid_i)
            dep_count += 1
        else:
            # 无数据依赖 + 不同 compute_unit → 可并行
            parallel_count += 1

    logger.info("调度完成。依赖关系: %d 条，可并行算子对: %d", dep_count, parallel_count)
    return graph


def post_validate(graph: Graph) -> list[str]:
    """scheduler 后的校验：所有节点有 schedule_order 且 dependencies 非 None。"""
    errors: list[str] = []
    for node in graph.nodes.values():
        if node.schedule_order is None:
            errors.append(f"节点 {node.id} 缺少 schedule_order")
        if node.dependencies is None:
            errors.append(f"节点 {node.id} 的 dependencies 为 None")
    return errors
