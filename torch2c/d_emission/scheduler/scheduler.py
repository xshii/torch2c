"""scheduler — Pass⑦：计算单元调度与依赖生成。"""

from __future__ import annotations

from torch2c.common import Graph, get_logger
from torch2c.common.opt_log import log_opt

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
        graph: 已完成校验的 Graph IR（内存尚未分配）。
        config: 可选配置（当前未使用）。

    Returns:
        填充了 schedule_order 和 dependencies 的 Graph。
    """
    logger.info("Pass 开始，输入图: %d 个节点", len(graph.nodes))

    # 1. 拓扑排序确定执行顺序（后续 memory_planner 将基于此顺序分配内存）
    topo_order = graph.topo_sort()
    graph.execution_order = topo_order
    for idx, nid in enumerate(topo_order):
        graph.nodes[nid].schedule_order = idx
        graph.nodes[nid].task_id = idx + 1  # globally unique, starting from 1
        graph.nodes[nid].dependencies = []

    # 2. 遍历算子对，确定依赖关系
    #    - 数据依赖：pred 的输出是 succ 的输入
    #    - 结构冒险：同一 compute_unit + 共享 tensor（L1 buffer 冲突）
    dep_count = 0
    parallel_count = 0

    for j in range(len(topo_order)):
        nid_j = topo_order[j]
        node_j = graph.nodes[nid_j]
        j_tids = set(node_j.inputs) | set(node_j.outputs)

        for i in range(j):
            nid_i = topo_order[i]
            node_i = graph.nodes[nid_i]

            if _has_data_dependency(graph, nid_i, nid_j):
                node_j.dependencies.append(nid_i)
                dep_count += 1
            elif (node_i.compute_unit == node_j.compute_unit
                  and (set(node_i.inputs) | set(node_i.outputs)) & j_tids):
                # 同单元 + 共享 tensor → 结构冒险，需串行
                node_j.dependencies.append(nid_i)
                dep_count += 1
            else:
                parallel_count += 1

    # scheduler 不是优化 pass，不记 opt_log

    # 3. 将 dependencies 转换为 per-unit TidInfo deps
    _assign_tid_deps(graph)

    logger.info("调度完成。依赖关系: %d 条，可并行算子对: %d", dep_count, parallel_count)
    return graph


def _assign_tid_deps(graph: Graph) -> None:
    """为每个节点计算 per-unit TidInfo 依赖（存入 node.params）。"""
    unit_keys = {"cube": "_tid_dep_cube", "vector": "_tid_dep_vector",
                 "dma": "_tid_dep_dma", "idma": "_tid_dep_idma"}
    for node in graph.nodes.values():
        deps_by_unit: dict[str, int] = {k: 0 for k in unit_keys}
        for dep_id in node.dependencies:
            dep_node = graph.nodes[dep_id]
            unit = (dep_node.compute_unit or "").lower()
            if unit in deps_by_unit:
                deps_by_unit[unit] = max(deps_by_unit[unit], dep_node.task_id)
        for unit, param_key in unit_keys.items():
            node.params[param_key] = deps_by_unit[unit]


def post_validate(graph: Graph) -> list[str]:
    """scheduler 后的校验：所有节点有 schedule_order 且 dependencies 非 None。"""
    errors: list[str] = []
    for node in graph.nodes.values():
        if node.schedule_order is None:
            errors.append(f"节点 {node.id} 缺少 schedule_order")
        if node.dependencies is None:
            errors.append(f"节点 {node.id} 的 dependencies 为 None")
    return errors
