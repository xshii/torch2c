"""策略 2：Per-op — L1 驻留感知 DMA + liveness 复用。

核心优化（L1 Residency-Aware DMA Elimination）：
  - Skip Load: 已在 L1 中的 tensor 不重新从 HBM 加载
  - Lazy Store: 所有 consumer 都在 L1 lifetime 内的中间 tensor 不写回 HBM
  - 维护 l1_resident 集合，跟踪当前在 L1 中的 tensor
"""

from __future__ import annotations

from torch2c.common import Graph, get_logger

from .._dma import DmaPlan, build_dma_plan_residency
from .._hbm_alloc import allocate_hbm, analyze_lifetimes
from .._l1_alloc import allocate_l1_global, analyze_l1_lifetimes, build_per_op_l1_layouts

logger = get_logger("memory_planner.strategy")


def strategy_perop(
    graph: Graph,
    l1_align: int,
    l1_cap: int,
    hbm_align: int,
    cube_size: int,
) -> tuple[bool, list[DmaPlan]]:
    """L1 驻留感知 DMA + liveness best-fit + HBM lifetime best-fit。"""
    # HBM 分配
    lifetimes = analyze_lifetimes(graph)
    reuse_count = allocate_hbm(graph, lifetimes, hbm_align, cube_size)

    # L1 全局 liveness 分配
    global_l1 = allocate_l1_global(graph, l1_align, l1_cap, cube_size)
    for tid, off in global_l1.items():
        t = graph.tensors.get(tid)
        if t:
            t.l1_offset = off

    # L1 生命周期 + 驻留集合
    l1_lifetimes = analyze_l1_lifetimes(graph)
    l1_resident: set[str] = set()

    # per-op DMA 计划（驻留感知）
    per_op_layouts = build_per_op_l1_layouts(graph, global_l1)
    dma_plans: list[DmaPlan] = []
    total_skipped_loads = 0
    total_skipped_stores = 0

    for op_idx, (nid, l1_layout) in enumerate(zip(graph.execution_order, per_op_layouts)):
        # 释放 L1 lifetime 已过期的 tensor
        expired = [tid for tid in l1_resident if l1_lifetimes.get(tid, (0, -1))[1] < op_idx]
        for tid in expired:
            l1_resident.discard(tid)

        # 统计优化前的指令数（用于日志）
        old_resident_size = len(l1_resident)

        plan = build_dma_plan_residency(
            graph, nid, l1_layout, cube_size,
            l1_resident, l1_lifetimes, op_idx,
        )
        dma_plans.append(plan)

        # 统计跳过的指令
        new_resident_size = len(l1_resident)
        node = graph.nodes[nid]
        n_inputs = len(node.inputs) + len(node.absorbed_inputs)
        n_outputs = len(node.outputs)
        actual_loads = len(plan.loads)
        actual_stores = len(plan.stores)
        total_skipped_loads += n_inputs - actual_loads
        total_skipped_stores += n_outputs - actual_stores

    allocated = sum(1 for t in graph.tensors.values() if t.hbm_offset is not None)
    total_loads = sum(len(p.loads) for p in dma_plans)
    total_stores = sum(len(p.stores) for p in dma_plans)
    logger.info(
        "strategy_perop: HBM %d 张量, 复用 %d, DMA %d loads + %d stores "
        "(eliminated: %d loads, %d stores)",
        allocated, reuse_count, total_loads, total_stores,
        total_skipped_loads, total_skipped_stores,
    )
    return True, dma_plans
