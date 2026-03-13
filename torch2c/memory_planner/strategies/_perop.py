"""策略 2：Per-op — L1 liveness 复用 + per-op DMA。"""

from __future__ import annotations

from torch2c.common import Graph, get_logger

from .._dma import DmaPlan, build_dma_plan
from .._hbm_alloc import allocate_hbm, analyze_lifetimes
from .._l1_alloc import allocate_l1_global, build_per_op_l1_layouts

logger = get_logger("memory_planner.strategy")


def strategy_perop(
    graph: Graph,
    l1_align: int,
    l1_cap: int,
    hbm_align: int,
    cube_size: int,
) -> tuple[bool, list[DmaPlan]]:
    """L1 liveness best-fit + HBM lifetime best-fit + per-op DMA。"""
    # HBM 分配
    lifetimes = analyze_lifetimes(graph)
    reuse_count = allocate_hbm(graph, lifetimes, hbm_align, cube_size)

    # L1 全局 liveness 分配
    global_l1 = allocate_l1_global(graph, l1_align, l1_cap, cube_size)
    for tid, off in global_l1.items():
        t = graph.tensors.get(tid)
        if t:
            t.l1_offset = off

    # per-op DMA 计划
    per_op_layouts = build_per_op_l1_layouts(graph, global_l1)
    dma_plans: list[DmaPlan] = []
    for nid, l1_layout in zip(graph.execution_order, per_op_layouts):
        plan = build_dma_plan(graph, nid, l1_layout, cube_size)
        dma_plans.append(plan)

    allocated = sum(1 for t in graph.tensors.values() if t.hbm_offset is not None)
    logger.info(
        "strategy_perop: HBM %d 张量, 复用 %d, DMA %d 算子",
        allocated, reuse_count, len(dma_plans),
    )
    return True, dma_plans
