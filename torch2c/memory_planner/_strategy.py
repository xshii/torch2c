"""内存分配策略。

每个策略函数签名：
    (graph, l1_align, l1_cap, hbm_align, cube_size) -> (bool, list[DmaPlan])
    返回 (成功, DMA计划列表)。成功时 graph 上的 tensor 已填充 l1_offset / hbm_offset。
"""

from __future__ import annotations

from torch2c.common import Graph, get_logger

from ._dma import DmaPlan, DmaInstruction, build_dma_plan, build_bulk_dma
from ._hbm_alloc import allocate_hbm, analyze_lifetimes
from ._l1_alloc import allocate_l1_global, build_per_op_l1_layouts
from ._utils import align_up, calc_padded_size

logger = get_logger("memory_planner.strategy")


# ── 策略 1：Bulk（全部 tensor 同时放 L1）──────────────────


def strategy_bulk(
    graph: Graph,
    l1_align: int,
    l1_cap: int,
    hbm_align: int,
    cube_size: int,
) -> tuple[bool, list[DmaPlan]]:
    """所有 tensor 同时放入 L1，HBM 线性分配，bulk DMA。"""
    # L1 简单累加
    offset = 0
    layout: dict[str, int] = {}
    for tid, t in graph.tensors.items():
        if t.storage == "pipe":
            continue
        offset = align_up(offset, l1_align)
        layout[tid] = offset
        offset += calc_padded_size(t.shape, t.dtype, t.format, cube_size)

    if offset > l1_cap:
        return False, []

    logger.info("strategy_bulk: 所有张量适配 L1（%d / %d 字节）", offset, l1_cap)

    # 填充 L1 offset + HBM 线性分配
    hbm_offset = 0
    for tid, l1_off in layout.items():
        t = graph.tensors[tid]
        t.l1_offset = l1_off
        size = calc_padded_size(t.shape, t.dtype, t.format, cube_size)
        if t.storage in ("local", "pipe"):
            continue
        t.hbm_size = size
        t.hbm_offset = hbm_offset
        hbm_offset = align_up(hbm_offset + size, hbm_align)

    dma_plans = build_bulk_dma(graph, cube_size)
    return True, dma_plans


# ── 策略 2：Per-op（L1 liveness 复用 + per-op DMA）────────


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
