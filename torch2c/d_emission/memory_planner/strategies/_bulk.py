"""策略 1：Bulk — 所有 tensor 同时放入 L1。"""

from __future__ import annotations

from torch2c.common import Graph, get_logger

from .._dma import DmaPlan, build_bulk_dma
from .._utils import align_up, calc_padded_size
from torch2c.common.sizing import get_dim_align

logger = get_logger("memory_planner.strategy")


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
        offset += calc_padded_size(t.shape, t.dtype, t.format, get_dim_align(t.format, t.dtype))

    if offset > l1_cap:
        return False, []

    logger.info("strategy_bulk: 所有张量适配 L1（%d / %d 字节）", offset, l1_cap)

    # 填充 L1 offset + HBM 线性分配
    hbm_offset = 0
    for tid, l1_off in layout.items():
        t = graph.tensors[tid]
        t.l1_offset = l1_off
        size = calc_padded_size(t.shape, t.dtype, t.format, get_dim_align(t.format, t.dtype))
        if t.storage in ("local", "pipe"):
            continue
        t.hbm_size = size
        t.hbm_offset = hbm_offset
        hbm_offset = align_up(hbm_offset + size, hbm_align)

    dma_plans = build_bulk_dma(graph, cube_size)
    return True, dma_plans
