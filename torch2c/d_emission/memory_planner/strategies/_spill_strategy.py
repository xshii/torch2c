"""策略 3：Spill — selective eviction + 按需 tiling。"""

from __future__ import annotations

from torch2c.common import Graph, MemoryPlanError, get_logger

from .._dma import DmaInstruction, DmaPlan, build_dma_plan
from .._hbm_alloc import allocate_hbm, analyze_lifetimes
from .._l1_alloc import collect_op_tensors
from .._spill import allocate_l1_with_spill
from .._tiling import analyze_tiling
from .._utils import align_up, calc_padded_size
from torch2c.common.sizing import get_dim_align
from ._tiled import _apply_tile_info

logger = get_logger("memory_planner.strategy")


def strategy_spill(
    graph: Graph,
    l1_align: int,
    l1_cap: int,
    hbm_align: int,
    cube_size: int,
    tile_override: dict | None = None,
) -> tuple[bool, list[DmaPlan]]:
    """Selective spill: 只驱逐导致 L1 超限的 tensor，保留可复用的。

    对单算子仍超限的节点做 M 维切分。
    """
    # 1 HBM 分配
    lifetimes = analyze_lifetimes(graph)
    allocate_hbm(graph, lifetimes, hbm_align, cube_size)

    # 2 L1 selective spill 分配
    per_op_layouts, per_op_loads, per_op_spills = allocate_l1_with_spill(
        graph, l1_align, l1_cap, cube_size,
    )

    # 3 检查单算子是否仍超限 -> tiling
    tile_map = analyze_tiling(graph, l1_cap, l1_align, cube_size, tile_override)

    # 3b 检查 overflow 但不可 tile 的节点
    for op_idx, nid in enumerate(graph.execution_order):
        if not per_op_layouts[op_idx] and nid not in tile_map:
            raise MemoryPlanError(
                f"L1 溢出: 节点 {nid} spill 后仍超限且不可 tile"
            )

    # 3c 降级 tiled 节点的 local tensor -> hbm
    demoted_tids: set[str] = set()
    for nid in tile_map:
        node = graph.nodes[nid]
        for tid in list(node.inputs) + list(node.outputs):
            t = graph.tensors.get(tid)
            if t and t.storage == "local":
                t.storage = "hbm"
                demoted_tids.add(tid)
    if demoted_tids:
        lifetimes = analyze_lifetimes(graph)
        allocate_hbm(graph, lifetimes, hbm_align, cube_size)
        logger.info("降级 %d 个 local tensor 为 hbm: %s", len(demoted_tids), demoted_tids)

    # 4 对需要 tiling 的节点，重新构建独立 L1 layout（含多级缓冲）
    for nid in list(tile_map):
        op_idx = graph.execution_order.index(nid)
        tile_info = tile_map[nid]
        l1_offset = 0
        l1_layout: dict[str, int] = {}
        tiled_sizes: dict[str, int] = {}
        for tid in collect_op_tensors(graph, nid):
            t = graph.tensors.get(tid)
            if not t or t.storage == "pipe":
                continue
            l1_offset = align_up(l1_offset, l1_align)
            l1_layout[tid] = l1_offset
            if tid in tile_info.tiled_tensors:
                dim_idx = tile_info.tiled_tensors[tid]
                tiled_shape = list(t.shape)
                tiled_shape[dim_idx] = tile_info.tile_size
                size = calc_padded_size(tiled_shape, t.dtype, t.format, get_dim_align(t.format, t.dtype))
                tiled_sizes[tid] = align_up(size, l1_align)
            else:
                size = calc_padded_size(t.shape, t.dtype, t.format, get_dim_align(t.format, t.dtype))
            l1_offset += align_up(size, l1_align)

        # 分配额外缓冲副本
        extra_l1_offsets: list[dict[str, int]] = []
        for buf_idx in range(1, tile_info.num_buffers):
            buf_offsets: dict[str, int] = {}
            for tid, sz in tiled_sizes.items():
                l1_offset = align_up(l1_offset, l1_align)
                buf_offsets[tid] = l1_offset
                l1_offset += sz
            extra_l1_offsets.append(buf_offsets)

        if l1_offset > l1_cap:
            raise MemoryPlanError(
                f"L1 溢出: 节点 {nid} tiled 后仍超限 ({l1_offset} > {l1_cap})"
            )
        per_op_layouts[op_idx] = l1_layout
        per_op_loads[op_idx] = set(l1_layout.keys())
        for tid, off in l1_layout.items():
            t = graph.tensors.get(tid)
            if t:
                t.l1_offset = off
        tile_info._extra_l1_offsets = extra_l1_offsets

    # 5 生成 DMA 计划
    dma_plans: list[DmaPlan] = []
    for op_idx, nid in enumerate(graph.execution_order):
        l1_layout = per_op_layouts[op_idx]
        loads_needed = per_op_loads[op_idx]
        spills = per_op_spills[op_idx]

        plan = build_dma_plan(graph, nid, l1_layout, cube_size)
        plan.l1_layout = l1_layout

        plan.loads = [i for i in plan.loads if i.tensor_id in loads_needed]

        # 添加 spill store
        for stid in spills:
            t = graph.tensors.get(stid)
            if t and t.hbm_offset is not None:
                plan.stores.insert(0, DmaInstruction(
                    op="store", tensor_id=stid,
                    hbm_offset=t.hbm_offset, l1_offset=t.l1_offset or 0,
                    size_bytes=calc_padded_size(t.shape, t.dtype, t.format, get_dim_align(t.format, t.dtype)),
                    src_format=t.format, dst_format=t.format, dtype=t.dtype,
                ))

        # tiled 节点的处理
        tile_info = tile_map.get(nid)
        if tile_info:
            _apply_tile_info(plan, graph, nid, tile_info, cube_size)

        dma_plans.append(plan)

    total_loads = sum(len(p.loads) for p in dma_plans)
    spill_count = sum(len(s) for s in per_op_spills)
    tiled_count = len(tile_map)
    logger.info(
        "strategy_spill: DMA loads %d, spills %d, tiled %d 个算子",
        total_loads, spill_count, tiled_count,
    )
    return True, dma_plans
