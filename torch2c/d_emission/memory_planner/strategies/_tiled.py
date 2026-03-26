"""策略 4：Tiled — per-op eviction + M 维切分。

包含 _set_tile_stride / _apply_tile_info 辅助函数（也被 _spill_strategy 使用）。
"""

from __future__ import annotations

from torch2c.common import Graph, MemoryPlanError, dtype_bytes, get_logger

from .._dma import DmaInstruction, DmaPlan, build_dma_plan
from .._hbm_alloc import allocate_hbm, analyze_lifetimes
from .._l1_alloc import collect_op_tensors
from .._tiling import analyze_tiling
from .._utils import align_up, calc_padded_size
from torch2c.common.sizing import get_dim_align

logger = get_logger("memory_planner.strategy")


# ── 辅助：tile stride 设置 ──────────────────────────────────


def _set_tile_stride(
    instr: DmaInstruction, t, dim_idx: int, tile_size: int, cube_size: int,
) -> None:
    """设置 tiled DMA 的步长信息，含 batch > 1 的 per-batch stride。"""
    inner = 1
    for d in t.shape[dim_idx + 1:]:
        inner *= d
    instr.tile_stride = tile_size * inner * dtype_bytes(t.dtype)

    # batch > 1 且 tiled dim 不是 dim 0 → HBM 非连续，需 per-batch DMA
    batch_count = 1
    for d in t.shape[:dim_idx]:
        batch_count *= d
    if batch_count > 1:
        per_batch_orig = list(t.shape[dim_idx:])
        per_batch_tiled = list(per_batch_orig)
        per_batch_tiled[0] = tile_size
        instr.batch_count = batch_count
        instr.hbm_batch_stride = calc_padded_size(
            per_batch_orig, t.dtype, t.format, get_dim_align(t.format, t.dtype))
        instr.l1_batch_stride = calc_padded_size(
            per_batch_tiled, t.dtype, t.format, get_dim_align(t.format, t.dtype))


# ── 共用辅助：tile info 写入 DMA plan ──────────────────────


def _apply_tile_info(
    plan: DmaPlan, graph: Graph, nid: str, tile_info, cube_size: int,
    *, extra_l1_offsets: list[dict[str, int]] | None = None,
) -> None:
    """将 tile_info 写入 node.params 并调整 DMA 指令的 size/stride。"""
    node = graph.nodes[nid]
    ti_dict: dict = {
        "tile_dim": tile_info.tile_dim,
        "tile_size": tile_info.tile_size,
        "num_tiles": tile_info.num_tiles,
        "original_size": tile_info.original_size,
        "tiled_tensors": tile_info.tiled_tensors,
        "num_buffers": tile_info.num_buffers,
        "pipe_group": tile_info.pipe_group,
    }
    elo = extra_l1_offsets
    if elo is None and hasattr(tile_info, "_extra_l1_offsets"):
        elo = tile_info._extra_l1_offsets
    if tile_info.num_buffers > 1 and elo:
        ti_dict["extra_l1_offsets"] = elo
    node.params["_tile_info"] = ti_dict

    for instr in plan.loads + plan.stores:
        if instr.tensor_id in tile_info.tiled_tensors:
            t = graph.tensors[instr.tensor_id]
            dim_idx = tile_info.tiled_tensors[instr.tensor_id]
            tiled_shape = list(t.shape)
            tiled_shape[dim_idx] = tile_info.tile_size
            instr.size_bytes = calc_padded_size(
                tiled_shape, t.dtype, t.format, get_dim_align(t.format, t.dtype),
            )
            _set_tile_stride(instr, t, dim_idx, tile_info.tile_size, cube_size)
    plan.tile_info = node.params["_tile_info"]


# ── 策略 4：Tiled ──────────────────────────────────────────


def strategy_tiled(
    graph: Graph,
    l1_align: int,
    l1_cap: int,
    hbm_align: int,
    cube_size: int,
    tile_override: dict | None = None,
) -> tuple[bool, list[DmaPlan]]:
    """Per-op eviction + M 维 tiling。

    每个算子独立分配 L1（eviction 模型），
    对单算子仍超限的节点做 M 维切分。
    storage=local 的 tensor 降级为 hbm（eviction 需要完全释放 L1）。
    """
    # 0 降级 local -> hbm
    for t in graph.tensors.values():
        if t.storage == "local":
            t.storage = "hbm"

    # 1 HBM 分配
    lifetimes = analyze_lifetimes(graph)
    allocate_hbm(graph, lifetimes, hbm_align, cube_size)

    # 2 分析需要 tiling 的节点
    tile_map = analyze_tiling(graph, l1_cap, l1_align, cube_size, tile_override)

    # 3 per-op eviction L1 分配 + DMA 计划
    dma_plans: list[DmaPlan] = []
    for nid in graph.execution_order:
        tile_info = tile_map.get(nid)

        l1_offset = 0
        l1_layout: dict[str, int] = {}
        tiled_sizes: dict[str, int] = {}
        for tid in collect_op_tensors(graph, nid):
            t = graph.tensors.get(tid)
            if not t or t.storage == "pipe":
                continue
            l1_offset = align_up(l1_offset, l1_align)
            l1_layout[tid] = l1_offset

            if tile_info and tid in tile_info.tiled_tensors:
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
        if tile_info and tile_info.num_buffers > 1:
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

        for tid, off in l1_layout.items():
            t = graph.tensors.get(tid)
            if t:
                t.l1_offset = off

        plan = build_dma_plan(graph, nid, l1_layout, cube_size)
        plan.l1_layout = l1_layout

        if tile_info:
            _apply_tile_info(plan, graph, nid, tile_info, cube_size,
                             extra_l1_offsets=extra_l1_offsets)

        dma_plans.append(plan)

    tiled_count = len(tile_map)
    logger.info("strategy_tiled: %d 个算子需要 tiling", tiled_count)
    return True, dma_plans
