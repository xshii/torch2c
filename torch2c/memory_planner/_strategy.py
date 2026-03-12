"""内存分配策略。

每个策略函数签名：
    (graph, l1_align, l1_cap, hbm_align, cube_size) -> (bool, list[DmaPlan])
    返回 (成功, DMA计划列表)。成功时 graph 上的 tensor 已填充 l1_offset / hbm_offset。
"""

from __future__ import annotations

from torch2c.common import Graph, MemoryPlanError, get_logger

from torch2c.common import dtype_bytes

from ._dma import DmaPlan, DmaInstruction, build_dma_plan, build_bulk_dma
from ._hbm_alloc import allocate_hbm, analyze_lifetimes
from ._l1_alloc import allocate_l1_global, build_per_op_l1_layouts, collect_op_tensors
from ._spill import allocate_l1_with_spill
from ._tiling import analyze_tiling
from ._utils import align_up, calc_padded_size

logger = get_logger("memory_planner.strategy")


def _set_tile_stride(instr: DmaInstruction, t, dim_idx: int, tile_size: int,
                     cube_size: int) -> None:
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
            per_batch_orig, t.dtype, t.format, cube_size)
        instr.l1_batch_stride = calc_padded_size(
            per_batch_tiled, t.dtype, t.format, cube_size)


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


# ── 策略 3：Spill（selective eviction + 按需 tiling）──────


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
    # ① HBM 分配
    lifetimes = analyze_lifetimes(graph)
    allocate_hbm(graph, lifetimes, hbm_align, cube_size)

    # ② L1 selective spill 分配
    per_op_layouts, per_op_loads, per_op_spills = allocate_l1_with_spill(
        graph, l1_align, l1_cap, cube_size,
    )

    # ③ 检查单算子是否仍超限 → tiling
    tile_map = analyze_tiling(graph, l1_cap, l1_align, cube_size, tile_override)

    # ③b 检查 overflow 但不可 tile 的节点（应抛错）
    for op_idx, nid in enumerate(graph.execution_order):
        if not per_op_layouts[op_idx] and nid not in tile_map:
            # 空 layout 且无 tiling 方案 → 真正无法分配
            raise MemoryPlanError(
                f"L1 溢出: 节点 {nid} spill 后仍超限且不可 tile"
            )

    # ③c 降级 tiled 节点的 local tensor → hbm（tiled 独立 L1 layout 无法保持 local 驻留）
    demoted_tids: set[str] = set()
    for nid in tile_map:
        node = graph.nodes[nid]
        for tid in list(node.inputs) + list(node.outputs):
            t = graph.tensors.get(tid)
            if t and t.storage == "local":
                t.storage = "hbm"
                demoted_tids.add(tid)
    if demoted_tids:
        # 重新分配 HBM（demoted tensors 需要 HBM 空间）
        lifetimes = analyze_lifetimes(graph)
        allocate_hbm(graph, lifetimes, hbm_align, cube_size)
        logger.info("降级 %d 个 local tensor 为 hbm: %s", len(demoted_tids), demoted_tids)

    # ④ 对需要 tiling 的节点，重新构建独立 L1 layout（含多级缓冲）
    for nid in list(tile_map):
        op_idx = graph.execution_order.index(nid)
        tile_info = tile_map[nid]
        l1_offset = 0
        l1_layout: dict[str, int] = {}
        # 记录每个 tiled tensor 的单份大小，用于分配额外缓冲
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
                size = calc_padded_size(tiled_shape, t.dtype, t.format, cube_size)
                tiled_sizes[tid] = align_up(size, l1_align)
            else:
                size = calc_padded_size(t.shape, t.dtype, t.format, cube_size)
            l1_offset += align_up(size, l1_align)

        # 分配额外缓冲副本（buf 1 = ping 已在上面分配，buf 2..N 追加）
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
        per_op_loads[op_idx] = set(l1_layout.keys())  # tiled op 全量加载
        # 写 l1_offset 到 tensor
        for tid, off in l1_layout.items():
            t = graph.tensors.get(tid)
            if t:
                t.l1_offset = off
        # 保存额外缓冲偏移供后续写入 tile_info
        tile_info._extra_l1_offsets = extra_l1_offsets

    # ⑤ 生成 DMA 计划
    dma_plans: list[DmaPlan] = []
    for op_idx, nid in enumerate(graph.execution_order):
        l1_layout = per_op_layouts[op_idx]
        loads_needed = per_op_loads[op_idx]
        spills = per_op_spills[op_idx]

        plan = build_dma_plan(graph, nid, l1_layout, cube_size)
        plan.l1_layout = l1_layout

        # 过滤掉不需要 load 的（已经驻留在 L1 的 tensor）
        plan.loads = [i for i in plan.loads if i.tensor_id in loads_needed]

        # 添加 spill store（被驱逐的 dirty tensor）
        for stid in spills:
            t = graph.tensors.get(stid)
            if t and t.hbm_offset is not None:
                plan.stores.insert(0, DmaInstruction(
                    op="store", tensor_id=stid,
                    hbm_offset=t.hbm_offset, l1_offset=t.l1_offset or 0,
                    size_bytes=calc_padded_size(t.shape, t.dtype, t.format, cube_size),
                    src_format=t.format, dst_format=t.format, dtype=t.dtype,
                ))

        # tiled 节点的处理
        tile_info = tile_map.get(nid)
        if tile_info:
            node = graph.nodes[nid]
            ti_dict: dict = {
                "tile_dim": tile_info.tile_dim,
                "tile_size": tile_info.tile_size,
                "num_tiles": tile_info.num_tiles,
                "original_size": tile_info.original_size,
                "tiled_tensors": tile_info.tiled_tensors,
                "num_buffers": tile_info.num_buffers,
            }
            if tile_info.num_buffers > 1:
                ti_dict["extra_l1_offsets"] = tile_info._extra_l1_offsets
            node.params["_tile_info"] = ti_dict
            for instr in plan.loads + plan.stores:
                if instr.tensor_id in tile_info.tiled_tensors:
                    t = graph.tensors[instr.tensor_id]
                    dim_idx = tile_info.tiled_tensors[instr.tensor_id]
                    tiled_shape = list(t.shape)
                    tiled_shape[dim_idx] = tile_info.tile_size
                    instr.size_bytes = calc_padded_size(
                        tiled_shape, t.dtype, t.format, cube_size,
                    )
                    _set_tile_stride(instr, t, dim_idx, tile_info.tile_size, cube_size)
            plan.tile_info = node.params["_tile_info"]

        dma_plans.append(plan)

    total_loads = sum(len(p.loads) for p in dma_plans)
    spill_count = sum(len(s) for s in per_op_spills)
    tiled_count = len(tile_map)
    logger.info(
        "strategy_spill: DMA loads %d, spills %d, tiled %d 个算子",
        total_loads, spill_count, tiled_count,
    )
    return True, dma_plans


# ── 策略 4：Tiled（per-op eviction + M 维切分）────────────


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
    # ⓪ 降级 local → hbm，确保 eviction 模型正确
    for t in graph.tensors.values():
        if t.storage == "local":
            t.storage = "hbm"

    # ① HBM 分配（全量 tensor，lifetime 复用）
    lifetimes = analyze_lifetimes(graph)
    allocate_hbm(graph, lifetimes, hbm_align, cube_size)

    # ② 分析需要 tiling 的节点
    tile_map = analyze_tiling(graph, l1_cap, l1_align, cube_size, tile_override)

    # ③ per-op eviction L1 分配 + DMA 计划
    dma_plans: list[DmaPlan] = []
    for nid in graph.execution_order:
        node = graph.nodes[nid]
        tile_info = tile_map.get(nid)

        # 为当前算子构建独立 L1 layout（从 0 开始）
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
                size = calc_padded_size(tiled_shape, t.dtype, t.format, cube_size)
                tiled_sizes[tid] = align_up(size, l1_align)
            else:
                size = calc_padded_size(t.shape, t.dtype, t.format, cube_size)
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

        # 写 l1_offset 到 tensor（后写覆盖先写）
        for tid, off in l1_layout.items():
            t = graph.tensors.get(tid)
            if t:
                t.l1_offset = off

        # 构建 DMA 计划
        plan = build_dma_plan(graph, nid, l1_layout, cube_size)
        plan.l1_layout = l1_layout  # 记录 per-op L1 layout 供 codegen 使用

        # 为 tiled 算子调整 DMA 指令的 size + 记录 tile_stride
        if tile_info:
            ti_dict: dict = {
                "tile_dim": tile_info.tile_dim,
                "tile_size": tile_info.tile_size,
                "num_tiles": tile_info.num_tiles,
                "original_size": tile_info.original_size,
                "tiled_tensors": tile_info.tiled_tensors,
                "num_buffers": tile_info.num_buffers,
            }
            if tile_info.num_buffers > 1:
                ti_dict["extra_l1_offsets"] = extra_l1_offsets
            node.params["_tile_info"] = ti_dict
            for instr in plan.loads + plan.stores:
                if instr.tensor_id in tile_info.tiled_tensors:
                    t = graph.tensors[instr.tensor_id]
                    dim_idx = tile_info.tiled_tensors[instr.tensor_id]
                    tiled_shape = list(t.shape)
                    tiled_shape[dim_idx] = tile_info.tile_size
                    instr.size_bytes = calc_padded_size(
                        tiled_shape, t.dtype, t.format, cube_size,
                    )
                    _set_tile_stride(instr, t, dim_idx, tile_info.tile_size, cube_size)
            plan.tile_info = node.params["_tile_info"]

        dma_plans.append(plan)

    tiled_count = len(tile_map)
    logger.info("strategy_tiled: %d 个算子需要 tiling", tiled_count)
    return True, dma_plans
