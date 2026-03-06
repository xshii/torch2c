"""memory_planner — Pass⑦：HBM 全局规划 + L1 局部排列。

DMA 计划生成逻辑见 _dma.py，工具函数见 _utils.py。
"""

from __future__ import annotations

from npu_compiler.common import Graph, MemoryPlanError, get_logger

from ._dma import DmaInstruction, DmaPlan, build_bulk_dma, build_dma_plan, try_global_l1_layout
from ._utils import align_up, calc_padded_size

# re-export for backward compat
__all__ = ["DmaInstruction", "DmaPlan", "align_up", "calc_padded_size", "post_validate", "run"]

logger = get_logger("memory_planner")


# ── HBM 分配 ─────────────────────────────────────────────


def _analyze_lifetimes(
    graph: Graph,
) -> dict[str, tuple[int, int]]:
    """分析每个 tensor 的生命周期 (first_use, last_use)。

    对于 weight/model_input，first_use=0；对于 model_output，last_use=max_order。
    无消费者且非 model_output 的 tensor 不分配 HBM（如 layernorm mean/rstd）。
    返回 {tensor_id: (first_use, last_use)}，跳过不需要分配的 tensor。
    """
    order_map: dict[str, int] = {}
    for idx, nid in enumerate(graph.execution_order):
        order_map[nid] = idx
    max_order = len(graph.execution_order) - 1

    lifetimes: dict[str, tuple[int, int]] = {}
    for tid, t in graph.tensors.items():
        # storage=local 的 tensor 不需要 HBM 分配
        if t.storage == "local":
            continue

        # 确定 first_use
        if t.is_weight or t.is_model_input:
            first_use = 0
        elif t.producer_node_id and t.producer_node_id in order_map:
            first_use = order_map[t.producer_node_id]
        else:
            first_use = 0

        # 确定 last_use
        if t.is_model_output:
            last_use = max_order
        elif t.consumer_node_ids:
            consumer_orders = [order_map[c] for c in t.consumer_node_ids if c in order_map]
            if consumer_orders:
                last_use = max(consumer_orders)
            else:
                continue  # 无有效消费者，跳过
        elif t.is_weight or t.is_model_input:
            # weight/input 被所有使用它的算子消费
            last_use = max_order
        else:
            continue  # 无消费者且非 model_output，不分配

        lifetimes[tid] = (first_use, last_use)
    return lifetimes


def _release_expired(
    graph: Graph,
    sorted_tids: list[str],
    lifetimes: dict[str, tuple[int, int]],
    current_first: int,
    freed_set: set[str],
    free_blocks: list[list[int]],
    hbm_alignment: int,
) -> None:
    """释放 lifetime 结束时间早于 current_first 的 tensor 到 free_blocks。"""
    for other_tid in sorted_tids:
        if other_tid in freed_set:
            continue
        other_t = graph.tensors[other_tid]
        if other_t.hbm_offset is None or other_t.hbm_size is None:
            continue
        if lifetimes[other_tid][1] < current_first:
            free_blocks.append([other_t.hbm_offset, align_up(other_t.hbm_size, hbm_alignment)])
            freed_set.add(other_tid)


def _best_fit_alloc(free_blocks: list[list[int]], aligned_size: int) -> int | None:
    """从 free_blocks 找最优空闲块，返回 offset 或 None。"""
    best_idx = -1
    best_fit_size = float("inf")
    for i, (_, sz) in enumerate(free_blocks):
        if sz >= aligned_size and sz < best_fit_size:
            best_idx = i
            best_fit_size = sz

    if best_idx < 0:
        return None

    blk_off, blk_sz = free_blocks[best_idx]
    remaining = blk_sz - aligned_size
    if remaining > 0:
        free_blocks[best_idx] = [blk_off + aligned_size, remaining]
    else:
        free_blocks.pop(best_idx)
    return blk_off


def _allocate_hbm(
    graph: Graph,
    lifetimes: dict[str, tuple[int, int]],
    hbm_alignment: int,
    cube_size: int,
) -> int:
    """Best-fit HBM 分配，支持空间复用。返回实际复用次数。"""
    sorted_tids = sorted(lifetimes.keys(), key=lambda t: lifetimes[t][0])
    free_blocks: list[list[int]] = []
    freed_set: set[str] = set()
    hbm_watermark = 0
    reuse_count = 0

    for tid in sorted_tids:
        t = graph.tensors[tid]
        size = calc_padded_size(t.shape, t.dtype, t.format, cube_size)
        t.hbm_size = size
        aligned_size = align_up(size, hbm_alignment)

        _release_expired(graph, sorted_tids, lifetimes, lifetimes[tid][0], freed_set, free_blocks, hbm_alignment)

        offset = _best_fit_alloc(free_blocks, aligned_size)
        if offset is not None:
            t.hbm_offset = offset
            reuse_count += 1
            logger.debug("HBM 复用: %s -> offset=%d, size=%d", tid, offset, size)
        else:
            t.hbm_offset = hbm_watermark
            hbm_watermark = t.hbm_offset + aligned_size
            logger.debug("HBM 新分配: %s -> offset=%d, size=%d", tid, t.hbm_offset, size)

    return reuse_count


# ── L1 布局 ──────────────────────────────────────────────


def _plan_l1_layout(
    graph: Graph,
    node_id: str,
    l1_alignment: int,
    l1_capacity: int,
    cube_size: int,
    local_l1_offsets: dict[str, int] | None = None,
) -> dict[str, int]:
    """为单个算子规划 L1 布局，返回 {tensor_id: l1_offset}。

    Args:
        local_l1_offsets: storage=local 的 tensor 已由 producer 分配的 L1 偏移。
            这些 tensor 作为输入时直接复用该偏移，不再重新分配。
    """
    if local_l1_offsets is None:
        local_l1_offsets = {}
    node = graph.nodes[node_id]
    offset = 0
    layout: dict[str, int] = {}

    # storage=local 的输入 tensor：复用 producer 分配的 L1 偏移，不占新空间
    for tid in node.inputs:
        t = graph.tensors.get(tid)
        if t and t.storage == "local" and tid in local_l1_offsets:
            layout[tid] = local_l1_offsets[tid]

    # 输入 tensor（非 weight，非 local）
    for tid in node.inputs:
        t = graph.tensors.get(tid)
        if t and not t.is_weight and t.storage != "local":
            offset = align_up(offset, l1_alignment)
            layout[tid] = offset
            offset += calc_padded_size(t.shape, t.dtype, t.format, cube_size)

    # 权重 tensor
    for tid in node.inputs:
        t = graph.tensors.get(tid)
        if t and t.is_weight:
            offset = align_up(offset, l1_alignment)
            layout[tid] = offset
            offset += calc_padded_size(t.shape, t.dtype, t.format, cube_size)

    # absorbed_inputs 中的 tensor
    for _, atid in sorted(node.absorbed_inputs.items()):
        t = graph.tensors.get(atid)
        if t:
            offset = align_up(offset, l1_alignment)
            layout[atid] = offset
            offset += calc_padded_size(t.shape, t.dtype, t.format, cube_size)

    # 输出 tensor
    for tid in node.outputs:
        t = graph.tensors.get(tid)
        if t:
            offset = align_up(offset, l1_alignment)
            layout[tid] = offset
            offset += calc_padded_size(t.shape, t.dtype, t.format, cube_size)

    total = align_up(offset, l1_alignment)
    if total > l1_capacity:
        raise MemoryPlanError(f"L1 溢出: 节点 {node_id} 需要 {total} 字节, 容量 {l1_capacity} 字节")
    return layout


# ── 主入口 ───────────────────────────────────────────────


def run(graph: Graph, config: dict) -> tuple[Graph, list[DmaPlan]]:
    """内存编排主函数。

    Args:
        graph: 校验通过的 Graph IR。
        config: hardware_config.yaml 解析后的字典。

    Returns:
        (编排后的 Graph, DMA 计划列表)
    """
    logger.info("Pass 开始，输入图: %d 个节点, %d 条张量", len(graph.nodes), len(graph.tensors))

    mem = config["memory"]
    hbm_align = mem["hbm"]["alignment_bytes"]
    l1_align = mem["l1"]["alignment_bytes"]
    l1_cap = mem["l1"]["total_size_bytes"]
    cube_size = config["fractal"]["cube_size"]

    if not graph.execution_order:
        graph.execution_order = graph.topo_sort()

    # 尝试全局 L1 布局：如果所有 tensor 同时放得下，跳过 per-op DMA
    if try_global_l1_layout(graph, l1_align, l1_cap, cube_size, hbm_align):
        dma_plans = build_bulk_dma(graph, cube_size)
        allocated = sum(1 for t in graph.tensors.values() if t.hbm_offset is not None)
        logger.info(
            "Pass 完成（L1 全局布局）。HBM 分配: %d 个张量, DMA: bulk load/store",
            allocated,
        )
        return graph, dma_plans

    # 常规路径：HBM 全局分配 + per-op L1 布局
    lifetimes = _analyze_lifetimes(graph)
    reuse_count = _allocate_hbm(graph, lifetimes, hbm_align, cube_size)

    # 跟踪 storage=local 的 tensor 的 L1 偏移（producer 写入，consumer 复用）
    local_l1_offsets: dict[str, int] = {}

    dma_plans = []
    for nid in graph.execution_order:
        l1_layout = _plan_l1_layout(graph, nid, l1_align, l1_cap, cube_size, local_l1_offsets)
        for tid, off in l1_layout.items():
            t = graph.tensors.get(tid)
            if t:
                t.l1_offset = off
                # 输出 tensor 且 storage=local → 记录偏移供下游复用
                if t.storage == "local" and tid in graph.nodes[nid].outputs:
                    local_l1_offsets[tid] = off
        plan = build_dma_plan(graph, nid, l1_layout, cube_size)
        dma_plans.append(plan)
        logger.debug("节点 %s: %d loads, %d stores", nid, len(plan.loads), len(plan.stores))

    allocated = sum(1 for t in graph.tensors.values() if t.hbm_offset is not None)
    logger.info(
        "Pass 完成。HBM 分配: %d 个张量, 复用: %d, DMA 计划: %d 个算子",
        allocated,
        reuse_count,
        len(dma_plans),
    )
    return graph, dma_plans


def post_validate(graph: Graph) -> list[str]:
    """memory_planner 后的校验：有消费者或是输出的 tensor 必须有内存偏移。

    storage=local 的 tensor 不需要 HBM 偏移，只需要 L1 偏移。
    """
    errors: list[str] = []
    for t in graph.tensors.values():
        needs_mem = t.consumer_node_ids or t.is_model_output
        if not needs_mem:
            continue
        if t.storage == "local":
            # local tensor 只需要 l1_offset
            if t.l1_offset is None:
                errors.append(f"tensor {t.id} (storage=local) 缺少 l1_offset")
            continue
        if t.hbm_offset is None:
            errors.append(f"tensor {t.id} 缺少 hbm_offset")
        if t.hbm_size is None:
            errors.append(f"tensor {t.id} 缺少 hbm_size")
        if t.l1_offset is None:
            errors.append(f"tensor {t.id} 缺少 l1_offset")
    return errors
