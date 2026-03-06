"""memory_planner — Pass⑦：HBM 全局规划 + L1 全局 liveness best-fit 分配。

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
        # storage=local/pipe 的 tensor 不需要 HBM 分配
        if t.storage in ("local", "pipe"):
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


# ── L1 全局 liveness + best-fit 分配 ─────────────────────


def _analyze_l1_lifetimes(
    graph: Graph,
) -> dict[str, tuple[int, int]]:
    """分析每个 tensor 的 L1 生命周期 (first_op_idx, last_op_idx)。

    - storage=local 的 tensor：从 producer 到最后一个 consumer（跨算子驻留）。
    - 普通 tensor：仅在使用它的单个算子内活跃（per-op load/compute/store）。
      普通输入 tensor 在消费算子处活跃；普通输出 tensor 在生产算子处活跃。
      若 tensor 被多个算子消费（如 weight 或被多次引用），每个算子独立加载，
      但分配器会在不同算子间复用空间。

    返回 {tensor_id: (first_op_idx, last_op_idx)}。
    """
    order_map: dict[str, int] = {}
    for idx, nid in enumerate(graph.execution_order):
        order_map[nid] = idx

    lifetimes: dict[str, tuple[int, int]] = {}
    for tid, t in graph.tensors.items():
        # pipe tensor 走硬件直连，不占 L1
        if t.storage == "pipe":
            continue
        if t.storage == "local":
            # local tensor 跨算子驻留：producer → last consumer
            first = order_map.get(t.producer_node_id, 0) if t.producer_node_id else 0
            if t.consumer_node_ids:
                last = max(order_map[c] for c in t.consumer_node_ids if c in order_map)
            else:
                first = order_map.get(t.producer_node_id, 0) if t.producer_node_id else 0
                last = first
            lifetimes[tid] = (first, last)
        else:
            # 普通 tensor：per-op 活跃。收集所有使用此 tensor 的算子。
            ops: set[int] = set()
            if t.producer_node_id and t.producer_node_id in order_map:
                ops.add(order_map[t.producer_node_id])
            for cid in t.consumer_node_ids:
                if cid in order_map:
                    ops.add(order_map[cid])
            # 也检查 absorbed_inputs
            for nid, node in graph.nodes.items():
                if tid in node.absorbed_inputs.values() and nid in order_map:
                    ops.add(order_map[nid])
            if ops:
                lifetimes[tid] = (min(ops), max(ops))
    return lifetimes


def _collect_op_tensors(graph: Graph, node_id: str) -> list[str]:
    """收集单个算子需要的所有 tensor id（输入 + absorbed + 输出）。"""
    node = graph.nodes[node_id]
    tids: list[str] = []
    # 输入（非 weight 优先，再 weight）
    for tid in node.inputs:
        t = graph.tensors.get(tid)
        if t and not t.is_weight:
            tids.append(tid)
    for tid in node.inputs:
        t = graph.tensors.get(tid)
        if t and t.is_weight:
            tids.append(tid)
    # absorbed_inputs
    for _, atid in sorted(node.absorbed_inputs.items()):
        if atid not in tids:
            tids.append(atid)
    # 输出
    for tid in node.outputs:
        tids.append(tid)
    return tids


def _allocate_l1_global(
    graph: Graph,
    l1_alignment: int,
    l1_capacity: int,
    cube_size: int,
) -> dict[str, int]:
    """L1 全局 liveness best-fit 分配，返回 {tensor_id: l1_offset}。

    按 execution_order 逐算子处理：
    1. 释放当前算子不再需要的 tensor（last_op < current_op）。
    2. 为当前算子的新 tensor 分配 L1 空间（best-fit 复用空闲块）。
    3. 检查 L1 容量约束。
    """
    l1_lifetimes = _analyze_l1_lifetimes(graph)
    result: dict[str, int] = {}          # tid → l1_offset（最终结果，不删除）
    live: dict[str, int] = {}            # tid → l1_offset（当前活跃）
    alloc_sizes: dict[str, int] = {}     # tid → aligned_size（已分配的）
    free_blocks: list[list[int]] = []    # [offset, size]
    l1_watermark = 0

    for op_idx, nid in enumerate(graph.execution_order):
        # ① 释放 last_op < op_idx 的已分配 tensor
        expired = [
            tid for tid in live
            if l1_lifetimes[tid][1] < op_idx
        ]
        for tid in expired:
            free_blocks.append([live[tid], alloc_sizes[tid]])
            del live[tid]
            del alloc_sizes[tid]
            logger.debug("L1 释放: %s (last_op < %d)", tid, op_idx)

        # ② 合并相邻空闲块减少碎片
        if free_blocks:
            free_blocks.sort(key=lambda b: b[0])
            merged: list[list[int]] = [free_blocks[0]]
            for blk in free_blocks[1:]:
                prev = merged[-1]
                if prev[0] + prev[1] == blk[0]:
                    prev[1] += blk[1]
                else:
                    merged.append(blk)
            free_blocks = merged

        # ③ 为当前算子需要的 tensor 分配 L1（跳过已分配的）
        op_tids = _collect_op_tensors(graph, nid)
        for tid in op_tids:
            if tid in live:
                continue  # 已分配（如 local tensor 由 producer 分配）
            t = graph.tensors.get(tid)
            if not t or tid not in l1_lifetimes:
                continue
            size = calc_padded_size(t.shape, t.dtype, t.format, cube_size)
            aligned_size = align_up(size, l1_alignment)

            offset = _best_fit_alloc(free_blocks, aligned_size)
            if offset is not None:
                live[tid] = offset
                result[tid] = offset
                alloc_sizes[tid] = aligned_size
                logger.debug("L1 复用: %s -> offset=%d, size=%d (op=%s)", tid, offset, size, nid)
            else:
                offset = align_up(l1_watermark, l1_alignment)
                live[tid] = offset
                result[tid] = offset
                alloc_sizes[tid] = aligned_size
                l1_watermark = offset + aligned_size
                logger.debug("L1 新分配: %s -> offset=%d, size=%d (op=%s)", tid, offset, size, nid)

        # ④ 检查当前 L1 峰值（已分配 tensor 的最大 end）
        if live:
            peak = max(live[t] + alloc_sizes[t] for t in live)
            if peak > l1_capacity:
                raise MemoryPlanError(
                    f"L1 溢出: 节点 {nid} 处峰值 {peak} 字节, 容量 {l1_capacity} 字节"
                )

    return result


def _build_per_op_l1_layouts(
    graph: Graph,
    global_l1: dict[str, int],
) -> list[dict[str, int]]:
    """从全局 L1 分配结果中提取每个算子的 l1_layout。"""
    layouts: list[dict[str, int]] = []
    for nid in graph.execution_order:
        op_tids = _collect_op_tensors(graph, nid)
        l1_layout = {tid: global_l1[tid] for tid in op_tids if tid in global_l1}
        layouts.append(l1_layout)
    return layouts


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

    # 常规路径：HBM 全局分配 + L1 全局 liveness best-fit 分配
    lifetimes = _analyze_lifetimes(graph)
    reuse_count = _allocate_hbm(graph, lifetimes, hbm_align, cube_size)

    # L1 全局分配
    global_l1 = _allocate_l1_global(graph, l1_align, l1_cap, cube_size)
    for tid, off in global_l1.items():
        t = graph.tensors.get(tid)
        if t:
            t.l1_offset = off

    # 生成 per-op DMA 计划
    per_op_layouts = _build_per_op_l1_layouts(graph, global_l1)
    dma_plans = []
    for nid, l1_layout in zip(graph.execution_order, per_op_layouts):
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
        if t.storage == "pipe":
            # pipe tensor 走硬件直连，不需要 HBM 和 L1
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
