"""L1 全局 liveness best-fit 分配。"""

from __future__ import annotations

from torch2c.common import Graph, MemoryPlanError, get_logger

from ._utils import align_up, best_fit_alloc, calc_padded_size

logger = get_logger("memory_planner.l1")


def _analyze_l1_lifetimes(graph: Graph) -> dict[str, tuple[int, int]]:
    """分析每个 tensor 的 L1 生命周期 (first_op_idx, last_op_idx)。

    - storage=local 的 tensor：从 producer 到最后一个 consumer（跨算子驻留）。
    - 普通 tensor：仅在使用它的单个算子内活跃（per-op load/compute/store）。
    """
    order_map: dict[str, int] = {}
    for idx, nid in enumerate(graph.execution_order):
        order_map[nid] = idx

    lifetimes: dict[str, tuple[int, int]] = {}
    for tid, t in graph.tensors.items():
        if t.storage == "pipe":
            continue
        if t.storage == "local":
            first = order_map.get(t.producer_node_id, 0) if t.producer_node_id else 0
            consumers = [order_map[c] for c in t.consumer_node_ids if c in order_map]
            # absorbed_inputs 也是消费者
            for nid, node in graph.nodes.items():
                if tid in node.absorbed_inputs.values() and nid in order_map:
                    consumers.append(order_map[nid])
            last = max(consumers) if consumers else first
            lifetimes[tid] = (first, last)
        else:
            ops: set[int] = set()
            if t.producer_node_id and t.producer_node_id in order_map:
                ops.add(order_map[t.producer_node_id])
            for cid in t.consumer_node_ids:
                if cid in order_map:
                    ops.add(order_map[cid])
            for nid, node in graph.nodes.items():
                if tid in node.absorbed_inputs.values() and nid in order_map:
                    ops.add(order_map[nid])
            if ops:
                lifetimes[tid] = (min(ops), max(ops))
    return lifetimes


def collect_op_tensors(graph: Graph, node_id: str) -> list[str]:
    """收集单个算子需要的所有 tensor id（非权重输入 → 权重输入 → absorbed → 输出）。"""
    node = graph.nodes[node_id]
    non_weights: list[str] = []
    weights: list[str] = []
    for tid in node.inputs:
        t = graph.tensors.get(tid)
        if not t:
            continue
        (weights if t.is_weight else non_weights).append(tid)
    tids = non_weights + weights
    for _, atid in sorted(node.absorbed_inputs.items()):
        if atid not in tids:
            tids.append(atid)
    for tid in node.outputs:
        tids.append(tid)
    return tids


def allocate_l1_global(
    graph: Graph,
    l1_alignment: int,
    l1_capacity: int,
    cube_size: int,
) -> dict[str, int]:
    """L1 全局 liveness best-fit 分配，返回 {tensor_id: l1_offset}。"""
    l1_lifetimes = _analyze_l1_lifetimes(graph)
    result: dict[str, int] = {}
    live: dict[str, int] = {}
    alloc_sizes: dict[str, int] = {}
    free_blocks: list[list[int]] = []
    l1_watermark = 0

    for op_idx, nid in enumerate(graph.execution_order):
        # ① 释放 last_op < op_idx 的已分配 tensor
        expired = [tid for tid in live if l1_lifetimes[tid][1] < op_idx]
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

        # ③ 为当前算子需要的 tensor 分配 L1
        op_tids = collect_op_tensors(graph, nid)
        for tid in op_tids:
            if tid in live:
                continue
            t = graph.tensors.get(tid)
            if not t or tid not in l1_lifetimes:
                continue
            size = calc_padded_size(t.shape, t.dtype, t.format, (cube_size, cube_size))
            aligned_size = align_up(size, l1_alignment)

            offset = best_fit_alloc(free_blocks, aligned_size)
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

        # ④ 检查当前 L1 峰值
        if live:
            peak = max(live[t] + alloc_sizes[t] for t in live)
            if peak > l1_capacity:
                raise MemoryPlanError(
                    f"L1 溢出: 节点 {nid} 处峰值 {peak} 字节, 容量 {l1_capacity} 字节"
                )

    return result


def build_per_op_l1_layouts(
    graph: Graph,
    global_l1: dict[str, int],
) -> list[dict[str, int]]:
    """从全局 L1 分配结果中提取每个算子的 l1_layout。"""
    layouts: list[dict[str, int]] = []
    for nid in graph.execution_order:
        op_tids = collect_op_tensors(graph, nid)
        l1_layout = {tid: global_l1[tid] for tid in op_tids if tid in global_l1}
        layouts.append(l1_layout)
    return layouts
