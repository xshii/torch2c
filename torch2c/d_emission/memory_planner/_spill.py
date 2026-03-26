"""Selective spill: L1 liveness 分配 + 按需驱逐。

与全量 eviction 不同，selective spill 只驱逐导致 L1 超限的 tensor，
保留仍被后续算子使用的 tensor 以减少 DMA 搬运。

算法: Belady's optimal — 驱逐下次使用最远的 tensor。
"""

from __future__ import annotations

from torch2c.common import Graph, MemoryPlanError, get_logger

from ._l1_alloc import collect_op_tensors
from ._utils import align_up, best_fit_alloc, calc_padded_size

logger = get_logger("memory_planner.spill")


def _build_use_map(graph: Graph) -> dict[str, list[int]]:
    """构建 tensor_id → 使用它的 op index 列表（升序）。"""
    use_map: dict[str, list[int]] = {}
    for op_idx, nid in enumerate(graph.execution_order):
        for tid in collect_op_tensors(graph, nid):
            use_map.setdefault(tid, []).append(op_idx)
    return use_map


def _next_use(use_map: dict[str, list[int]], tid: str, after: int) -> int:
    """返回 tensor 在 after 之后的下次使用 op_idx，无则返回 inf。"""
    ops = use_map.get(tid, [])
    for op_idx in ops:
        if op_idx > after:
            return op_idx
    return 999999


def _merge_free_blocks(free_blocks: list[list[int]]) -> list[list[int]]:
    """合并相邻空闲块。"""
    if len(free_blocks) <= 1:
        return free_blocks
    free_blocks.sort(key=lambda b: b[0])
    merged: list[list[int]] = [free_blocks[0]]
    for blk in free_blocks[1:]:
        prev = merged[-1]
        if prev[0] + prev[1] == blk[0]:
            prev[1] += blk[1]
        else:
            merged.append(blk)
    return merged


def _tensor_l1_size(graph: Graph, tid: str, cube_size: int, l1_align: int) -> int:
    """计算 tensor 的 L1 对齐大小。"""
    t = graph.tensors[tid]
    return align_up(calc_padded_size(t.shape, t.dtype, t.format, (cube_size, cube_size)), l1_align)


def _select_victims(
    graph: Graph,
    resident: dict[str, int],
    resident_sizes: dict[str, int],
    needed_tids: set[str],
    overflow: int,
    use_map: dict[str, list[int]],
    current_op: int,
) -> list[str]:
    """Belady 最优替换: 选择驱逐目标，优先驱逐下次使用最远的。

    只从 resident 中不在 needed_tids 的 tensor 中选择。
    返回足够释放 overflow 字节的 victim 列表。
    """
    candidates = [
        (tid, _next_use(use_map, tid, current_op), resident_sizes[tid])
        for tid in resident
        if tid not in needed_tids
        and graph.tensors[tid].storage != "local"  # local 不可驱逐
    ]
    # 按下次使用距离降序，同距离按大小降序
    candidates.sort(key=lambda x: (-x[1], -x[2]))

    victims: list[str] = []
    freed = 0
    for tid, _, size in candidates:
        victims.append(tid)
        freed += size
        if freed >= overflow:
            break
    return victims


def allocate_l1_with_spill(
    graph: Graph,
    l1_align: int,
    l1_cap: int,
    cube_size: int,
) -> tuple[list[dict[str, int]], list[set[str]], list[set[str]]]:
    """L1 selective spill 分配。

    Returns:
        (per_op_layouts, per_op_loads, per_op_spills)
        - per_op_layouts: 每个 op 的 {tid: l1_offset}
        - per_op_loads: 每个 op 需要从 HBM load 的 tensor id 集合
        - per_op_spills: 每个 op 开始前需要 spill 回 HBM 的 tensor id 集合
    """
    use_map = _build_use_map(graph)

    resident: dict[str, int] = {}       # tid -> l1_offset
    resident_sizes: dict[str, int] = {}  # tid -> aligned_size
    dirty: set[str] = set()              # 产出后未 store 的 tensor
    free_blocks: list[list[int]] = []
    watermark = 0

    per_op_layouts: list[dict[str, int]] = []
    per_op_loads: list[set[str]] = []
    per_op_spills: list[set[str]] = []

    for op_idx, nid in enumerate(graph.execution_order):
        needed_tids = set()
        for tid in collect_op_tensors(graph, nid):
            t = graph.tensors.get(tid)
            if t and t.storage != "pipe":
                needed_tids.add(tid)

        # ① 释放生命周期已结束的 tensor（不再被任何后续 op 使用）
        expired = [
            tid for tid in list(resident)
            if tid not in needed_tids and _next_use(use_map, tid, op_idx - 1) > op_idx
            and _next_use(use_map, tid, op_idx) == 999999
        ]
        for tid in expired:
            free_blocks.append([resident[tid], resident_sizes[tid]])
            del resident[tid]
            del resident_sizes[tid]
            dirty.discard(tid)
        free_blocks = _merge_free_blocks(free_blocks)

        # ② 计算新增 tensor 的空间需求
        new_tids = [tid for tid in needed_tids if tid not in resident]
        new_total = sum(_tensor_l1_size(graph, tid, cube_size, l1_align) for tid in new_tids)
        current_used = sum(resident_sizes.values())
        projected = current_used + new_total
        logger.debug(
            "op %d (%s): resident_used=%d new_total=%d projected=%d",
            op_idx, nid, current_used, new_total, projected,
        )

        # ③ 如果超限，选择性驱逐
        spilled: set[str] = set()
        if projected > l1_cap:
            overflow = projected - l1_cap
            victims = _select_victims(
                graph, resident, resident_sizes, needed_tids,
                overflow, use_map, op_idx,
            )
            for tid in victims:
                free_blocks.append([resident[tid], resident_sizes[tid]])
                del resident[tid]
                del resident_sizes[tid]
                if tid in dirty:
                    spilled.add(tid)
                    dirty.discard(tid)
            free_blocks = _merge_free_blocks(free_blocks)

        # ③b 检查 spill 后是否仍超限 → 标记为需要 tiling，跳过分配
        current_used_after = sum(resident_sizes.values())
        new_total_after = sum(
            _tensor_l1_size(graph, tid, cube_size, l1_align)
            for tid in new_tids if tid not in resident
        )
        if current_used_after + new_total_after > l1_cap:
            logger.info(
                "节点 %s spill 后仍超限 (%d > %d)，标记为 tiling 候选",
                nid, current_used_after + new_total_after, l1_cap,
            )
            # 跳过分配，记录空 layout，让 strategy_spill 的 tiling 步骤处理
            per_op_layouts.append({})
            per_op_loads.append(set())
            per_op_spills.append(spilled)
            # 标记输出为 dirty
            node = graph.nodes[nid]
            for tid in node.outputs:
                if tid in resident:
                    dirty.add(tid)
            continue

        # ④ 为新 tensor 分配 L1 空间
        loads: set[str] = set()
        for tid in new_tids:
            if tid in resident:
                continue  # 被驱逐的 tensor 如果在 needed 中会在下面重新分配
            size = _tensor_l1_size(graph, tid, cube_size, l1_align)
            offset = best_fit_alloc(free_blocks, size)
            if offset is None:
                offset = align_up(watermark, l1_align)
                watermark = offset + size
            resident[tid] = offset
            resident_sizes[tid] = size
            loads.add(tid)

        # ⑤ 记录 layout
        l1_layout = {tid: resident[tid] for tid in needed_tids if tid in resident}
        per_op_layouts.append(l1_layout)
        per_op_loads.append(loads)
        per_op_spills.append(spilled)

        # ⑥ 标记输出为 dirty
        node = graph.nodes[nid]
        for tid in node.outputs:
            if tid in resident:
                dirty.add(tid)

        # ⑦ 写 l1_offset 到 tensor（供 codegen 和 viz 使用）
        for tid, off in l1_layout.items():
            t = graph.tensors.get(tid)
            if t:
                t.l1_offset = off

    return per_op_layouts, per_op_loads, per_op_spills
