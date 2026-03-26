"""HBM 全局分配：lifetime 分析 + best-fit 空间复用。"""

from __future__ import annotations

from torch2c.common import Graph, get_logger

from ._utils import align_up, best_fit_alloc, calc_padded_size

logger = get_logger("memory_planner.hbm")


def analyze_lifetimes(graph: Graph) -> dict[str, tuple[int, int]]:
    """分析每个 tensor 的生命周期 (first_use, last_use)。

    对于 weight/model_input，first_use=0；对于 model_output，last_use=max_order。
    无消费者且非 model_output 的 tensor 不分配 HBM（如 layernorm mean/rstd）。
    返回 {tensor_id: (first_use, last_use)}，跳过不需要分配的 tensor。
    """
    order_map: dict[str, int] = {}
    for idx, nid in enumerate(graph.execution_order):
        order_map[nid] = idx
    max_order = len(graph.execution_order) - 1

    # 构建 tensor → 吸收它的节点 映射（absorbed tensor 也算消费者）
    absorbed_consumers: dict[str, list[str]] = {}
    for nid, node in graph.nodes.items():
        for atid in node.absorbed_inputs.values():
            absorbed_consumers.setdefault(atid, []).append(nid)

    lifetimes: dict[str, tuple[int, int]] = {}
    for tid, t in graph.tensors.items():
        if t.storage in ("local", "pipe"):
            continue

        if t.is_weight or t.is_model_input:
            first_use = 0
        elif t.producer_node_id and t.producer_node_id in order_map:
            first_use = order_map[t.producer_node_id]
        else:
            first_use = 0

        # 合并 consumer_node_ids 和 absorbed_inputs 中的消费者
        all_consumers = list(t.consumer_node_ids) + absorbed_consumers.get(tid, [])

        if t.is_model_output:
            last_use = max_order
        elif all_consumers:
            consumer_orders = [order_map[c] for c in all_consumers if c in order_map]
            if consumer_orders:
                last_use = max(consumer_orders)
            else:
                continue
        elif t.is_weight or t.is_model_input:
            last_use = max_order
        else:
            continue

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


def allocate_hbm(
    graph: Graph,
    lifetimes: dict[str, tuple[int, int]],
    hbm_alignment: int,
    cube_size: int,
    *,
    allow_reuse: bool = True,
) -> int:
    """Best-fit HBM 分配。allow_reuse=True 时支持空间复用。返回实际复用次数。"""
    sorted_tids = sorted(lifetimes.keys(), key=lambda t: lifetimes[t][0])
    free_blocks: list[list[int]] = []
    freed_set: set[str] = set()
    hbm_watermark = 0
    reuse_count = 0

    for tid in sorted_tids:
        t = graph.tensors[tid]
        size = calc_padded_size(t.shape, t.dtype, t.format, (cube_size, cube_size))
        t.hbm_size = size
        aligned_size = align_up(size, hbm_alignment)

        if allow_reuse:
            _release_expired(graph, sorted_tids, lifetimes, lifetimes[tid][0], freed_set, free_blocks, hbm_alignment)
            offset = best_fit_alloc(free_blocks, aligned_size)
        else:
            offset = None

        if offset is not None:
            t.hbm_offset = offset
            reuse_count += 1
            logger.debug("HBM 复用: %s -> offset=%d, size=%d", tid, offset, size)
        else:
            t.hbm_offset = hbm_watermark
            hbm_watermark = t.hbm_offset + aligned_size
            logger.debug("HBM 新分配: %s -> offset=%d, size=%d", tid, t.hbm_offset, size)

    return reuse_count
