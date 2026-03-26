"""组内联合 tile 决策 — 融合组内所有节点共享 tile_size。

与 global_tiler 的逐节点独立 tile 不同：
  - 组内所有 tileable 节点共享同一 tile_size
  - tile_size 受 L1 容量约束（组内所有 tensor 的 tiled 切片 ≤ L1）
  - 支持 ping-pong 双 buffer（如果 2× peak ≤ L1）
"""

from __future__ import annotations

import math

from torch2c.common import get_logger
from torch2c.optpass.cd_block_fuser._block_graph import BlockGraph
from torch2c.optpass.cd_block_fuser._fusion import FusionGroup

logger = get_logger(__name__)


def assign_tile_sizes(
    group: FusionGroup,
    block_graph: BlockGraph,
    l1_capacity: int,
    cube_size: int = 16,
) -> dict:
    """为融合组选择共享 tile_size。

    Returns:
        {"tile_size": int, "num_buffers": int, "num_tiles": int}
        或 {} 表示不 tile。
    """
    # 收集组内所有 tileable 节点的 tile_dim_size
    tileable_nodes = []
    for nid in group.node_ids:
        cb = block_graph.compute_blocks.get(nid)
        if cb and cb.tileable and cb.tile_dim_size > cube_size:
            tileable_nodes.append(cb)

    if not tileable_nodes:
        return {}

    # 共享 tile_dim_size = 最小的那个（约束最严的）
    shared_dim_size = min(cb.tile_dim_size for cb in tileable_nodes)
    if shared_dim_size <= cube_size:
        return {}

    # 候选 tile sizes：shared_dim_size 的因子，降序
    candidates = _divisor_candidates(shared_dim_size, cube_size)
    if not candidates:
        return {}

    # 二分搜索最大可行 tile_size
    best_tile = None
    for tile_size in candidates:
        num_tiles = math.ceil(shared_dim_size / tile_size)
        peak = _estimate_tiled_l1_peak(group, block_graph, tile_size, shared_dim_size)

        # 先尝试 ping-pong（2× peak）
        if peak * 2 <= l1_capacity:
            best_tile = {"tile_size": tile_size, "num_buffers": 2, "num_tiles": num_tiles}
            break
        # 再尝试单 buffer
        if peak <= l1_capacity:
            best_tile = {"tile_size": tile_size, "num_buffers": 1, "num_tiles": num_tiles}
            break

    if best_tile is None:
        logger.debug("fusion group %s: no feasible tile size found", group.id)
        return {}

    logger.debug(
        "fusion group %s: tile_size=%d, num_tiles=%d, num_buffers=%d",
        group.id, best_tile["tile_size"], best_tile["num_tiles"],
        best_tile["num_buffers"],
    )
    return best_tile


def _divisor_candidates(dim_size: int, min_tile: int) -> list[int]:
    """返回 dim_size 的因子列表，降序排列，过滤掉 < min_tile 的。"""
    divs = []
    for d in range(1, int(math.sqrt(dim_size)) + 1):
        if dim_size % d == 0:
            if d >= min_tile:
                divs.append(d)
            complement = dim_size // d
            if complement >= min_tile and complement != d:
                divs.append(complement)
    divs.sort(reverse=True)
    return divs


def _estimate_tiled_l1_peak(
    group: FusionGroup,
    bg: BlockGraph,
    tile_size: int,
    original_dim_size: int,
) -> int:
    """估算 tile 后融合组的 L1 峰值占用。

    简化模型：所有组关联 tensor（内部 + 外部 IO）的 tiled 切片同时存在。
    Tiled tensor：大小按 tile_size / original_dim_size 比例缩放。
    非 tiled tensor（如 weight）：保持原大小。
    """
    ratio = tile_size / original_dim_size if original_dim_size > 0 else 1.0
    peak = 0

    all_tids = group.internal_block_ids | group.external_input_ids | group.external_output_ids
    for tid in all_tids:
        db = bg.data_blocks.get(tid)
        if db is None:
            continue
        # weight/constant 不 tile
        if db.is_external and not db.consumer_ids:
            peak += db.size_bytes
            continue
        # 有 producer 在组内 → 按 ratio 缩放
        if db.producer_id and db.producer_id in set(group.node_ids):
            peak += int(db.size_bytes * ratio)
        elif any(cid in set(group.node_ids) for cid in db.consumer_ids):
            # 外部输入按 ratio 缩放（DMA load tiled 部分）
            peak += int(db.size_bytes * ratio)
        else:
            peak += db.size_bytes

    return peak
