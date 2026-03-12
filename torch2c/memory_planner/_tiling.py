"""Auto-tiling: M 维切分，解决单算子 L1 溢出。

当 per-op 策略因为单个算子的 tensor 总量超限时，
对该算子做 M 维切分（矩阵乘法的行维度），使每个 tile 峰值 ≤ L1。

术语：
  tile_dim    — 切分的维度索引（对 [B, M, K] 即 dim 1）
  tile_size   — 每块的元素数
  num_tiles   — 切分块数
  tiled_tensors — {tensor_id: dim_idx} 参与切分的张量

设计：
  - 权重 / bias 不切分（跨 tile 共享）
  - 第一个非权重输入（A）及输出（C）的 M 维切分
  - 第二个输入（B / weight）不切分
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from torch2c.common import Graph, Node, get_logger

from ._l1_alloc import collect_op_tensors
from ._utils import align_up, calc_padded_size

logger = get_logger("memory_planner.tiling")


@dataclass
class TileInfo:
    """单个节点的切分描述。"""

    tile_dim: int
    tile_size: int
    num_tiles: int
    original_size: int
    tiled_tensors: dict[str, int] = field(default_factory=dict)
    num_buffers: int = 1  # 1=无流水, 2=ping-pong, N=N 级流水


# ---- 内部工具 -------------------------------------------------------


# op → 可切分维度（相对于 shape 末尾的偏移，-2 = M 维）
# None = 使用默认逻辑（_classify_matmul_tensors）
_TILEABLE_OPS: dict[str, int | None] = {
    "cube_matmul": -2,
    "cube_matmul_bias": -2,
    "dma_reformat": -2,
}


def _is_tileable(node: Node) -> bool:
    return node.npu_op in _TILEABLE_OPS


def _get_tile_dim(node: Node) -> int:
    """返回 op 的切分维度偏移（相对于 shape 末尾）。"""
    return _TILEABLE_OPS.get(node.npu_op, -2) or -2


def _classify_matmul_tensors(
    graph: Graph, node: Node,
) -> dict[str, int]:
    """将 matmul 的 tensor 分为 tiled / untiled。

    cube_matmul/cube_matmul_bias:
      A（第一个非权重非吸收输入）：tiled
      B（第二个输入 / weight）：untiled
      C（输出）：tiled
      bias：untiled（已吸收）

    dma_reformat:
      所有非权重 tensor 均 tiled
    """
    absorbed = set(node.absorbed_inputs.values())
    tiled: dict[str, int] = {}
    tile_dim_offset = _get_tile_dim(node)

    if node.npu_op == "dma_reformat":
        for tid in list(node.inputs) + list(node.outputs):
            t = graph.tensors.get(tid)
            if t and len(t.shape) >= 2 and not t.is_weight:
                tiled[tid] = len(t.shape) + tile_dim_offset
        return tiled

    for tid in node.inputs:
        if tid in absorbed:
            continue
        t = graph.tensors.get(tid)
        if not t or t.is_weight:
            continue
        tiled[tid] = len(t.shape) + tile_dim_offset
        break

    for tid in node.outputs:
        t = graph.tensors.get(tid)
        if t:
            tiled[tid] = len(t.shape) + tile_dim_offset

    return tiled


def _calc_op_peak(
    graph: Graph, node_id: str, cube_size: int, l1_align: int,
) -> int:
    """单个算子所有 tensor 的 L1 峰值。"""
    total = 0
    for tid in collect_op_tensors(graph, node_id):
        t = graph.tensors.get(tid)
        if not t or t.storage == "pipe":
            continue
        size = calc_padded_size(t.shape, t.dtype, t.format, cube_size)
        total += align_up(size, l1_align)
    return total


def _calc_tiled_peak(
    graph: Graph,
    node: Node,
    tile_size: int,
    tiled_tensors: dict[str, int],
    cube_size: int,
    l1_align: int,
) -> int:
    """给定 tile_size 时的 L1 峰值。"""
    total = 0
    for tid in collect_op_tensors(graph, node.id):
        t = graph.tensors.get(tid)
        if not t or t.storage == "pipe":
            continue
        if tid in tiled_tensors:
            dim_idx = tiled_tensors[tid]
            s = list(t.shape)
            s[dim_idx] = tile_size
            size = calc_padded_size(s, t.dtype, t.format, cube_size)
        else:
            size = calc_padded_size(t.shape, t.dtype, t.format, cube_size)
        total += align_up(size, l1_align)
    return total


def _calc_multi_buffer_peak(
    graph: Graph,
    node: Node,
    tile_size: int,
    tiled_tensors: dict[str, int],
    num_buffers: int,
    cube_size: int,
    l1_align: int,
) -> int:
    """N 级缓冲时的 L1 峰值：tiled tensor × num_buffers，untiled tensor × 1。"""
    total = 0
    for tid in collect_op_tensors(graph, node.id):
        t = graph.tensors.get(tid)
        if not t or t.storage == "pipe":
            continue
        if tid in tiled_tensors:
            dim_idx = tiled_tensors[tid]
            s = list(t.shape)
            s[dim_idx] = tile_size
            size = calc_padded_size(s, t.dtype, t.format, cube_size)
            total += align_up(size, l1_align) * num_buffers
        else:
            size = calc_padded_size(t.shape, t.dtype, t.format, cube_size)
            total += align_up(size, l1_align)
    return total


def _find_tile_size(
    graph: Graph,
    node: Node,
    tiled_tensors: dict[str, int],
    l1_cap: int,
    cube_size: int,
    l1_align: int,
) -> tuple[int, int]:
    """二分查找最大可行 tile_size，优先整除。

    Returns: (tile_size, num_tiles)
    """
    first_tid = next(iter(tiled_tensors))
    dim_idx = tiled_tensors[first_tid]
    original = graph.tensors[first_tid].shape[dim_idx]

    # 二分搜索最大 tile_size
    lo, hi, best = 1, original, 1
    while lo <= hi:
        mid = (lo + hi) // 2
        peak = _calc_tiled_peak(
            graph, node, mid, tiled_tensors, cube_size, l1_align,
        )
        if peak <= l1_cap:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    # 优先找能整除的最大值
    for d in range(best, 0, -1):
        if original % d == 0:
            peak = _calc_tiled_peak(
                graph, node, d, tiled_tensors, cube_size, l1_align,
            )
            if peak <= l1_cap:
                return d, original // d

    return best, math.ceil(original / best)


def _find_tile_size_for_multi_buffer(
    graph: Graph,
    node: Node,
    tiled_tensors: dict[str, int],
    l1_cap: int,
    cube_size: int,
    l1_align: int,
    original: int,
    num_buffers: int = 2,
) -> tuple[int | None, int]:
    """缩小 tile_size 以在多级缓冲下适配 L1。返回 (tile_size, num_tiles) 或 (None, 0)。"""
    lo, hi, best = 1, original, None
    while lo <= hi:
        mid = (lo + hi) // 2
        peak = _calc_multi_buffer_peak(
            graph, node, mid, tiled_tensors, num_buffers, cube_size, l1_align,
        )
        if peak <= l1_cap:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    if best is None:
        return None, 0

    # 优先找整除的最大值
    for d in range(best, 0, -1):
        if original % d == 0:
            peak = _calc_multi_buffer_peak(
                graph, node, d, tiled_tensors, num_buffers, cube_size, l1_align,
            )
            if peak <= l1_cap:
                return d, original // d

    return best, math.ceil(original / best)


# ---- 公共 API -------------------------------------------------------


def analyze_tiling(
    graph: Graph,
    l1_cap: int,
    l1_align: int,
    cube_size: int,
    tile_override: dict | None = None,
) -> dict[str, TileInfo]:
    """分析图中哪些节点需要 M 维 tiling。

    tile_override: {node_id: {"tile_size": int, "num_buffers": int}} 手动覆盖。

    Returns: {node_id: TileInfo}
    """
    tile_override = tile_override or {}
    result: dict[str, TileInfo] = {}

    for nid in graph.execution_order:
        node = graph.nodes[nid]
        override = tile_override.get(nid, {})
        peak = _calc_op_peak(graph, nid, cube_size, l1_align)
        if peak <= l1_cap and nid not in tile_override:
            continue

        if not _is_tileable(node):
            continue

        tiled_tensors = _classify_matmul_tensors(graph, node)
        if not tiled_tensors:
            continue

        if "tile_size" in override:
            first_tid = next(iter(tiled_tensors))
            dim_idx = tiled_tensors[first_tid]
            original = graph.tensors[first_tid].shape[dim_idx]
            tile_size = override["tile_size"]
            num_tiles = math.ceil(original / tile_size)
        else:
            tile_size, num_tiles = _find_tile_size(
                graph, node, tiled_tensors, l1_cap, cube_size, l1_align,
            )

        first_tid = next(iter(tiled_tensors))
        dim_idx = tiled_tensors[first_tid]
        original = graph.tensors[first_tid].shape[dim_idx]

        # 确定可用缓冲级数，必要时缩小 tile_size 以启用 double buffer
        if "num_buffers" in override:
            num_buffers = override["num_buffers"]
        else:
            num_buffers = 1
            for nb in (3, 2):
                peak_nb = _calc_multi_buffer_peak(
                    graph, node, tile_size, tiled_tensors, nb, cube_size, l1_align,
                )
                if peak_nb <= l1_cap:
                    num_buffers = nb
                    break

            # 单缓冲且 tile 数 >= 4 时，尝试缩小 tile_size 以启用 ping-pong
            if num_buffers == 1 and num_tiles >= 4:
                db_ts, db_nt = _find_tile_size_for_multi_buffer(
                    graph, node, tiled_tensors, l1_cap, cube_size, l1_align,
                    original, num_buffers=2,
                )
                if db_ts and db_nt >= 2:
                    tile_size, num_tiles, num_buffers = db_ts, db_nt, 2

        info = TileInfo(
            tile_dim=dim_idx,
            tile_size=tile_size,
            num_tiles=num_tiles,
            original_size=original,
            tiled_tensors=tiled_tensors,
            num_buffers=num_buffers,
        )
        result[nid] = info
        buf_label = f", {num_buffers}-buffer" if num_buffers > 1 else ""
        src = " (manual)" if nid in tile_override else ""
        logger.info(
            "节点 %s 需要 tiling: dim=%d, %d → %d × %d%s%s",
            nid, dim_idx, original, tile_size, num_tiles, buf_label, src,
        )

    return result
