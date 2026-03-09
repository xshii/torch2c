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


# ---- 内部工具 -------------------------------------------------------


_TILEABLE_OPS = frozenset({
    "cube_matmul", "cube_matmul_bias",
    "dma_reformat",
})


def _is_tileable(node: Node) -> bool:
    return node.npu_op in _TILEABLE_OPS


def _classify_matmul_tensors(
    graph: Graph, node: Node,
) -> dict[str, int]:
    """将 matmul 的 tensor 分为 tiled / untiled。

    cube_matmul/cube_matmul_bias:
      A（第一个非权重非吸收输入）：tiled（dim -2）
      B（第二个输入 / weight）：untiled
      C（输出）：tiled（dim -2）
      bias：untiled（已吸收）

    dma_reformat:
      所有非权重 tensor 均 tiled（dim -2）
    """
    absorbed = set(node.absorbed_inputs.values())
    tiled: dict[str, int] = {}

    if node.npu_op == "dma_reformat":
        # reformat: 所有 activation 输入输出均可切分
        for tid in list(node.inputs) + list(node.outputs):
            t = graph.tensors.get(tid)
            if t and len(t.shape) >= 2 and not t.is_weight:
                tiled[tid] = len(t.shape) - 2
        return tiled

    for tid in node.inputs:
        if tid in absorbed:
            continue
        t = graph.tensors.get(tid)
        if not t or t.is_weight:
            continue
        # 第一个 activation 输入 = A → tiled
        tiled[tid] = len(t.shape) - 2
        break

    for tid in node.outputs:
        t = graph.tensors.get(tid)
        if t:
            tiled[tid] = len(t.shape) - 2

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


# ---- 公共 API -------------------------------------------------------


def analyze_tiling(
    graph: Graph,
    l1_cap: int,
    l1_align: int,
    cube_size: int,
) -> dict[str, TileInfo]:
    """分析图中哪些节点需要 M 维 tiling。

    Returns: {node_id: TileInfo}
    """
    result: dict[str, TileInfo] = {}

    for nid in graph.execution_order:
        node = graph.nodes[nid]
        peak = _calc_op_peak(graph, nid, cube_size, l1_align)
        if peak <= l1_cap:
            continue

        if not _is_tileable(node):
            continue

        tiled_tensors = _classify_matmul_tensors(graph, node)
        if not tiled_tensors:
            continue

        tile_size, num_tiles = _find_tile_size(
            graph, node, tiled_tensors, l1_cap, cube_size, l1_align,
        )

        first_tid = next(iter(tiled_tensors))
        dim_idx = tiled_tensors[first_tid]
        original = graph.tensors[first_tid].shape[dim_idx]

        info = TileInfo(
            tile_dim=dim_idx,
            tile_size=tile_size,
            num_tiles=num_tiles,
            original_size=original,
            tiled_tensors=tiled_tensors,
        )
        result[nid] = info
        logger.info(
            "节点 %s 需要 M 维 tiling: dim=%d, %d → %d × %d",
            nid, dim_idx, original, tile_size, num_tiles,
        )

    return result
