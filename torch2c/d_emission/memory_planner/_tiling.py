"""Auto-tiling: M 维切分，解决单算子 L1 溢出。

当 per-op 策略因为单个算子的 tensor 总量超限时，
对该算子做 M 维切分（矩阵乘法的行维度），使每个 tile 峰值 ≤ L1。

术语：
  tile_dim    — 切分的维度索引（对 [B, M, K] 即 dim 1）
  tile_size   — 每块的元素数
  num_tiles   — 切分块数
  tiled_tensors — {tensor_id: dim_idx} 参与切分的张量
  pipe_group  — pipe 连接的节点组编号（共享 tile 循环）

设计：
  - 权重 / bias 不切分（跨 tile 共享）
  - 第一个非权重输入（A）及输出（C）的 M 维切分
  - 第二个输入（B / weight）不切分
  - pipe 连接的节点必须统一 tile_size（共享同一个 tile 循环）
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
    pipe_group: int | None = None  # pipe 连接的节点组编号，共享 tile 循环


# ---- 内部工具 -------------------------------------------------------


# op → 可切分维度（相对于 shape 末尾的偏移，-2 = M 维）
_TILEABLE_OPS: dict[str, int | None] = {
    # cube ops
    "cube_matmul": -2,
    "cube_matmul_bias": -2,
    # vector: arithmetic
    "vector_add": -2,
    "vector_sub": -2,
    "vector_mul": -2,
    "vector_div": -2,
    "vector_mul_scalar": -2,
    "vector_fill": -2,
    # vector: activation
    "vector_gelu": -2,
    "vector_dropout": -2,
    # vector: normalization
    "vector_softmax": -2,
    "vector_softmax_part1": -2,
    "vector_softmax_part2": -2,
    "vector_layernorm": -2,
    "vector_layernorm_part1": -2,
    "vector_layernorm_part2": -2,
    "vector_rmsnorm": -2,
    "vector_rmsnorm_part1": -2,
    "vector_rmsnorm_part2": -2,
    # vector: shape
    "vector_transpose": -2,
    "vector_transpose_2d": -2,
    # DMA / iDMA ops
    "dma_reformat": -2,
    "dma_move": -2,
    "idma_broadcast": -2,
    "idma_reshape": -2,
    "idma_move": -2,
    "idma_slice": -2,
    "idma_concat": -2,
}

# matmul 类 op: 只切分 A（第一个输入）和 C（输出），B（权重）不切分
from torch2c.common.constants import MATMUL_OPS as _MATMUL_OPS

# element-wise 类 op: 所有非权重 tensor 统一切分（同 dma_reformat）
_ELEMENTWISE_OPS = _TILEABLE_OPS.keys() - _MATMUL_OPS


def _is_tileable(node: Node) -> bool:
    return node.npu_op in _TILEABLE_OPS


def _get_tile_dim(node: Node) -> int:
    """返回 op 的切分维度偏移（相对于 shape 末尾）。"""
    return _TILEABLE_OPS.get(node.npu_op, -2) or -2


def _classify_tiled_tensors(
    graph: Graph, node: Node,
) -> dict[str, int]:
    """将节点的 tensor 分为 tiled / untiled。

    matmul 类（cube_matmul/cube_matmul_bias）:
      A（第一个非权重非吸收输入）：tiled
      B（第二个输入 / weight）：untiled
      C（输出）：tiled
      bias：untiled（已吸收）

    element-wise 类（softmax/add/layernorm/gelu/reformat 等）:
      所有非权重 tensor 均 tiled
    """
    absorbed = set(node.absorbed_inputs.values())
    tiled: dict[str, int] = {}
    tile_dim_offset = _get_tile_dim(node)

    if node.npu_op in _ELEMENTWISE_OPS:
        for tid in list(node.inputs) + list(node.outputs):
            if tid in absorbed:
                continue
            t = graph.tensors.get(tid)
            if t and len(t.shape) >= 2 and not t.is_weight:
                tiled[tid] = len(t.shape) + tile_dim_offset
        return tiled

    # matmul: 只 tile A（第一个非权重输入）和 C（输出）
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


# ---- pipe 统一 tiling ------------------------------------------------


def _find_pipe_groups(graph: Graph) -> list[list[str]]:
    """找出被 pipe tensor 连接的节点组（按执行序排列）。

    返回连通分量列表，每个分量是共享 pipe tensor 的节点 ID 列表。
    """
    adj: dict[str, set[str]] = {}
    for t in graph.tensors.values():
        if t.storage != "pipe" or not t.producer_node_id:
            continue
        adj.setdefault(t.producer_node_id, set())
        for cid in t.consumer_node_ids:
            adj.setdefault(cid, set())
            adj[t.producer_node_id].add(cid)
            adj[cid].add(t.producer_node_id)
    if not adj:
        return []

    order_idx = {nid: i for i, nid in enumerate(graph.execution_order)}
    visited: set[str] = set()
    groups: list[list[str]] = []
    for nid in graph.execution_order:
        if nid not in adj or nid in visited:
            continue
        group: list[str] = []
        stack = [nid]
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            group.append(n)
            for nb in adj.get(n, ()):
                if nb not in visited:
                    stack.append(nb)
        group.sort(key=lambda x: order_idx.get(x, 0))
        groups.append(group)
    return groups


def _degrade_group_pipes(graph: Graph, group: list[str]) -> None:
    """将组内所有节点关联的 pipe tensor 降级为 hbm。"""
    nid_set = set(group)
    for t in graph.tensors.values():
        if t.storage != "pipe":
            continue
        touches = (t.producer_node_id in nid_set
                   or any(c in nid_set for c in t.consumer_node_ids))
        if touches:
            t.storage = "hbm"
            logger.info("pipe 降级: tensor %s → hbm", t.id)


def _unify_pipe_tile_sizes(
    graph: Graph,
    tile_map: dict[str, TileInfo],
    l1_cap: int,
    l1_align: int,
    cube_size: int,
) -> None:
    """统一 pipe 连接节点组的 tile_size，原地修改 tile_map。

    规则：pipe 连接的节点共享 tile 循环，必须使用相同 tile_size。
    统一值 = 组内所有需 tiling 节点的 tile_size 最小值。
    原本不需 tiling 的节点被"拉入"（tiled 后 L1 更小，保证 fit）。
    """
    for gid, group in enumerate(_find_pipe_groups(graph)):
        tiled_nids = [n for n in group if n in tile_map]
        if not tiled_nids:
            continue

        # 组内有不可 tile 的节点 → 降级整组 pipe
        untileable = [n for n in group if not _is_tileable(graph.nodes[n])]
        if untileable:
            logger.warning(
                "pipe 组 %d 含不可 tile 节点 %s，降级 pipe → hbm", gid, untileable,
            )
            _degrade_group_pipes(graph, group)
            continue

        ref = tile_map[tiled_nids[0]]
        original = ref.original_size
        unified = min(tile_map[n].tile_size for n in tiled_nids)
        num_tiles = math.ceil(original / unified)

        for nid in group:
            _apply_unified_tile(
                graph, tile_map, nid, gid, unified, num_tiles, original,
                l1_cap, l1_align, cube_size,
            )

        logger.info(
            "pipe 组 %d (%d 节点): 统一 tile_size=%d, num_tiles=%d",
            gid, len(group), unified, num_tiles,
        )


def _apply_unified_tile(
    graph: Graph,
    tile_map: dict[str, TileInfo],
    nid: str,
    gid: int,
    unified_ts: int,
    num_tiles: int,
    original: int,
    l1_cap: int,
    l1_align: int,
    cube_size: int,
) -> None:
    """为组内单个节点应用统一 tile_size，更新 tile_map。"""
    node = graph.nodes[nid]
    already_tiled = nid in tile_map

    if already_tiled:
        tt = tile_map[nid].tiled_tensors
        old_ts = tile_map[nid].tile_size
    else:
        tt = _classify_tiled_tensors(graph, node)
        old_ts = original
        if not tt:
            return

    # 重新计算 num_buffers
    nb = 1
    for try_nb in (3, 2):
        peak = _calc_multi_buffer_peak(
            graph, node, unified_ts, tt, try_nb, cube_size, l1_align,
        )
        if peak <= l1_cap:
            nb = try_nb
            break

    first_tid = next(iter(tt))
    tile_map[nid] = TileInfo(
        tile_dim=tt[first_tid],
        tile_size=unified_ts,
        num_tiles=num_tiles,
        original_size=original,
        tiled_tensors=tt,
        num_buffers=nb,
        pipe_group=gid,
    )

    if not already_tiled:
        logger.info("pipe 组 %d: 节点 %s 加入 tiling (tile_size=%d)", gid, nid, unified_ts)
    elif old_ts != unified_ts:
        logger.info(
            "pipe 组 %d: 节点 %s tile_size %d → %d", gid, nid, old_ts, unified_ts,
        )


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

        tiled_tensors = _classify_tiled_tensors(graph, node)
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

    # pipe 连接的节点组统一 tile_size
    _unify_pipe_tile_sizes(graph, result, l1_cap, l1_align, cube_size)

    return result
