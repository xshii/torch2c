"""global_tiler — 基于 cost model 的全局 tiling 决策。

与 memory_planner 的 _tiling.py 区别：
- _tiling.py 是「被动式」：仅在 L1 溢出时触发，目标是「能放下」
- global_tiler 是「主动式」：即使能放下，也评估 tiling+ping-pong 是否更快

决策流程：
1. 对每个融合组（或单独 op），评估 no-tile vs tiled 的总 cycles
2. 如果 tiled 更快，设置 _tile_config
3. memory_planner 读取 _tile_config 作为 tile_override

输出：node.params["_tile_config"] = {tile_size, num_buffers, num_tiles}
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from torch2c.common import Graph, Node, get_logger
from torch2c.common.opt_log import log_opt
from torch2c.common.sizing import calc_padded_size

logger = get_logger("global_tiler")

# global tiler 支持的 op（仅 cube + reformat，与 memory_planner._tiling 的全量列表不同）
_GLOBAL_TILEABLE_OPS = frozenset({"cube_matmul", "cube_matmul_bias", "dma_reformat"})


@dataclass
class TileCandidate:
    """一个 tiling 候选方案。"""

    tile_size: int
    num_tiles: int
    num_buffers: int
    total_cycles: float
    l1_peak: int


# ── 辅助 ────────────────────────────────────────────────


def _prod(shape: list[int]) -> int:
    r = 1
    for s in shape:
        r *= s
    return r


def _dtype_bytes(dtype: str) -> int:
    return {"fp16": 2, "bf16": 2, "fp32": 4, "int8": 1, "int32": 4}.get(dtype, 2)


def _estimate_compute_cycles(
    node: Node, graph: Graph, tile_size: int | None,
    cube_ops: int, vector_ops: int,
) -> float:
    """估算单 tile 或全量计算的 cycle 数。"""
    op = node.npu_op or node.op_type
    cu = (node.compute_unit or "vector").lower()

    if op in ("cube_matmul", "cube_matmul_bias") and len(node.inputs) >= 2:
        a = graph.tensors.get(node.inputs[0])
        b = graph.tensors.get(node.inputs[1])
        if a and b:
            m = tile_size if tile_size else (a.shape[-2] if len(a.shape) >= 2 else 1)
            k = a.shape[-1] if len(a.shape) >= 1 else 1
            n = b.shape[-1] if len(b.shape) >= 1 else 1
            batch = _prod(a.shape[:-2]) if len(a.shape) > 2 else 1
            return 2 * batch * m * n * k / cube_ops

    # vector/other
    for tid in node.outputs:
        t = graph.tensors.get(tid)
        if t:
            elem = _prod(t.shape)
            if tile_size and len(t.shape) >= 2:
                ratio = tile_size / t.shape[-2] if t.shape[-2] > 0 else 1
                elem = int(elem * ratio)
            ops_per = vector_ops if cu == "vector" else cube_ops
            return max(1.0, elem / ops_per)
    return 1.0


def _estimate_dma_cycles(
    node: Node, graph: Graph, tile_size: int | None,
    cube_size: int, dma_bw: int,
) -> float:
    """估算单 tile 或全量 DMA 搬运的 cycle 数。"""
    total_bytes = 0
    for tid in list(node.inputs) + list(node.outputs):
        t = graph.tensors.get(tid)
        if not t:
            continue
        if tile_size and not t.is_weight and len(t.shape) >= 2:
            s = list(t.shape)
            s[-2] = tile_size
            total_bytes += calc_padded_size(s, t.dtype, t.format, (cube_size, cube_size))
        else:
            total_bytes += calc_padded_size(t.shape, t.dtype, t.format, (cube_size, cube_size))
    return max(1.0, total_bytes / dma_bw)


def _estimate_l1_peak(
    node: Node, graph: Graph, tile_size: int | None,
    num_buffers: int, cube_size: int,
) -> int:
    """估算 L1 峰值占用。"""
    total = 0
    for tid in list(node.inputs) + list(node.outputs):
        t = graph.tensors.get(tid)
        if not t or t.storage == "pipe":
            continue
        if tile_size and not t.is_weight and len(t.shape) >= 2:
            s = list(t.shape)
            s[-2] = tile_size
            size = calc_padded_size(s, t.dtype, t.format, (cube_size, cube_size))
            total += size * num_buffers
        else:
            total += calc_padded_size(t.shape, t.dtype, t.format, (cube_size, cube_size))
    return total


def _get_tile_dim_size(node: Node, graph: Graph) -> int | None:
    """取首个非权重输入的 M 维大小。"""
    for tid in node.inputs:
        t = graph.tensors.get(tid)
        if t and not t.is_weight and len(t.shape) >= 2:
            return t.shape[-2]
    return None


# ── 核心：评估单个节点 ───────────────────────────────────


def _evaluate_node(
    node: Node, graph: Graph, l1_cap: int, cube_size: int,
    cube_ops: int, vector_ops: int, dma_bw: int,
) -> dict | None:
    """评估节点是否应主动 tiling，返回 _tile_config 或 None。"""
    op = node.npu_op or node.op_type
    if op not in _GLOBAL_TILEABLE_OPS:
        return None

    dim_size = _get_tile_dim_size(node, graph)
    if dim_size is None or dim_size <= cube_size:
        return None

    # no-tile baseline
    t_compute_full = _estimate_compute_cycles(node, graph, None, cube_ops, vector_ops)
    t_dma_full = _estimate_dma_cycles(node, graph, None, cube_size, dma_bw)
    t_no_tile = t_compute_full + t_dma_full

    best: TileCandidate | None = None

    # 枚举 tile_size（整除优先，从大到小）
    divisors = [d for d in range(dim_size, cube_size - 1, -1) if dim_size % d == 0]
    if not divisors:
        divisors = list(range(dim_size, cube_size - 1, -1))
    # 只评估合理的候选（最多 20 个）
    step = max(1, len(divisors) // 20)
    candidates = divisors[::step]

    for ts in candidates:
        num_tiles = math.ceil(dim_size / ts)
        if num_tiles < 2:
            continue  # 不如不 tile

        for nb in [2, 1]:
            l1 = _estimate_l1_peak(node, graph, ts, nb, cube_size)
            if l1 > l1_cap:
                continue
            if nb >= 2 and num_tiles < 4:
                continue  # ping-pong 需要足够 tile 才值得

            t_comp = _estimate_compute_cycles(node, graph, ts, cube_ops, vector_ops)
            t_dma = _estimate_dma_cycles(node, graph, ts, cube_size, dma_bw)

            # ping-pong: DMA 和计算重叠
            overlap = min(t_dma, t_comp) if nb >= 2 else 0
            per_tile = t_comp + t_dma - overlap
            # prologue/epilogue overhead
            overhead = t_dma if nb >= 2 else 0
            total = per_tile * num_tiles + overhead

            c = TileCandidate(ts, num_tiles, nb, total, l1)
            if best is None or total < best.total_cycles:
                best = c
            break  # 如果 2-buffer 放得下就不试 1-buffer

    if best is None:
        return None

    # 只有当 tiling 比 no-tile 快时才采用（阈值 10%）
    speedup = t_no_tile / best.total_cycles if best.total_cycles > 0 else 0
    if speedup < 1.10:
        return None

    return {
        "tile_size": best.tile_size,
        "num_buffers": best.num_buffers,
        "num_tiles": best.num_tiles,
        "speedup": round(speedup, 2),
    }


# ── 公开 API ────────────────────────────────────────────


def run(graph: Graph, config: dict) -> Graph:
    """全局 tiling 决策 pass。

    在 scheduler 之前运行。为适合主动 tiling 的节点设置
    node.params["_tile_config"]，供 memory_planner 通过
    tile_override 读取。
    """
    hw_config = config.get("hardware", config)
    l1_cap = 16 * 1024 * 1024  # 默认 16 MB
    cube_size = 16
    if "memory" in hw_config:
        l1_cap = hw_config["memory"]["l1"]["total_size_bytes"]
    if "fractal" in hw_config:
        cube_size = hw_config["fractal"]["cube_size"]

    compute_cfg = hw_config.get("compute", {})
    cube_ops = compute_cfg.get("cube_ops_per_cycle", 4096)
    vector_ops = compute_cfg.get("vector_ops_per_cycle", 128)
    dma_bw = compute_cfg.get("dma_bytes_per_cycle", 256)

    tiled_count = 0
    for nid, node in graph.nodes.items():
        tile_cfg = _evaluate_node(node, graph, l1_cap, cube_size,
                                  cube_ops, vector_ops, dma_bw)
        if tile_cfg:
            node.params["_tile_config"] = tile_cfg
            tiled_count += 1
            dim_size = _get_tile_dim_size(node, graph) or 0
            nb = tile_cfg['num_buffers']
            log_opt(
                node, "global_tiler", "分块计算",
                f"原始数据超出 L1 容量，切为 {tile_cfg['num_tiles']}×{tile_cfg['tile_size']} 块，"
                f"{'双 buffer ping-pong: DMA 搬运与计算流水线重叠，隐藏访存延迟' if nb >= 2 else '单 buffer 顺序执行'}，"
                f"预计加速 {tile_cfg['speedup']:.1f}x",
            )
            logger.info(
                "节点 %s 主动 tiling: %d → %d×%d, %d-buf, speedup=%.2fx",
                nid, dim_size,
                tile_cfg["tile_size"], tile_cfg["num_tiles"],
                tile_cfg["num_buffers"], tile_cfg["speedup"],
            )

    logger.info("全局 tiling 决策：%d / %d 节点主动 tiling", tiled_count, len(graph.nodes))
    return graph
