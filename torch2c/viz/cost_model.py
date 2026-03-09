"""cost_model — 算子延迟估算模型。

为 pipeline schedule 可视化提供每个算子的预估执行周期数（cycles）。

估算层次：
1. 算子专用公式（COST_FORMULAS 注册表，按 npu_op 匹配）
2. 通用 fallback（按 compute_unit 使用默认计算量公式）

硬件参数从 hardware_config.yaml 读取，也可直接传入。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from operator import mul

from torch2c.common import Graph, Node
from torch2c.memory_planner._utils import calc_padded_size


# ── 硬件参数 ─────────────────────────────────────────────


@dataclass
class HwParams:
    """硬件性能参数（用于延迟估算）。"""

    cube_ops_per_cycle: int = 4096       # 16x16x16 MAC
    vector_ops_per_cycle: int = 128      # SIMD 宽度
    dma_bytes_per_cycle: int = 256       # DMA 带宽
    idma_bytes_per_cycle: int = 512      # IDMA 带宽（L1 内部）
    cube_size: int = 16

    @classmethod
    def from_config(cls, hw_config: dict | None) -> HwParams:
        """从 hardware_config.yaml 构建，缺失字段用默认值。"""
        hw = cls()
        if hw_config and "fractal" in hw_config:
            hw.cube_size = hw_config["fractal"].get("cube_size", hw.cube_size)
        return hw


# ── 辅助函数 ─────────────────────────────────────────────


def _prod(shape: list[int]) -> int:
    return reduce(mul, shape, 1)


def _dtype_bytes(dtype: str) -> int:
    return {"fp16": 2, "bf16": 2, "fp32": 4, "int8": 1, "int16": 2, "int32": 4}.get(dtype, 2)


# ── 算子专用公式 ─────────────────────────────────────────
#
# 签名: (node, graph, hw: HwParams) -> int (cycles)


def _cube_matmul_cost(node: Node, graph: Graph, hw: HwParams) -> int:
    """cube_matmul: 2*M*N*K / ops_per_cycle。"""
    inputs = node.inputs
    if len(inputs) >= 2:
        a = graph.tensors.get(inputs[0])
        b = graph.tensors.get(inputs[1])
        if a and b:
            m = a.shape[-2] if len(a.shape) >= 2 else 1
            k = a.shape[-1] if len(a.shape) >= 1 else 1
            n = b.shape[-1] if len(b.shape) >= 1 else 1
            batch = _prod(a.shape[:-2]) if len(a.shape) > 2 else 1
            flops = 2 * batch * m * n * k
            return max(1, flops // hw.cube_ops_per_cycle)
    return _generic_cube_cost(node, graph, hw)


def _cube_matmul_bias_cost(node: Node, graph: Graph, hw: HwParams) -> int:
    """cube_matmul_bias: 同 matmul（bias add 开销忽略不计）。"""
    return _cube_matmul_cost(node, graph, hw)


def _vector_elementwise_cost(node: Node, graph: Graph, hw: HwParams) -> int:
    """通用 vector element-wise: element_count / ops_per_cycle。"""
    for tid in node.outputs:
        t = graph.tensors.get(tid)
        if t:
            return max(1, _prod(t.shape) // hw.vector_ops_per_cycle)
    return 1


def _vector_layernorm_cost(node: Node, graph: Graph, hw: HwParams) -> int:
    """layernorm: ~5x element-wise（mean + var + normalize + scale + shift）。"""
    return _vector_elementwise_cost(node, graph, hw) * 5


def _vector_softmax_cost(node: Node, graph: Graph, hw: HwParams) -> int:
    """softmax: ~3x element-wise（max + exp + sum + div）。"""
    return _vector_elementwise_cost(node, graph, hw) * 3


def _dma_cost(node: Node, graph: Graph, hw: HwParams) -> int:
    """DMA/IDMA: bytes / bytes_per_cycle。"""
    cu = (node.compute_unit or "dma").lower()
    bw = hw.idma_bytes_per_cycle if cu == "idma" else hw.dma_bytes_per_cycle
    for tid in node.inputs:
        t = graph.tensors.get(tid)
        if t:
            size = calc_padded_size(t.shape, t.dtype, t.format, hw.cube_size)
            return max(1, size // bw)
    return 1


def _generic_cube_cost(node: Node, graph: Graph, hw: HwParams) -> int:
    """Cube fallback: 用输出 tensor 尺寸估算。"""
    for tid in node.outputs:
        t = graph.tensors.get(tid)
        if t:
            return max(1, _prod(t.shape) * 2 // hw.cube_ops_per_cycle)
    return 1


# ── 注册表 ───────────────────────────────────────────────

CostFn = Callable[[Node, Graph, HwParams], int]

COST_FORMULAS: dict[str, CostFn] = {
    # cube
    "cube_matmul": _cube_matmul_cost,
    "cube_matmul_bias": _cube_matmul_bias_cost,
    # vector: arithmetic
    "vector_add": _vector_elementwise_cost,
    "vector_sub": _vector_elementwise_cost,
    "vector_mul": _vector_elementwise_cost,
    "vector_div": _vector_elementwise_cost,
    "vector_mul_scalar": _vector_elementwise_cost,
    "vector_fill": _vector_elementwise_cost,
    # vector: activation
    "vector_gelu": _vector_elementwise_cost,
    "vector_dropout": _vector_elementwise_cost,
    # vector: normalization
    "vector_softmax": _vector_softmax_cost,
    "vector_softmax_part1": _vector_softmax_cost,
    "vector_softmax_part2": _vector_elementwise_cost,
    "vector_layernorm": _vector_layernorm_cost,
    "vector_layernorm_part1": _vector_layernorm_cost,
    "vector_layernorm_part2": _vector_elementwise_cost,
    "vector_rmsnorm": _vector_layernorm_cost,
    "vector_rmsnorm_part1": _vector_layernorm_cost,
    "vector_rmsnorm_part2": _vector_elementwise_cost,
    # vector: shape
    "vector_transpose": _vector_elementwise_cost,
    "vector_transpose_2d": _vector_elementwise_cost,
    # dma / idma
    "dma_move": _dma_cost,
    "dma_reformat": _dma_cost,
    "idma_move": _dma_cost,
    "idma_reshape": _dma_cost,
    "idma_broadcast": _dma_cost,
    "idma_concat": _dma_cost,
}

_GENERIC_FALLBACK: dict[str, CostFn] = {
    "cube": _generic_cube_cost,
    "vector": _vector_elementwise_cost,
    "dma": _dma_cost,
    "idma": _dma_cost,
}


# ── 公开 API ─────────────────────────────────────────────


def estimate_cost(node: Node, graph: Graph, hw_config: dict | None = None) -> int:
    """估算单个节点的执行周期数。至少返回 1。"""
    hw = HwParams.from_config(hw_config)

    op = node.npu_op or node.op_type
    if op in COST_FORMULAS:
        return COST_FORMULAS[op](node, graph, hw)

    cu = (node.compute_unit or "vector").lower()
    fallback = _GENERIC_FALLBACK.get(cu, _vector_elementwise_cost)
    return fallback(node, graph, hw)


def estimate_all(graph: Graph, hw_config: dict | None = None) -> dict[str, int]:
    """估算图中所有节点的执行周期（tiled 算子自动乘 num_tiles）。"""
    hw = HwParams.from_config(hw_config)
    result: dict[str, int] = {}
    for nid, node in graph.nodes.items():
        op = node.npu_op or node.op_type
        if op in COST_FORMULAS:
            cost = COST_FORMULAS[op](node, graph, hw)
        else:
            cu = (node.compute_unit or "vector").lower()
            fallback = _GENERIC_FALLBACK.get(cu, _vector_elementwise_cost)
            cost = fallback(node, graph, hw)
        # tiled 算子：单 tile compute cost × num_tiles
        tile_info = node.params.get("_tile_info") if node.params else None
        if tile_info:
            cost = cost * tile_info.get("num_tiles", 1)
        result[nid] = cost
    return result
