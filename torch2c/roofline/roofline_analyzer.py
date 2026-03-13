"""roofline_analyzer — 标注每个算子的计算强度与瓶颈类型。

为每个 Node 注入 params["_roofline"] = {
    "flops": int,          # 浮点运算量
    "bytes": int,          # 输入+输出总字节数
    "oi": float,           # operational intensity = flops/bytes
    "bottleneck": str,     # "compute" | "memory"
    "achievable_ratio": float,  # 可达性能 / 峰值性能 (0-1)
}
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul

from torch2c.common import Graph, Node, get_logger
from torch2c.common.sizing import calc_padded_size


logger = get_logger("roofline")


# ── 硬件参数 ─────────────────────────────────────────────


@dataclass
class RooflineHwParams:
    """Roofline 模型使用的硬件参数。"""

    cube_ops_per_cycle: int = 4096
    vector_ops_per_cycle: int = 128
    dma_bytes_per_cycle: int = 256
    cube_size: int = 16

    @classmethod
    def from_config(cls, hw_config: dict | None) -> RooflineHwParams:
        """从 hardware_config 字典构建，缺失字段用默认值。"""
        hw = cls()
        if hw_config and "fractal" in hw_config:
            hw.cube_size = hw_config["fractal"].get("cube_size", hw.cube_size)
        return hw


# ── 辅助函数 ─────────────────────────────────────────────


def _prod(shape: list[int]) -> int:
    """shape 各维乘积。"""
    return reduce(mul, shape, 1)


def _dtype_bytes(dtype: str) -> int:
    """每个元素的字节数。"""
    return {
        "fp16": 2, "bf16": 2, "fp32": 4,
        "int8": 1, "int16": 2, "int32": 4,
    }.get(dtype, 2)


# ── FLOPS 估算 ───────────────────────────────────────────

# npu_op 前缀 → FLOPS 乘数（相对于元素数）
_VECTOR_FLOPS_MULTIPLIER: dict[str, int] = {
    "vector_softmax": 3,
    "vector_layernorm": 5,
    "vector_rmsnorm": 5,
}


def estimate_flops(node: Node, graph: Graph, hw: RooflineHwParams) -> int:
    """估算节点的浮点运算量。"""
    op = node.npu_op or node.op_type
    cu = (node.compute_unit or "vector").lower()

    # cube_matmul: 2*M*N*K (x batch)
    if op in ("cube_matmul", "cube_matmul_bias"):
        return _matmul_flops(node, graph)

    # dma: 无计算
    if cu in ("dma", "idma"):
        return 0

    # vector: 按乘数计算
    multiplier = 1
    for prefix, m in _VECTOR_FLOPS_MULTIPLIER.items():
        if op.startswith(prefix):
            multiplier = m
            break

    elem_count = _output_element_count(node, graph)
    return elem_count * multiplier


def _matmul_flops(node: Node, graph: Graph) -> int:
    """cube_matmul 的 2*M*N*K*batch 公式。"""
    if len(node.inputs) < 2:
        return 0
    a = graph.tensors.get(node.inputs[0])
    b = graph.tensors.get(node.inputs[1])
    if not a or not b:
        return 0
    m = a.shape[-2] if len(a.shape) >= 2 else 1
    k = a.shape[-1] if len(a.shape) >= 1 else 1
    n = b.shape[-1] if len(b.shape) >= 1 else 1
    batch = _prod(a.shape[:-2]) if len(a.shape) > 2 else 1
    return 2 * batch * m * n * k


def _output_element_count(node: Node, graph: Graph) -> int:
    """取第一个输出 tensor 的元素数量。"""
    for tid in node.outputs:
        t = graph.tensors.get(tid)
        if t:
            return _prod(t.shape)
    return 0


# ── 字节量估算 ───────────────────────────────────────────


def estimate_bytes(node: Node, graph: Graph, hw: RooflineHwParams) -> int:
    """估算节点所有输入+输出 tensor 的总字节数（含 padding）。"""
    total = 0
    for tid in list(node.inputs) + list(node.outputs):
        t = graph.tensors.get(tid)
        if t:
            total += calc_padded_size(t.shape, t.dtype, t.format, hw.cube_size)
    return total


# ── Ridge Point ──────────────────────────────────────────


def _ridge_point(compute_unit: str, hw: RooflineHwParams) -> float:
    """计算 ridge point（计算/带宽平衡点）。

    load + store 双向带宽，所以 dma_bytes_per_cycle * 2。
    """
    bw = hw.dma_bytes_per_cycle * 2
    if compute_unit == "cube":
        return hw.cube_ops_per_cycle / bw
    return hw.vector_ops_per_cycle / bw


# ── 主入口 ───────────────────────────────────────────────


def run(graph: Graph, config: dict | None = None) -> Graph:
    """为每个节点标注 roofline 分析结果。"""
    hw_config = config.get("hardware", None) if config else None
    hw = RooflineHwParams.from_config(hw_config)

    compute_count = 0
    memory_count = 0

    for nid, node in graph.nodes.items():
        flops = estimate_flops(node, graph, hw)
        nbytes = estimate_bytes(node, graph, hw)
        oi = flops / nbytes if nbytes > 0 else 0.0

        cu = (node.compute_unit or "vector").lower()
        if cu in ("dma", "idma"):
            cu_class = "dma"
        elif cu == "cube":
            cu_class = "cube"
        else:
            cu_class = "vector"

        ridge = _ridge_point(cu_class, hw)
        bottleneck = "compute" if oi >= ridge else "memory"

        # achievable_ratio: min(oi / ridge, 1.0)
        achievable_ratio = min(oi / ridge, 1.0) if ridge > 0 else 0.0

        node.params["_roofline"] = {
            "flops": flops,
            "bytes": nbytes,
            "oi": round(oi, 4),
            "bottleneck": bottleneck,
            "achievable_ratio": round(achievable_ratio, 4),
        }

        if bottleneck == "compute":
            compute_count += 1
        else:
            memory_count += 1

    total = len(graph.nodes)
    logger.info(
        "roofline analysis: %d nodes — %d compute-bound, %d memory-bound",
        total, compute_count, memory_count,
    )

    return graph
