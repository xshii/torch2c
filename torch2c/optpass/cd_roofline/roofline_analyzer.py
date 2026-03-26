"""roofline_analyzer — 标注每个算子的计算强度、瓶颈类型与时延估算。

代价模型三层结构（优先级从高到低）：
  1. Python 函数覆盖 — register_cost_fn() 注册，接收完整 CostContext
  2. YAML per-op 覆盖 — cost_model_config.yaml op_overrides
  3. YAML unit 默认    — cost_model_config.yaml unit_defaults

CostContext 包含完整的算子上下文：
  - inputs/outputs: Tensor 列表（shape, dtype, format, storage）
  - compute_dtype: 计算精度
  - params: 算子参数
  - is_fused / fusion_role: 融合状态
  - hw: 硬件参数

为每个 Node 注入 params["_roofline"] = {
    "flops": int,          # 浮点运算量
    "bytes": int,          # 输入+输出总字节数（含 padding）
    "dma_bytes": int,      # 实际 HBM DMA 搬运字节数（排除 local/pipe）
    "oi": float,           # operational intensity = flops/bytes
    "bottleneck": str,     # "compute" | "memory"
    "achievable_ratio": float,  # 可达性能 / 峰值性能 (0-1)
    "compute_cycles": int, # 计算时延（cycles）
    "dma_cycles": int,     # DMA 搬运时延（cycles）
    "node_cycles": int,    # 节点总时延 = max(compute, dma)
}
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import reduce
from operator import mul

from torch2c.common import Graph, Node, Tensor, get_logger
from torch2c.common.opt_log import log_opt
from torch2c.common.sizing import calc_padded_size, get_dim_align


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
        if not hw_config:
            return hw
        if "fractal" in hw_config:
            hw.cube_size = hw_config["fractal"].get("cube_size", hw.cube_size)
        compute = hw_config.get("compute", {})
        hw.cube_ops_per_cycle = compute.get("cube_ops_per_cycle", hw.cube_ops_per_cycle)
        hw.vector_ops_per_cycle = compute.get("vector_ops_per_cycle", hw.vector_ops_per_cycle)
        hw.dma_bytes_per_cycle = compute.get("dma_bytes_per_cycle", hw.dma_bytes_per_cycle)
        return hw


# ── CostContext / CostResult / 函数注册 ──────────────────


@dataclass
class CostContext:
    """传给 cost function 的完整上下文。

    包含算子的输入输出 tensor、计算精度、格式、存储位置、融合状态等信息，
    供 Python 级 cost function 做精确的代价估算。
    """

    node: Node                     # 当前节点
    inputs: list[Tensor]           # 输入 tensor 列表
    outputs: list[Tensor]          # 输出 tensor 列表
    hw: RooflineHwParams           # 硬件参数
    params: dict                   # node.params（算子参数 + 优化元数据）
    compute_dtype: str             # 计算精度 (fp16/fp32/...)
    input_formats: list[str]       # 输入 format 列表 (nd/nz/zz/nn)
    output_formats: list[str]      # 输出 format 列表
    input_storage: list[str]       # 输入 storage 列表 (hbm/local/pipe)
    output_storage: list[str]      # 输出 storage 列表
    is_fused: bool                 # 是否在融合组内
    fusion_role: str | None        # head/middle/tail/None

    @property
    def elem_count(self) -> int:
        """第一个输出 tensor 的元素数。"""
        if self.outputs:
            return _prod(self.outputs[0].shape)
        return 0

    @property
    def M(self) -> int:
        """matmul 的 M 维度（第一个输入 shape[-2]）。"""
        if self.inputs and len(self.inputs[0].shape) >= 2:
            return self.inputs[0].shape[-2]
        return 1

    @property
    def N(self) -> int:
        """matmul 的 N 维度（第二个输入 shape[-1]）。"""
        if len(self.inputs) >= 2 and len(self.inputs[1].shape) >= 1:
            return self.inputs[1].shape[-1]
        return 1

    @property
    def K(self) -> int:
        """matmul 的 K 维度（第一个输入 shape[-1]）。"""
        if self.inputs and len(self.inputs[0].shape) >= 1:
            return self.inputs[0].shape[-1]
        return 1

    @property
    def batch(self) -> int:
        """batch 维度乘积（shape[:-2]）。"""
        if self.inputs and len(self.inputs[0].shape) > 2:
            return _prod(self.inputs[0].shape[:-2])
        return 1


@dataclass
class CostResult:
    """cost function 返回值。"""

    flops: int = 0                 # 浮点运算量
    launch_cycles: int = 0         # 管线启动开销（cycles）


# Python 函数注册表
_COST_FN_REGISTRY: dict[str, Callable[[CostContext], CostResult]] = {}


def register_cost_fn(op_name: str) -> Callable:
    """装饰器：注册 per-op cost function。

    用法::

        @register_cost_fn("cube_matmul")
        def _cube_matmul_cost(ctx: CostContext) -> CostResult:
            flops = 2 * ctx.batch * ctx.M * ctx.N * ctx.K
            launch = 80 if ctx.input_formats[0] == "zz" else 100
            return CostResult(flops=flops, launch_cycles=launch)
    """
    def decorator(fn: Callable[[CostContext], CostResult]) -> Callable:
        _COST_FN_REGISTRY[op_name] = fn
        return fn
    return decorator


def _build_cost_context(
    node: Node, graph: Graph, hw: RooflineHwParams,
) -> CostContext:
    """从 node + graph 构建 CostContext。"""
    inputs = [graph.tensors[tid] for tid in node.inputs if tid in graph.tensors]
    outputs = [graph.tensors[tid] for tid in node.outputs if tid in graph.tensors]
    compute_dtype = node.params.get("compute_dtype", "fp16")

    return CostContext(
        node=node,
        inputs=inputs,
        outputs=outputs,
        hw=hw,
        params=node.params,
        compute_dtype=compute_dtype,
        input_formats=[t.format or "nd" for t in inputs],
        output_formats=[t.format or "nd" for t in outputs],
        input_storage=[t.storage or "hbm" for t in inputs],
        output_storage=[t.storage or "hbm" for t in outputs],
        is_fused="_fusion_group" in node.params,
        fusion_role=node.params.get("_fusion_role"),
    )


# ── 代价模型配置 ─────────────────────────────────────────


# 内置默认值（无 cost_model_config.yaml 时使用）
_DEFAULT_UNIT_COSTS: dict[str, dict] = {
    "cube": {"flops_formula": "matmul", "launch_cycles": 100},
    "vector": {"flops_formula": "elementwise", "flops_multiplier": 1, "launch_cycles": 10},
    "dma": {"flops_formula": "none", "launch_cycles": 5},
    "idma": {"flops_formula": "none", "launch_cycles": 5},
}


@dataclass
class OpCostParams:
    """单个算子的代价参数。"""

    flops_formula: str = "elementwise"  # matmul | elementwise | none
    flops_multiplier: int = 1           # elementwise 模式下每元素运算量
    launch_cycles: int = 10             # 管线启动/填充开销


@dataclass
class CostModel:
    """代价模型：unit 默认 + per-op 覆盖。"""

    unit_defaults: dict[str, OpCostParams] = field(default_factory=dict)
    op_overrides: dict[str, OpCostParams] = field(default_factory=dict)

    def get(self, npu_op: str, compute_unit: str) -> OpCostParams:
        """按 npu_op 查找代价参数，fallback 到 unit 默认。"""
        if npu_op in self.op_overrides:
            return self.op_overrides[npu_op]
        cu = compute_unit.lower() if compute_unit else "vector"
        return self.unit_defaults.get(cu, OpCostParams())


def _parse_cost_params(spec: dict, base: OpCostParams | None = None) -> OpCostParams:
    """从 dict 解析 OpCostParams，缺省字段继承 base。"""
    if base is None:
        base = OpCostParams()
    return OpCostParams(
        flops_formula=spec.get("flops_formula", base.flops_formula),
        flops_multiplier=spec.get("flops_multiplier", base.flops_multiplier),
        launch_cycles=spec.get("launch_cycles", base.launch_cycles),
    )


def parse_cost_model(config: dict | None) -> CostModel:
    """从 cost_model_config 构建 CostModel。

    config 为 None 或空时使用内置默认值。
    """
    model = CostModel()

    # unit defaults
    raw_defaults = (config or {}).get("unit_defaults", _DEFAULT_UNIT_COSTS)
    for unit, spec in raw_defaults.items():
        model.unit_defaults[unit] = _parse_cost_params(spec)

    # 确保所有 unit 类型都有默认值
    for unit, spec in _DEFAULT_UNIT_COSTS.items():
        if unit not in model.unit_defaults:
            model.unit_defaults[unit] = _parse_cost_params(spec)

    # per-op overrides（继承 unit 默认）
    raw_overrides = (config or {}).get("op_overrides", {})
    for op_name, spec in raw_overrides.items():
        # 从 op 名前缀推断 unit 类型
        unit = _infer_unit(op_name)
        base = model.unit_defaults.get(unit, OpCostParams())
        model.op_overrides[op_name] = _parse_cost_params(spec, base)

    return model


def _infer_unit(op_name: str) -> str:
    """从 npu_op 名推断计算单元类型。"""
    if op_name.startswith("cube"):
        return "cube"
    if op_name.startswith("dma"):
        return "dma"
    if op_name.startswith("idma"):
        return "idma"
    return "vector"


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


def _try_cost_fn(
    node: Node, graph: Graph, hw: RooflineHwParams,
) -> CostResult | None:
    """尝试调用注册的 Python cost function，未注册返回 None。"""
    op = node.npu_op or node.op_type
    fn = _COST_FN_REGISTRY.get(op)
    if fn is None:
        return None
    ctx = _build_cost_context(node, graph, hw)
    return fn(ctx)


def estimate_flops(
    node: Node, graph: Graph, hw: RooflineHwParams,
    cost_model: CostModel | None = None,
) -> int:
    """估算节点的浮点运算量。

    优先级：Python 函数注册 > YAML per-op > YAML unit 默认 > 硬编码。
    """
    # 1. Python 函数注册
    result = _try_cost_fn(node, graph, hw)
    if result is not None:
        return result.flops

    op = node.npu_op or node.op_type
    cu = (node.compute_unit or "vector").lower()

    # 2. YAML 配置
    if cost_model:
        params = cost_model.get(op, cu)
        if params.flops_formula == "matmul":
            return _matmul_flops(node, graph)
        if params.flops_formula == "none":
            return 0
        return _output_element_count(node, graph) * params.flops_multiplier

    # 3. 硬编码 fallback
    if op in ("cube_matmul", "cube_matmul_bias"):
        return _matmul_flops(node, graph)
    if cu in ("dma", "idma"):
        return 0
    return _output_element_count(node, graph)


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
            total += calc_padded_size(t.shape, t.dtype, t.format, get_dim_align(t.format, t.dtype))
    return total


def estimate_dma_bytes(node: Node, graph: Graph, hw: RooflineHwParams) -> int:
    """估算节点实际的 HBM DMA 搬运字节数。

    storage=local 的 tensor 已在 L1 中，不产生 HBM DMA。
    storage=pipe 的 tensor 走硬件直连，不产生任何搬运。
    只有 storage=hbm 的 tensor 需要 DMA load/store。
    """
    total = 0
    for tid in list(node.inputs) + list(node.outputs):
        t = graph.tensors.get(tid)
        if not t:
            continue
        if t.storage in ("local", "pipe"):
            continue  # 不产生 HBM DMA
        total += calc_padded_size(t.shape, t.dtype, t.format, get_dim_align(t.format, t.dtype))
    return total


# ── Cycle 估算 ───────────────────────────────────────────


def estimate_node_cycles(
    node: Node, graph: Graph, hw: RooflineHwParams,
    cost_model: CostModel | None = None,
) -> tuple[int, int, int]:
    """估算节点的计算、DMA 和总时延（cycles）。

    优先级：Python 函数注册 > YAML per-op > YAML unit 默认 > 硬编码。

    Returns:
        (compute_cycles, dma_cycles, node_cycles)
        node_cycles = max(compute_cycles, dma_cycles)，即瓶颈决定时延。
    """
    op = node.npu_op or node.op_type
    cu = (node.compute_unit or "vector").lower()
    dma_bytes = estimate_dma_bytes(node, graph, hw)

    # 1. Python 函数注册 — 优先级最高
    cost_result = _try_cost_fn(node, graph, hw)
    if cost_result is not None:
        flops = cost_result.flops
        launch = cost_result.launch_cycles
    else:
        flops = estimate_flops(node, graph, hw, cost_model)
        # launch_cycles 从 YAML
        launch = 0
        if cost_model:
            params = cost_model.get(op, cu)
            launch = params.launch_cycles

    # 计算 cycles
    if cu == "cube":
        ops_rate = hw.cube_ops_per_cycle
    elif cu in ("dma", "idma"):
        ops_rate = 1
    else:
        ops_rate = hw.vector_ops_per_cycle

    compute_cycles = (flops // ops_rate if ops_rate > 0 else 0) + launch

    # DMA cycles
    dma_cycles = dma_bytes // hw.dma_bytes_per_cycle if hw.dma_bytes_per_cycle > 0 else 0

    node_cycles = max(compute_cycles, dma_cycles)
    return compute_cycles, dma_cycles, node_cycles


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
    """为每个节点标注 roofline 分析结果 + 时延估算。"""
    hw_config = config.get("hardware", None) if config else None
    hw = RooflineHwParams.from_config(hw_config)
    cost_config = config.get("cost_model", None) if config else None
    cost_model = parse_cost_model(cost_config)

    compute_count = 0
    memory_count = 0
    total_compute_cycles = 0
    total_dma_cycles = 0
    total_node_cycles = 0

    for nid, node in graph.nodes.items():
        flops = estimate_flops(node, graph, hw, cost_model)
        nbytes = estimate_bytes(node, graph, hw)
        dma_nbytes = estimate_dma_bytes(node, graph, hw)
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
        achievable_ratio = min(oi / ridge, 1.0) if ridge > 0 else 0.0

        comp_cy, dma_cy, node_cy = estimate_node_cycles(node, graph, hw, cost_model)
        total_compute_cycles += comp_cy
        total_dma_cycles += dma_cy
        total_node_cycles += node_cy

        node.params["_roofline"] = {
            "flops": flops,
            "bytes": nbytes,
            "dma_bytes": dma_nbytes,
            "oi": round(oi, 4),
            "bottleneck": bottleneck,
            "achievable_ratio": round(achievable_ratio, 4),
            "compute_cycles": comp_cy,
            "dma_cycles": dma_cy,
            "node_cycles": node_cy,
        }

        bnd = "计算受限" if bottleneck == "compute" else "访存受限"
        hint = ("优化方向：减少计算量（算子融合/低精度）" if bottleneck == "compute"
                else "优化方向：减少数据搬运（tiling/bypass/融合）")
        log_opt(
            node, "roofline_analyzer", bnd,
            f"算术强度 {round(oi, 2)} {'≥' if bottleneck == 'compute' else '<'} "
            f"拐点 {round(ridge, 2)}，硬件利用率 {round(achievable_ratio * 100)}%。"
            f"时延: compute={comp_cy} dma={dma_cy} total={node_cy} cycles。{hint}",
        )

        if bottleneck == "compute":
            compute_count += 1
        else:
            memory_count += 1

    # graph 级别汇总（动态属性）
    object.__setattr__(graph, "_roofline_summary", {
        "total_cycles": total_node_cycles,
        "total_compute_cycles": total_compute_cycles,
        "total_dma_cycles": total_dma_cycles,
        "compute_bound_nodes": compute_count,
        "memory_bound_nodes": memory_count,
    })

    total = len(graph.nodes)
    logger.info(
        "roofline analysis: %d nodes — %d compute-bound, %d memory-bound, "
        "total %d cycles (compute=%d, dma=%d)",
        total, compute_count, memory_count,
        total_node_cycles, total_compute_cycles, total_dma_cycles,
    )

    return graph
