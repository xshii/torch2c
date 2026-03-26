"""Tests for roofline analyzer."""

from __future__ import annotations

from torch2c.common import Graph, Node, Tensor
from torch2c.optpass.cd_roofline.roofline_analyzer import (
    CostContext,
    CostModel,
    CostResult,
    OpCostParams,
    RooflineHwParams,
    _COST_FN_REGISTRY,
    _build_cost_context,
    _ridge_point,
    estimate_bytes,
    estimate_dma_bytes,
    estimate_flops,
    estimate_node_cycles,
    parse_cost_model,
    register_cost_fn,
    run,
)


# ── helpers ──────────────────────────────────────────────


def _make_graph(
    node: Node,
    tensors: list[Tensor],
) -> Graph:
    """构建包含单个节点的测试图。"""
    g = Graph()
    for t in tensors:
        g.add_tensor(t)
    g.add_node(node)
    return g


def _default_hw() -> RooflineHwParams:
    return RooflineHwParams()


# ── tests ────────────────────────────────────────────────


def test_matmul_is_compute_bound() -> None:
    """大矩阵乘法应被判定为 compute-bound。"""
    # M=256, K=256, N=256 → flops = 2*256*256*256 = 33554432
    a = Tensor(id="a", shape=[256, 256], dtype="fp16")
    b = Tensor(id="b", shape=[256, 256], dtype="fp16")
    c = Tensor(id="c", shape=[256, 256], dtype="fp16")
    node = Node(
        id="mm0", op_type="aten.mm.default",
        inputs=["a", "b"], outputs=["c"],
        compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
    )
    graph = _make_graph(node, [a, b, c])
    hw = _default_hw()

    flops = estimate_flops(node, graph, hw)
    nbytes = estimate_bytes(node, graph, hw)
    oi = flops / nbytes
    ridge = _ridge_point("cube", hw)

    assert flops == 2 * 256 * 256 * 256
    assert oi > ridge, f"oi={oi:.2f} should exceed ridge={ridge:.2f}"

    run(graph)
    assert graph.nodes["mm0"].params["_roofline"]["bottleneck"] == "compute"


def test_vector_add_is_memory_bound() -> None:
    """element-wise add 应被判定为 memory-bound。"""
    a = Tensor(id="a", shape=[1, 32], dtype="fp16")
    b = Tensor(id="b", shape=[1, 32], dtype="fp16")
    c = Tensor(id="c", shape=[1, 32], dtype="fp16")
    node = Node(
        id="add0", op_type="aten.add.Tensor",
        inputs=["a", "b"], outputs=["c"],
        compute_unit="vector", npu_op="vector_add", is_mapped=True,
    )
    graph = _make_graph(node, [a, b, c])
    run(graph)

    info = graph.nodes["add0"].params["_roofline"]
    assert info["bottleneck"] == "memory"


def test_dma_is_memory_bound() -> None:
    """DMA 搬运无计算，应为 memory-bound。"""
    a = Tensor(id="a", shape=[4, 64], dtype="fp16")
    b = Tensor(id="b", shape=[4, 64], dtype="fp16")
    node = Node(
        id="dma0", op_type="dma_move",
        inputs=["a"], outputs=["b"],
        compute_unit="dma", npu_op="dma_move", is_mapped=True,
    )
    graph = _make_graph(node, [a, b])
    run(graph)

    info = graph.nodes["dma0"].params["_roofline"]
    assert info["flops"] == 0
    assert info["bottleneck"] == "memory"


def test_flops_matmul() -> None:
    """验证 matmul FLOPS 公式 2*M*N*K。"""
    m, k, n = 64, 128, 32
    a = Tensor(id="a", shape=[m, k], dtype="fp16")
    b = Tensor(id="b", shape=[k, n], dtype="fp16")
    c = Tensor(id="c", shape=[m, n], dtype="fp16")
    node = Node(
        id="mm0", op_type="aten.mm.default",
        inputs=["a", "b"], outputs=["c"],
        compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
    )
    graph = _make_graph(node, [a, b, c])
    hw = _default_hw()

    flops = estimate_flops(node, graph, hw)
    assert flops == 2 * m * n * k


def test_bytes_includes_all_tensors() -> None:
    """验证 bytes 计算包含所有输入和输出 tensor。"""
    a = Tensor(id="a", shape=[4, 16], dtype="fp16", format="nd")
    b = Tensor(id="b", shape=[4, 16], dtype="fp16", format="nd")
    c = Tensor(id="c", shape=[4, 16], dtype="fp16", format="nd")
    node = Node(
        id="add0", op_type="aten.add.Tensor",
        inputs=["a", "b"], outputs=["c"],
        compute_unit="vector", npu_op="vector_add", is_mapped=True,
    )
    graph = _make_graph(node, [a, b, c])
    hw = _default_hw()

    nbytes = estimate_bytes(node, graph, hw)
    # ND format fp16: dim_align=(1,16), dim[-2]=4 not padded, dim[-1]=16 already aligned
    # 3 tensors x 4x16 x 2 bytes = 384
    expected = 3 * (4 * 16 * 2)
    assert nbytes == expected


# ── cycle estimation tests ──────────────────────────────


def test_cycle_estimation_matmul() -> None:
    """matmul cycle 估算：compute_cycles = flops / ops_per_cycle + launch。"""
    a = Tensor(id="a", shape=[256, 256], dtype="fp16")
    b = Tensor(id="b", shape=[256, 256], dtype="fp16")
    c = Tensor(id="c", shape=[256, 256], dtype="fp16")
    node = Node(
        id="mm0", op_type="aten.mm.default",
        inputs=["a", "b"], outputs=["c"],
        compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
    )
    graph = _make_graph(node, [a, b, c])
    hw = _default_hw()

    comp_cy, dma_cy, node_cy = estimate_node_cycles(node, graph, hw)
    expected_flops = 2 * 256 * 256 * 256
    # Python cost fn 注册后 launch=100（ND format 默认）
    assert comp_cy == expected_flops // hw.cube_ops_per_cycle + 100
    assert node_cy == max(comp_cy, dma_cy)


def test_dma_bytes_excludes_local() -> None:
    """storage=local 的 tensor 不计入 DMA 字节数。"""
    a = Tensor(id="a", shape=[4, 16], dtype="fp16", format="nd", storage="local")
    b = Tensor(id="b", shape=[4, 16], dtype="fp16", format="nd")  # hbm
    c = Tensor(id="c", shape=[4, 16], dtype="fp16", format="nd")  # hbm
    node = Node(
        id="add0", op_type="aten.add.Tensor",
        inputs=["a", "b"], outputs=["c"],
        compute_unit="vector", npu_op="vector_add", is_mapped=True,
    )
    graph = _make_graph(node, [a, b, c])
    hw = _default_hw()

    total_bytes = estimate_bytes(node, graph, hw)
    dma_bytes = estimate_dma_bytes(node, graph, hw)
    # a is local → excluded from dma_bytes
    assert dma_bytes < total_bytes
    assert dma_bytes == 2 * (4 * 16 * 2)  # only b + c


def test_dma_bytes_excludes_pipe() -> None:
    """storage=pipe 的 tensor 不计入 DMA 字节数。"""
    a = Tensor(id="a", shape=[4, 16], dtype="fp16", format="nd", storage="pipe")
    c = Tensor(id="c", shape=[4, 16], dtype="fp16", format="nd")
    node = Node(
        id="relu0", op_type="relu",
        inputs=["a"], outputs=["c"],
        compute_unit="vector", npu_op="vector_relu", is_mapped=True,
    )
    graph = _make_graph(node, [a, c])
    hw = _default_hw()

    dma_bytes = estimate_dma_bytes(node, graph, hw)
    assert dma_bytes == 4 * 16 * 2  # only c (hbm)


def test_roofline_summary_on_graph() -> None:
    """run 后 graph 上有 _roofline_summary。"""
    a = Tensor(id="a", shape=[64, 64], dtype="fp16")
    b = Tensor(id="b", shape=[64, 64], dtype="fp16")
    c = Tensor(id="c", shape=[64, 64], dtype="fp16")
    node = Node(
        id="mm0", op_type="aten.mm.default",
        inputs=["a", "b"], outputs=["c"],
        compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
    )
    graph = _make_graph(node, [a, b, c])
    run(graph)

    summary = graph.metadata.get("roofline_summary")
    assert summary is not None
    assert summary["total_cycles"] > 0
    assert "compute_bound_nodes" in summary
    assert "memory_bound_nodes" in summary


def test_node_roofline_has_cycle_fields() -> None:
    """每个节点的 _roofline 包含 cycle 字段。"""
    a = Tensor(id="a", shape=[32, 32], dtype="fp16")
    b = Tensor(id="b", shape=[32, 32], dtype="fp16")
    node = Node(
        id="relu0", op_type="relu",
        inputs=["a"], outputs=["b"],
        compute_unit="vector", npu_op="vector_relu", is_mapped=True,
    )
    graph = _make_graph(node, [a, b])
    run(graph)

    info = graph.nodes["relu0"].params["_roofline"]
    assert "compute_cycles" in info
    assert "dma_cycles" in info
    assert "node_cycles" in info
    assert "dma_bytes" in info


# ── cost model config tests ─────────────────────────────


def test_cost_model_flops_multiplier() -> None:
    """per-op flops_multiplier 覆盖默认值。"""
    cost_config = {
        "unit_defaults": {
            "vector": {"flops_formula": "elementwise", "flops_multiplier": 1},
        },
        "op_overrides": {
            "vector_gelu": {"flops_multiplier": 8},
        },
    }
    cm = parse_cost_model(cost_config)
    hw = _default_hw()

    # gelu: 8x multiplier
    a = Tensor(id="a", shape=[1, 32], dtype="fp16")
    b = Tensor(id="b", shape=[1, 32], dtype="fp16")
    gelu = Node(id="g0", op_type="gelu", inputs=["a"], outputs=["b"],
                compute_unit="vector", npu_op="vector_gelu", is_mapped=True)
    graph = _make_graph(gelu, [a, b])
    flops = estimate_flops(gelu, graph, hw, cm)
    assert flops == 32 * 8

    # add: default 1x
    add = Node(id="a0", op_type="add", inputs=["a"], outputs=["b"],
               compute_unit="vector", npu_op="vector_add", is_mapped=True)
    flops_add = estimate_flops(add, graph, hw, cm)
    assert flops_add == 32 * 1


def test_cost_model_launch_cycles() -> None:
    """YAML launch_cycles 对未注册 Python fn 的算子生效。"""
    cost_config = {
        "op_overrides": {
            # 用一个没有 Python 注册的算子来测 YAML 覆盖
            "vector_relu": {"launch_cycles": 200},
        },
    }
    cm = parse_cost_model(cost_config)
    hw = _default_hw()

    a = Tensor(id="a", shape=[1, 128], dtype="fp16")
    b = Tensor(id="b", shape=[1, 128], dtype="fp16")
    node = Node(id="r0", op_type="relu", inputs=["a"], outputs=["b"],
                compute_unit="vector", npu_op="vector_relu", is_mapped=True)
    graph = _make_graph(node, [a, b])

    comp_cy, _, _ = estimate_node_cycles(node, graph, hw, cm)
    flops = 128  # elem_count * 1
    expected = flops // hw.vector_ops_per_cycle + 200
    assert comp_cy == expected


def test_cost_model_none_fallback() -> None:
    """cost_model=None 时使用硬编码 fallback。"""
    hw = _default_hw()
    a = Tensor(id="a", shape=[1, 64], dtype="fp16")
    b = Tensor(id="b", shape=[1, 64], dtype="fp16")
    node = Node(id="r0", op_type="relu", inputs=["a"], outputs=["b"],
                compute_unit="vector", npu_op="vector_relu", is_mapped=True)
    graph = _make_graph(node, [a, b])

    # 无 cost_model → fallback 到硬编码 multiplier=1
    flops = estimate_flops(node, graph, hw, None)
    assert flops == 64


def test_parse_cost_model_inherits_unit_defaults() -> None:
    """op_overrides 缺省字段从 unit_defaults 继承。"""
    cost_config = {
        "unit_defaults": {
            "vector": {"flops_formula": "elementwise", "flops_multiplier": 1, "launch_cycles": 15},
        },
        "op_overrides": {
            "vector_relu": {"flops_multiplier": 2},  # 只覆盖 multiplier
        },
    }
    cm = parse_cost_model(cost_config)
    params = cm.get("vector_relu", "vector")
    assert params.flops_multiplier == 2
    assert params.launch_cycles == 15   # 继承
    assert params.flops_formula == "elementwise"  # 继承


# ── Python cost function tests ──────────────────────────


def test_python_cost_fn_overrides_yaml() -> None:
    """注册的 Python function 优先于 YAML 配置。"""
    hw = _default_hw()

    # cube_matmul 已通过 _builtin_costs.py 注册
    assert "cube_matmul" in _COST_FN_REGISTRY

    a = Tensor(id="a", shape=[32, 64], dtype="fp16", format="zz")
    b = Tensor(id="b", shape=[64, 16], dtype="fp16", format="nz")
    c = Tensor(id="c", shape=[32, 16], dtype="fp16")
    node = Node(id="mm0", op_type="mm", inputs=["a", "b"], outputs=["c"],
                compute_unit="cube", npu_op="cube_matmul", is_mapped=True)
    graph = _make_graph(node, [a, b, c])

    # Python function 应该被调用
    flops = estimate_flops(node, graph, hw)
    assert flops == 2 * 32 * 16 * 64  # 2*M*N*K


def test_python_cost_fn_format_aware() -> None:
    """Python cost function 可以根据 format 调整 launch_cycles。"""
    hw = _default_hw()

    # ZZ format → launch=80, ND format → launch=100
    a_zz = Tensor(id="a", shape=[16, 16], dtype="fp16", format="zz")
    a_nd = Tensor(id="a", shape=[16, 16], dtype="fp16", format="nd")
    b = Tensor(id="b", shape=[16, 16], dtype="fp16", format="nz")
    c = Tensor(id="c", shape=[16, 16], dtype="fp16")

    node = Node(id="mm0", op_type="mm", inputs=["a", "b"], outputs=["c"],
                compute_unit="cube", npu_op="cube_matmul", is_mapped=True)

    # ZZ format
    g_zz = _make_graph(node, [a_zz, b, c])
    comp_zz, _, _ = estimate_node_cycles(node, g_zz, hw)

    # ND format
    g_nd = _make_graph(node, [a_nd, b, c])
    comp_nd, _, _ = estimate_node_cycles(node, g_nd, hw)

    # ZZ 应该更快（launch=80 vs 100）
    assert comp_zz < comp_nd


def test_cost_context_has_all_fields() -> None:
    """CostContext 包含完整的上下文信息。"""
    hw = _default_hw()
    a = Tensor(id="a", shape=[4, 32], dtype="fp16", format="zz", storage="local")
    b = Tensor(id="b", shape=[32, 16], dtype="fp16", format="nz")
    c = Tensor(id="c", shape=[4, 16], dtype="fp16", format="nd")
    node = Node(id="mm0", op_type="mm", inputs=["a", "b"], outputs=["c"],
                compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
                params={"compute_dtype": "fp32", "_fusion_group": "fg_0",
                        "_fusion_role": "head"})
    graph = _make_graph(node, [a, b, c])

    ctx = _build_cost_context(node, graph, hw)
    assert len(ctx.inputs) == 2
    assert len(ctx.outputs) == 1
    assert ctx.compute_dtype == "fp32"
    assert ctx.input_formats == ["zz", "nz"]
    assert ctx.output_formats == ["nd"]
    assert ctx.input_storage == ["local", "hbm"]
    assert ctx.is_fused is True
    assert ctx.fusion_role == "head"
    assert ctx.M == 4
    assert ctx.N == 16
    assert ctx.K == 32
    assert ctx.batch == 1
    assert ctx.elem_count == 4 * 16


def test_custom_cost_fn_registration() -> None:
    """自定义 cost function 注册和使用。"""
    hw = _default_hw()

    # 注册一个自定义算子
    @register_cost_fn("_test_custom_op")
    def _custom_cost(ctx: CostContext) -> CostResult:
        return CostResult(flops=ctx.elem_count * 42, launch_cycles=999)

    a = Tensor(id="a", shape=[2, 8], dtype="fp16")
    b = Tensor(id="b", shape=[2, 8], dtype="fp16")
    node = Node(id="c0", op_type="custom", inputs=["a"], outputs=["b"],
                compute_unit="vector", npu_op="_test_custom_op", is_mapped=True)
    graph = _make_graph(node, [a, b])

    flops = estimate_flops(node, graph, hw)
    assert flops == 16 * 42

    comp_cy, _, _ = estimate_node_cycles(node, graph, hw)
    expected = (16 * 42) // hw.vector_ops_per_cycle + 999
    assert comp_cy == expected

    # 清理
    del _COST_FN_REGISTRY["_test_custom_op"]
