"""Tests for roofline analyzer."""

from __future__ import annotations

from torch2c.common import Graph, Node, Tensor
from torch2c.roofline.roofline_analyzer import (
    RooflineHwParams,
    _ridge_point,
    estimate_bytes,
    estimate_flops,
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
    # 3 tensors x 4x16 x 2 bytes = 384
    expected = 3 * (4 * 16 * 2)
    assert nbytes == expected
