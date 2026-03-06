"""op_mapping 单元测试。"""

from __future__ import annotations

import pathlib

from ...common.config_loader import load_config
from ...common.graph_ir import Graph, Node, Tensor
from ..op_mapping import post_validate, run

_CONFIG_PATH = str(
    pathlib.Path(__file__).resolve().parents[2] / "integration" / "config" / "direct_mappings.yaml"
)


def _make_5node_graph() -> Graph:
    """创建包含 5 个已知算子的测试图。"""
    g = Graph()

    # tensors
    for tid, shape, is_w, is_in in [
        ("t_in", [1, 32, 64], False, True),
        ("t_w", [64, 64], True, False),
        ("t_mm", [1, 32, 64], False, False),
        ("t_bias", [64], True, False),
        ("t_add", [1, 32, 64], False, False),
        ("t_scale", [1], True, False),
        ("t_mul", [1, 32, 64], False, False),
        ("t_gelu", [1, 32, 64], False, False),
        ("t_out", [1, 32, 64], False, False),
    ]:
        g.add_tensor(
            Tensor(id=tid, shape=shape, dtype="fp16", is_weight=is_w, is_model_input=is_in)
        )

    for nid, op, ins, outs in [
        ("n0", "aten.mm.default", ["t_in", "t_w"], ["t_mm"]),
        ("n1", "aten.add.Tensor", ["t_mm", "t_bias"], ["t_add"]),
        ("n2", "aten.mul.Tensor", ["t_add", "t_scale"], ["t_mul"]),
        ("n3", "aten.gelu.default", ["t_mul"], ["t_gelu"]),
        ("n4", "aten.view.default", ["t_gelu"], ["t_out"]),
    ]:
        g.add_node(Node(id=nid, op_type=op, inputs=ins, outputs=outs))

    return g


# ---- tests ----


def test_basic_mapping():
    """5 个算子全部映射成功。"""
    graph = _make_5node_graph()
    config = load_config(_CONFIG_PATH, required_keys=["mappings"])
    run(graph, config)

    for node in graph.nodes.values():
        assert node.is_mapped, f"{node.id} ({node.op_type}) should be mapped"
        assert node.npu_op is not None
        assert node.compute_unit is not None


def test_unmapped_preserved():
    """不在配置中的算子保持未映射。"""
    g = Graph()
    g.add_tensor(Tensor(id="t0", shape=[1], dtype="fp16"))
    g.add_tensor(Tensor(id="t1", shape=[1], dtype="fp16"))
    g.add_node(Node(id="unknown", op_type="aten.fake_op.default", inputs=["t0"], outputs=["t1"]))

    config = load_config(_CONFIG_PATH, required_keys=["mappings"])
    run(g, config)

    n = g.get_node("unknown")
    assert n is not None
    assert not n.is_mapped
    assert n.npu_op is None


def test_compute_unit():
    """验证 matmul→cube, add→vector。"""
    graph = _make_5node_graph()
    config = load_config(_CONFIG_PATH, required_keys=["mappings"])
    run(graph, config)

    n0 = graph.get_node("n0")
    assert n0 is not None
    assert n0.compute_unit == "cube"  # mm
    n1 = graph.get_node("n1")
    assert n1 is not None
    assert n1.compute_unit == "vector"  # add
    n4 = graph.get_node("n4")
    assert n4 is not None
    assert n4.compute_unit == "scalar"  # view/reshape


def test_empty_graph():
    """空图不报错。"""
    graph = Graph()
    result = run(graph, {"mappings": {}})
    assert len(result.nodes) == 0


def test_already_mapped_skipped():
    """已映射的节点不会被重复映射。"""
    g = Graph()
    g.add_tensor(Tensor(id="t0", shape=[1, 32, 64], dtype="fp16"))
    g.add_tensor(Tensor(id="t1", shape=[1, 32, 64], dtype="fp16"))
    g.add_node(
        Node(
            id="n0",
            op_type="aten.mm.default",
            inputs=["t0"],
            outputs=["t1"],
            npu_op="custom_op",
            compute_unit="cube",
            is_mapped=True,
        )
    )

    config = load_config(_CONFIG_PATH, required_keys=["mappings"])
    run(g, config)

    # 应保持原有映射，不被覆盖
    n0 = g.get_node("n0")
    assert n0 is not None
    assert n0.npu_op == "custom_op"


def test_mixed_mapped_unmapped():
    """混合已映射和未映射节点的图。"""
    g = Graph()
    for tid in ["t0", "t1", "t2", "t3"]:
        g.add_tensor(Tensor(id=tid, shape=[1], dtype="fp16"))

    g.add_node(Node(id="n0", op_type="aten.mm.default", inputs=["t0"], outputs=["t1"]))
    g.add_node(
        Node(id="n1", op_type="aten.unknown_custom_op.default", inputs=["t1"], outputs=["t2"])
    )
    g.add_node(Node(id="n2", op_type="aten.add.Tensor", inputs=["t2"], outputs=["t3"]))

    config = load_config(_CONFIG_PATH, required_keys=["mappings"])
    run(g, config)

    n0 = g.get_node("n0")
    assert n0 is not None
    assert n0.is_mapped  # mm → cube_matmul
    n1 = g.get_node("n1")
    assert n1 is not None
    assert not n1.is_mapped  # unknown op 不在直接映射表
    n2 = g.get_node("n2")
    assert n2 is not None
    assert n2.is_mapped  # add → vector_add


class TestPostValidate:
    def test_valid(self):
        g = Graph()
        g.add_tensor(Tensor(id="t0", shape=[1], dtype="fp16"))
        g.add_tensor(Tensor(id="t1", shape=[1], dtype="fp16", producer_node_id="n0"))
        g.add_node(
            Node(
                id="n0",
                op_type="vector_add",
                inputs=["t0"],
                outputs=["t1"],
                is_mapped=True,
                npu_op="vector_add",
            )
        )
        assert post_validate(g) == []

    def test_missing_npu_op(self):
        g = Graph()
        g.add_tensor(Tensor(id="t0", shape=[1], dtype="fp16"))
        g.add_tensor(Tensor(id="t1", shape=[1], dtype="fp16", producer_node_id="n0"))
        g.add_node(Node(id="n0", op_type="vector_add", inputs=["t0"], outputs=["t1"], is_mapped=True))
        errors = post_validate(g)
        assert any("npu_op" in e for e in errors)
