"""format_annotator 单元测试。"""

from __future__ import annotations

from torch2c.common import Graph, Node, Tensor
from torch2c.format_annotator import post_validate, run


def _make_graph(dtype: str = "fp16") -> Graph:
    """创建 matmul → add 的测试图。"""
    g = Graph()
    g.add_tensor(Tensor(id="t_a", shape=[1, 32, 64], dtype=dtype, consumer_node_ids=["n_mm"]))
    g.add_tensor(
        Tensor(id="t_b", shape=[64, 64], dtype=dtype, is_weight=True, consumer_node_ids=["n_mm"])
    )
    g.add_tensor(
        Tensor(
            id="t_mm_out",
            shape=[1, 32, 64],
            dtype=dtype,
            producer_node_id="n_mm",
            consumer_node_ids=["n_add"],
        )
    )
    g.add_tensor(
        Tensor(id="t_bias", shape=[64], dtype=dtype, is_weight=True, consumer_node_ids=["n_add"])
    )
    g.add_tensor(Tensor(id="t_add_out", shape=[1, 32, 64], dtype=dtype, producer_node_id="n_add"))

    g.add_node(
        Node(
            id="n_mm",
            op_type="aten.mm",
            npu_op="cube_matmul",
            inputs=["t_a", "t_b"],
            outputs=["t_mm_out"],
            is_mapped=True,
        )
    )
    g.add_node(
        Node(
            id="n_add",
            op_type="aten.add",
            npu_op="vector_add",
            inputs=["t_mm_out", "t_bias"],
            outputs=["t_add_out"],
            is_mapped=True,
        )
    )
    return g


def test_inherit_model_dtype():
    """默认继承模型原始 dtype。"""
    g = _make_graph(dtype="fp32")
    result = run(g, {})
    mm_out = result.get_tensor("t_mm_out")
    assert mm_out is not None
    assert mm_out.dtype == "fp32"
    add_out = result.get_tensor("t_add_out")
    assert add_out is not None
    assert add_out.dtype == "fp32"


def test_inherit_model_format():
    """默认继承模型原始 format（nd）。"""
    g = _make_graph()
    result = run(g, {})
    mm_out = result.get_tensor("t_mm_out")
    assert mm_out is not None
    assert mm_out.format == "nd"


def test_target_dtype_override():
    """target_dtype 覆盖模型 dtype。"""
    g = _make_graph(dtype="fp32")
    result = run(g, {"target_dtype": "fp16"})
    mm_out = result.get_tensor("t_mm_out")
    assert mm_out is not None
    assert mm_out.dtype == "fp16"
    add_out = result.get_tensor("t_add_out")
    assert add_out is not None
    assert add_out.dtype == "fp16"


def test_target_format_override():
    """target_format 覆盖模型 format。"""
    g = _make_graph()
    result = run(g, {"target_format": "nz"})
    mm_out = result.get_tensor("t_mm_out")
    assert mm_out is not None
    assert mm_out.format == "nz"
    add_out = result.get_tensor("t_add_out")
    assert add_out is not None
    assert add_out.format == "nz"


def test_annotation_structure():
    """format_annotation 字段结构正确。"""
    g = _make_graph()
    result = run(g, {"target_dtype": "fp16", "target_format": "nz"})

    mm = result.get_node("n_mm")
    assert mm is not None
    ann = mm.format_annotation
    assert ann is not None
    assert "inputs" in ann
    assert "outputs" in ann
    assert ann["inputs"][0]["format"] == "nz"
    assert ann["inputs"][0]["dtype"] == "fp16"
    assert ann["outputs"][0]["format"] == "nz"
    assert ann["outputs"][0]["dtype"] == "fp16"

    add_node = result.get_node("n_add")
    assert add_node is not None
    add_ann = add_node.format_annotation
    assert add_ann is not None
    assert add_ann["inputs"][0]["format"] == "nz"


def test_empty_config_inherits():
    """空 config 完全继承模型值。"""
    g = _make_graph(dtype="fp16")
    result = run(g, {})
    for t in result.tensors.values():
        assert t.dtype == "fp16"
        assert t.format == "nd"


class TestPostValidate:
    def test_valid(self):
        g = Graph()
        g.add_tensor(Tensor(id="t0", shape=[1], dtype="fp16", producer_node_id="n0"))
        g.add_node(Node(id="n0", op_type="op", outputs=["t0"]))
        assert post_validate(g) == []

    def test_missing_dtype(self):
        g = Graph()
        g.add_tensor(Tensor(id="t0", shape=[1], dtype="", producer_node_id="n0"))
        g.add_node(Node(id="n0", op_type="op", outputs=["t0"]))
        errors = post_validate(g)
        assert any("dtype" in e for e in errors)
