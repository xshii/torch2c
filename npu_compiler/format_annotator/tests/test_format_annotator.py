"""format_annotator 单元测试。"""

from __future__ import annotations

from npu_compiler.common import Graph, Node, Tensor
from npu_compiler.format_annotator import run

_CONFIG = {
    "op_format_requirements": {
        "npu_matmul": {
            "inputs": [{"format": "nz", "dtype": "fp16"}, {"format": "nz", "dtype": "fp16"}],
            "outputs": [{"format": "nz", "dtype": "fp16"}],
            "compute_dtype": "fp16",
            "supports_format_convert": True,
            "supports_dtype_cast": True,
        },
        "npu_add": {
            "inputs": [{"format": "nd", "dtype": "fp16"}, {"format": "nd", "dtype": "fp16"}],
            "outputs": [{"format": "nd", "dtype": "fp16"}],
            "compute_dtype": "fp16",
            "supports_format_convert": True,
            "supports_dtype_cast": True,
        },
    }
}


def _make_graph() -> Graph:
    """创建 matmul → add 的测试图。"""
    g = Graph()
    g.add_tensor(Tensor(id="t_a", shape=[1, 32, 64], dtype="fp16",
                        consumer_node_ids=["n_mm"]))
    g.add_tensor(Tensor(id="t_b", shape=[64, 64], dtype="fp16",
                        is_weight=True, consumer_node_ids=["n_mm"]))
    g.add_tensor(Tensor(id="t_mm_out", shape=[1, 32, 64], dtype="fp16",
                        producer_node_id="n_mm", consumer_node_ids=["n_add"]))
    g.add_tensor(Tensor(id="t_bias", shape=[64], dtype="fp16",
                        is_weight=True, consumer_node_ids=["n_add"]))
    g.add_tensor(Tensor(id="t_add_out", shape=[1, 32, 64], dtype="fp16",
                        producer_node_id="n_add"))

    g.add_node(Node(id="n_mm", op_type="aten.mm", npu_op="npu_matmul",
                    inputs=["t_a", "t_b"], outputs=["t_mm_out"], is_mapped=True))
    g.add_node(Node(id="n_add", op_type="aten.add", npu_op="npu_add",
                    inputs=["t_mm_out", "t_bias"], outputs=["t_add_out"], is_mapped=True))
    return g


def test_matmul_format():
    """matmul 输出 tensor 标注为 nz。"""
    g = _make_graph()
    result = run(g, _CONFIG)
    mm_out = result.get_tensor("t_mm_out")
    assert mm_out is not None
    assert mm_out.format == "nz"


def test_vector_format():
    """add 输出 tensor 标注为 nd。"""
    g = _make_graph()
    result = run(g, _CONFIG)
    add_out = result.get_tensor("t_add_out")
    assert add_out is not None
    assert add_out.format == "nd"


def test_annotation_structure():
    """format_annotation 字段结构正确。"""
    g = _make_graph()
    result = run(g, _CONFIG)

    mm = result.get_node("n_mm")
    assert mm is not None
    ann = mm.format_annotation
    assert ann is not None

    # 检查必要字段
    assert "inputs" in ann
    assert "outputs" in ann
    assert "compute_dtype" in ann
    assert "supports_format_convert" in ann
    assert "supports_dtype_cast" in ann

    # 检查值
    assert ann["inputs"][0]["format"] == "nz"
    assert ann["inputs"][0]["dtype"] == "fp16"
    assert ann["outputs"][0]["format"] == "nz"
    assert ann["compute_dtype"] == "fp16"
    assert ann["supports_format_convert"] is True

    # add 节点
    add_node = result.get_node("n_add")
    assert add_node is not None
    add_ann = add_node.format_annotation
    assert add_ann is not None
    assert add_ann["inputs"][0]["format"] == "nd"
