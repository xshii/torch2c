"""validator 单元测试。"""

from __future__ import annotations

import pytest

from torch2c.common import Graph, Node, Tensor, ValidationError
from torch2c.c_backend.validator import run

_CONFIG = {
    "supported_ops": [
        "cube_matmul",
        "vector_add",
        "vector_mul",
        "vector_mul_scalar",
        "vector_gelu",
        "vector_transpose",
        "vector_transpose_2d",
        "scalar_reshape",
        "vector_layernorm_part1",
        "vector_layernorm_part2",
        "vector_softmax_part1",
        "vector_softmax_part2",
    ]
}


def _make_valid_graph() -> Graph:
    """创建全部合法的测试图。"""
    g = Graph()
    g.add_tensor(Tensor(id="t_a", shape=[32, 64], dtype="fp16", consumer_node_ids=["n_mm"]))
    g.add_tensor(
        Tensor(id="t_b", shape=[64, 64], dtype="fp16", is_weight=True, consumer_node_ids=["n_mm"])
    )
    g.add_tensor(Tensor(id="t_out", shape=[32, 64], dtype="fp16", producer_node_id="n_mm"))
    g.add_node(
        Node(
            id="n_mm",
            op_type="aten.mm",
            npu_op="cube_matmul",
            inputs=["t_a", "t_b"],
            outputs=["t_out"],
            is_mapped=True,
        )
    )
    return g


def test_all_supported():
    """全部通过。"""
    g = _make_valid_graph()
    result = run(g, _CONFIG)
    assert len(result.nodes) == 1


def test_one_unsupported():
    """报错且错误信息包含具体算子名。"""
    g = _make_valid_graph()
    g.add_tensor(Tensor(id="t_bad_out", shape=[32], dtype="fp16", producer_node_id="n_bad"))
    g.add_node(
        Node(
            id="n_bad",
            op_type="aten.custom",
            npu_op="npu_unknown",
            inputs=["t_out"],
            outputs=["t_bad_out"],
            is_mapped=True,
        )
    )

    with pytest.raises(ValidationError, match="npu_unknown"):
        run(g, _CONFIG)


def test_multiple_unsupported():
    """报错且列出所有未支持算子（不只第一个）。"""
    g = _make_valid_graph()
    g.add_tensor(Tensor(id="t_bad1_out", shape=[32], dtype="fp16", producer_node_id="n_bad1"))
    g.add_tensor(Tensor(id="t_bad2_out", shape=[32], dtype="fp16", producer_node_id="n_bad2"))
    g.add_node(
        Node(
            id="n_bad1",
            op_type="aten.foo",
            npu_op="npu_foo",
            inputs=["t_out"],
            outputs=["t_bad1_out"],
            is_mapped=True,
        )
    )
    g.add_node(
        Node(
            id="n_bad2",
            op_type="aten.bar",
            npu_op="npu_bar",
            inputs=["t_out"],
            outputs=["t_bad2_out"],
            is_mapped=True,
        )
    )

    with pytest.raises(ValidationError) as exc_info:
        run(g, _CONFIG)

    err_msg = str(exc_info.value)
    assert "npu_foo" in err_msg
    assert "npu_bar" in err_msg
