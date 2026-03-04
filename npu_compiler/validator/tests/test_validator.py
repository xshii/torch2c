"""validator 单元测试。"""

from __future__ import annotations

import pytest

from npu_compiler.common import Graph, Node, Tensor, ValidationError
from npu_compiler.validator import run

_CONFIG = {
    "supported_ops": [
        "npu_matmul",
        "npu_add",
        "npu_mul",
        "npu_mul_scalar",
        "npu_gelu",
        "npu_transpose",
        "npu_transpose_2d",
        "npu_reshape",
        "npu_layernorm_part1",
        "npu_layernorm_part2",
        "npu_softmax_part1",
        "npu_softmax_part2",
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
            npu_op="npu_matmul",
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
