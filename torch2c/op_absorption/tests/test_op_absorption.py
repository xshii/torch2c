"""op_absorption 单元测试。"""

from __future__ import annotations

from torch2c.common import Graph, Node, Tensor
from torch2c.op_absorption.op_absorption import post_validate, run


def _make_absorption_graph() -> Graph:
    """创建 add(scores, mask) → softmax_part1 → softmax_part2 的测试图。"""
    g = Graph()

    # tensors
    g.add_tensor(
        Tensor(
            id="t_scores",
            shape=[1, 1, 32, 32],
            dtype="fp16",
            is_model_input=True,
            consumer_node_ids=["n_add"],
        )
    )
    g.add_tensor(
        Tensor(
            id="t_mask",
            shape=[1, 1, 32, 32],
            dtype="fp16",
            is_model_input=True,
            consumer_node_ids=["n_add"],
        )
    )
    g.add_tensor(
        Tensor(
            id="t_add_out",
            shape=[1, 1, 32, 32],
            dtype="fp16",
            producer_node_id="n_add",
            consumer_node_ids=["n_sp1"],
        )
    )
    g.add_tensor(
        Tensor(
            id="t_sp1_out",
            shape=[1, 1, 32, 32],
            dtype="fp16",
            producer_node_id="n_sp1",
            consumer_node_ids=["n_sp2"],
        )
    )
    g.add_tensor(
        Tensor(id="t_sp2_out", shape=[1, 1, 32, 32], dtype="fp16", producer_node_id="n_sp2")
    )

    # nodes
    g.add_node(
        Node(
            id="n_add",
            op_type="aten.add",
            npu_op="vector_add",
            inputs=["t_scores", "t_mask"],
            outputs=["t_add_out"],
            is_mapped=True,
        )
    )
    g.add_node(
        Node(
            id="n_sp1",
            op_type="aten.softmax",
            npu_op="vector_softmax_part1",
            inputs=["t_add_out"],
            outputs=["t_sp1_out"],
            is_mapped=True,
        )
    )
    g.add_node(
        Node(
            id="n_sp2",
            op_type="aten.softmax",
            npu_op="vector_softmax_part2",
            inputs=["t_sp1_out"],
            outputs=["t_sp2_out"],
            is_mapped=True,
        )
    )
    return g


_RULES = {
    "absorptions": [
        {
            "absorbed_op": "vector_add",
            "target_op": "vector_softmax_part1",
            "param_name": "mask",
            "absorbed_input_index": 1,
            "passthrough_input_index": 0,
        }
    ]
}


def test_mask_absorption():
    """add 被吸收进 softmax_part1。"""
    g = _make_absorption_graph()
    result = run(g, _RULES)
    assert "n_add" not in result.nodes
    assert "n_sp1" in result.nodes
    assert "n_sp2" in result.nodes
    assert len(result.nodes) == 2


def test_absorbed_input_recorded():
    """softmax_part1 的 absorbed_inputs 包含 mask。"""
    g = _make_absorption_graph()
    result = run(g, _RULES)
    sp1 = result.get_node("n_sp1")
    assert sp1 is not None
    assert sp1.absorbed_inputs == {"mask": "t_mask"}
    assert sp1.inputs[0] == "t_scores"


def test_intermediate_removed():
    """add 的输出 tensor 被删除。"""
    g = _make_absorption_graph()
    result = run(g, _RULES)
    assert "t_add_out" not in result.tensors
    # 其他 tensor 保留
    assert "t_scores" in result.tensors
    assert "t_mask" in result.tensors
    assert "t_sp1_out" in result.tensors


def test_no_match_preserved():
    """不匹配规则的 add 保持不变（如残差 add）。"""
    g = Graph()
    g.add_tensor(Tensor(id="t_a", shape=[32], dtype="fp16", consumer_node_ids=["n_residual_add"]))
    g.add_tensor(Tensor(id="t_b", shape=[32], dtype="fp16", consumer_node_ids=["n_residual_add"]))
    g.add_tensor(Tensor(id="t_c", shape=[32], dtype="fp16", producer_node_id="n_residual_add"))
    g.add_node(
        Node(
            id="n_residual_add",
            op_type="aten.add",
            npu_op="vector_add",
            inputs=["t_a", "t_b"],
            outputs=["t_c"],
            is_mapped=True,
        )
    )
    result = run(g, _RULES)
    # add 没有消费者是 softmax_part1，不吸收
    assert "n_residual_add" in result.nodes
    assert len(result.nodes) == 1


def test_consumer_ids_cleaned():
    """吸收后 consumer_node_ids 不含已删除节点，且新消费关系正确。"""
    g = _make_absorption_graph()
    result = run(g, _RULES)

    # scores: 原消费者 n_add 已删除，新消费者是 n_sp1
    t_scores = result.get_tensor("t_scores")
    assert t_scores is not None
    assert "n_add" not in t_scores.consumer_node_ids
    assert "n_sp1" in t_scores.consumer_node_ids

    # mask: 原消费者 n_add 已删除，作为 absorbed_input 被 n_sp1 消费
    t_mask = result.get_tensor("t_mask")
    assert t_mask is not None
    assert "n_add" not in t_mask.consumer_node_ids
    assert "n_sp1" in t_mask.consumer_node_ids


def test_empty_rules():
    """absorptions 为空列表时图不变。"""
    g = _make_absorption_graph()
    node_count = len(g.nodes)
    tensor_count = len(g.tensors)
    result = run(g, {"absorptions": []})
    assert len(result.nodes) == node_count
    assert len(result.tensors) == tensor_count


def test_post_validate_clean():
    """吸收后校验通过。"""
    g = _make_absorption_graph()
    result = run(g, _RULES)
    errors = post_validate(result)
    assert errors == []


def test_post_validate_missing_tensor():
    """absorbed_inputs 引用不存在的 tensor 报错。"""
    g = Graph()
    g.add_tensor(Tensor(id="t_in", shape=[32], dtype="fp16", consumer_node_ids=["n"]))
    g.add_tensor(Tensor(id="t_out", shape=[32], dtype="fp16", producer_node_id="n"))
    g.add_node(
        Node(
            id="n",
            op_type="cube_matmul",
            inputs=["t_in"],
            outputs=["t_out"],
            absorbed_inputs={"bias": "t_nonexistent"},
        )
    )
    errors = post_validate(g)
    assert len(errors) == 1
    assert "t_nonexistent" in errors[0]
