"""op_decomposition 单元测试。"""

from __future__ import annotations

from torch2c.common import INTEGRATION_CONFIG_DIR, load_config
from torch2c.common.graph_ir import Graph, Node, Tensor
from ..op_decomposition import post_validate, run

_CONFIG_PATH = str(INTEGRATION_CONFIG_DIR / "decompositions.yaml")


def _load_rules() -> dict:
    return load_config(_CONFIG_PATH, required_keys=["decompositions"])


def _make_layernorm_graph() -> Graph:
    """创建包含一个未映射 layer_norm 的测试图。"""
    g = Graph()
    g.add_tensor(Tensor(id="t_in", shape=[1, 32, 64], dtype="fp16", is_model_input=True))
    g.add_tensor(Tensor(id="t_w", shape=[64], dtype="fp16", is_weight=True))
    g.add_tensor(Tensor(id="t_b", shape=[64], dtype="fp16", is_weight=True))
    g.add_tensor(
        Tensor(
            id="t_out", shape=[1, 32, 64], dtype="fp16", producer_node_id="n0", consumer_node_ids=[]
        )
    )
    g.add_tensor(Tensor(id="t_mean", shape=[1, 32, 1], dtype="fp32", producer_node_id="n0"))
    g.add_tensor(Tensor(id="t_rstd", shape=[1, 32, 1], dtype="fp32", producer_node_id="n0"))

    g.add_node(
        Node(
            id="n0",
            op_type="aten.native_layer_norm.default",
            npu_op="vector_layernorm",
            compute_unit="vector",
            inputs=["t_in", "t_w", "t_b"],
            outputs=["t_out", "t_mean", "t_rstd"],
            params={"p0": [64], "p1": 1e-5},
        )
    )
    g.tensors["t_in"].consumer_node_ids = ["n0"]
    g.tensors["t_w"].consumer_node_ids = ["n0"]
    g.tensors["t_b"].consumer_node_ids = ["n0"]
    return g


def _make_softmax_graph() -> Graph:
    """创建包含一个未映射 softmax 的测试图。"""
    g = Graph()
    g.add_tensor(
        Tensor(
            id="t_in",
            shape=[1, 4, 32, 32],
            dtype="fp16",
            is_model_input=True,
            consumer_node_ids=["n0"],
        )
    )
    g.add_tensor(
        Tensor(
            id="t_out",
            shape=[1, 4, 32, 32],
            dtype="fp16",
            producer_node_id="n0",
            is_model_output=True,
        )
    )

    g.add_node(
        Node(
            id="n0",
            op_type="aten._softmax.default",
            npu_op="vector_softmax",
            compute_unit="vector",
            inputs=["t_in"],
            outputs=["t_out"],
            params={"p0": -1, "p1": False},
        )
    )
    return g


# ---- tests ----


def test_layernorm_decompose():
    """layer_norm 裂解为 2 个 part。"""
    g = _make_layernorm_graph()
    config = _load_rules()
    run(g, config)

    npu_ops = [n.npu_op for n in g.nodes.values()]
    assert "vector_layernorm_part1" in npu_ops
    assert "vector_layernorm_part2" in npu_ops
    assert "n0" not in g.nodes  # 原节点已删除


def test_softmax_decompose():
    """softmax 裂解为 2 个 part。"""
    g = _make_softmax_graph()
    config = _load_rules()
    run(g, config)

    npu_ops = [n.npu_op for n in g.nodes.values()]
    assert "vector_softmax_part1" in npu_ops
    assert "vector_softmax_part2" in npu_ops


def test_intermediate_tensor_created():
    """中间 tensor 存在且 shape 正确（= 源算子输入 shape）。"""
    g = _make_layernorm_graph()
    original_tids = set(g.tensors.keys())
    config = _load_rules()
    run(g, config)

    new_tids = set(g.tensors.keys()) - original_tids
    assert len(new_tids) >= 1, "Expected at least 1 intermediate tensor"

    for tid in new_tids:
        t = g.get_tensor(tid)
        assert t is not None
        assert t.shape == [1, 32, 64], f"Intermediate shape should be [1,32,64], got {t.shape}"


def test_already_mapped_skipped():
    """已映射的节点不被裂解。"""
    g = Graph()
    g.add_tensor(Tensor(id="t0", shape=[1, 32, 64], dtype="fp16", consumer_node_ids=["n0"]))
    g.add_tensor(Tensor(id="t1", shape=[1, 32, 64], dtype="fp16", producer_node_id="n0"))

    g.add_node(
        Node(
            id="n0",
            op_type="aten.native_layer_norm.default",
            inputs=["t0"],
            outputs=["t1"],
            npu_op="cube_matmul",
            compute_unit="cube",
            is_mapped=True,
        )
    )
    config = _load_rules()
    run(g, config)

    assert "n0" in g.nodes  # 未被裂解
    n0 = g.get_node("n0")
    assert n0 is not None
    assert n0.is_mapped


def test_no_rule_preserved():
    """没有裂解规则的未映射节点保持原样。"""
    g = Graph()
    g.add_tensor(Tensor(id="t0", shape=[1], dtype="fp16", consumer_node_ids=["n0"]))
    g.add_tensor(Tensor(id="t1", shape=[1], dtype="fp16", producer_node_id="n0"))

    g.add_node(
        Node(
            id="n0",
            op_type="aten.unknown_op.default",
            inputs=["t0"],
            outputs=["t1"],
        )
    )
    config = _load_rules()
    run(g, config)

    assert "n0" in g.nodes
    n0 = g.get_node("n0")
    assert n0 is not None
    assert not n0.is_mapped


def test_execution_order_after_decompose():
    """裂解后 execution_order 正确反映新节点顺序。"""
    g = Graph()
    g.add_tensor(Tensor(id="t0", shape=[1, 32, 64], dtype="fp16", consumer_node_ids=["n0"]))
    g.add_tensor(
        Tensor(
            id="t1",
            shape=[1, 32, 64],
            dtype="fp16",
            producer_node_id="n0",
            consumer_node_ids=["n1"],
        )
    )
    g.add_tensor(Tensor(id="t2", shape=[1, 32, 64], dtype="fp16", producer_node_id="n1"))

    g.add_node(Node(id="n0", op_type="aten._softmax.default",
                    npu_op="vector_softmax", compute_unit="vector",
                    inputs=["t0"], outputs=["t1"]))
    g.add_node(
        Node(
            id="n1",
            op_type="aten.mm.default",
            inputs=["t1"],
            outputs=["t2"],
            npu_op="cube_matmul",
            compute_unit="cube",
            is_mapped=True,
        )
    )

    config = _load_rules()
    run(g, config)

    # n0 被裂解为 n0_step_0, n0_step_1，n1 保持
    assert g.execution_order == ["n0_step_0", "n0_step_1", "n1"]


def test_producer_consumer_after_decompose():
    """裂解后 producer/consumer 引用正确更新。"""
    g = _make_softmax_graph()
    config = _load_rules()
    run(g, config)

    # 输入 tensor 的 consumer 应从原节点转移到 step_0
    t_in = g.get_tensor("t_in")
    assert t_in is not None
    assert "n0" not in t_in.consumer_node_ids
    assert "n0_step_0" in t_in.consumer_node_ids

    # 输出 tensor 的 producer 应转移到最后一个 step
    t_out = g.get_tensor("t_out")
    assert t_out is not None
    assert t_out.producer_node_id == "n0_step_1"

    # 中间 tensor 的 producer/consumer 正确
    inter = g.get_tensor("n0_inter_0")
    assert inter is not None
    assert inter.producer_node_id == "n0_step_0"
    assert "n0_step_1" in inter.consumer_node_ids


def test_layernorm_part2_has_orig_input():
    """layernorm_part2 应包含原始输入作为第二输入。"""
    g = _make_layernorm_graph()
    config = _load_rules()
    run(g, config)

    part2_nodes = [n for n in g.nodes.values() if n.npu_op == "vector_layernorm_part2"]
    assert len(part2_nodes) == 1
    part2 = part2_nodes[0]
    # part2 应有 2 个输入: [intermediate, orig_input]
    assert len(part2.inputs) == 2, f"layernorm_part2 should have 2 inputs, got {part2.inputs}"
    assert part2.inputs[1] == "t_in", (
        f"layernorm_part2 input[1] should be orig input 't_in', got {part2.inputs[1]}"
    )
    # 原始输入 tensor 的 consumer 应包含 part2 节点
    t_in = g.get_tensor("t_in")
    assert t_in is not None
    assert part2.id in t_in.consumer_node_ids


def test_multi_output_decompose():
    """多输出节点裂解后，次要输出 tensor 的 producer 被清空。"""
    g = _make_layernorm_graph()
    config = _load_rules()
    run(g, config)

    # 第一个输出 (t_out) 的 producer 应是最后一个 step
    t_out = g.get_tensor("t_out")
    assert t_out is not None
    assert t_out.producer_node_id == "n0_step_1"

    # mean 和 rstd 的 producer 应为 None（已失去生产者）
    t_mean = g.get_tensor("t_mean")
    assert t_mean is not None
    t_rstd = g.get_tensor("t_rstd")
    assert t_rstd is not None
    assert t_mean.producer_node_id is None
    assert t_rstd.producer_node_id is None


class TestPostValidate:
    def test_valid(self):
        g = Graph()
        g.add_tensor(Tensor(id="t0", shape=[1], dtype="fp16"))
        g.add_tensor(Tensor(id="t1", shape=[1], dtype="fp16", producer_node_id="n0"))
        g.add_node(
            Node(
                id="n0",
                op_type="vector_gelu",
                inputs=["t0"],
                outputs=["t1"],
                is_mapped=True,
                npu_op="vector_gelu",
                compute_unit="vector",
            )
        )
        assert post_validate(g) == []

    def test_missing_compute_unit(self):
        g = Graph()
        g.add_tensor(Tensor(id="t0", shape=[1], dtype="fp16"))
        g.add_tensor(Tensor(id="t1", shape=[1], dtype="fp16", producer_node_id="n0"))
        g.add_node(
            Node(
                id="n0",
                op_type="vector_gelu",
                inputs=["t0"],
                outputs=["t1"],
                is_mapped=True,
                npu_op="vector_gelu",
            )
        )
        errors = post_validate(g)
        assert any("compute_unit" in e for e in errors)
