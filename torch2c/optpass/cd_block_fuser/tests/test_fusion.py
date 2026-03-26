"""Tests for _fusion — greedy block-level fusion algorithm."""

from __future__ import annotations

from torch2c.common import Graph, Node, Tensor
from torch2c.optpass.cd_block_fuser._block_graph import BlockGraph
from torch2c.optpass.cd_block_fuser._fusion import fuse_blocks
from torch2c.optpass.cd_roofline.roofline_analyzer import RooflineHwParams, parse_cost_model


def _hw():
    return RooflineHwParams()


def _cm():
    return parse_cost_model(None)


def _linear_chain_graph() -> Graph:
    """matmul → relu → add (线性链)。"""
    g = Graph()
    g.add_tensor(Tensor(id="x", shape=[1, 32, 64], dtype="fp16", is_model_input=True))
    g.add_tensor(Tensor(id="w", shape=[64, 32], dtype="fp16", is_weight=True))
    g.add_tensor(Tensor(id="bias", shape=[1, 1, 32], dtype="fp16", is_weight=True))
    g.add_tensor(Tensor(id="mm_out", shape=[1, 32, 32], dtype="fp16",
                        producer_node_id="n_mm", consumer_node_ids=["n_relu"]))
    g.add_tensor(Tensor(id="relu_out", shape=[1, 32, 32], dtype="fp16",
                        producer_node_id="n_relu", consumer_node_ids=["n_add"]))
    g.add_tensor(Tensor(id="add_out", shape=[1, 32, 32], dtype="fp16",
                        producer_node_id="n_add", is_model_output=True))

    g.add_node(Node(id="n_mm", op_type="mm", inputs=["x", "w"], outputs=["mm_out"],
                    npu_op="cube_matmul", compute_unit="cube", is_mapped=True))
    g.add_node(Node(id="n_relu", op_type="relu", inputs=["mm_out"], outputs=["relu_out"],
                    npu_op="vector_relu", compute_unit="vector", is_mapped=True))
    g.add_node(Node(id="n_add", op_type="add", inputs=["relu_out", "bias"], outputs=["add_out"],
                    npu_op="vector_add", compute_unit="vector", is_mapped=True))
    return g


def _fanout_graph() -> Graph:
    """matmul → (relu, gelu) DAG (fan-out=2)。"""
    g = Graph()
    g.add_tensor(Tensor(id="x", shape=[1, 32, 64], dtype="fp16", is_model_input=True))
    g.add_tensor(Tensor(id="w", shape=[64, 32], dtype="fp16", is_weight=True))
    g.add_tensor(Tensor(id="mm_out", shape=[1, 32, 32], dtype="fp16",
                        producer_node_id="n_mm", consumer_node_ids=["n_relu", "n_gelu"]))
    g.add_tensor(Tensor(id="relu_out", shape=[1, 32, 32], dtype="fp16",
                        producer_node_id="n_relu", is_model_output=True))
    g.add_tensor(Tensor(id="gelu_out", shape=[1, 32, 32], dtype="fp16",
                        producer_node_id="n_gelu", is_model_output=True))

    g.add_node(Node(id="n_mm", op_type="mm", inputs=["x", "w"], outputs=["mm_out"],
                    npu_op="cube_matmul", compute_unit="cube", is_mapped=True))
    g.add_node(Node(id="n_relu", op_type="relu", inputs=["mm_out"], outputs=["relu_out"],
                    npu_op="vector_relu", compute_unit="vector", is_mapped=True))
    g.add_node(Node(id="n_gelu", op_type="gelu", inputs=["mm_out"], outputs=["gelu_out"],
                    npu_op="vector_gelu", compute_unit="vector", is_mapped=True))
    return g


class TestLinearChainFusion:
    def test_single_group(self):
        """线性链 matmul→relu→add 应融合为一组。"""
        g = _linear_chain_graph()
        bg = BlockGraph.from_graph(g, _hw(), _cm())
        groups = fuse_blocks(bg, l1_capacity=1024 * 1024)

        assert len(groups) == 1
        grp = groups[0]
        assert set(grp.node_ids) == {"n_mm", "n_relu", "n_add"}

    def test_internal_blocks(self):
        """mm_out 和 relu_out 应为 internal（留 L1）。"""
        g = _linear_chain_graph()
        bg = BlockGraph.from_graph(g, _hw(), _cm())
        groups = fuse_blocks(bg, l1_capacity=1024 * 1024)

        assert "mm_out" in groups[0].internal_block_ids
        assert "relu_out" in groups[0].internal_block_ids

    def test_external_io(self):
        """x, w, bias 为外部输入，add_out 为外部输出。"""
        g = _linear_chain_graph()
        bg = BlockGraph.from_graph(g, _hw(), _cm())
        groups = fuse_blocks(bg, l1_capacity=1024 * 1024)

        grp = groups[0]
        assert "x" in grp.external_input_ids
        assert "w" in grp.external_input_ids
        assert "bias" in grp.external_input_ids
        assert "add_out" in grp.external_output_ids

    def test_benefit_positive(self):
        g = _linear_chain_graph()
        bg = BlockGraph.from_graph(g, _hw(), _cm())
        groups = fuse_blocks(bg, l1_capacity=1024 * 1024)
        assert groups[0].total_benefit > 0

    def test_topo_order_preserved(self):
        """组内节点按拓扑序排列。"""
        g = _linear_chain_graph()
        bg = BlockGraph.from_graph(g, _hw(), _cm())
        groups = fuse_blocks(bg, l1_capacity=1024 * 1024)
        assert groups[0].node_ids == ["n_mm", "n_relu", "n_add"]


class TestDAGFusion:
    def test_fanout_fused(self):
        """fan-out=2 的 DAG 也能融合（和 fusion_planner 不同）。"""
        g = _fanout_graph()
        bg = BlockGraph.from_graph(g, _hw(), _cm())
        groups = fuse_blocks(bg, l1_capacity=1024 * 1024)

        assert len(groups) == 1
        assert set(groups[0].node_ids) == {"n_mm", "n_relu", "n_gelu"}
        assert "mm_out" in groups[0].internal_block_ids


class TestL1CapacityConstraint:
    def test_l1_too_small_no_fusion(self):
        """L1 容量太小 → 不融合。"""
        g = _linear_chain_graph()
        bg = BlockGraph.from_graph(g, _hw(), _cm())
        # 设 L1 = 1 byte，任何 tensor 都放不下
        groups = fuse_blocks(bg, l1_capacity=1)
        assert len(groups) == 0

    def test_partial_fusion(self):
        """L1 只够一个 intermediate → 只融合一对。"""
        g = _linear_chain_graph()
        bg = BlockGraph.from_graph(g, _hw(), _cm())
        # mm_out size = 1*32*32*2 = 2048 bytes
        # relu_out size = 1*32*32*2 = 2048 bytes
        # 设 L1 = 3000，够一个但不够两个
        groups = fuse_blocks(bg, l1_capacity=3000)
        # 应该融合了最大收益的一对
        assert len(groups) >= 1
        total_internals = sum(len(g.internal_block_ids) for g in groups)
        assert total_internals >= 1  # 至少融合了一个 tensor


class TestNoFusionCases:
    def test_single_node(self):
        """单节点图 → 无融合。"""
        g = Graph()
        g.add_tensor(Tensor(id="x", shape=[1, 32], dtype="fp16", is_model_input=True))
        g.add_tensor(Tensor(id="y", shape=[1, 32], dtype="fp16",
                            producer_node_id="n0", is_model_output=True))
        g.add_node(Node(id="n0", op_type="relu", inputs=["x"], outputs=["y"],
                        npu_op="vector_relu", compute_unit="vector", is_mapped=True))
        bg = BlockGraph.from_graph(g, _hw(), _cm())
        groups = fuse_blocks(bg, l1_capacity=1024 * 1024)
        assert len(groups) == 0

    def test_all_external(self):
        """所有 tensor 都是外部 → 无可融合边。"""
        g = Graph()
        g.add_tensor(Tensor(id="x", shape=[1, 32], dtype="fp16",
                            is_model_input=True, consumer_node_ids=["n0"]))
        g.add_tensor(Tensor(id="y", shape=[1, 32], dtype="fp16",
                            producer_node_id="n0", is_model_output=True))
        g.add_node(Node(id="n0", op_type="relu", inputs=["x"], outputs=["y"],
                        npu_op="vector_relu", compute_unit="vector", is_mapped=True))
        bg = BlockGraph.from_graph(g, _hw(), _cm())
        edges = bg.get_fusible_edges()
        assert len(edges) == 0
