"""Tests for _block_graph — BlockGraph construction."""

from __future__ import annotations

from torch2c.common import Graph, Node, Tensor
from torch2c.optpass.cd_block_fuser._block_graph import BlockGraph, _tensor_padded_bytes
from torch2c.optpass.cd_roofline.roofline_analyzer import RooflineHwParams, parse_cost_model


def _hw():
    return RooflineHwParams()


def _cm():
    return parse_cost_model(None)


def _simple_graph() -> Graph:
    """matmul → relu → output。"""
    g = Graph()
    g.add_tensor(Tensor(id="x", shape=[1, 32, 64], dtype="fp16", is_model_input=True))
    g.add_tensor(Tensor(id="w", shape=[64, 32], dtype="fp16", is_weight=True))
    g.add_tensor(Tensor(id="mm_out", shape=[1, 32, 32], dtype="fp16",
                        producer_node_id="n_mm", consumer_node_ids=["n_relu"]))
    g.add_tensor(Tensor(id="relu_out", shape=[1, 32, 32], dtype="fp16",
                        producer_node_id="n_relu", is_model_output=True))

    g.add_node(Node(id="n_mm", op_type="mm", inputs=["x", "w"], outputs=["mm_out"],
                    npu_op="cube_matmul", compute_unit="cube", is_mapped=True))
    g.add_node(Node(id="n_relu", op_type="relu", inputs=["mm_out"], outputs=["relu_out"],
                    npu_op="vector_relu", compute_unit="vector", is_mapped=True))
    return g


class TestBlockGraphConstruction:
    def test_data_blocks_created(self):
        g = _simple_graph()
        bg = BlockGraph.from_graph(g, _hw(), _cm())
        assert len(bg.data_blocks) == 4  # x, w, mm_out, relu_out

    def test_compute_blocks_created(self):
        g = _simple_graph()
        bg = BlockGraph.from_graph(g, _hw(), _cm())
        assert len(bg.compute_blocks) == 2  # n_mm, n_relu

    def test_external_flags(self):
        g = _simple_graph()
        bg = BlockGraph.from_graph(g, _hw(), _cm())
        assert bg.data_blocks["x"].is_external is True
        assert bg.data_blocks["w"].is_external is True
        assert bg.data_blocks["mm_out"].is_external is False
        assert bg.data_blocks["relu_out"].is_external is True

    def test_elimination_benefit(self):
        g = _simple_graph()
        bg = BlockGraph.from_graph(g, _hw(), _cm())
        # mm_out 是中间 tensor, storage=hbm → benefit > 0
        assert bg.data_blocks["mm_out"].elimination_benefit > 0
        # x 是 model_input → benefit = 0
        assert bg.data_blocks["x"].elimination_benefit == 0
        # relu_out 是 model_output → benefit = 0
        assert bg.data_blocks["relu_out"].elimination_benefit == 0

    def test_fusible_edges(self):
        g = _simple_graph()
        bg = BlockGraph.from_graph(g, _hw(), _cm())
        edges = bg.get_fusible_edges()
        # 只有 mm_out 是可融合边
        assert len(edges) == 1
        assert edges[0].tensor_id == "mm_out"

    def test_tileable(self):
        g = _simple_graph()
        bg = BlockGraph.from_graph(g, _hw(), _cm())
        assert bg.compute_blocks["n_mm"].tileable is True
        assert bg.compute_blocks["n_relu"].tileable is True

    def test_local_tensor_no_benefit(self):
        """storage=local 的 tensor 无融合收益（已在 L1）。"""
        g = _simple_graph()
        g.tensors["mm_out"].storage = "local"
        bg = BlockGraph.from_graph(g, _hw(), _cm())
        assert bg.data_blocks["mm_out"].elimination_benefit == 0

    def test_topo_order(self):
        g = _simple_graph()
        bg = BlockGraph.from_graph(g, _hw(), _cm())
        assert bg.topo_order == ["n_mm", "n_relu"]
