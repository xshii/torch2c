"""graph_ir 单元测试。"""

from npu_compiler.common.graph_ir import Graph, Node, Tensor


def _make_linear_graph():
    """构建线性链: n1 -> t1 -> n2 -> t2 -> n3。"""
    g = Graph()
    g.add_tensor(Tensor(id="t_in", shape=[1, 32], dtype="fp16"))
    g.add_tensor(Tensor(id="t1", shape=[1, 64], dtype="fp16"))
    g.add_tensor(Tensor(id="t2", shape=[1, 64], dtype="fp16"))

    g.add_node(Node(id="n1", op_type="aten.mm", inputs=["t_in"], outputs=["t1"]))
    g.add_node(Node(id="n2", op_type="aten.relu", inputs=["t1"], outputs=["t2"]))
    g.add_node(Node(id="n3", op_type="aten.add", inputs=["t2"], outputs=[]))
    return g


class TestAddRemoveNode:
    def test_add_remove_node(self):
        g = Graph()
        n = Node(id="n1", op_type="aten.mm")
        g.add_node(n)
        assert len(g.nodes) == 1
        assert "n1" in g.execution_order

        g.remove_node("n1")
        assert len(g.nodes) == 0
        assert "n1" not in g.execution_order

    def test_remove_nonexistent(self):
        g = Graph()
        g.remove_node("does_not_exist")  # 不抛异常


class TestTopoSort:
    def test_topo_sort_linear(self):
        g = _make_linear_graph()
        order = g.topo_sort()
        assert order.index("n1") < order.index("n2") < order.index("n3")

    def test_topo_sort_diamond(self):
        """菱形依赖: n1 -> (n2, n3) -> n4。"""
        g = Graph()
        g.add_tensor(Tensor(id="t0", shape=[1], dtype="fp32"))
        g.add_tensor(Tensor(id="t1", shape=[1], dtype="fp32"))
        g.add_tensor(Tensor(id="t2", shape=[1], dtype="fp32"))
        g.add_tensor(Tensor(id="t3", shape=[1], dtype="fp32"))

        g.add_node(Node(id="n1", op_type="op", inputs=["t0"], outputs=["t1"]))
        g.add_node(Node(id="n2", op_type="op", inputs=["t1"], outputs=["t2"]))
        g.add_node(Node(id="n3", op_type="op", inputs=["t1"], outputs=["t3"]))
        g.add_node(Node(id="n4", op_type="op", inputs=["t2", "t3"], outputs=[]))

        order = g.topo_sort()
        assert order.index("n1") < order.index("n2")
        assert order.index("n1") < order.index("n3")
        assert order.index("n2") < order.index("n4")
        assert order.index("n3") < order.index("n4")


class TestSerialize:
    def test_to_dict_from_dict(self):
        g = _make_linear_graph()
        d = g.to_dict()
        g2 = Graph.from_dict(d)

        assert set(g2.nodes.keys()) == set(g.nodes.keys())
        assert set(g2.tensors.keys()) == set(g.tensors.keys())
        assert g2.execution_order == g.execution_order
        for nid in g.nodes:
            assert g2.nodes[nid].op_type == g.nodes[nid].op_type


class TestValidate:
    def test_valid_graph(self):
        g = _make_linear_graph()
        assert g.validate() == []

    def test_dangling_input(self):
        g = Graph()
        g.add_node(Node(id="n1", op_type="op", inputs=["missing_tensor"]))
        errors = g.validate()
        assert len(errors) == 1
        assert "missing_tensor" in errors[0]

    def test_dangling_output(self):
        g = Graph()
        g.add_node(Node(id="n1", op_type="op", outputs=["missing_tensor"]))
        errors = g.validate()
        assert len(errors) == 1


class TestSummary:
    def test_summary_content(self):
        g = _make_linear_graph()
        s = g.summary()
        assert "3 nodes" in s
        assert "3 tensors" in s
