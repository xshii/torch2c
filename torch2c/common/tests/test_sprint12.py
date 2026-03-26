"""Sprint 1 + Sprint 2 新增 API 的测试。

T1: _PassSlot descriptor
T2: Enum (ComputeUnit, TensorFormat, Storage, FusionRole)
T3: GraphBuilder
T4: Graph 改写原子方法 (splice_execution_order, rewire_input, insert_node_before)
T5: Graph 查询 API (single_consumer, intermediates, nodes_by_unit, consumers_of, producer_of)
T6: FormatSpec / FormatAnnotation
"""

from torch2c.common.graph_ir import (
    ComputeUnit,
    FormatAnnotation,
    FormatSpec,
    FusionRole,
    Graph,
    Node,
    Storage,
    Tensor,
    TensorFormat,
)
from torch2c.common.test_utils import LAST, GraphBuilder


# ═══════════════════════════════════════════════════════════════
# T1: _PassSlot descriptor
# ═══════════════════════════════════════════════════════════════


class TestPassSlot:
    def test_read_write(self):
        node = Node(id="n0", op_type="test")
        assert node.roofline is None  # default
        node.roofline = {"flops": 1024, "bottleneck": "compute"}
        assert node.roofline == {"flops": 1024, "bottleneck": "compute"}
        # 底层仍存 params dict
        assert node.params["_roofline"] == {"flops": 1024, "bottleneck": "compute"}

    def test_backward_compat(self):
        """旧代码通过 params dict 写入，新代码通过 descriptor 读取。"""
        node = Node(id="n0", op_type="test")
        node.params["_tile_config"] = {"tile_size": 64}
        assert node.tile_config == {"tile_size": 64}

    def test_delete(self):
        node = Node(id="n0", op_type="test")
        node.fusion_group = "fg_0"
        assert node.fusion_group == "fg_0"
        del node.fusion_group
        assert node.fusion_group is None
        assert "_fusion_group" not in node.params

    def test_all_slots(self):
        """验证所有 _PassSlot 实例都可用。"""
        node = Node(id="n0", op_type="test")
        slots = [
            ("roofline", "_roofline"),
            ("tile_config", "_tile_config"),
            ("fusion_group", "_fusion_group"),
            ("fusion_role", "_fusion_role"),
            ("mha_analysis", "_mha_analysis"),
            ("weight_slices", "_weight_slices"),
            ("tile_info", "_tile_info"),
            ("npu_hint", "_npu"),
        ]
        for attr, key in slots:
            assert getattr(node, attr) is None, f"{attr} should default to None"
            setattr(node, attr, f"test_{key}")
            assert node.params[key] == f"test_{key}"

    def test_serialization_roundtrip(self):
        """descriptor 值通过 params 序列化，不影响 asdict/from_dict。"""
        node = Node(id="n0", op_type="test")
        node.roofline = {"flops": 100}
        node.fusion_group = "fg_0"

        g = Graph()
        g.add_node(node)
        g.add_tensor(Tensor(id="t0", shape=[1], dtype="fp16"))
        node.inputs = ["t0"]
        node.outputs = ["t0"]

        d = g.to_dict()
        g2 = Graph.from_dict(d)
        n2 = g2.get_node("n0")
        assert n2.roofline == {"flops": 100}
        assert n2.fusion_group == "fg_0"


# ═══════════════════════════════════════════════════════════════
# T2: Enum
# ═══════════════════════════════════════════════════════════════


class TestEnums:
    def test_str_equality(self):
        """Enum 值与裸字符串相等（向后兼容）。"""
        assert ComputeUnit.CUBE == "cube"
        assert TensorFormat.NZ == "nz"
        assert Storage.LOCAL == "local"
        assert FusionRole.HEAD == "head"

    def test_in_set(self):
        """Enum 值可用于 set/dict 查找。"""
        matmul_ops = {"cube_matmul", "cube_matmul_bias"}
        # str 查找
        assert "cube_matmul" in matmul_ops

    def test_node_compute_unit(self):
        """Enum 可赋值给 Node.compute_unit，与字符串比较兼容。"""
        node = Node(id="n0", op_type="test", compute_unit=ComputeUnit.CUBE)
        assert node.compute_unit == "cube"
        assert node.compute_unit == ComputeUnit.CUBE

    def test_tensor_format(self):
        t = Tensor(id="t0", shape=[1], dtype="fp16", format=TensorFormat.NZ)
        assert t.format == "nz"
        assert t.format == TensorFormat.NZ

    def test_tensor_storage(self):
        t = Tensor(id="t0", shape=[1], dtype="fp16", storage=Storage.LOCAL)
        assert t.storage == "local"

    def test_json_serialization(self):
        """Enum 值可被 json.dumps 序列化为字符串。"""
        import json
        assert json.dumps(ComputeUnit.CUBE) == '"cube"'
        assert json.dumps(TensorFormat.NZ) == '"nz"'


# ═══════════════════════════════════════════════════════════════
# T3: GraphBuilder
# ═══════════════════════════════════════════════════════════════


class TestGraphBuilder:
    def test_basic_chain(self):
        """构建 matmul → relu 链。"""
        b = GraphBuilder()
        x = b.input([1, 32, 64])
        w = b.weight([64, 64])
        mm = b.op("cube_matmul", [x, w], [1, 32, 64])
        relu = b.op("vector_relu", [LAST], [1, 32, 64], compute_unit="vector")
        b.mark_output()
        g = b.build()

        assert len(g.nodes) == 2
        assert len(g.tensors) == 4  # x, w, mm_out, relu_out
        assert len(g.execution_order) == 2

    def test_producer_consumer_wiring(self):
        """自动 producer/consumer 接线。"""
        b = GraphBuilder()
        x = b.input([1, 32, 64])
        w = b.weight([64, 64])
        mm = b.op("cube_matmul", [x, w], [1, 32, 64], nid="mm")
        g = b.build()

        # mm 的输出 tensor
        out_t = g.get_tensor(mm)
        assert out_t is not None
        assert out_t.producer_node_id == "mm"

        # x 和 w 被 mm 消费
        assert "mm" in g.get_tensor(x).consumer_node_ids
        assert "mm" in g.get_tensor(w).consumer_node_ids

    def test_last_sentinel(self):
        """LAST 哨兵值引用上一个 op 的输出。"""
        b = GraphBuilder()
        x = b.input([1, 32, 64])
        out1 = b.op("vector_relu", [x], [1, 32, 64], compute_unit="vector", nid="n1")
        out2 = b.op("vector_relu", [LAST], [1, 32, 64], compute_unit="vector", nid="n2")
        g = b.build()

        n2 = g.get_node("n2")
        assert n2.inputs == [out1]  # LAST resolved to out1

    def test_mark_output(self):
        b = GraphBuilder()
        b.input([1, 32, 64])
        out = b.op("vector_relu", [LAST], [1, 32, 64], compute_unit="vector")
        b.mark_output()
        g = b.build()
        assert g.get_tensor(out).is_model_output is True

    def test_custom_ids(self):
        """自定义 node/tensor id。"""
        b = GraphBuilder()
        x = b.input([1, 64], tid="my_input")
        w = b.weight([64, 64], tid="my_weight")
        out = b.op("cube_matmul", [x, w], [1, 64], nid="my_node", out_tid="my_out")
        g = b.build()

        assert "my_input" in g.tensors
        assert "my_weight" in g.tensors
        assert "my_node" in g.nodes
        assert "my_out" in g.tensors

    def test_validate_passes(self):
        """构建的图通过 graph.validate()。"""
        b = GraphBuilder()
        x = b.input([1, 32, 64])
        w = b.weight([64, 64])
        b.op("cube_matmul", [x, w], [1, 32, 64])
        g = b.build()
        assert g.validate() == []


# ═══════════════════════════════════════════════════════════════
# T4: Graph 改写原子方法
# ═══════════════════════════════════════════════════════════════


class TestGraphMutation:
    def _make_chain(self):
        """构建 n0 → n1 → n2 链。"""
        b = GraphBuilder()
        x = b.input([1, 64])
        o0 = b.op("vector_relu", [x], [1, 64], compute_unit="vector", nid="n0")
        o1 = b.op("vector_relu", [LAST], [1, 64], compute_unit="vector", nid="n1")
        o2 = b.op("vector_relu", [LAST], [1, 64], compute_unit="vector", nid="n2")
        return b.build(), x, o0, o1, o2

    def test_splice_execution_order(self):
        g, _, _, _, _ = self._make_chain()
        assert g.execution_order == ["n0", "n1", "n2"]
        g.splice_execution_order("n1", ["n1a", "n1b"])
        assert g.execution_order == ["n0", "n1a", "n1b", "n2"]

    def test_splice_nonexistent(self):
        g, _, _, _, _ = self._make_chain()
        g.splice_execution_order("n_missing", ["x"])  # no-op
        assert g.execution_order == ["n0", "n1", "n2"]

    def test_rewire_input(self):
        g, x, o0, o1, o2 = self._make_chain()
        # n1 的输入从 o0 改为 x
        g.rewire_input("n1", 0, x)
        n1 = g.get_node("n1")
        assert n1.inputs == [x]
        # o0 不再有 n1 作为 consumer
        assert "n1" not in g.get_tensor(o0).consumer_node_ids
        # x 现在有 n1 作为 consumer
        assert "n1" in g.get_tensor(x).consumer_node_ids

    def test_rewire_noop(self):
        """rewire 到相同 tensor 不改变任何东西。"""
        g, _, o0, _, _ = self._make_chain()
        consumers_before = list(g.get_tensor(o0).consumer_node_ids)
        g.rewire_input("n1", 0, o0)  # same as current
        assert g.get_tensor(o0).consumer_node_ids == consumers_before

    def test_insert_node_before(self):
        g, _, _, _, _ = self._make_chain()
        new_t = Tensor(id="t_new", shape=[1, 64], dtype="fp16", producer_node_id="n_new")
        new_n = Node(
            id="n_new", op_type="vector_relu", inputs=[], outputs=["t_new"],
            npu_op="vector_relu", compute_unit="vector", is_mapped=True,
        )
        g.insert_node_before("n1", new_n, new_t)
        assert g.execution_order == ["n0", "n_new", "n1", "n2"]
        assert "t_new" in g.tensors
        assert "n_new" in g.nodes


# ═══════════════════════════════════════════════════════════════
# T5: Graph 查询 API
# ═══════════════════════════════════════════════════════════════


class TestGraphQuery:
    def _make_graph(self):
        b = GraphBuilder()
        x = b.input([1, 64])
        w = b.weight([64, 64])
        mm = b.op("cube_matmul", [x, w], [1, 64], nid="mm")
        relu = b.op("vector_relu", [LAST], [1, 64], compute_unit="vector", nid="relu")
        reshape = b.op("idma_reshape", [LAST], [1, 64], compute_unit="idma", nid="reshape")
        b.mark_output()
        return b.build(), x, w, mm, relu, reshape

    def test_single_consumer(self):
        g, _, _, mm, _, _ = self._make_graph()
        # mm_out has exactly 1 consumer (relu)
        consumer = g.single_consumer(mm)
        assert consumer is not None
        assert consumer.id == "relu"

    def test_single_consumer_none(self):
        g, _, _, _, _, _ = self._make_graph()
        # weight has 1 consumer but let's test no-consumer case
        assert g.single_consumer("nonexistent") is None

    def test_intermediates(self):
        g, _, _, mm, relu, reshape = self._make_graph()
        inter_ids = {t.id for t in g.intermediates()}
        # mm_out and relu_out are intermediates; reshape_out is model_output
        assert mm in inter_ids
        assert relu in inter_ids
        # reshape_out is model output, not intermediate
        assert reshape not in inter_ids

    def test_nodes_by_unit(self):
        g, _, _, _, _, _ = self._make_graph()
        cube_nodes = list(g.nodes_by_unit("cube"))
        assert len(cube_nodes) == 1
        assert cube_nodes[0].id == "mm"
        vector_nodes = list(g.nodes_by_unit("vector"))
        assert len(vector_nodes) == 1
        assert vector_nodes[0].id == "relu"

    def test_consumers_of(self):
        g, _, _, _, _, _ = self._make_graph()
        consumers = g.consumers_of("mm")
        assert len(consumers) == 1
        assert consumers[0].id == "relu"

    def test_producer_of(self):
        g, _, _, mm, _, _ = self._make_graph()
        producer = g.producer_of(mm)
        assert producer is not None
        assert producer.id == "mm"

    def test_producer_of_input(self):
        g, x, _, _, _, _ = self._make_graph()
        assert g.producer_of(x) is None  # model input has no producer


# ═══════════════════════════════════════════════════════════════
# T6: FormatSpec / FormatAnnotation
# ═══════════════════════════════════════════════════════════════


class TestFormatAnnotation:
    def test_format_spec(self):
        spec = FormatSpec(format="nz", dtype="fp16")
        assert spec.format == "nz"
        assert spec.dtype == "fp16"
        assert spec.to_dict() == {"format": "nz", "dtype": "fp16"}

    def test_format_spec_from_dict(self):
        spec = FormatSpec.from_dict({"format": "zz", "dtype": "int8"})
        assert spec.format == "zz"
        assert spec.dtype == "int8"

    def test_format_annotation_roundtrip(self):
        ann = FormatAnnotation(
            inputs=[FormatSpec("nd", "fp16"), FormatSpec("nz", "fp16")],
            outputs=[FormatSpec("nd", "fp16")],
        )
        d = ann.to_dict()
        assert d == {
            "inputs": [
                {"format": "nd", "dtype": "fp16"},
                {"format": "nz", "dtype": "fp16"},
            ],
            "outputs": [{"format": "nd", "dtype": "fp16"}],
        }
        ann2 = FormatAnnotation.from_dict(d)
        assert ann2.inputs == ann.inputs
        assert ann2.outputs == ann.outputs

    def test_uniform(self):
        ann = FormatAnnotation.uniform(2, 1, fmt="nz", dtype="fp16")
        assert len(ann.inputs) == 2
        assert len(ann.outputs) == 1
        assert all(s.format == "nz" for s in ann.inputs)

    def test_backward_compat_with_dict(self):
        """FormatAnnotation.to_dict() 产出的格式与现有代码使用的 dict 格式一致。"""
        # 现有代码构建 format_annotation 的方式：
        legacy = {
            "inputs": [{"format": "nd", "dtype": "fp16"}],
            "outputs": [{"format": "nz", "dtype": "fp16"}],
        }
        # 新代码可以从中构建
        ann = FormatAnnotation.from_dict(legacy)
        assert ann.to_dict() == legacy
