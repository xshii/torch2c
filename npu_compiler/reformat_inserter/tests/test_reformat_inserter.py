"""reformat_inserter 单元测试。"""

import pytest

from npu_compiler.common import Graph, Node, Tensor
from npu_compiler.reformat_inserter import run as run_reformat_inserter
from npu_compiler.reformat_inserter.reformat_inserter import post_validate


def _make_matmul_add() -> Graph:
    """matmul(cube,nz) → add(vector,nd)，需要 nz→nd 转换。

    t_input(fp16,nd) ─┐
    t_weight(fp16,nz) ─┤→ [cube_matmul] → t_mm_out(fp16,nz)
                                                │ (nz → nd 格式不匹配)
                                     [vector_add] → t_out(fp16,nd)
    """
    shape = [1, 32, 64]
    g = Graph()

    g.add_tensor(Tensor(
        id="t_input", shape=shape, dtype="fp16", format="nd",
        is_model_input=True, consumer_node_ids=["node_matmul"],
    ))
    g.add_tensor(Tensor(
        id="t_weight", shape=shape, dtype="fp16", format="nz",
        is_weight=True, consumer_node_ids=["node_matmul"],
    ))
    g.add_tensor(Tensor(
        id="t_mm_out", shape=shape, dtype="fp16", format="nz",
        producer_node_id="node_matmul", consumer_node_ids=["node_add"],
    ))
    g.add_tensor(Tensor(
        id="t_out", shape=shape, dtype="fp16", format="nd",
        producer_node_id="node_add", is_model_output=True,
    ))

    g.add_node(Node(
        id="node_matmul", op_type="cube_matmul",
        inputs=["t_input", "t_weight"], outputs=["t_mm_out"],
        compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
    ))
    g.add_node(Node(
        id="node_add", op_type="vector_add",
        inputs=["t_mm_out"], outputs=["t_out"],
        compute_unit="vector", npu_op="vector_add", is_mapped=True,
    ))
    g.execution_order = ["node_matmul", "node_add"]
    return g


class TestBasicInsertion:
    """基础格式转换插入测试。"""

    def test_inserts_reformat_nz_to_nd(self):
        """matmul(nz) → add(nd) 之间应插入 reformat 节点。"""
        g = _make_matmul_add()
        g = run_reformat_inserter(g, {})

        # 应存在 reformat 节点
        reformat_nodes = [n for n in g.nodes.values() if n.op_type == "dma_reformat"]
        assert len(reformat_nodes) == 1

        rf = reformat_nodes[0]
        assert rf.compute_unit == "idma"
        assert rf.npu_op == "dma_reformat"
        assert rf.is_mapped is True

    def test_reformat_annotation(self):
        """reformat 节点的 format_annotation 应标注 nz→nd。"""
        g = _make_matmul_add()
        g = run_reformat_inserter(g, {})

        rf = [n for n in g.nodes.values() if n.op_type == "dma_reformat"][0]
        assert rf.format_annotation["inputs"][0]["format"] == "nz"
        assert rf.format_annotation["outputs"][0]["format"] == "nd"

    def test_new_tensor_created(self):
        """应创建 nd 格式的中间 tensor。"""
        g = _make_matmul_add()
        g = run_reformat_inserter(g, {})

        rf = [n for n in g.nodes.values() if n.op_type == "dma_reformat"][0]
        new_tid = rf.outputs[0]
        new_t = g.tensors[new_tid]

        assert new_t.format == "nd"
        assert new_t.dtype == "fp16"
        assert new_t.shape == [1, 32, 64]
        assert new_t.producer_node_id == rf.id

    def test_consumer_rewired(self):
        """add 节点的输入应指向 reformat 的输出。"""
        g = _make_matmul_add()
        g = run_reformat_inserter(g, {})

        add_node = g.nodes["node_add"]
        rf = [n for n in g.nodes.values() if n.op_type == "dma_reformat"][0]

        # add 的输入是 reformat 的输出
        assert rf.outputs[0] in add_node.inputs
        # add 的输入不再是 t_mm_out
        assert "t_mm_out" not in add_node.inputs

    def test_producer_consumer_chain(self):
        """tensor 的 consumer 链应正确更新。"""
        g = _make_matmul_add()
        g = run_reformat_inserter(g, {})

        rf = [n for n in g.nodes.values() if n.op_type == "dma_reformat"][0]

        # t_mm_out 的 consumer 应指向 reformat（不再是 node_add）
        assert rf.id in g.tensors["t_mm_out"].consumer_node_ids
        assert "node_add" not in g.tensors["t_mm_out"].consumer_node_ids

    def test_execution_order(self):
        """reformat 应在消费者之前执行。"""
        g = _make_matmul_add()
        g = run_reformat_inserter(g, {})

        rf = [n for n in g.nodes.values() if n.op_type == "dma_reformat"][0]
        order = g.execution_order
        assert order.index(rf.id) < order.index("node_add")
        assert order.index("node_matmul") < order.index(rf.id)


class TestNoInsertion:
    """不需要插入 reformat 的场景。"""

    def test_format_matches(self):
        """格式匹配时不插入 reformat。"""
        g = Graph()
        shape = [1, 32, 64]
        g.add_tensor(Tensor(
            id="t_in", shape=shape, dtype="fp16", format="nd",
            is_model_input=True, consumer_node_ids=["n0"],
        ))
        g.add_tensor(Tensor(
            id="t_mid", shape=shape, dtype="fp16", format="nd",
            producer_node_id="n0", consumer_node_ids=["n1"],
        ))
        g.add_tensor(Tensor(
            id="t_out", shape=shape, dtype="fp16", format="nd",
            producer_node_id="n1", is_model_output=True,
        ))
        g.add_node(Node(
            id="n0", op_type="vector_add", inputs=["t_in"], outputs=["t_mid"],
            compute_unit="vector", npu_op="vector_add", is_mapped=True,
        ))
        g.add_node(Node(
            id="n1", op_type="vector_gelu", inputs=["t_mid"], outputs=["t_out"],
            compute_unit="vector", npu_op="vector_gelu", is_mapped=True,
        ))
        g.execution_order = ["n0", "n1"]

        g = run_reformat_inserter(g, {})

        reformat_nodes = [n for n in g.nodes.values() if n.op_type == "dma_reformat"]
        assert len(reformat_nodes) == 0

    def test_weight_skipped(self):
        """权重的格式转换由 DMA load 处理，不插入 reformat。"""
        g = Graph()
        shape = [1, 32, 64]
        g.add_tensor(Tensor(
            id="t_in", shape=shape, dtype="fp16", format="nd",
            is_model_input=True, consumer_node_ids=["n0"],
        ))
        g.add_tensor(Tensor(
            id="t_w", shape=shape, dtype="fp16", format="nd",
            is_weight=True, consumer_node_ids=["n0"],
        ))
        g.add_tensor(Tensor(
            id="t_out", shape=shape, dtype="fp16", format="nz",
            producer_node_id="n0", is_model_output=True,
        ))
        # cube 要求 nz，但 weight(nd) 不应触发 reformat
        g.add_node(Node(
            id="n0", op_type="cube_matmul",
            inputs=["t_in", "t_w"], outputs=["t_out"],
            compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
        ))
        g.execution_order = ["n0"]

        g = run_reformat_inserter(g, {})

        reformat_nodes = [n for n in g.nodes.values() if n.op_type == "dma_reformat"]
        assert len(reformat_nodes) == 0

    def test_model_input_skipped(self):
        """模型输入的格式转换由 DMA load 处理，不插入 reformat。"""
        g = Graph()
        shape = [1, 32, 64]
        g.add_tensor(Tensor(
            id="t_in", shape=shape, dtype="fp16", format="nd",
            is_model_input=True, consumer_node_ids=["n0"],
        ))
        g.add_tensor(Tensor(
            id="t_out", shape=shape, dtype="fp16", format="nz",
            producer_node_id="n0", is_model_output=True,
        ))
        # cube 要求 nz，但 model_input(nd) 不应触发 reformat
        g.add_node(Node(
            id="n0", op_type="cube_matmul",
            inputs=["t_in"], outputs=["t_out"],
            compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
        ))
        g.execution_order = ["n0"]

        g = run_reformat_inserter(g, {})

        reformat_nodes = [n for n in g.nodes.values() if n.op_type == "dma_reformat"]
        assert len(reformat_nodes) == 0


class TestChainInsertion:
    """多节点链路的 reformat 插入测试。"""

    def _make_matmul_add_matmul(self) -> Graph:
        """matmul(cube) → add(vector) → matmul(cube)，需要两次 reformat。

        matmul 输出 nz → add 要求 nd：插入 reformat1(nz→nd)
        add 输出 nd → matmul 要求 nz：插入 reformat2(nd→nz)
        """
        shape = [1, 32, 64]
        g = Graph()

        g.add_tensor(Tensor(
            id="t_in", shape=shape, dtype="fp16", format="nd",
            is_model_input=True, consumer_node_ids=["n_mm1"],
        ))
        g.add_tensor(Tensor(
            id="t_w1", shape=shape, dtype="fp16", format="nz",
            is_weight=True, consumer_node_ids=["n_mm1"],
        ))
        g.add_tensor(Tensor(
            id="t_mm1_out", shape=shape, dtype="fp16", format="nz",
            producer_node_id="n_mm1", consumer_node_ids=["n_add"],
        ))
        g.add_tensor(Tensor(
            id="t_add_out", shape=shape, dtype="fp16", format="nd",
            producer_node_id="n_add", consumer_node_ids=["n_mm2"],
        ))
        g.add_tensor(Tensor(
            id="t_w2", shape=shape, dtype="fp16", format="nz",
            is_weight=True, consumer_node_ids=["n_mm2"],
        ))
        g.add_tensor(Tensor(
            id="t_out", shape=shape, dtype="fp16", format="nz",
            producer_node_id="n_mm2", is_model_output=True,
        ))

        g.add_node(Node(
            id="n_mm1", op_type="cube_matmul",
            inputs=["t_in", "t_w1"], outputs=["t_mm1_out"],
            compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
        ))
        g.add_node(Node(
            id="n_add", op_type="vector_add",
            inputs=["t_mm1_out"], outputs=["t_add_out"],
            compute_unit="vector", npu_op="vector_add", is_mapped=True,
        ))
        g.add_node(Node(
            id="n_mm2", op_type="cube_matmul",
            inputs=["t_add_out", "t_w2"], outputs=["t_out"],
            compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
        ))
        g.execution_order = ["n_mm1", "n_add", "n_mm2"]
        return g

    def test_two_reformats_inserted(self):
        """matmul→add→matmul 链应插入 2 个 reformat（nz→nd 和 nd→nz）。"""
        g = self._make_matmul_add_matmul()
        g = run_reformat_inserter(g, {})

        reformat_nodes = [n for n in g.nodes.values() if n.op_type == "dma_reformat"]
        assert len(reformat_nodes) == 2

    def test_execution_order_correct(self):
        """execution_order 应为 matmul → rf1 → add → rf2 → matmul。"""
        g = self._make_matmul_add_matmul()
        g = run_reformat_inserter(g, {})

        order = g.execution_order
        assert order.index("n_mm1") < order.index("n_add")
        assert order.index("n_add") < order.index("n_mm2")
        assert len(order) == 5  # 3 原始 + 2 reformat

    def test_format_annotations_correct(self):
        """两个 reformat 的 format_annotation 应分别为 nz→nd 和 nd→nz。"""
        g = self._make_matmul_add_matmul()
        g = run_reformat_inserter(g, {})

        rfs = sorted(
            [n for n in g.nodes.values() if n.op_type == "dma_reformat"],
            key=lambda n: g.execution_order.index(n.id),
        )

        # 第一个 reformat: nz → nd（matmul 输出 → add 输入）
        assert rfs[0].format_annotation["inputs"][0]["format"] == "nz"
        assert rfs[0].format_annotation["outputs"][0]["format"] == "nd"

        # 第二个 reformat: nd → nz（add 输出 → matmul 输入）
        assert rfs[1].format_annotation["inputs"][0]["format"] == "nd"
        assert rfs[1].format_annotation["outputs"][0]["format"] == "nz"


class TestCustomConfig:
    """自定义 compute_unit_formats 配置。"""

    def test_custom_format_mapping(self):
        """使用自定义格式映射时应按新规则插入 reformat。"""
        g = _make_matmul_add()
        # 全部都要求 nd → matmul(cube) 输入的 nz 不需要转换
        # 因为 input/weight 跳过，t_mm_out(nz) → add(要求nd) 仍然需要
        g = run_reformat_inserter(g, {"compute_unit_formats": {"vector": "nd"}})

        reformat_nodes = [n for n in g.nodes.values() if n.op_type == "dma_reformat"]
        # 只在 t_mm_out(nz) → add(nd) 处插入
        assert len(reformat_nodes) == 1

    def test_empty_format_mapping(self):
        """空的格式映射时不插入任何 reformat。"""
        g = _make_matmul_add()
        g = run_reformat_inserter(g, {"compute_unit_formats": {}})

        reformat_nodes = [n for n in g.nodes.values() if n.op_type == "dma_reformat"]
        assert len(reformat_nodes) == 0


class TestPostValidate:
    """post_validate 校验测试。"""

    def test_valid_reformat(self):
        """合法的 reformat 节点不报错。"""
        g = _make_matmul_add()
        g = run_reformat_inserter(g, {})

        errors = post_validate(g)
        assert errors == []

    def test_same_format_reformat_invalid(self):
        """输入输出格式相同的 reformat 应报错。"""
        g = Graph()
        g.add_node(Node(
            id="rf_bad", op_type="dma_reformat",
            inputs=["t0"], outputs=["t1"],
            compute_unit="idma", npu_op="dma_reformat", is_mapped=True,
            format_annotation={
                "inputs": [{"format": "nd", "dtype": "fp16"}],
                "outputs": [{"format": "nd", "dtype": "fp16"}],
            },
        ))

        errors = post_validate(g)
        assert len(errors) == 1
        assert "格式相同" in errors[0]

    def test_missing_annotation_invalid(self):
        """缺少 format_annotation 的 reformat 应报错。"""
        g = Graph()
        g.add_node(Node(
            id="rf_bad", op_type="dma_reformat",
            inputs=["t0"], outputs=["t1"],
            compute_unit="idma", npu_op="dma_reformat", is_mapped=True,
        ))

        errors = post_validate(g)
        assert len(errors) == 1
        assert "缺少" in errors[0]
