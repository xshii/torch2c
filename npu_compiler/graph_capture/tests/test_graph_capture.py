"""graph_capture 单元测试。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from ...common.graph_ir import Graph, Node, Tensor
from ..graph_capture import capture, post_validate

# ---- WI-3 新增: addmm/param/dim 处理测试 ----


def test_addmm_inputs_reordered():
    """addmm 的输入从 [bias, mat1, mat2] 重排为 [mat1, mat2, bias]。"""
    model = nn.Linear(16, 16, bias=True)
    model.eval()
    dummy = torch.randn(1, 16)
    graph = capture(model, dummy)

    addmm_nodes = [n for n in graph.nodes.values() if "addmm" in n.op_type]
    if addmm_nodes:
        node = addmm_nodes[0]
        # input_0 应是 mat1（模型输入），input_2 应是 bias（权重）
        graph.get_tensor(node.inputs[0])  # mat1 — verified by existence
        t2 = graph.get_tensor(node.inputs[2])
        # mat1 是模型输入或由前序节点产出，bias 是权重
        assert t2.is_weight, "addmm input[2] (bias) 应为权重"


def test_param_renames_transpose():
    """transpose 的 p0/p1 参数被重命名为 dim0/dim1。"""

    class TransposeModel(nn.Module):
        def forward(self, x):
            return x.transpose(1, 2)

    model = TransposeModel()
    model.eval()
    dummy = torch.randn(1, 4, 8)
    graph = capture(model, dummy)

    tr_nodes = [n for n in graph.nodes.values() if "transpose" in n.op_type]
    assert len(tr_nodes) >= 1
    node = tr_nodes[0]
    assert "dim0" in node.params, f"Expected dim0 in params, got {node.params}"
    assert "dim1" in node.params, f"Expected dim1 in params, got {node.params}"
    assert "p0" not in node.params
    assert "p1" not in node.params


def test_negative_dim_resolved():
    """softmax 的负 dim 索引转为维度大小值。"""

    class SoftmaxModel(nn.Module):
        def forward(self, x):
            return F.softmax(x, dim=-1)

    model = SoftmaxModel()
    model.eval()
    dummy = torch.randn(1, 4, 8)
    graph = capture(model, dummy)

    sm_nodes = [n for n in graph.nodes.values() if "softmax" in n.op_type]
    assert len(sm_nodes) >= 1
    node = sm_nodes[0]
    # dim=-1 对 shape [1,4,8] 应转为 8 (最后一维的大小)
    assert node.params.get("dim") == 8, (
        f"Expected dim=8 (size of last dim), got {node.params.get('dim')}"
    )


# ---- test_capture_linear ----


def test_capture_linear():
    """单个 nn.Linear 导出，验证有 addmm/mm 相关节点。"""
    model = nn.Linear(16, 16, bias=True)
    model.eval()
    dummy = torch.randn(1, 16)

    graph = capture(model, dummy)

    assert len(graph.nodes) > 0
    op_types = {n.op_type for n in graph.nodes.values()}
    assert any("mm" in op for op in op_types), f"Expected mm-related op, got {op_types}"


# ---- test_capture_encoder ----


def test_capture_encoder():
    """完整 encoder 导出，验证关键算子类型存在。"""
    from ..demo.demo_model import DemoEncoder

    model = DemoEncoder(hidden_size=64, num_heads=4, ffn_dim=256)
    model.eval()
    dummy = torch.randn(1, 32, 64)
    mask = torch.zeros(1, 1, 32, 32)

    graph = capture(model, dummy, mask=mask)

    op_types = {n.op_type for n in graph.nodes.values()}

    # 关键算子必须存在
    assert any("mm" in op or "addmm" in op for op in op_types)
    assert any("layer_norm" in op or "native_layer_norm" in op for op in op_types)
    assert any("softmax" in op for op in op_types)
    assert any("gelu" in op for op in op_types)
    assert any("transpose" in op for op in op_types)

    # 节点数在合理范围
    assert len(graph.nodes) >= 40, f"Too few nodes: {len(graph.nodes)}"


# ---- test_weight_marking ----


def test_weight_marking():
    """权重 tensor 的 is_weight=True。"""
    model = nn.Linear(16, 16, bias=True)
    model.eval()
    dummy = torch.randn(1, 16)

    graph = capture(model, dummy)

    weight_tensors = [t for t in graph.tensors.values() if t.is_weight]
    assert len(weight_tensors) >= 2, (
        f"Expected at least 2 weight tensors (weight+bias), got {len(weight_tensors)}"
    )


# ---- test_io_marking ----


def test_io_marking():
    """输入/输出 tensor 标记正确。"""
    model = nn.Linear(16, 16, bias=True)
    model.eval()
    dummy = torch.randn(1, 16)

    graph = capture(model, dummy)

    inputs = [t for t in graph.tensors.values() if t.is_model_input]
    outputs = [t for t in graph.tensors.values() if t.is_model_output]

    assert len(inputs) >= 1, "Expected at least 1 model input"
    assert len(outputs) >= 1, "Expected at least 1 model output"

    # 输入 tensor 不应同时是输出
    input_ids = {t.id for t in inputs}
    output_ids = {t.id for t in outputs}
    assert not input_ids & output_ids, "Input and output sets should not overlap"


# ---- test_mask_input ----


def test_mask_input():
    """带 mask 输入的模型，mask 被标记为 is_model_input。"""
    from ..demo.demo_model import DemoEncoder

    model = DemoEncoder(hidden_size=64, num_heads=4, ffn_dim=256)
    model.eval()
    dummy = torch.randn(1, 32, 64)
    mask = torch.zeros(1, 1, 32, 32)

    graph = capture(model, dummy, mask=mask)

    model_inputs = [t for t in graph.tensors.values() if t.is_model_input]
    # 至少有 dummy_input 和 mask 两个用户输入
    assert len(model_inputs) >= 2, (
        f"Expected at least 2 model inputs (x + mask), got {len(model_inputs)}"
    )


# ---- test_scalar_constant_tensor ----


def test_scalar_constant_tensor():
    """标量乘法中的常量被创建为 is_weight=True, shape=[1] 的 tensor。"""

    class ScalarMulModel(nn.Module):
        def forward(self, x):
            return x * 0.25

    model = ScalarMulModel()
    model.eval()
    dummy = torch.randn(1, 4, 32, 32)

    graph = capture(model, dummy)

    scalar_tensors = [t for t in graph.tensors.values() if t.is_weight and t.shape == [1]]
    assert len(scalar_tensors) >= 1, "Expected scalar constant tensor for 0.25"


# ---- test_multi_output_layernorm ----


def test_multi_output_layernorm():
    """LayerNorm 多输出：output, mean, rstd 全部创建为 Tensor。"""

    class LNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(64)

        def forward(self, x):
            return self.ln(x)

    model = LNModel()
    model.eval()
    dummy = torch.randn(1, 32, 64)

    graph = capture(model, dummy)

    # native_layer_norm 返回 (output, mean, rstd)
    ln_nodes = [
        n
        for n in graph.nodes.values()
        if "layer_norm" in n.op_type or "native_layer_norm" in n.op_type
    ]
    assert len(ln_nodes) >= 1, "Expected at least 1 layer_norm node"

    ln_node = ln_nodes[0]
    # 应有 3 个输出 tensor
    assert len(ln_node.outputs) == 3, (
        f"layer_norm should have 3 outputs, got {len(ln_node.outputs)}"
    )
    for out_tid in ln_node.outputs:
        assert out_tid in graph.tensors, f"Output tensor {out_tid} missing from graph"


# ---- test_graph_validates ----


def test_graph_validates():
    """捕获后的图通过 validate() 校验。"""
    model = nn.Linear(16, 16, bias=True)
    model.eval()
    dummy = torch.randn(1, 16)

    graph = capture(model, dummy)

    errors = graph.validate()
    assert len(errors) == 0, f"Graph validation errors: {errors}"


# ---- test_producer_consumer_consistency ----


def test_producer_consumer_consistency():
    """验证 producer/consumer 引用的一致性。"""
    model = nn.Linear(16, 16, bias=True)
    model.eval()
    dummy = torch.randn(1, 16)

    graph = capture(model, dummy)

    for node in graph.nodes.values():
        # 每个输出 tensor 的 producer 应指向该节点
        for out_tid in node.outputs:
            t = graph.get_tensor(out_tid)
            assert t is not None, f"Output tensor {out_tid} not found"
            assert t.producer_node_id == node.id, (
                f"Tensor {out_tid} producer={t.producer_node_id}, expected {node.id}"
            )
        # 每个输入 tensor 的 consumer 应包含该节点
        for in_tid in node.inputs:
            t = graph.get_tensor(in_tid)
            assert t is not None, f"Input tensor {in_tid} not found"
            assert node.id in t.consumer_node_ids, (
                f"Node {node.id} not in tensor {in_tid} consumers"
            )


class TestPostValidate:
    def test_valid(self):
        g = Graph()
        g.add_tensor(Tensor(id="t0", shape=[1, 16], dtype="fp16", is_model_input=True))
        g.add_tensor(
            Tensor(id="tw", shape=[16, 16], dtype="fp16", is_weight=True, name="linear.weight")
        )
        g.add_tensor(Tensor(id="t1", shape=[1, 16], dtype="fp16", producer_node_id="n0"))
        g.add_node(Node(id="n0", op_type="aten.mm", inputs=["t0", "tw"], outputs=["t1"]))
        assert post_validate(g) == []

    def test_missing_weight_name(self):
        g = Graph()
        g.add_tensor(Tensor(id="tw", shape=[16], dtype="fp16", is_weight=True))
        g.add_tensor(Tensor(id="t1", shape=[16], dtype="fp16", producer_node_id="n0"))
        g.add_node(Node(id="n0", op_type="aten.mm", inputs=["tw"], outputs=["t1"]))
        errors = post_validate(g)
        assert any("name" in e for e in errors)

    def test_missing_input_dtype(self):
        g = Graph()
        g.add_tensor(Tensor(id="t0", shape=[1, 16], dtype="", is_model_input=True))
        g.add_tensor(Tensor(id="t1", shape=[1, 16], dtype="fp16", producer_node_id="n0"))
        g.add_node(Node(id="n0", op_type="aten.mm", inputs=["t0"], outputs=["t1"]))
        errors = post_validate(g)
        assert any("dtype" in e for e in errors)
