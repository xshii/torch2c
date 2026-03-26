"""Tests for _fused_emitter — fusion group code generation."""

from __future__ import annotations

from torch2c.common import Node, Tensor
from torch2c.d_emission.codegen._fused_emitter import gen_fused_block, segment_by_fusion


# ---- helpers ----


def _make_node(nid, npu_op, inputs, outputs, compute_unit="vector", **params):
    return Node(
        id=nid, op_type=npu_op, inputs=inputs, outputs=outputs,
        npu_op=npu_op, compute_unit=compute_unit, is_mapped=True,
        params=params,
    )


def _make_tensor(tid, shape=None, storage="hbm", producer=None, consumers=None):
    return Tensor(
        id=tid, shape=shape or [1, 32], dtype="fp16",
        storage=storage, producer_node_id=producer,
        consumer_node_ids=consumers or [],
    )


def _dummy_dma_line(instr):
    return f"dma_move({instr['op']}, {instr.get('tensor_id', '?')});"


def _dummy_op_call(npu_op, sig, node, tensors, c_names=None,
                   struct_prefix="", dim_replace=None, l1_layout=None):
    return f"{npu_op}(/*{node.id}*/)"


_SIG = {"compute_ops": {
    "cube_matmul": {"params": []},
    "vector_relu": {"params": []},
    "vector_add": {"params": []},
}}


# ---- segment_by_fusion tests ----


class TestSegmentByFusion:
    def test_no_fusion(self):
        """无融合标注 → 每个节点独立段。"""
        nodes = {
            "n0": _make_node("n0", "vector_relu", ["a"], ["b"]),
            "n1": _make_node("n1", "vector_add", ["b", "c"], ["d"]),
        }
        segments = segment_by_fusion(["n0", "n1"], nodes)
        assert len(segments) == 2
        assert all(not is_fused for is_fused, _ in segments)

    def test_two_fused(self):
        """两个节点同组 → 一个 fused 段。"""
        nodes = {
            "n0": _make_node("n0", "vector_relu", ["a"], ["b"],
                             _fusion_group="fg_0", _fusion_role="head"),
            "n1": _make_node("n1", "vector_add", ["b"], ["c"],
                             _fusion_group="fg_0", _fusion_role="tail"),
        }
        segments = segment_by_fusion(["n0", "n1"], nodes)
        assert len(segments) == 1
        is_fused, nids = segments[0]
        assert is_fused is True
        assert nids == ["n0", "n1"]

    def test_mixed(self):
        """融合组 + 非融合节点混合。"""
        nodes = {
            "n0": _make_node("n0", "cube_matmul", ["a"], ["b"],
                             _fusion_group="fg_0", _fusion_role="head"),
            "n1": _make_node("n1", "vector_relu", ["b"], ["c"],
                             _fusion_group="fg_0", _fusion_role="tail"),
            "n2": _make_node("n2", "vector_add", ["c", "d"], ["e"]),
        }
        segments = segment_by_fusion(["n0", "n1", "n2"], nodes)
        assert len(segments) == 2
        assert segments[0] == (True, ["n0", "n1"])
        assert segments[1] == (False, ["n2"])

    def test_two_groups(self):
        """两个不同融合组。"""
        nodes = {
            "n0": _make_node("n0", "cube_matmul", ["a"], ["b"],
                             _fusion_group="fg_0", _fusion_role="head"),
            "n1": _make_node("n1", "vector_relu", ["b"], ["c"],
                             _fusion_group="fg_0", _fusion_role="tail"),
            "n2": _make_node("n2", "cube_matmul", ["d"], ["e"],
                             _fusion_group="fg_1", _fusion_role="head"),
            "n3": _make_node("n3", "vector_add", ["e"], ["f"],
                             _fusion_group="fg_1", _fusion_role="tail"),
        }
        segments = segment_by_fusion(["n0", "n1", "n2", "n3"], nodes)
        assert len(segments) == 2
        assert segments[0] == (True, ["n0", "n1"])
        assert segments[1] == (True, ["n2", "n3"])

    def test_single_node_group_not_fused(self):
        """单节点组不算融合。"""
        nodes = {
            "n0": _make_node("n0", "vector_relu", ["a"], ["b"],
                             _fusion_group="fg_0", _fusion_role="head"),
        }
        segments = segment_by_fusion(["n0"], nodes)
        assert len(segments) == 1
        assert segments[0] == (False, ["n0"])


# ---- gen_fused_block tests ----


class TestGenFusedBlock:
    def test_internal_tensor_no_dma(self):
        """组内 storage=local 的 tensor 不生成 DMA。"""
        nodes = {
            "n0": _make_node("n0", "cube_matmul", ["x", "w"], ["mm_out"],
                             compute_unit="cube",
                             _fusion_group="fg_0", _fusion_role="head"),
            "n1": _make_node("n1", "vector_relu", ["mm_out"], ["relu_out"],
                             _fusion_group="fg_0", _fusion_role="tail"),
        }
        tensors = {
            "x": _make_tensor("x", storage="hbm"),
            "w": _make_tensor("w", storage="hbm"),
            "mm_out": _make_tensor("mm_out", storage="local",
                                   producer="n0", consumers=["n1"]),
            "relu_out": _make_tensor("relu_out", storage="hbm", producer="n1"),
        }
        dma_plans = {
            "n0": {
                "loads": [
                    {"op": "load", "tensor_id": "x", "l1_offset": 0,
                     "hbm_offset": 0, "size_bytes": 64},
                    {"op": "load", "tensor_id": "w", "l1_offset": 64,
                     "hbm_offset": 64, "size_bytes": 64},
                ],
                "stores": [
                    {"op": "store", "tensor_id": "mm_out", "l1_offset": 128,
                     "hbm_offset": 128, "size_bytes": 64},
                ],
            },
            "n1": {
                "loads": [
                    {"op": "load", "tensor_id": "mm_out", "l1_offset": 128,
                     "hbm_offset": 128, "size_bytes": 64},
                ],
                "stores": [
                    {"op": "store", "tensor_id": "relu_out", "l1_offset": 192,
                     "hbm_offset": 192, "size_bytes": 64},
                ],
            },
        }

        code = gen_fused_block(
            ["n0", "n1"], nodes, tensors, dma_plans, _SIG,
            gen_dma_line_fn=_dummy_dma_line,
            gen_op_call_fn=_dummy_op_call,
        )

        # 应该有 x 和 w 的 load
        assert "load, x" in code
        assert "load, w" in code
        # mm_out 的 store 和 load 都不应该有（internal）
        assert "store, mm_out" not in code
        assert "load, mm_out" not in code
        # relu_out 的 store 应该有
        assert "store, relu_out" in code
        # 两个 compute op 都应该有
        assert "cube_matmul(/*n0*/)" in code
        assert "vector_relu(/*n1*/)" in code

    def test_header_comment(self):
        """生成的代码包含 fusion group 注释。"""
        nodes = {
            "n0": _make_node("n0", "cube_matmul", ["a"], ["b"],
                             _fusion_group="fg_0"),
            "n1": _make_node("n1", "vector_relu", ["b"], ["c"],
                             _fusion_group="fg_0"),
        }
        tensors = {
            "a": _make_tensor("a"),
            "b": _make_tensor("b", storage="local", producer="n0", consumers=["n1"]),
            "c": _make_tensor("c", producer="n1"),
        }
        code = gen_fused_block(
            ["n0", "n1"], nodes, tensors, {}, _SIG,
            gen_dma_line_fn=_dummy_dma_line,
            gen_op_call_fn=_dummy_op_call,
        )
        assert "Fusion Group fg_0" in code
        assert "cube_matmul → vector_relu" in code

    def test_external_only_tensors(self):
        """没有 internal tensor（所有 storage=hbm）→ 所有 DMA 保留。"""
        nodes = {
            "n0": _make_node("n0", "cube_matmul", ["a"], ["b"],
                             _fusion_group="fg_0"),
            "n1": _make_node("n1", "vector_relu", ["b"], ["c"],
                             _fusion_group="fg_0"),
        }
        tensors = {
            "a": _make_tensor("a"),
            "b": _make_tensor("b", storage="hbm", producer="n0", consumers=["n1"]),
            "c": _make_tensor("c", producer="n1"),
        }
        dma_plans = {
            "n0": {"loads": [{"op": "load", "tensor_id": "a",
                              "l1_offset": 0, "hbm_offset": 0, "size_bytes": 64}],
                   "stores": [{"op": "store", "tensor_id": "b",
                               "l1_offset": 64, "hbm_offset": 64, "size_bytes": 64}]},
            "n1": {"loads": [{"op": "load", "tensor_id": "b",
                              "l1_offset": 64, "hbm_offset": 64, "size_bytes": 64}],
                   "stores": [{"op": "store", "tensor_id": "c",
                               "l1_offset": 128, "hbm_offset": 128, "size_bytes": 64}]},
        }
        code = gen_fused_block(
            ["n0", "n1"], nodes, tensors, dma_plans, _SIG,
            gen_dma_line_fn=_dummy_dma_line,
            gen_op_call_fn=_dummy_op_call,
        )
        # b 不是 internal（storage=hbm）→ 所有 DMA 保留
        assert "store, b" in code
        assert "load, b" in code
