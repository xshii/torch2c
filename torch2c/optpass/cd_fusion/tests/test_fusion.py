"""Tests for the fusion_planner module."""

from __future__ import annotations

import pytest

from torch2c.common import Graph, Node, Tensor
from torch2c.optpass.cd_fusion.fusion_planner import (
    FusionGroup,
    _build_tensor_fanout,
    _find_linear_chains,
    _is_fusible_pair,
    run,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tensor(
    tid: str,
    shape: list[int] | None = None,
    producer: str | None = None,
    consumers: list[str] | None = None,
    *,
    is_weight: bool = False,
    is_model_input: bool = False,
    is_model_output: bool = False,
) -> Tensor:
    return Tensor(
        id=tid,
        shape=shape or [1, 32, 32],
        dtype="fp16",
        producer_node_id=producer,
        consumer_node_ids=consumers or [],
        is_weight=is_weight,
        is_model_input=is_model_input,
        is_model_output=is_model_output,
    )


def _make_node(
    nid: str,
    inputs: list[str],
    outputs: list[str],
    compute_unit: str = "vector",
    npu_op: str = "vector_add",
) -> Node:
    return Node(
        id=nid,
        op_type=npu_op,
        inputs=inputs,
        outputs=outputs,
        compute_unit=compute_unit,
        npu_op=npu_op,
        is_mapped=True,
    )


def _build_matmul_bias_gelu() -> Graph:
    """matmul -> bias_add -> gelu, a classic fusible 3-op chain."""
    g = Graph()

    # External inputs
    g.add_tensor(_make_tensor("x", [1, 32, 64], consumers=["n_mm"], is_model_input=True))
    g.add_tensor(_make_tensor("w", [1, 64, 32], consumers=["n_mm"], is_weight=True))
    g.add_tensor(_make_tensor("bias", [1, 1, 32], consumers=["n_add"], is_weight=True))

    # Intermediate tensors
    g.add_tensor(_make_tensor("mm_out", [1, 32, 32], producer="n_mm", consumers=["n_add"]))
    g.add_tensor(_make_tensor("add_out", [1, 32, 32], producer="n_add", consumers=["n_gelu"]))

    # Output
    g.add_tensor(_make_tensor("gelu_out", [1, 32, 32], producer="n_gelu", is_model_output=True))

    # Nodes
    g.add_node(_make_node("n_mm", ["x", "w"], ["mm_out"], "cube", "cube_matmul"))
    g.add_node(_make_node("n_add", ["mm_out", "bias"], ["add_out"], "vector", "vector_add"))
    g.add_node(_make_node("n_gelu", ["add_out"], ["gelu_out"], "vector", "vector_gelu"))

    return g


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMatmulBiasGeluFused:
    """3-op chain (cube_matmul -> vector_add -> vector_gelu) -> single fusion group."""

    def test_single_group_found(self) -> None:
        g = _build_matmul_bias_gelu()
        g = run(g, {})
        groups = g.metadata.get("fusion_groups", [])
        assert len(groups) == 1

    def test_group_contains_all_three(self) -> None:
        g = _build_matmul_bias_gelu()
        g = run(g, {})
        groups = g.metadata.get("fusion_groups", [])
        assert groups[0]["node_ids"] == ["n_mm", "n_add", "n_gelu"]

    def test_savings_positive(self) -> None:
        g = _build_matmul_bias_gelu()
        g = run(g, {})
        groups = g.metadata.get("fusion_groups", [])
        assert groups[0]["estimated_dma_savings"] > 0


class TestFanoutBreaksChain:
    """matmul output feeds 2 consumers -> no fusion."""

    def test_no_fusion(self) -> None:
        g = Graph()
        g.add_tensor(_make_tensor("x", consumers=["n_mm"], is_model_input=True))
        g.add_tensor(_make_tensor("w", consumers=["n_mm"], is_weight=True))
        # mm_out feeds two consumers
        g.add_tensor(_make_tensor("mm_out", producer="n_mm", consumers=["n_a", "n_b"]))
        g.add_tensor(_make_tensor("a_out", producer="n_a", is_model_output=True))
        g.add_tensor(_make_tensor("b_out", producer="n_b", is_model_output=True))

        g.add_node(_make_node("n_mm", ["x", "w"], ["mm_out"], "cube", "cube_matmul"))
        g.add_node(_make_node("n_a", ["mm_out"], ["a_out"], "vector", "vector_relu"))
        g.add_node(_make_node("n_b", ["mm_out"], ["b_out"], "vector", "vector_gelu"))

        g = run(g, {})
        groups = g.metadata.get("fusion_groups", [])
        assert len(groups) == 0


class TestModelOutputBreaksChain:
    """Intermediate is model_output -> no fusion."""

    def test_no_fusion(self) -> None:
        g = Graph()
        g.add_tensor(_make_tensor("x", consumers=["n_a"], is_model_input=True))
        # a_out is a model output but also feeds n_b
        g.add_tensor(_make_tensor(
            "a_out", producer="n_a", consumers=["n_b"], is_model_output=True,
        ))
        g.add_tensor(_make_tensor("b_out", producer="n_b", is_model_output=True))

        g.add_node(_make_node("n_a", ["x"], ["a_out"], "vector", "vector_relu"))
        g.add_node(_make_node("n_b", ["a_out"], ["b_out"], "vector", "vector_gelu"))

        g = run(g, {})
        groups = g.metadata.get("fusion_groups", [])
        assert len(groups) == 0


class TestDmaNotFused:
    """dma_reformat is never part of fusion group."""

    def test_no_fusion(self) -> None:
        g = Graph()
        g.add_tensor(_make_tensor("x", consumers=["n_dma"], is_model_input=True))
        g.add_tensor(_make_tensor("dma_out", producer="n_dma", consumers=["n_v"]))
        g.add_tensor(_make_tensor("v_out", producer="n_v", is_model_output=True))

        g.add_node(_make_node("n_dma", ["x"], ["dma_out"], "idma", "dma_reformat"))
        g.add_node(_make_node("n_v", ["dma_out"], ["v_out"], "vector", "vector_relu"))

        g = run(g, {})
        groups = g.metadata.get("fusion_groups", [])
        assert len(groups) == 0


class TestSingleOpNotFused:
    """Chains of length 1 are not fusion groups."""

    def test_no_fusion(self) -> None:
        g = Graph()
        g.add_tensor(_make_tensor("x", consumers=["n_a"], is_model_input=True))
        g.add_tensor(_make_tensor("a_out", producer="n_a", is_model_output=True))
        g.add_node(_make_node("n_a", ["x"], ["a_out"], "vector", "vector_relu"))

        g = run(g, {})
        groups = g.metadata.get("fusion_groups", [])
        assert len(groups) == 0


class TestTwoSeparateChains:
    """Two independent chains -> two fusion groups."""

    def test_two_groups(self) -> None:
        g = Graph()

        # Chain 1: mm1 -> relu1
        g.add_tensor(_make_tensor("x1", consumers=["n_mm1"], is_model_input=True))
        g.add_tensor(_make_tensor("w1", consumers=["n_mm1"], is_weight=True))
        g.add_tensor(_make_tensor("mm1_out", producer="n_mm1", consumers=["n_relu1"]))
        g.add_tensor(_make_tensor("relu1_out", producer="n_relu1", is_model_output=True))
        g.add_node(_make_node("n_mm1", ["x1", "w1"], ["mm1_out"], "cube", "cube_matmul"))
        g.add_node(_make_node("n_relu1", ["mm1_out"], ["relu1_out"], "vector", "vector_relu"))

        # Chain 2: mm2 -> gelu2
        g.add_tensor(_make_tensor("x2", consumers=["n_mm2"], is_model_input=True))
        g.add_tensor(_make_tensor("w2", consumers=["n_mm2"], is_weight=True))
        g.add_tensor(_make_tensor("mm2_out", producer="n_mm2", consumers=["n_gelu2"]))
        g.add_tensor(_make_tensor("gelu2_out", producer="n_gelu2", is_model_output=True))
        g.add_node(_make_node("n_mm2", ["x2", "w2"], ["mm2_out"], "cube", "cube_matmul"))
        g.add_node(_make_node("n_gelu2", ["mm2_out"], ["gelu2_out"], "vector", "vector_gelu"))

        g = run(g, {})
        groups = g.metadata.get("fusion_groups", [])
        assert len(groups) == 2
        all_node_ids = {nid for grp in groups for nid in grp["node_ids"]}
        assert all_node_ids == {"n_mm1", "n_relu1", "n_mm2", "n_gelu2"}


class TestInternalTensorsIdentified:
    """Verify internal_tensors list is correct for a 3-op chain."""

    def test_internal_tensors(self) -> None:
        g = _build_matmul_bias_gelu()
        g = run(g, {})
        groups = g.metadata.get("fusion_groups", [])
        assert len(groups) == 1
        grp = groups[0]
        assert set(grp["internal_tensors"]) == {"mm_out", "add_out"}
        assert "gelu_out" not in grp["internal_tensors"]


class TestFusionAnnotations:
    """Verify _fusion_group and _fusion_role in node.params."""

    def test_annotations_present(self) -> None:
        g = _build_matmul_bias_gelu()
        g = run(g, {})

        mm = g.get_node("n_mm")
        add = g.get_node("n_add")
        gelu = g.get_node("n_gelu")

        assert mm is not None and mm.params["_fusion_group"] == "fg_0"
        assert mm.params["_fusion_role"] == "head"

        assert add is not None and add.params["_fusion_group"] == "fg_0"
        assert add.params["_fusion_role"] == "middle"

        assert gelu is not None and gelu.params["_fusion_group"] == "fg_0"
        assert gelu.params["_fusion_role"] == "tail"


# ---- Phase 1: aggressive local promotion ----


class TestAggressiveLocalPromotion:
    """Phase 1：所有合格中间 tensor 默认标记为 local。"""

    def test_single_consumer_not_promoted(self) -> None:
        """fan-out == 1 的中间 tensor 不被 Phase 1 提升（尊重 storage_assigner）。"""
        g = _build_matmul_bias_gelu()
        run(g, {})
        # mm_out, add_out 是 fan-out == 1 的链内 tensor
        # Phase 1 不提升它们（storage_assigner 已处理 fan-out == 1 的情况）
        assert g.tensors["mm_out"].storage == "hbm"
        assert g.tensors["add_out"].storage == "hbm"

    def test_model_io_stays_hbm(self) -> None:
        """model_input / model_output 必须保持 hbm。"""
        g = _build_matmul_bias_gelu()
        run(g, {})
        assert g.tensors["x"].storage == "hbm"
        assert g.tensors["gelu_out"].storage == "hbm"

    def test_weight_stays_hbm(self) -> None:
        """权重必须保持 hbm。"""
        g = _build_matmul_bias_gelu()
        run(g, {})
        assert g.tensors["w"].storage == "hbm"
        assert g.tensors["bias"].storage == "hbm"

    def test_fanout_tensor_promoted(self) -> None:
        """fan-out > 1 的中间 tensor 被 Phase 1 标记为 local。"""
        g = Graph()
        g.add_tensor(_make_tensor("x", consumers=["n_mm"], is_model_input=True))
        g.add_tensor(_make_tensor("w", consumers=["n_mm"], is_weight=True))
        # mm_out feeds two consumers — storage_assigner 不处理，Phase 1 会标记为 local
        g.add_tensor(_make_tensor("mm_out", producer="n_mm", consumers=["n_a", "n_b"]))
        g.add_tensor(_make_tensor("a_out", producer="n_a", is_model_output=True))
        g.add_tensor(_make_tensor("b_out", producer="n_b", is_model_output=True))

        g.add_node(_make_node("n_mm", ["x", "w"], ["mm_out"], "cube", "cube_matmul"))
        g.add_node(_make_node("n_a", ["mm_out"], ["a_out"], "vector", "vector_relu"))
        g.add_node(_make_node("n_b", ["mm_out"], ["b_out"], "vector", "vector_gelu"))

        run(g, {})
        # fan-out > 1 tensor 被提升为 local
        assert g.tensors["mm_out"].storage == "local"

    def test_single_consumer_hbm_not_overridden(self) -> None:
        """fan-out == 1 且 storage_assigner 留在 hbm 的 tensor 不被覆盖。"""
        g = Graph()
        g.add_tensor(_make_tensor("x", consumers=["n_a"], is_model_input=True))
        # mm_out 是 fan-out == 1，storage_assigner 可能因硬件约束留在 hbm
        g.add_tensor(_make_tensor("mm_out", producer="n_a", consumers=["n_b"]))
        g.add_tensor(_make_tensor("b_out", producer="n_b", is_model_output=True))

        g.add_node(_make_node("n_a", ["x"], ["mm_out"], "cube", "cube_matmul"))
        g.add_node(_make_node("n_b", ["mm_out"], ["b_out"], "cube", "cube_matmul"))

        # 模拟 storage_assigner 决策：cube→cube 不在 allowed_pairs → 留 hbm
        g.tensors["mm_out"].storage = "hbm"

        run(g, {})
        # Phase 1 不应覆盖 storage_assigner 的 hbm 决策
        assert g.tensors["mm_out"].storage == "hbm"

    def test_pipe_not_downgraded(self) -> None:
        """已经是 pipe 的 tensor 不应被降级为 local。"""
        g = _build_matmul_bias_gelu()
        g.tensors["mm_out"].storage = "pipe"
        run(g, {})
        assert g.tensors["mm_out"].storage == "pipe"

    def test_disable_aggressive(self) -> None:
        """aggressive_local=False 时不做 Phase 1。"""
        g = _build_matmul_bias_gelu()
        run(g, {"aggressive_local": False})
        assert g.tensors["mm_out"].storage == "hbm"
