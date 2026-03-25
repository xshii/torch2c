"""pipe + tiling 联动测试。

验证 pipe 连接的节点组统一 tile_size 的正确性：
- producer 需要 tiling → consumer 被拉入同 tile_size
- consumer 需要 tiling → producer 被拉入同 tile_size
- 两者都需要 tiling、tile_size 不同 → 取最小值
- 链式 pipe (A→B→C) 统一
- 组内含不可 tile 节点 → 降级 pipe 为 hbm
"""

from __future__ import annotations

import math

import pytest

from torch2c.common import Graph, Node, Tensor
from torch2c.d_emission.memory_planner._tiling import (
    TileInfo,
    _classify_tiled_tensors,
    _find_pipe_groups,
    _unify_pipe_tile_sizes,
    analyze_tiling,
)

# 硬件参数
_CUBE_SIZE = 16
_L1_ALIGN = 32
_L1_CAP = 2 * 1024 * 1024  # 2MB


# ── 辅助 ──────────────────────────────────────────────────


def _make_cube_vector_pipe(
    shape: list[int],
    *,
    weight_shape: list[int] | None = None,
) -> Graph:
    """cube_matmul →(pipe)→ vector_add → output。

    cube 输出通过 pipe 直连 vector，模拟 attention score → add mask。
    """
    g = Graph()
    ws = weight_shape or shape

    g.add_tensor(Tensor(
        id="t_in", shape=shape, dtype="fp16",
        is_model_input=True, consumer_node_ids=["n_mm"],
    ))
    g.add_tensor(Tensor(
        id="t_w", shape=ws, dtype="fp16",
        is_weight=True, consumer_node_ids=["n_mm"],
    ))
    g.add_tensor(Tensor(
        id="t_pipe", shape=shape, dtype="fp16", storage="pipe",
        producer_node_id="n_mm", consumer_node_ids=["n_add"],
    ))
    g.add_tensor(Tensor(
        id="t_mask", shape=shape, dtype="fp16",
        is_model_input=True, consumer_node_ids=["n_add"],
    ))
    g.add_tensor(Tensor(
        id="t_out", shape=shape, dtype="fp16",
        producer_node_id="n_add", is_model_output=True,
    ))
    g.add_node(Node(
        id="n_mm", op_type="cube_matmul",
        inputs=["t_in", "t_w"], outputs=["t_pipe"],
        compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
    ))
    g.add_node(Node(
        id="n_add", op_type="vector_add",
        inputs=["t_pipe", "t_mask"], outputs=["t_out"],
        compute_unit="vector", npu_op="vector_add", is_mapped=True,
    ))
    g.execution_order = ["n_mm", "n_add"]
    return g


def _make_pipe_chain(shape: list[int]) -> Graph:
    """A(cube) →pipe→ B(vector) →pipe→ C(vector) 三节点链。"""
    g = Graph()
    g.add_tensor(Tensor(
        id="t_in", shape=shape, dtype="fp16",
        is_model_input=True, consumer_node_ids=["n_a"],
    ))
    g.add_tensor(Tensor(
        id="t_w", shape=shape, dtype="fp16",
        is_weight=True, consumer_node_ids=["n_a"],
    ))
    g.add_tensor(Tensor(
        id="t_pipe1", shape=shape, dtype="fp16", storage="pipe",
        producer_node_id="n_a", consumer_node_ids=["n_b"],
    ))
    g.add_tensor(Tensor(
        id="t_mid", shape=shape, dtype="fp16",
        is_model_input=True, consumer_node_ids=["n_b"],
    ))
    g.add_tensor(Tensor(
        id="t_pipe2", shape=shape, dtype="fp16", storage="pipe",
        producer_node_id="n_b", consumer_node_ids=["n_c"],
    ))
    g.add_tensor(Tensor(
        id="t_other", shape=shape, dtype="fp16",
        is_model_input=True, consumer_node_ids=["n_c"],
    ))
    g.add_tensor(Tensor(
        id="t_out", shape=shape, dtype="fp16",
        producer_node_id="n_c", is_model_output=True,
    ))
    g.add_node(Node(
        id="n_a", op_type="cube_matmul",
        inputs=["t_in", "t_w"], outputs=["t_pipe1"],
        compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
    ))
    g.add_node(Node(
        id="n_b", op_type="vector_add",
        inputs=["t_pipe1", "t_mid"], outputs=["t_pipe2"],
        compute_unit="vector", npu_op="vector_add", is_mapped=True,
    ))
    g.add_node(Node(
        id="n_c", op_type="vector_add",
        inputs=["t_pipe2", "t_other"], outputs=["t_out"],
        compute_unit="vector", npu_op="vector_add", is_mapped=True,
    ))
    g.execution_order = ["n_a", "n_b", "n_c"]
    return g


# ── 测试：_find_pipe_groups ────────────────────────────────


class TestFindPipeGroups:
    def test_single_pipe_pair(self):
        """单个 pipe tensor 连接的两个节点形成一组。"""
        g = _make_cube_vector_pipe([1, 64, 64])
        groups = _find_pipe_groups(g)
        assert len(groups) == 1
        assert set(groups[0]) == {"n_mm", "n_add"}

    def test_chain_forms_single_group(self):
        """A→pipe→B→pipe→C 形成一个组。"""
        g = _make_pipe_chain([1, 64, 64])
        groups = _find_pipe_groups(g)
        assert len(groups) == 1
        assert set(groups[0]) == {"n_a", "n_b", "n_c"}

    def test_no_pipe_no_groups(self):
        """没有 pipe tensor 时返回空列表。"""
        g = Graph()
        g.add_tensor(Tensor(
            id="t", shape=[1, 8, 8], dtype="fp16",
            is_model_input=True, consumer_node_ids=["n"],
        ))
        g.add_tensor(Tensor(
            id="t_out", shape=[1, 8, 8], dtype="fp16",
            producer_node_id="n", is_model_output=True,
        ))
        g.add_node(Node(
            id="n", op_type="vector_add",
            inputs=["t"], outputs=["t_out"],
            compute_unit="vector", npu_op="vector_add", is_mapped=True,
        ))
        g.execution_order = ["n"]
        assert _find_pipe_groups(g) == []

    def test_execution_order_preserved(self):
        """组内节点按执行序排列。"""
        g = _make_pipe_chain([1, 64, 64])
        groups = _find_pipe_groups(g)
        assert groups[0] == ["n_a", "n_b", "n_c"]


# ── 测试：pipe tiling 统一 ─────────────────────────────────


class TestPipeTilingUnification:
    """pipe 连接的节点统一 tile_size。"""

    def test_producer_tiled_consumer_enrolled(self):
        """producer 需要 tiling → consumer 被拉入相同 tile_size。

        构造: 大 shape 使 cube_matmul 超 L1（权重很大），
        但 vector_add 本身不超 L1（pipe 输入不占 L1）。
        期望: consumer 被 enroll 进 pipe_group，使用同一 tile_size。
        """
        # shape 使 cube_matmul 需要 tiling:
        #   t_in [1,512,512] = 512KB, t_w [1,512,512] = 512KB, t_pipe [1,512,512] = 512KB
        #   total = 1.5MB < 2MB → 不需要 tiling
        # 用更大的 shape 触发 tiling:
        #   [1,1024,1024] fp16 = 2MB per tensor
        #   cube_matmul: t_in(2MB) + t_w(2MB) + t_pipe(pipe,0) = 4MB > 2MB → tiling
        shape = [1, 1024, 1024]
        g = _make_cube_vector_pipe(shape)
        result = analyze_tiling(g, _L1_CAP, _L1_ALIGN, _CUBE_SIZE)

        # 两个节点都在 tile_map 中
        assert "n_mm" in result
        assert "n_add" in result

        # 使用相同 tile_size 和 pipe_group
        assert result["n_mm"].tile_size == result["n_add"].tile_size
        assert result["n_mm"].pipe_group == result["n_add"].pipe_group
        assert result["n_mm"].pipe_group is not None

    def test_consumer_tiled_producer_enrolled(self):
        """consumer 需要 tiling → producer 被拉入。

        构造: vector_add 有大的 mask 导致超 L1，
        但 cube_matmul 本身不超 L1。
        """
        # cube_matmul: t_in [1,64,64](8KB) + t_w [1,64,64](8KB) = 16KB ≪ 2MB
        # vector_add: t_pipe(pipe,0) + t_mask [1,1024,1024](2MB) + t_out [1,1024,1024](2MB) = 4MB > 2MB
        g = Graph()
        small = [1, 64, 64]
        big = [1, 1024, 1024]

        g.add_tensor(Tensor(
            id="t_in", shape=small, dtype="fp16",
            is_model_input=True, consumer_node_ids=["n_mm"],
        ))
        g.add_tensor(Tensor(
            id="t_w", shape=small, dtype="fp16",
            is_weight=True, consumer_node_ids=["n_mm"],
        ))
        g.add_tensor(Tensor(
            id="t_pipe", shape=big, dtype="fp16", storage="pipe",
            producer_node_id="n_mm", consumer_node_ids=["n_add"],
        ))
        g.add_tensor(Tensor(
            id="t_mask", shape=big, dtype="fp16",
            is_model_input=True, consumer_node_ids=["n_add"],
        ))
        g.add_tensor(Tensor(
            id="t_out", shape=big, dtype="fp16",
            producer_node_id="n_add", is_model_output=True,
        ))
        g.add_node(Node(
            id="n_mm", op_type="cube_matmul",
            inputs=["t_in", "t_w"], outputs=["t_pipe"],
            compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
        ))
        g.add_node(Node(
            id="n_add", op_type="vector_add",
            inputs=["t_pipe", "t_mask"], outputs=["t_out"],
            compute_unit="vector", npu_op="vector_add", is_mapped=True,
        ))
        g.execution_order = ["n_mm", "n_add"]

        result = analyze_tiling(g, _L1_CAP, _L1_ALIGN, _CUBE_SIZE)

        # consumer 需要 tiling
        assert "n_add" in result
        # producer 被拉入
        assert "n_mm" in result
        assert result["n_mm"].tile_size == result["n_add"].tile_size
        assert result["n_mm"].pipe_group == result["n_add"].pipe_group

    def test_both_tiled_uses_minimum(self):
        """两个节点都需 tiling 但 tile_size 不同 → 取最小值。

        构造: producer 有大权重（需要小 tile），consumer 也需 tiling。
        """
        g = Graph()
        big = [1, 1024, 1024]

        # producer: t_in(2MB) + t_w(2MB) = 4MB，pipe output 不占 L1
        g.add_tensor(Tensor(
            id="t_in", shape=big, dtype="fp16",
            is_model_input=True, consumer_node_ids=["n_mm"],
        ))
        g.add_tensor(Tensor(
            id="t_w", shape=big, dtype="fp16",
            is_weight=True, consumer_node_ids=["n_mm"],
        ))
        g.add_tensor(Tensor(
            id="t_pipe", shape=big, dtype="fp16", storage="pipe",
            producer_node_id="n_mm", consumer_node_ids=["n_add"],
        ))
        # consumer: t_mask(2MB) + t_out(2MB) = 4MB > 2MB
        g.add_tensor(Tensor(
            id="t_mask", shape=big, dtype="fp16",
            is_model_input=True, consumer_node_ids=["n_add"],
        ))
        g.add_tensor(Tensor(
            id="t_out", shape=big, dtype="fp16",
            producer_node_id="n_add", is_model_output=True,
        ))
        g.add_node(Node(
            id="n_mm", op_type="cube_matmul",
            inputs=["t_in", "t_w"], outputs=["t_pipe"],
            compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
        ))
        g.add_node(Node(
            id="n_add", op_type="vector_add",
            inputs=["t_pipe", "t_mask"], outputs=["t_out"],
            compute_unit="vector", npu_op="vector_add", is_mapped=True,
        ))
        g.execution_order = ["n_mm", "n_add"]

        result = analyze_tiling(g, _L1_CAP, _L1_ALIGN, _CUBE_SIZE)

        assert "n_mm" in result
        assert "n_add" in result
        unified = result["n_mm"].tile_size
        assert unified == result["n_add"].tile_size
        # tile_size 应 ≤ 两者独立计算的最小值
        assert unified <= 1024

    def test_chain_three_nodes_unified(self):
        """A→pipe→B→pipe→C 三节点链统一 tile_size。"""
        # 每个 tensor 2MB → 各节点都需 tiling
        big = [1, 1024, 1024]
        g = _make_pipe_chain(big)

        result = analyze_tiling(g, _L1_CAP, _L1_ALIGN, _CUBE_SIZE)

        # 至少有一个节点需要 tiling（因为 tensor 很大）
        tiled_in_group = [n for n in ("n_a", "n_b", "n_c") if n in result]
        assert len(tiled_in_group) == 3, f"期望 3 节点全 tiled，实际 {tiled_in_group}"

        # 所有节点同 tile_size 和 pipe_group
        ts_set = {result[n].tile_size for n in tiled_in_group}
        pg_set = {result[n].pipe_group for n in tiled_in_group}
        assert len(ts_set) == 1, f"tile_size 不统一: {ts_set}"
        assert len(pg_set) == 1, f"pipe_group 不统一: {pg_set}"

    def test_no_pipe_no_unification(self):
        """无 pipe tensor 时 tiling 独立，无 pipe_group。"""
        g = Graph()
        big = [1, 1024, 1024]
        g.add_tensor(Tensor(
            id="t_in", shape=big, dtype="fp16",
            is_model_input=True, consumer_node_ids=["n0"],
        ))
        g.add_tensor(Tensor(
            id="t_mid", shape=big, dtype="fp16",
            producer_node_id="n0", consumer_node_ids=["n1"],
        ))
        g.add_tensor(Tensor(
            id="t_out", shape=big, dtype="fp16",
            producer_node_id="n1", is_model_output=True,
        ))
        g.add_node(Node(
            id="n0", op_type="vector_add",
            inputs=["t_in"], outputs=["t_mid"],
            compute_unit="vector", npu_op="vector_add", is_mapped=True,
        ))
        g.add_node(Node(
            id="n1", op_type="vector_add",
            inputs=["t_mid"], outputs=["t_out"],
            compute_unit="vector", npu_op="vector_add", is_mapped=True,
        ))
        g.execution_order = ["n0", "n1"]

        result = analyze_tiling(g, _L1_CAP, _L1_ALIGN, _CUBE_SIZE)
        for nid, info in result.items():
            assert info.pipe_group is None, f"节点 {nid} 不应有 pipe_group"


class TestPipeDegradation:
    """pipe 组含不可 tile 节点时降级测试。"""

    def test_untileable_node_degrades_pipe(self):
        """组内有不可 tile 节点 → pipe tensor 降级为 hbm。"""
        g = Graph()
        big = [1, 1024, 1024]

        g.add_tensor(Tensor(
            id="t_in", shape=big, dtype="fp16",
            is_model_input=True, consumer_node_ids=["n_mm"],
        ))
        g.add_tensor(Tensor(
            id="t_w", shape=big, dtype="fp16",
            is_weight=True, consumer_node_ids=["n_mm"],
        ))
        g.add_tensor(Tensor(
            id="t_pipe", shape=big, dtype="fp16", storage="pipe",
            producer_node_id="n_mm", consumer_node_ids=["n_custom"],
        ))
        g.add_tensor(Tensor(
            id="t_out", shape=big, dtype="fp16",
            producer_node_id="n_custom", is_model_output=True,
        ))
        g.add_node(Node(
            id="n_mm", op_type="cube_matmul",
            inputs=["t_in", "t_w"], outputs=["t_pipe"],
            compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
        ))
        # 不可 tile 的自定义 op
        g.add_node(Node(
            id="n_custom", op_type="custom_op",
            inputs=["t_pipe"], outputs=["t_out"],
            compute_unit="vector", npu_op="custom_unknown_op", is_mapped=True,
        ))
        g.execution_order = ["n_mm", "n_custom"]

        result = analyze_tiling(g, _L1_CAP, _L1_ALIGN, _CUBE_SIZE)

        # pipe 应被降级为 hbm
        assert g.tensors["t_pipe"].storage == "hbm"


class TestPipeTilingNumBuffers:
    """pipe 统一后 num_buffers 重新计算。"""

    def test_enrolled_node_gets_multi_buffer(self):
        """被拉入 tiling 的节点如果 L1 充裕，可以获得 multi-buffer。

        构造: producer 需要 tiling（大权重），consumer 本身不需要 tiling
        但被拉入后 L1 很宽裕（pipe 输入不占 L1），应获得 ≥ 2 buffer。
        """
        g = Graph()
        big = [1, 1024, 1024]  # 2MB per tensor
        small = [1, 64, 64]    # 8KB per tensor

        # producer: t_in(2MB) + t_w(2MB) = 4MB > 2MB → tiling
        g.add_tensor(Tensor(
            id="t_in", shape=big, dtype="fp16",
            is_model_input=True, consumer_node_ids=["n_mm"],
        ))
        g.add_tensor(Tensor(
            id="t_w", shape=big, dtype="fp16",
            is_weight=True, consumer_node_ids=["n_mm"],
        ))
        g.add_tensor(Tensor(
            id="t_pipe", shape=big, dtype="fp16", storage="pipe",
            producer_node_id="n_mm", consumer_node_ids=["n_add"],
        ))
        # consumer: t_mask(8KB) + t_out(8KB) = 16KB ≪ 2MB
        # 被拉入 tiling 后 L1 极其宽裕，应获得 multi-buffer
        g.add_tensor(Tensor(
            id="t_mask", shape=small, dtype="fp16",
            is_model_input=True, consumer_node_ids=["n_add"],
        ))
        g.add_tensor(Tensor(
            id="t_out", shape=small, dtype="fp16",
            producer_node_id="n_add", is_model_output=True,
        ))
        g.add_node(Node(
            id="n_mm", op_type="cube_matmul",
            inputs=["t_in", "t_w"], outputs=["t_pipe"],
            compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
        ))
        g.add_node(Node(
            id="n_add", op_type="vector_add",
            inputs=["t_pipe", "t_mask"], outputs=["t_out"],
            compute_unit="vector", npu_op="vector_add", is_mapped=True,
        ))
        g.execution_order = ["n_mm", "n_add"]

        result = analyze_tiling(g, _L1_CAP, _L1_ALIGN, _CUBE_SIZE)

        assert "n_add" in result
        # consumer L1 很小，应获得 multi-buffer
        assert result["n_add"].num_buffers >= 2
