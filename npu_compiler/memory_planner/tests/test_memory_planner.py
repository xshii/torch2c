"""memory_planner 单元测试。"""

import os

import pytest

from npu_compiler.common import Graph, Node, Tensor, load_config
from npu_compiler.memory_planner import run
from npu_compiler.memory_planner.memory_planner import (
    DmaPlan,
    align_up,
    calc_padded_size,
)
from npu_compiler.common.errors import MemoryPlanError

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "hardware_config.yaml"
)


def _load_config() -> dict:
    return load_config(CONFIG_PATH)


def _make_linear_chain(n_ops: int = 5) -> Graph:
    """创建一个 n_ops 算子的线性链图。"""
    g = Graph()
    # 输入 tensor
    t_in = Tensor(
        id="tensor_input", shape=[1, 32, 64], dtype="fp16",
        is_model_input=True, consumer_node_ids=["node_0"],
    )
    g.add_tensor(t_in)

    prev_tid = "tensor_input"
    for i in range(n_ops):
        out_tid = f"tensor_{i}" if i < n_ops - 1 else "tensor_output"
        node = Node(
            id=f"node_{i}", op_type="npu_add",
            inputs=[prev_tid], outputs=[out_tid],
            compute_unit="vector", npu_op="npu_add", is_mapped=True,
        )
        g.add_node(node)

        is_last = i == n_ops - 1
        t = Tensor(
            id=out_tid, shape=[1, 32, 64], dtype="fp16",
            producer_node_id=f"node_{i}",
            consumer_node_ids=[] if is_last else [f"node_{i + 1}"],
            is_model_output=is_last,
        )
        g.add_tensor(t)
        prev_tid = out_tid

    g.execution_order = [f"node_{i}" for i in range(n_ops)]
    return g


class TestCalcPaddedSize:
    def test_nd_format(self):
        # [1, 32, 64] fp16 = 1*32*64*2 = 4096
        assert calc_padded_size([1, 32, 64], "fp16", "nd", 16) == 4096

    def test_nz_aligned(self):
        # [1, 32, 64] 已对齐到 16 → 不变
        assert calc_padded_size([1, 32, 64], "fp16", "nz", 16) == 4096

    def test_nz_unaligned(self):
        # [1, 30, 60] → pad 到 [1, 32, 64], 1*32*64*2 = 4096
        assert calc_padded_size([1, 30, 60], "fp16", "nz", 16) == 4096

    def test_nz_small(self):
        # [1, 3, 5] → pad 到 [1, 16, 16], 1*16*16*2 = 512
        assert calc_padded_size([1, 3, 5], "fp16", "nz", 16) == 512

    def test_fp32_dtype(self):
        # [1, 32, 64] fp32 = 1*32*64*4 = 8192
        assert calc_padded_size([1, 32, 64], "fp32", "nd", 16) == 8192


class TestAlignUp:
    def test_already_aligned(self):
        assert align_up(512, 512) == 512

    def test_needs_alignment(self):
        assert align_up(100, 512) == 512

    def test_one_past(self):
        assert align_up(513, 512) == 1024

    def test_zero(self):
        assert align_up(0, 32) == 0


class TestNoOverlap:
    def test_no_overlap(self):
        """同时活跃的 tensor 地址不重叠。"""
        g = _make_linear_chain(5)
        config = _load_config()
        g, _ = run(g, config)

        # 检查同时活跃的 tensor 不重叠
        tensors_with_hbm = [
            t for t in g.tensors.values() if t.hbm_offset is not None
        ]
        for i, t1 in enumerate(tensors_with_hbm):
            for t2 in tensors_with_hbm[i + 1:]:
                # 判断是否同时活跃：取生命周期交集
                t1_start = _get_first_use(g, t1)
                t1_end = _get_last_use(g, t1)
                t2_start = _get_first_use(g, t2)
                t2_end = _get_last_use(g, t2)
                # 如果有交集
                if t1_start <= t2_end and t2_start <= t1_end:
                    end1 = t1.hbm_offset + t1.hbm_size
                    end2 = t2.hbm_offset + t2.hbm_size
                    overlap = (
                        t1.hbm_offset < end2 and t2.hbm_offset < end1
                    )
                    if overlap:
                        assert t1.hbm_offset == t2.hbm_offset, (
                            f"同时活跃的 {t1.id} 和 {t2.id} 地址重叠"
                            " (不同偏移但区间交叉)"
                        )


def _get_first_use(g: Graph, t: Tensor) -> int:
    order_map = {nid: i for i, nid in enumerate(g.execution_order)}
    if t.is_model_input or t.is_weight:
        return 0
    if t.producer_node_id and t.producer_node_id in order_map:
        return order_map[t.producer_node_id]
    return 0


def _get_last_use(g: Graph, t: Tensor) -> int:
    order_map = {nid: i for i, nid in enumerate(g.execution_order)}
    max_order = len(g.execution_order) - 1
    if t.is_model_output:
        return max_order
    if t.consumer_node_ids:
        orders = [order_map[c] for c in t.consumer_node_ids if c in order_map]
        return max(orders) if orders else max_order
    return max_order


class TestAlignment:
    def test_alignment(self):
        """HBM offset 是 512 的倍数，L1 offset 是 32 的倍数。"""
        g = _make_linear_chain(5)
        config = _load_config()
        g, _ = run(g, config)

        for t in g.tensors.values():
            if t.hbm_offset is not None:
                assert t.hbm_offset % 512 == 0, (
                    f"{t.id} HBM offset {t.hbm_offset} 不是 512 的倍数"
                )
            if t.l1_offset is not None:
                assert t.l1_offset % 32 == 0, (
                    f"{t.id} L1 offset {t.l1_offset} 不是 32 的倍数"
                )


class TestReuse:
    def test_reuse(self):
        """线性链中 dead tensor 的空间被后续 tensor 复用。"""
        g = _make_linear_chain(5)
        config = _load_config()
        g, _ = run(g, config)

        offsets = [
            t.hbm_offset for t in g.tensors.values()
            if t.hbm_offset is not None
        ]
        # 线性链中有 6 个 tensor，但不是所有都需要独立空间
        unique_offsets = set(offsets)
        # 至少应有一些复用（unique < total）
        assert len(unique_offsets) < len(offsets), (
            f"期望有复用，但所有 {len(offsets)} 个 tensor 都有独立偏移"
        )


class TestDmaPlan:
    def test_dma_plan(self):
        """每个算子有正确数量的 load 和 store 指令。"""
        g = _make_linear_chain(5)
        config = _load_config()
        g, dma_plans = run(g, config)

        assert len(dma_plans) == 5
        for plan in dma_plans:
            node = g.nodes[plan.node_id]
            # load 数 = 输入 tensor 数
            expected_loads = len(node.inputs)
            assert len(plan.loads) == expected_loads, (
                f"{plan.node_id}: 期望 {expected_loads} loads, 实际 {len(plan.loads)}"
            )
            # store 数 = 输出 tensor 数
            expected_stores = len(node.outputs)
            assert len(plan.stores) == expected_stores, (
                f"{plan.node_id}: 期望 {expected_stores} stores, 实际 {len(plan.stores)}"
            )


class TestL1CapacityCheck:
    def test_l1_capacity_check(self):
        """L1 溢出时抛出 MemoryPlanError。"""
        g = Graph()
        # 创建一个输入非常大的算子
        huge_tensor = Tensor(
            id="huge_input", shape=[1, 4096, 4096], dtype="fp32",
            is_model_input=True, consumer_node_ids=["node_0"],
        )
        huge_output = Tensor(
            id="huge_output", shape=[1, 4096, 4096], dtype="fp32",
            is_model_output=True, producer_node_id="node_0",
        )
        node = Node(
            id="node_0", op_type="npu_add",
            inputs=["huge_input"], outputs=["huge_output"],
            compute_unit="vector", npu_op="npu_add", is_mapped=True,
        )
        g.add_tensor(huge_tensor)
        g.add_tensor(huge_output)
        g.add_node(node)
        g.execution_order = ["node_0"]

        config = _load_config()
        with pytest.raises(MemoryPlanError, match="L1 溢出"):
            run(g, config)
