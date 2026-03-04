"""memory_planner 单元测试。"""

import os

import pytest

from npu_compiler.common import Graph, Node, Tensor, load_config
from npu_compiler.common.errors import MemoryPlanError
from npu_compiler.memory_planner import run
from npu_compiler.memory_planner.memory_planner import (
    align_up,
    calc_padded_size,
    post_validate,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "hardware_config.yaml")


def _load_config() -> dict:
    return load_config(CONFIG_PATH)


# 小 shape（适配 L1 全局布局）
_SMALL_SHAPE = [1, 32, 64]
# 中等 shape：每个 tensor ~4MB。
# 线性链 5 算子 = 6 tensor * 4MB = 24MB > L1(16MB) → per-op 路径
# 单个算子 per-op: 2 tensor * 4MB = 8MB < L1(16MB) → 不溢出
_MEDIUM_SHAPE = [1, 1024, 2048]


def _make_linear_chain(
    n_ops: int = 5,
    shape: list[int] | None = None,
) -> Graph:
    """创建一个 n_ops 算子的线性链图。"""
    if shape is None:
        shape = _SMALL_SHAPE
    g = Graph()
    t_in = Tensor(
        id="tensor_input",
        shape=shape,
        dtype="fp16",
        is_model_input=True,
        consumer_node_ids=["node_0"],
    )
    g.add_tensor(t_in)

    prev_tid = "tensor_input"
    for i in range(n_ops):
        out_tid = f"tensor_{i}" if i < n_ops - 1 else "tensor_output"
        node = Node(
            id=f"node_{i}",
            op_type="npu_add",
            inputs=[prev_tid],
            outputs=[out_tid],
            compute_unit="vector",
            npu_op="npu_add",
            is_mapped=True,
        )
        g.add_node(node)

        is_last = i == n_ops - 1
        t = Tensor(
            id=out_tid,
            shape=shape,
            dtype="fp16",
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
        """同时活跃的 tensor 地址不重叠（per-op 路径）。"""
        g = _make_linear_chain(5, shape=_MEDIUM_SHAPE)
        config = _load_config()
        g, _ = run(g, config)

        # 检查同时活跃的 tensor 不重叠
        tensors_with_hbm = [t for t in g.tensors.values() if t.hbm_offset is not None]
        for i, t1 in enumerate(tensors_with_hbm):
            for t2 in tensors_with_hbm[i + 1 :]:
                t1_start = _get_first_use(g, t1)
                t1_end = _get_last_use(g, t1)
                t2_start = _get_first_use(g, t2)
                t2_end = _get_last_use(g, t2)
                # 生命周期有交集 → 地址区间不得重叠
                if t1_start <= t2_end and t2_start <= t1_end:
                    assert t1.hbm_offset is not None
                    assert t1.hbm_size is not None
                    assert t2.hbm_offset is not None
                    assert t2.hbm_size is not None
                    end1 = t1.hbm_offset + t1.hbm_size
                    end2 = t2.hbm_offset + t2.hbm_size
                    overlap = t1.hbm_offset < end2 and t2.hbm_offset < end1
                    assert not overlap, (
                        f"同时活跃的 {t1.id}[{t1.hbm_offset}:{end1}] 和"
                        f" {t2.id}[{t2.hbm_offset}:{end2}] 地址重叠"
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
        """HBM offset 是 512 的倍数，L1 offset 是 32 的倍数（per-op 路径）。"""
        g = _make_linear_chain(5, shape=_MEDIUM_SHAPE)
        config = _load_config()
        g, _ = run(g, config)

        for t in g.tensors.values():
            if t.hbm_offset is not None:
                assert t.hbm_offset % 512 == 0, f"{t.id} HBM offset {t.hbm_offset} 不是 512 的倍数"
            if t.l1_offset is not None:
                assert t.l1_offset % 32 == 0, f"{t.id} L1 offset {t.l1_offset} 不是 32 的倍数"


class TestReuse:
    def test_reuse(self):
        """线性链中 dead tensor 的空间被后续 tensor 复用（per-op 路径）。"""
        g = _make_linear_chain(5, shape=_MEDIUM_SHAPE)
        config = _load_config()
        g, _ = run(g, config)

        offsets = [t.hbm_offset for t in g.tensors.values() if t.hbm_offset is not None]
        # 线性链中有 6 个 tensor，但不是所有都需要独立空间
        unique_offsets = set(offsets)
        # 至少应有一些复用（unique < total）
        assert len(unique_offsets) < len(offsets), (
            f"期望有复用，但所有 {len(offsets)} 个 tensor 都有独立偏移"
        )


class TestDmaPlan:
    def test_dma_plan(self):
        """每个算子有正确数量的 load 和 store 指令（per-op 路径）。"""
        g = _make_linear_chain(5, shape=_MEDIUM_SHAPE)
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
        huge_tensor = Tensor(
            id="huge_input",
            shape=[1, 4096, 4096],
            dtype="fp32",
            is_model_input=True,
            consumer_node_ids=["node_0"],
        )
        huge_output = Tensor(
            id="huge_output",
            shape=[1, 4096, 4096],
            dtype="fp32",
            is_model_output=True,
            producer_node_id="node_0",
        )
        node = Node(
            id="node_0",
            op_type="npu_add",
            inputs=["huge_input"],
            outputs=["huge_output"],
            compute_unit="vector",
            npu_op="npu_add",
            is_mapped=True,
        )
        g.add_tensor(huge_tensor)
        g.add_tensor(huge_output)
        g.add_node(node)
        g.execution_order = ["node_0"]

        config = _load_config()
        with pytest.raises(MemoryPlanError, match="L1 溢出"):
            run(g, config)


class TestNoOverlapLongChain:
    """针对 HBM 空闲块重复释放 bug 的回归测试。

    使用 10 个算子的长链，确保 freed_set 正确防止已复用块被重新释放。
    """

    def test_long_chain_no_overlap(self):
        g = _make_linear_chain(10, shape=_MEDIUM_SHAPE)
        config = _load_config()
        g, _ = run(g, config)

        tensors_with_hbm = [t for t in g.tensors.values() if t.hbm_offset is not None]
        for i, t1 in enumerate(tensors_with_hbm):
            for t2 in tensors_with_hbm[i + 1 :]:
                t1_start = _get_first_use(g, t1)
                t1_end = _get_last_use(g, t1)
                t2_start = _get_first_use(g, t2)
                t2_end = _get_last_use(g, t2)
                if t1_start <= t2_end and t2_start <= t1_end:
                    assert t1.hbm_offset is not None
                    assert t1.hbm_size is not None
                    assert t2.hbm_offset is not None
                    assert t2.hbm_size is not None
                    end1 = t1.hbm_offset + t1.hbm_size
                    end2 = t2.hbm_offset + t2.hbm_size
                    overlap = t1.hbm_offset < end2 and t2.hbm_offset < end1
                    assert not overlap, (
                        f"同时活跃的 {t1.id}[{t1.hbm_offset}:{end1}] 和"
                        f" {t2.id}[{t2.hbm_offset}:{end2}] 地址重叠"
                    )


class TestDmaStoreFormat:
    """DMA store 的 src_format 应使用 format_annotation 标注的计算格式。"""

    def test_store_uses_annotated_format(self):
        """构建 3 节点图（总 tensor > L1 容量）使其走 per-op 路径。"""
        g = Graph()
        g.add_tensor(
            Tensor(
                id="t_in",
                shape=_MEDIUM_SHAPE,
                dtype="fp16",
                format="nd",
                is_model_input=True,
                consumer_node_ids=["node_0"],
            )
        )
        g.add_tensor(
            Tensor(
                id="t_mid1",
                shape=_MEDIUM_SHAPE,
                dtype="fp16",
                format="nd",
                producer_node_id="node_0",
                consumer_node_ids=["node_1"],
            )
        )
        g.add_tensor(
            Tensor(
                id="t_mid2",
                shape=_MEDIUM_SHAPE,
                dtype="fp16",
                format="nd",
                producer_node_id="node_1",
                consumer_node_ids=["node_2"],
            )
        )
        g.add_tensor(
            Tensor(
                id="t_mid3",
                shape=_MEDIUM_SHAPE,
                dtype="fp16",
                format="nd",
                producer_node_id="node_2",
                consumer_node_ids=["node_3"],
            )
        )
        g.add_tensor(
            Tensor(
                id="t_out",
                shape=_MEDIUM_SHAPE,
                dtype="fp16",
                format="nd",
                producer_node_id="node_3",
                is_model_output=True,
            )
        )
        # node_0 有 format_annotation，是测试目标
        g.add_node(
            Node(
                id="node_0",
                op_type="npu_matmul",
                inputs=["t_in"],
                outputs=["t_mid1"],
                compute_unit="cube",
                npu_op="npu_matmul",
                is_mapped=True,
                format_annotation={
                    "inputs": [{"format": "nz", "dtype": "fp16"}],
                    "outputs": [{"format": "nz", "dtype": "fp16"}],
                },
            )
        )
        g.add_node(
            Node(
                id="node_1",
                op_type="npu_add",
                inputs=["t_mid1"],
                outputs=["t_mid2"],
                compute_unit="vector",
                npu_op="npu_add",
                is_mapped=True,
            )
        )
        g.add_node(
            Node(
                id="node_2",
                op_type="npu_gelu",
                inputs=["t_mid2"],
                outputs=["t_mid3"],
                compute_unit="vector",
                npu_op="npu_gelu",
                is_mapped=True,
            )
        )
        g.add_node(
            Node(
                id="node_3",
                op_type="npu_add",
                inputs=["t_mid3"],
                outputs=["t_out"],
                compute_unit="vector",
                npu_op="npu_add",
                is_mapped=True,
            )
        )
        g.execution_order = ["node_0", "node_1", "node_2", "node_3"]

        config = _load_config()
        g, dma_plans = run(g, config)

        # node_0 的 DMA plan
        plan = dma_plans[0]
        assert plan.node_id == "node_0"
        assert plan.loads[0].dst_format == "nz"
        assert plan.stores[0].src_format == "nz"
        assert plan.stores[0].dst_format == "nd"


class TestL1OnlyOptimization:
    """所有 tensor 可同时放入 L1 时，使用全局 L1 布局优化。"""

    def test_small_graph_uses_global_layout(self):
        """小图所有 tensor 直接常驻 L1，DMA 只有首尾的 bulk load/store。"""
        g = Graph()
        g.add_tensor(
            Tensor(
                id="t_in",
                shape=[1, 4, 4],
                dtype="fp16",
                is_model_input=True,
                consumer_node_ids=["node_0"],
            )
        )
        g.add_tensor(
            Tensor(
                id="t_mid",
                shape=[1, 4, 4],
                dtype="fp16",
                producer_node_id="node_0",
                consumer_node_ids=["node_1"],
            )
        )
        g.add_tensor(
            Tensor(
                id="t_out",
                shape=[1, 4, 4],
                dtype="fp16",
                producer_node_id="node_1",
                is_model_output=True,
            )
        )
        g.add_node(
            Node(
                id="node_0",
                op_type="npu_add",
                inputs=["t_in"],
                outputs=["t_mid"],
                compute_unit="vector",
                npu_op="npu_add",
                is_mapped=True,
            )
        )
        g.add_node(
            Node(
                id="node_1",
                op_type="npu_gelu",
                inputs=["t_mid"],
                outputs=["t_out"],
                compute_unit="vector",
                npu_op="npu_gelu",
                is_mapped=True,
            )
        )
        g.execution_order = ["node_0", "node_1"]

        config = _load_config()
        g, dma_plans = run(g, config)

        # 全局布局模式: 只有 2 个 DMA plan（bulk_load + bulk_store）
        assert len(dma_plans) == 2
        assert dma_plans[0].node_id == "__bulk_load__"
        assert dma_plans[1].node_id == "__bulk_store__"
        # 所有 tensor 有 l1_offset
        for t in g.tensors.values():
            assert t.l1_offset is not None


class TestPostValidate:
    def test_valid(self):
        g = Graph()
        g.add_tensor(
            Tensor(
                id="t0",
                shape=[1],
                dtype="fp16",
                is_model_output=True,
                hbm_offset=0,
                hbm_size=2,
                l1_offset=0,
            )
        )
        assert post_validate(g) == []

    def test_missing_offsets(self):
        g = Graph()
        g.add_tensor(Tensor(id="t0", shape=[1], dtype="fp16", is_model_output=True))
        errors = post_validate(g)
        assert any("hbm_offset" in e for e in errors)
        assert any("hbm_size" in e for e in errors)
        assert any("l1_offset" in e for e in errors)
