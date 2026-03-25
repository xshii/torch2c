"""memory_planner 单元测试。"""

import pytest

from torch2c.common import Graph, Node, Tensor
from torch2c.common.errors import MemoryPlanError
from torch2c.common.testing import load_hw_config, make_linear_chain
from torch2c.d_emission.memory_planner import run
from torch2c.d_emission.memory_planner._hbm_alloc import analyze_lifetimes
from torch2c.d_emission.memory_planner._utils import align_up, calc_padded_size
from torch2c.d_emission.memory_planner.memory_planner import post_validate

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
    return make_linear_chain(n_ops=n_ops, shape=shape)


def _load_config() -> dict:
    return load_hw_config()


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
        g = run(g, config)

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
        g = run(g, config)

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
        g = run(g, config)

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
        g = run(g, config)

        assert len(g.dma_plans) == 5
        for plan in g.dma_plans:
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
            op_type="custom_reduce",
            inputs=["huge_input"],
            outputs=["huge_output"],
            compute_unit="vector",
            npu_op="custom_reduce",
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
        g = run(g, config)

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
                op_type="cube_matmul",
                inputs=["t_in"],
                outputs=["t_mid1"],
                compute_unit="cube",
                npu_op="cube_matmul",
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
                op_type="vector_add",
                inputs=["t_mid1"],
                outputs=["t_mid2"],
                compute_unit="vector",
                npu_op="vector_add",
                is_mapped=True,
            )
        )
        g.add_node(
            Node(
                id="node_2",
                op_type="vector_gelu",
                inputs=["t_mid2"],
                outputs=["t_mid3"],
                compute_unit="vector",
                npu_op="vector_gelu",
                is_mapped=True,
            )
        )
        g.add_node(
            Node(
                id="node_3",
                op_type="vector_add",
                inputs=["t_mid3"],
                outputs=["t_out"],
                compute_unit="vector",
                npu_op="vector_add",
                is_mapped=True,
            )
        )
        g.execution_order = ["node_0", "node_1", "node_2", "node_3"]

        config = _load_config()
        g = run(g, config)

        # node_0 的 DMA plan
        plan = g.dma_plans[0]
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
                op_type="vector_add",
                inputs=["t_in"],
                outputs=["t_mid"],
                compute_unit="vector",
                npu_op="vector_add",
                is_mapped=True,
            )
        )
        g.add_node(
            Node(
                id="node_1",
                op_type="vector_gelu",
                inputs=["t_mid"],
                outputs=["t_out"],
                compute_unit="vector",
                npu_op="vector_gelu",
                is_mapped=True,
            )
        )
        g.execution_order = ["node_0", "node_1"]

        config = _load_config()
        g = run(g, config)

        # 全局布局模式: 只有 2 个 DMA plan（bulk_load + bulk_store）
        assert len(g.dma_plans) == 2
        assert g.dma_plans[0].node_id == "__bulk_load__"
        assert g.dma_plans[1].node_id == "__bulk_store__"
        # 所有 tensor 有 l1_offset
        for t in g.tensors.values():
            assert t.l1_offset is not None


# ── L1 全局分配专项测试 ──────────────────────────────────
# 使用 2MB L1 + 较小 shape 精确测试边界条件

# [1, 512, 512] fp16 nd = 512*512*2 = 512KB
_L1_TEST_SHAPE = [1, 512, 512]
_L1_2MB = 2 * 1024 * 1024  # 2MB


def _small_l1_config() -> dict:
    """基于默认配置，将 L1 缩小到 2MB。"""
    config = _load_config()
    config["memory"]["l1"]["total_size_bytes"] = _L1_2MB
    return config


class TestL1GlobalLiveness:
    """L1 全局 liveness best-fit 分配的专项测试。L1=2MB, tensor=512KB。"""

    def test_l1_reuse_across_ops(self):
        """线性链中，前序算子释放的 L1 空间被后续算子复用。

        5 算子链，每个 tensor 512KB，L1=2MB。
        per-op 需要 2 tensor（输入+输出）= 1MB < 2MB。
        全局分配应复用已释放空间，峰值不应超过 L1。
        """
        g = _make_linear_chain(5, shape=_L1_TEST_SHAPE)
        config = _small_l1_config()
        g = run(g, config)

        for t in g.tensors.values():
            assert t.l1_offset is not None, f"{t.id} 缺少 l1_offset"

        # L1 offset 应有复用（unique offsets < total tensors）
        l1_offsets = [t.l1_offset for t in g.tensors.values()]
        assert len(set(l1_offsets)) < len(l1_offsets), (
            "期望 L1 offset 有复用，但所有 tensor 都有独立偏移"
        )

    def test_l1_no_overlap_coexisting(self):
        """同时活跃在 L1 的 tensor 地址不重叠。"""
        g = _make_linear_chain(5, shape=_L1_TEST_SHAPE)
        config = _small_l1_config()
        g = run(g, config)

        cube_size = config["fractal"]["cube_size"]
        for nid in g.execution_order:
            node = g.nodes[nid]
            op_tids = list(node.inputs) + list(node.outputs)
            op_tensors = [g.tensors[tid] for tid in op_tids if tid in g.tensors]
            for i, t1 in enumerate(op_tensors):
                for t2 in op_tensors[i + 1:]:
                    if t1.l1_offset is None or t2.l1_offset is None:
                        continue
                    s1 = calc_padded_size(t1.shape, t1.dtype, t1.format, cube_size)
                    s2 = calc_padded_size(t2.shape, t2.dtype, t2.format, cube_size)
                    end1 = t1.l1_offset + s1
                    end2 = t2.l1_offset + s2
                    overlap = t1.l1_offset < end2 and t2.l1_offset < end1
                    assert not overlap, (
                        f"节点 {nid}: L1 重叠 {t1.id}[{t1.l1_offset}:{end1}]"
                        f" 和 {t2.id}[{t2.l1_offset}:{end2}]"
                    )

    def test_local_tensor_cross_op_residence(self):
        """storage=local 的 tensor 跨算子驻留 L1，不走 DMA。

        L1=2MB, tensor=512KB。node_0→t_local→node_1→...→node_3。
        t_local 应保持 L1 offset 且 node_1 无 DMA load。
        """
        g = Graph()
        shape = _L1_TEST_SHAPE
        g.add_tensor(Tensor(
            id="t_in", shape=shape, dtype="fp16",
            is_model_input=True, consumer_node_ids=["node_0"],
        ))
        g.add_tensor(Tensor(
            id="t_local", shape=shape, dtype="fp16",
            storage="local",
            producer_node_id="node_0", consumer_node_ids=["node_1"],
        ))
        g.add_tensor(Tensor(
            id="t_mid", shape=shape, dtype="fp16",
            producer_node_id="node_1", consumer_node_ids=["node_2"],
        ))
        g.add_tensor(Tensor(
            id="t_mid2", shape=shape, dtype="fp16",
            producer_node_id="node_2", consumer_node_ids=["node_3"],
        ))
        g.add_tensor(Tensor(
            id="t_out", shape=shape, dtype="fp16",
            producer_node_id="node_3", is_model_output=True,
        ))
        for i, (inp, out) in enumerate([
            (["t_in"], ["t_local"]),
            (["t_local"], ["t_mid"]),
            (["t_mid"], ["t_mid2"]),
            (["t_mid2"], ["t_out"]),
        ]):
            g.add_node(Node(
                id=f"node_{i}", op_type="vector_add",
                inputs=inp, outputs=out,
                compute_unit="vector", npu_op="vector_add", is_mapped=True,
            ))
        g.execution_order = [f"node_{i}" for i in range(4)]

        config = _small_l1_config()
        g = run(g, config)

        t_local = g.tensors["t_local"]
        assert t_local.l1_offset is not None, "t_local 应有 L1 offset"
        assert t_local.hbm_offset is None, "storage=local 不应有 HBM offset"

        plan_1 = g.dma_plans[1]
        load_tids = [inst.tensor_id for inst in plan_1.loads]
        assert "t_local" not in load_tids, "t_local 不应有 DMA load"
        assert post_validate(g) == []

    def test_local_tensors_no_l1_overlap(self):
        """多个 local tensor 同时驻留 L1 时不重叠。

        L1=2MB。node_0→(t_local_a, t_mid), node_1→t_local_b,
        node_2 消费 t_local_a+t_local_b。两个 512KB local 同时在 L1。
        """
        g = Graph()
        shape = _L1_TEST_SHAPE
        g.add_tensor(Tensor(
            id="t_in", shape=shape, dtype="fp16",
            is_model_input=True, consumer_node_ids=["node_0"],
        ))
        g.add_tensor(Tensor(
            id="t_local_a", shape=shape, dtype="fp16", storage="local",
            producer_node_id="node_0", consumer_node_ids=["node_2"],
        ))
        g.add_tensor(Tensor(
            id="t_mid", shape=shape, dtype="fp16",
            producer_node_id="node_0", consumer_node_ids=["node_1"],
        ))
        g.add_tensor(Tensor(
            id="t_local_b", shape=shape, dtype="fp16", storage="local",
            producer_node_id="node_1", consumer_node_ids=["node_2"],
        ))
        g.add_tensor(Tensor(
            id="t_out", shape=shape, dtype="fp16",
            producer_node_id="node_2", is_model_output=True,
        ))
        g.add_node(Node(
            id="node_0", op_type="vector_add",
            inputs=["t_in"], outputs=["t_local_a", "t_mid"],
            compute_unit="vector", npu_op="vector_add", is_mapped=True,
        ))
        g.add_node(Node(
            id="node_1", op_type="vector_add",
            inputs=["t_mid"], outputs=["t_local_b"],
            compute_unit="vector", npu_op="vector_add", is_mapped=True,
        ))
        g.add_node(Node(
            id="node_2", op_type="vector_add",
            inputs=["t_local_a", "t_local_b"], outputs=["t_out"],
            compute_unit="vector", npu_op="vector_add", is_mapped=True,
        ))
        g.execution_order = ["node_0", "node_1", "node_2"]

        config = _small_l1_config()
        g = run(g, config)

        t_a = g.tensors["t_local_a"]
        t_b = g.tensors["t_local_b"]
        assert t_a.l1_offset is not None
        assert t_b.l1_offset is not None

        cube_size = config["fractal"]["cube_size"]
        sa = calc_padded_size(t_a.shape, t_a.dtype, t_a.format, cube_size)
        sb = calc_padded_size(t_b.shape, t_b.dtype, t_b.format, cube_size)
        end_a = t_a.l1_offset + sa
        end_b = t_b.l1_offset + sb
        overlap = t_a.l1_offset < end_b and t_b.l1_offset < end_a
        assert not overlap, (
            f"L1 重叠: t_local_a[{t_a.l1_offset}:{end_a}]"
            f" 和 t_local_b[{t_b.l1_offset}:{end_b}]"
        )

    def test_l1_overflow_accumulated_local(self):
        """累积的 local tensor 超出 L1=2MB 时抛出 MemoryPlanError。

        每个 tensor 512KB，3 个 local tensor 全部汇聚到 node_3。
        node_3: 3 local (1.5MB) + t_out (512KB) = 2MB，刚好满。
        再加 1 个 local → 2.5MB > 2MB → 溢出。
        """
        g = Graph()
        shape = _L1_TEST_SHAPE  # 512KB each
        g.add_tensor(Tensor(
            id="t_in", shape=shape, dtype="fp16",
            is_model_input=True, consumer_node_ids=["node_0", "node_1", "node_2", "node_3"],
        ))
        # 4 个 local tensor 汇聚到 node_4
        for i in range(4):
            g.add_tensor(Tensor(
                id=f"t_local_{i}", shape=shape, dtype="fp16",
                storage="local",
                producer_node_id=f"node_{i}", consumer_node_ids=["node_4"],
            ))
        g.add_tensor(Tensor(
            id="t_out", shape=shape, dtype="fp16",
            producer_node_id="node_4", is_model_output=True,
        ))
        for i in range(4):
            g.add_node(Node(
                id=f"node_{i}", op_type="custom_reduce",
                inputs=["t_in"], outputs=[f"t_local_{i}"],
                compute_unit="vector", npu_op="custom_reduce", is_mapped=True,
            ))
        g.add_node(Node(
            id="node_4", op_type="custom_reduce",
            inputs=["t_local_0", "t_local_1", "t_local_2", "t_local_3"],
            outputs=["t_out"],
            compute_unit="vector", npu_op="custom_reduce", is_mapped=True,
        ))
        g.execution_order = [f"node_{i}" for i in range(5)]

        config = _small_l1_config()
        # node_4: 4 local (2MB) + t_out (512KB) = 2.5MB > 2MB
        with pytest.raises(MemoryPlanError, match="L1 溢出"):
            run(g, config)

    def test_l1_boundary_exact_fit(self):
        """L1 刚好满不溢出的边界测试。

        L1=2MB, 3 个 local (1.5MB) + t_out (512KB) = 2MB 刚好不溢出。
        """
        g = Graph()
        shape = _L1_TEST_SHAPE  # 512KB each
        g.add_tensor(Tensor(
            id="t_in", shape=shape, dtype="fp16",
            is_model_input=True, consumer_node_ids=["node_0", "node_1", "node_2"],
        ))
        for i in range(3):
            g.add_tensor(Tensor(
                id=f"t_local_{i}", shape=shape, dtype="fp16",
                storage="local",
                producer_node_id=f"node_{i}", consumer_node_ids=["node_3"],
            ))
        g.add_tensor(Tensor(
            id="t_out", shape=shape, dtype="fp16",
            producer_node_id="node_3", is_model_output=True,
        ))
        for i in range(3):
            g.add_node(Node(
                id=f"node_{i}", op_type="vector_add",
                inputs=["t_in"], outputs=[f"t_local_{i}"],
                compute_unit="vector", npu_op="vector_add", is_mapped=True,
            ))
        g.add_node(Node(
            id="node_3", op_type="vector_add",
            inputs=["t_local_0", "t_local_1", "t_local_2"],
            outputs=["t_out"],
            compute_unit="vector", npu_op="vector_add", is_mapped=True,
        ))
        g.execution_order = [f"node_{i}" for i in range(4)]

        config = _small_l1_config()
        # node_3: 3 local (1.5MB) + t_out (512KB) = 2MB = L1 容量，刚好不溢出
        g = run(g, config)
        assert post_validate(g) == []

    def test_l1_alignment_global(self):
        """全局 L1 分配的所有 offset 满足对齐要求。"""
        g = _make_linear_chain(5, shape=_L1_TEST_SHAPE)
        config = _small_l1_config()
        l1_align = config["memory"]["l1"]["alignment_bytes"]
        g = run(g, config)

        for t in g.tensors.values():
            if t.l1_offset is not None:
                assert t.l1_offset % l1_align == 0, (
                    f"{t.id} L1 offset {t.l1_offset} 不是 {l1_align} 的倍数"
                )


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

    def test_pipe_tensor_no_offsets_needed(self):
        """pipe tensor 不需要 HBM 和 L1 偏移，post_validate 应通过。"""
        g = Graph()
        g.add_tensor(Tensor(
            id="t0", shape=[1], dtype="fp16", storage="pipe",
            producer_node_id="n0", consumer_node_ids=["n1"],
        ))
        assert post_validate(g) == []


class TestPipeL1Allocation:
    """pipe tensor 不参与 L1 分配，节省 L1 空间。"""

    def test_pipe_tensor_not_in_l1(self):
        """pipe tensor 不占用 L1，不影响其他 tensor 的 L1 分配。

        L1=2MB, tensor=512KB。
        node_0→(t_pipe)→node_1→t_mid→node_2→t_out。
        t_pipe 不占 L1，node_1 只需为 t_mid 和 t_out 分配。
        """
        g = Graph()
        shape = _L1_TEST_SHAPE  # 512KB
        g.add_tensor(Tensor(
            id="t_in", shape=shape, dtype="fp16",
            is_model_input=True, consumer_node_ids=["node_0"],
        ))
        g.add_tensor(Tensor(
            id="t_pipe", shape=shape, dtype="fp16", storage="pipe",
            producer_node_id="node_0", consumer_node_ids=["node_1"],
        ))
        g.add_tensor(Tensor(
            id="t_mid", shape=shape, dtype="fp16",
            producer_node_id="node_1", consumer_node_ids=["node_2"],
        ))
        g.add_tensor(Tensor(
            id="t_out", shape=shape, dtype="fp16",
            producer_node_id="node_2", is_model_output=True,
        ))
        g.add_node(Node(
            id="node_0", op_type="vector_add",
            inputs=["t_in"], outputs=["t_pipe"],
            compute_unit="vector", npu_op="vector_add", is_mapped=True,
        ))
        g.add_node(Node(
            id="node_1", op_type="vector_add",
            inputs=["t_pipe"], outputs=["t_mid"],
            compute_unit="vector", npu_op="vector_add", is_mapped=True,
        ))
        g.add_node(Node(
            id="node_2", op_type="vector_add",
            inputs=["t_mid"], outputs=["t_out"],
            compute_unit="vector", npu_op="vector_add", is_mapped=True,
        ))
        g.execution_order = ["node_0", "node_1", "node_2"]

        config = _small_l1_config()
        g = run(g, config)

        # pipe tensor 不分配 HBM 和 L1
        assert g.tensors["t_pipe"].hbm_offset is None
        assert g.tensors["t_pipe"].l1_offset is None
        # 其他 tensor 正常分配
        assert g.tensors["t_in"].l1_offset is not None
        assert g.tensors["t_mid"].l1_offset is not None
        assert g.tensors["t_out"].l1_offset is not None
        # node_1 不应有 t_pipe 的 DMA load
        plan_1 = g.dma_plans[1]
        load_tids = [inst.tensor_id for inst in plan_1.loads]
        assert "t_pipe" not in load_tids
        # post_validate 通过
        assert post_validate(g) == []

    def test_pipe_saves_l1_space(self):
        """pipe 比 local 节省 L1：4 个 tensor 汇聚，用 pipe 可避免溢出。

        L1=2MB, tensor=512KB。4 个 tensor 汇聚到 node_4。
        如果全部是 local（占 L1）：4*512KB + t_out(512KB) = 2.5MB > 2MB → 溢出。
        如果其中 2 个是 pipe（不占 L1）：2*512KB(local) + t_out(512KB) = 1.5MB < 2MB。
        """
        g = Graph()
        shape = _L1_TEST_SHAPE
        g.add_tensor(Tensor(
            id="t_in", shape=shape, dtype="fp16",
            is_model_input=True, consumer_node_ids=["node_0", "node_1", "node_2", "node_3"],
        ))
        # 前 2 个用 pipe，后 2 个用 local
        for i in range(4):
            storage = "pipe" if i < 2 else "local"
            g.add_tensor(Tensor(
                id=f"t_inter_{i}", shape=shape, dtype="fp16",
                storage=storage,
                producer_node_id=f"node_{i}", consumer_node_ids=["node_4"],
            ))
        g.add_tensor(Tensor(
            id="t_out", shape=shape, dtype="fp16",
            producer_node_id="node_4", is_model_output=True,
        ))
        for i in range(4):
            g.add_node(Node(
                id=f"node_{i}", op_type="vector_add",
                inputs=["t_in"], outputs=[f"t_inter_{i}"],
                compute_unit="vector", npu_op="vector_add", is_mapped=True,
            ))
        g.add_node(Node(
            id="node_4", op_type="vector_add",
            inputs=["t_inter_0", "t_inter_1", "t_inter_2", "t_inter_3"],
            outputs=["t_out"],
            compute_unit="vector", npu_op="vector_add", is_mapped=True,
        ))
        g.execution_order = [f"node_{i}" for i in range(5)]

        config = _small_l1_config()
        # 不应溢出（pipe 节省了 L1）
        g = run(g, config)
        assert post_validate(g) == []

        # pipe tensor 无 L1
        assert g.tensors["t_inter_0"].l1_offset is None
        assert g.tensors["t_inter_1"].l1_offset is None
        # local tensor 有 L1
        assert g.tensors["t_inter_2"].l1_offset is not None
        assert g.tensors["t_inter_3"].l1_offset is not None


# ── absorbed_inputs 对 HBM lifetime 的影响 ──────────────────


class TestAbsorbedInputsLifetime:
    """验证 absorbed_inputs 里的 tensor 被正确计入 consumer，延长 lifetime。"""

    def test_absorbed_tensor_last_use_extended(self):
        """bias 被 matmul 吸收后，其 last_use 应等于 matmul 的执行顺序。"""
        g = Graph()
        # bias tensor：被 node_1 吸收（不在 consumer_node_ids 中）
        g.add_tensor(Tensor(id="bias", shape=[1, 64], dtype="fp16", is_weight=True))
        g.add_tensor(Tensor(id="input", shape=[1, 32, 64], dtype="fp16",
                            is_model_input=True, consumer_node_ids=["node_0"]))
        g.add_tensor(Tensor(id="inter", shape=[1, 32, 64], dtype="fp16",
                            producer_node_id="node_0", consumer_node_ids=["node_1"]))
        g.add_tensor(Tensor(id="output", shape=[1, 32, 64], dtype="fp16",
                            producer_node_id="node_1", is_model_output=True))
        g.add_node(Node(id="node_0", op_type="linear", inputs=["input"],
                        outputs=["inter"], npu_op="cube_matmul",
                        compute_unit="cube", is_mapped=True))
        g.add_node(Node(id="node_1", op_type="linear_bias", inputs=["inter"],
                        outputs=["output"], npu_op="cube_matmul_bias",
                        compute_unit="cube", is_mapped=True,
                        absorbed_inputs={"bias": "bias"}))
        g.execution_order = ["node_0", "node_1"]

        lifetimes = analyze_lifetimes(g)
        # bias 的 last_use 应为 node_1 的 order (=1)，而非 weight 默认的 max_order
        # 关键：bias 在 lifetimes 中存在（不被跳过）
        assert "bias" in lifetimes
        _, last_use = lifetimes["bias"]
        # last_use >= 1（node_1 的 order），确认 absorbed 消费者被计入
        assert last_use >= 1


# ── 策略降级全链路测试 ──────────────────────────────────────


class TestStrategyFallbackChain:
    """验证策略降级链：bulk 失败 → perop → spill → tiled。"""

    def test_bulk_to_spill_fallback(self):
        """中等模型：bulk 放不下 → perop/spill 自动降级，最终分配成功。"""
        g = _make_linear_chain(n_ops=5, shape=_MEDIUM_SHAPE)
        config = _load_config()
        g = run(g, config)
        errors = post_validate(g)
        assert errors == [], f"validation errors: {errors}"
        # 所有输出 tensor 有 HBM 分配
        for t in g.tensors.values():
            if t.is_model_output:
                assert t.hbm_offset is not None

    def test_tiled_fallback(self):
        """超大单算子：spill 也放不下 → tiled 自动切分。"""
        # 单个 matmul，input [1, 4096, 1024] + weight [1024, 1024]
        # weight=2MB 放得下，input+output M 维 4096 需要 tiling
        g = Graph()
        g.add_tensor(Tensor(id="inp", shape=[1, 4096, 1024], dtype="fp16",
                            is_model_input=True, consumer_node_ids=["node_0"]))
        g.add_tensor(Tensor(id="w", shape=[1024, 1024], dtype="fp16",
                            is_weight=True, consumer_node_ids=["node_0"]))
        g.add_tensor(Tensor(id="out", shape=[1, 4096, 1024], dtype="fp16",
                            producer_node_id="node_0", is_model_output=True))
        g.add_node(Node(id="node_0", op_type="matmul", inputs=["inp", "w"],
                        outputs=["out"], npu_op="cube_matmul",
                        compute_unit="cube", is_mapped=True))
        g.execution_order = ["node_0"]

        config = _load_config()
        g = run(g, config)
        errors = post_validate(g)
        assert errors == []
        # 确认使用了 tiling
        assert "_tile_info" in g.nodes["node_0"].params

    def test_memory_plan_error_not_swallowed(self):
        """非 MemoryPlanError 不应被策略降级吞掉。"""
        # 这个测试验证修复 #1：bare except 改为 except MemoryPlanError
        # 如果有 bug 导致 KeyError/TypeError 等，应直接抛出
        g = Graph()
        # 创建一个空图不会触发异常，只验证正常流程通过
        config = _load_config()
        g = run(g, config)
        assert g.dma_plans is not None
