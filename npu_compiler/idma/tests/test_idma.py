"""idma 单元测试。"""

import os

import pytest

from npu_compiler.common import Graph, Node, Tensor, load_config
from npu_compiler.memory_planner import run as run_memory_planner
from npu_compiler.idma import run as run_idma
from npu_compiler.idma.idma import post_validate

HARDWARE_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "memory_planner", "config", "hardware_config.yaml"
)

_SMALL_SHAPE = [1, 32, 64]
# 足够大使 4 tensor 超出 L1 (16MB)，强制走 per-op 路径
# 每 tensor ~4MB，4 tensor = 16MB = L1 容量 → 刚好不够（对齐后溢出）
_LARGE_SHAPE = [1, 1024, 2048]


def _make_reformat_chain(shape: list[int] | None = None) -> Graph:
    """构建 conv → reformat → add 的图，中间 tensor 是裂解产生的。

    conv → t_mid(local候选) → add → t_mid2 → gelu → t_out

    使用 5+ tensors（每个 ~4MB）确保超出 L1 (16MB)，走 per-op 路径。
    """
    if shape is None:
        shape = _LARGE_SHAPE
    g = Graph()
    g.add_tensor(
        Tensor(
            id="t_in",
            shape=shape,
            dtype="fp16",
            format="nd",
            is_model_input=True,
            consumer_node_ids=["node_conv"],
        )
    )
    g.add_tensor(
        Tensor(
            id="t_weight",
            shape=shape,
            dtype="fp16",
            format="nz",
            is_weight=True,
            consumer_node_ids=["node_conv"],
        )
    )
    # 中间 tensor：conv 的输出 → add 的输入（local 候选）
    g.add_tensor(
        Tensor(
            id="t_mid",
            shape=shape,
            dtype="fp16",
            format="nz",
            producer_node_id="node_conv",
            consumer_node_ids=["node_add"],
        )
    )
    g.add_tensor(
        Tensor(
            id="t_mid2",
            shape=shape,
            dtype="fp16",
            format="nd",
            producer_node_id="node_add",
            consumer_node_ids=["node_gelu"],
        )
    )
    g.add_tensor(
        Tensor(
            id="t_out",
            shape=shape,
            dtype="fp16",
            format="nd",
            producer_node_id="node_gelu",
            is_model_output=True,
        )
    )

    g.add_node(
        Node(
            id="node_conv",
            op_type="cube_matmul",
            inputs=["t_in", "t_weight"],
            outputs=["t_mid"],
            compute_unit="cube",
            npu_op="cube_matmul",
            is_mapped=True,
        )
    )
    g.add_node(
        Node(
            id="node_add",
            op_type="vector_add",
            inputs=["t_mid"],
            outputs=["t_mid2"],
            compute_unit="vector",
            npu_op="vector_add",
            is_mapped=True,
        )
    )
    g.add_node(
        Node(
            id="node_gelu",
            op_type="vector_gelu",
            inputs=["t_mid2"],
            outputs=["t_out"],
            compute_unit="vector",
            npu_op="vector_gelu",
            is_mapped=True,
        )
    )
    g.execution_order = ["node_conv", "node_add", "node_gelu"]
    return g


class TestStorageAssignment:
    def test_intermediate_tensor_marked_local(self):
        """单消费者的中间 tensor 应被标记为 local。"""
        g = _make_reformat_chain()
        g = run_idma(g, {})
        assert g.tensors["t_mid"].storage == "local"

    def test_model_input_stays_hbm(self):
        """模型输入不能标记为 local。"""
        g = _make_reformat_chain()
        g = run_idma(g, {})
        assert g.tensors["t_in"].storage == "hbm"

    def test_weight_stays_hbm(self):
        """权重不能标记为 local。"""
        g = _make_reformat_chain()
        g = run_idma(g, {})
        assert g.tensors["t_weight"].storage == "hbm"

    def test_model_output_stays_hbm(self):
        """模型输出不能标记为 local。"""
        g = _make_reformat_chain()
        g = run_idma(g, {})
        assert g.tensors["t_out"].storage == "hbm"

    def test_multi_consumer_stays_hbm(self):
        """多消费者的中间 tensor 不能标记为 local。"""
        g = _make_reformat_chain()
        # 添加第二个消费者
        g.tensors["t_mid"].consumer_node_ids.append("node_extra")
        g = run_idma(g, {})
        assert g.tensors["t_mid"].storage == "hbm"

    def test_disable_local_storage(self):
        """enable_local_storage=False 时不做任何标记。"""
        g = _make_reformat_chain()
        g = run_idma(g, {"enable_local_storage": False})
        assert g.tensors["t_mid"].storage == "hbm"


class TestPostValidate:
    def test_valid_local_tensor(self):
        g = Graph()
        g.add_tensor(
            Tensor(id="t0", shape=[1], dtype="fp16", storage="local",
                   producer_node_id="n0", consumer_node_ids=["n1"])
        )
        assert post_validate(g) == []

    def test_local_model_input_invalid(self):
        g = Graph()
        g.add_tensor(
            Tensor(id="t0", shape=[1], dtype="fp16", storage="local",
                   is_model_input=True, consumer_node_ids=["n0"])
        )
        errors = post_validate(g)
        assert any("模型输入" in e for e in errors)

    def test_local_weight_invalid(self):
        g = Graph()
        g.add_tensor(
            Tensor(id="t0", shape=[1], dtype="fp16", storage="local",
                   is_weight=True, consumer_node_ids=["n0"])
        )
        errors = post_validate(g)
        assert any("权重" in e for e in errors)


class TestMemoryPlannerWithLocalStorage:
    """验证 memory_planner 正确处理 storage=local 的 tensor。"""

    def _load_hw_config(self) -> dict:
        return load_config(HARDWARE_CONFIG_PATH)

    def test_local_tensor_no_hbm(self):
        """storage=local 的 tensor 不应分配 HBM。"""
        g = _make_reformat_chain()
        g = run_idma(g, {})
        config = self._load_hw_config()
        g, dma_plans = run_memory_planner(g, config)

        t_mid = g.tensors["t_mid"]
        assert t_mid.storage == "local"
        assert t_mid.hbm_offset is None
        assert t_mid.l1_offset is not None

    def test_local_tensor_no_dma_store(self):
        """producer 不应为 storage=local 的输出生成 DMA store。"""
        g = _make_reformat_chain()
        g = run_idma(g, {})
        config = self._load_hw_config()
        g, dma_plans = run_memory_planner(g, config)

        # node_conv 的 DMA plan
        conv_plan = [p for p in dma_plans if p.node_id == "node_conv"][0]
        # 不应有 t_mid 的 store
        store_tids = [s.tensor_id for s in conv_plan.stores]
        assert "t_mid" not in store_tids

    def test_local_tensor_no_dma_load(self):
        """consumer 不应为 storage=local 的输入生成 DMA load。"""
        g = _make_reformat_chain()
        g = run_idma(g, {})
        config = self._load_hw_config()
        g, dma_plans = run_memory_planner(g, config)

        # node_add 的 DMA plan
        add_plan = [p for p in dma_plans if p.node_id == "node_add"][0]
        # 不应有 t_mid 的 load
        load_tids = [ld.tensor_id for ld in add_plan.loads]
        assert "t_mid" not in load_tids

    def test_local_tensor_l1_offset_shared(self):
        """local tensor 的 L1 偏移在 producer 输出和 consumer 输入间一致。"""
        g = _make_reformat_chain()
        g = run_idma(g, {})
        config = self._load_hw_config()
        g, _ = run_memory_planner(g, config)

        # t_mid 只有一个 l1_offset，producer 和 consumer 共用
        t_mid = g.tensors["t_mid"]
        assert t_mid.l1_offset is not None

    def test_post_validate_local_passes(self):
        """memory_planner post_validate 不应对 local tensor 报 hbm 缺失错误。"""
        g = _make_reformat_chain()
        g = run_idma(g, {})
        config = self._load_hw_config()
        g, _ = run_memory_planner(g, config)

        from npu_compiler.memory_planner.memory_planner import post_validate as mp_post_validate
        errors = mp_post_validate(g)
        assert errors == [], f"不应有错误: {errors}"
