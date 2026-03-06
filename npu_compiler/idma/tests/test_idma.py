"""idma 单元测试。"""

import os

import pytest

from npu_compiler.common import Graph, Node, Tensor, load_config
from npu_compiler.memory_planner import run as run_memory_planner
from npu_compiler.memory_planner.memory_planner import post_validate as mp_post_validate
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
    """构建 matmul(cube) → reformat(idma) → add(vector) → gelu(vector) 的图。

    t_in(nd,hbm) ─┐
    t_weight(nz,hbm) ─┤→ [cube_matmul] → t_mid(nz)
                                              │ cube→idma
                                  [dma_reformat(nz→nd)] → t_reformatted(nd)
                                                                │ idma→vector
                                                    [vector_add] → t_mid2(nd)
                                                                      │ vector→vector
                                                          [vector_gelu] → t_out(nd,hbm)

    allowed_pairs 边：
      - t_mid:          cube → idma    (matmul → reformat)
      - t_reformatted:  idma → vector  (reformat → add)
      - t_mid2:         vector → vector (add → gelu)

    6+ tensors × 4MB > 16MB L1 → 走 per-op 路径。
    """
    if shape is None:
        shape = _LARGE_SHAPE
    g = Graph()
    g.add_tensor(Tensor(
        id="t_in", shape=shape, dtype="fp16", format="nd",
        is_model_input=True, consumer_node_ids=["node_conv"],
    ))
    g.add_tensor(Tensor(
        id="t_weight", shape=shape, dtype="fp16", format="nz",
        is_weight=True, consumer_node_ids=["node_conv"],
    ))
    # matmul 输出 nz → reformat 输入
    g.add_tensor(Tensor(
        id="t_mid", shape=shape, dtype="fp16", format="nz",
        producer_node_id="node_conv", consumer_node_ids=["node_reformat"],
    ))
    # reformat 输出 nd → add 输入
    g.add_tensor(Tensor(
        id="t_reformatted", shape=shape, dtype="fp16", format="nd",
        producer_node_id="node_reformat", consumer_node_ids=["node_add"],
    ))
    g.add_tensor(Tensor(
        id="t_mid2", shape=shape, dtype="fp16", format="nd",
        producer_node_id="node_add", consumer_node_ids=["node_gelu"],
    ))
    g.add_tensor(Tensor(
        id="t_out", shape=shape, dtype="fp16", format="nd",
        producer_node_id="node_gelu", is_model_output=True,
    ))

    g.add_node(Node(
        id="node_conv", op_type="cube_matmul",
        inputs=["t_in", "t_weight"], outputs=["t_mid"],
        compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
    ))
    g.add_node(Node(
        id="node_reformat", op_type="dma_reformat",
        inputs=["t_mid"], outputs=["t_reformatted"],
        compute_unit="idma", npu_op="dma_reformat", is_mapped=True,
    ))
    g.add_node(Node(
        id="node_add", op_type="vector_add",
        inputs=["t_reformatted"], outputs=["t_mid2"],
        compute_unit="vector", npu_op="vector_add", is_mapped=True,
    ))
    g.add_node(Node(
        id="node_gelu", op_type="vector_gelu",
        inputs=["t_mid2"], outputs=["t_out"],
        compute_unit="vector", npu_op="vector_gelu", is_mapped=True,
    ))
    g.execution_order = ["node_conv", "node_reformat", "node_add", "node_gelu"]
    return g


class TestStorageAssignment:
    """基础 idma 标记测试，使用 cube→idma→vector→vector 图。"""

    def test_intermediate_tensors_marked_local(self):
        """单消费者的中间 tensor 应被标记为 local。"""
        g = _make_reformat_chain()
        g = run_idma(g, {})
        # cube→idma: t_mid 可 local
        assert g.tensors["t_mid"].storage == "local"
        # idma→vector: t_reformatted 可 local
        assert g.tensors["t_reformatted"].storage == "local"
        # vector→vector: t_mid2 可 local
        assert g.tensors["t_mid2"].storage == "local"

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
        g.tensors["t_mid"].consumer_node_ids.append("node_extra")
        g = run_idma(g, {})
        assert g.tensors["t_mid"].storage == "hbm"

    def test_disable_local_storage(self):
        """enable_local_storage=False 时不做任何标记。"""
        g = _make_reformat_chain()
        g = run_idma(g, {"enable_local_storage": False})
        assert g.tensors["t_mid"].storage == "hbm"
        assert g.tensors["t_reformatted"].storage == "hbm"
        assert g.tensors["t_mid2"].storage == "hbm"

    def test_allowed_pairs_restrict(self):
        """只允许 vector→vector 时，cube→idma 和 idma→vector 不标 local。"""
        g = _make_reformat_chain()
        g = run_idma(g, {"allowed_pairs": [["vector", "vector"]]})
        # cube→idma: 不允许
        assert g.tensors["t_mid"].storage == "hbm"
        # idma→vector: 不允许
        assert g.tensors["t_reformatted"].storage == "hbm"
        # vector→vector: 允许
        assert g.tensors["t_mid2"].storage == "local"

    def test_allowed_pairs_empty(self):
        """allowed_pairs 为空列表时，所有中间 tensor 都不标 local。"""
        g = _make_reformat_chain()
        g = run_idma(g, {"allowed_pairs": []})
        assert g.tensors["t_mid"].storage == "hbm"
        assert g.tensors["t_reformatted"].storage == "hbm"
        assert g.tensors["t_mid2"].storage == "hbm"

    def test_allowed_pairs_cube_cube(self):
        """显式添加 cube→cube 对时，两个 cube 节点间的 tensor 可标 local。"""
        g = Graph()
        shape = _LARGE_SHAPE
        g.add_tensor(Tensor(
            id="t_in", shape=shape, dtype="fp16",
            is_model_input=True, consumer_node_ids=["n0"],
        ))
        g.add_tensor(Tensor(
            id="t_inter", shape=shape, dtype="fp16",
            producer_node_id="n0", consumer_node_ids=["n1"],
        ))
        g.add_tensor(Tensor(
            id="t_out", shape=shape, dtype="fp16",
            producer_node_id="n1", is_model_output=True,
        ))
        g.add_node(Node(
            id="n0", op_type="cube_matmul", inputs=["t_in"], outputs=["t_inter"],
            compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
        ))
        g.add_node(Node(
            id="n1", op_type="cube_matmul", inputs=["t_inter"], outputs=["t_out"],
            compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
        ))
        g.execution_order = ["n0", "n1"]

        # 默认不包含 cube→cube
        g_default = run_idma(g, {})
        assert g_default.tensors["t_inter"].storage == "hbm"

        # 重置后显式允许 cube→cube
        g.tensors["t_inter"].storage = "hbm"
        g = run_idma(g, {"allowed_pairs": [["cube", "cube"]]})
        assert g.tensors["t_inter"].storage == "local"


class TestIdmaComputeUnit:
    """idma 作为计算单元参与 allowed_pairs 判定。

    场景：matmul(cube) → reformat(idma) → add(vector)
    """

    def _make_cube_idma_vector_chain(self) -> Graph:
        """cube → idma → vector 三节点图。"""
        shape = _LARGE_SHAPE
        g = Graph()
        g.add_tensor(Tensor(
            id="t_in", shape=shape, dtype="fp16", format="nd",
            is_model_input=True, consumer_node_ids=["n_matmul"],
        ))
        # matmul 输出 nz → reformat 的输入
        g.add_tensor(Tensor(
            id="t_mm_out", shape=shape, dtype="fp16", format="nz",
            producer_node_id="n_matmul", consumer_node_ids=["n_reformat"],
        ))
        # reformat 输出 nd → add 的输入
        g.add_tensor(Tensor(
            id="t_reformatted", shape=shape, dtype="fp16", format="nd",
            producer_node_id="n_reformat", consumer_node_ids=["n_add"],
        ))
        g.add_tensor(Tensor(
            id="t_out", shape=shape, dtype="fp16", format="nd",
            producer_node_id="n_add", is_model_output=True,
        ))

        g.add_node(Node(
            id="n_matmul", op_type="cube_matmul",
            inputs=["t_in"], outputs=["t_mm_out"],
            compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
        ))
        g.add_node(Node(
            id="n_reformat", op_type="dma_reformat",
            inputs=["t_mm_out"], outputs=["t_reformatted"],
            compute_unit="idma", npu_op="dma_reformat", is_mapped=True,
        ))
        g.add_node(Node(
            id="n_add", op_type="vector_add",
            inputs=["t_reformatted"], outputs=["t_out"],
            compute_unit="vector", npu_op="vector_add", is_mapped=True,
        ))
        g.execution_order = ["n_matmul", "n_reformat", "n_add"]
        return g

    def test_default_allows_cube_idma_and_idma_vector(self):
        """默认配置：cube→idma 和 idma→vector 都允许 local。"""
        g = self._make_cube_idma_vector_chain()
        g = run_idma(g, {})
        # cube→idma：t_mm_out 可 local
        assert g.tensors["t_mm_out"].storage == "local"
        # idma→vector：t_reformatted 可 local
        assert g.tensors["t_reformatted"].storage == "local"

    def test_block_idma_to_vector(self):
        """只允许 cube→idma，不允许 idma→vector 时，reformat 输出必须落 hbm。"""
        g = self._make_cube_idma_vector_chain()
        g = run_idma(g, {"allowed_pairs": [["cube", "idma"]]})
        # cube→idma：允许
        assert g.tensors["t_mm_out"].storage == "local"
        # idma→vector：不允许
        assert g.tensors["t_reformatted"].storage == "hbm"

    def test_block_cube_to_idma(self):
        """只允许 idma→vector，不允许 cube→idma 时，matmul 输出必须落 hbm。"""
        g = self._make_cube_idma_vector_chain()
        g = run_idma(g, {"allowed_pairs": [["idma", "vector"]]})
        # cube→idma：不允许
        assert g.tensors["t_mm_out"].storage == "hbm"
        # idma→vector：允许
        assert g.tensors["t_reformatted"].storage == "local"

    def test_idma_to_cube_blocked_by_default(self):
        """默认不允许 idma→cube。"""
        shape = _LARGE_SHAPE
        g = Graph()
        g.add_tensor(Tensor(
            id="t_in", shape=shape, dtype="fp16",
            is_model_input=True, consumer_node_ids=["n_reformat"],
        ))
        g.add_tensor(Tensor(
            id="t_reformatted", shape=shape, dtype="fp16",
            producer_node_id="n_reformat", consumer_node_ids=["n_matmul"],
        ))
        g.add_tensor(Tensor(
            id="t_out", shape=shape, dtype="fp16",
            producer_node_id="n_matmul", is_model_output=True,
        ))
        g.add_node(Node(
            id="n_reformat", op_type="dma_reformat",
            inputs=["t_in"], outputs=["t_reformatted"],
            compute_unit="idma", npu_op="dma_reformat", is_mapped=True,
        ))
        g.add_node(Node(
            id="n_matmul", op_type="cube_matmul",
            inputs=["t_reformatted"], outputs=["t_out"],
            compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
        ))
        g.execution_order = ["n_reformat", "n_matmul"]

        g = run_idma(g, {})
        # idma→cube 默认不允许
        assert g.tensors["t_reformatted"].storage == "hbm"


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
    """验证 memory_planner 正确处理 storage=local 的 tensor（含 idma 节点）。"""

    def _load_hw_config(self) -> dict:
        return load_config(HARDWARE_CONFIG_PATH)

    def test_local_tensor_no_hbm(self):
        """storage=local 的 tensor 不应分配 HBM。"""
        g = _make_reformat_chain()
        g = run_idma(g, {})
        config = self._load_hw_config()
        g, _ = run_memory_planner(g, config)

        for tid in ("t_mid", "t_reformatted", "t_mid2"):
            t = g.tensors[tid]
            assert t.storage == "local"
            assert t.hbm_offset is None, f"{tid} 不应有 hbm_offset"
            assert t.l1_offset is not None, f"{tid} 应有 l1_offset"

    def test_local_tensor_no_dma_store(self):
        """producer 不应为 storage=local 的输出生成 DMA store。"""
        g = _make_reformat_chain()
        g = run_idma(g, {})
        config = self._load_hw_config()
        g, dma_plans = run_memory_planner(g, config)

        # node_conv 不 store t_mid（cube→idma local）
        conv_plan = [p for p in dma_plans if p.node_id == "node_conv"][0]
        store_tids = [s.tensor_id for s in conv_plan.stores]
        assert "t_mid" not in store_tids

        # node_reformat 不 store t_reformatted（idma→vector local）
        reformat_plan = [p for p in dma_plans if p.node_id == "node_reformat"][0]
        store_tids = [s.tensor_id for s in reformat_plan.stores]
        assert "t_reformatted" not in store_tids

    def test_local_tensor_no_dma_load(self):
        """consumer 不应为 storage=local 的输入生成 DMA load。"""
        g = _make_reformat_chain()
        g = run_idma(g, {})
        config = self._load_hw_config()
        g, dma_plans = run_memory_planner(g, config)

        # node_reformat 不 load t_mid（local from matmul）
        reformat_plan = [p for p in dma_plans if p.node_id == "node_reformat"][0]
        load_tids = [ld.tensor_id for ld in reformat_plan.loads]
        assert "t_mid" not in load_tids

        # node_add 不 load t_reformatted（local from reformat）
        add_plan = [p for p in dma_plans if p.node_id == "node_add"][0]
        load_tids = [ld.tensor_id for ld in add_plan.loads]
        assert "t_reformatted" not in load_tids

    def test_local_tensor_l1_offset_shared(self):
        """local tensor 的 L1 偏移在 producer 输出和 consumer 输入间一致。"""
        g = _make_reformat_chain()
        g = run_idma(g, {})
        config = self._load_hw_config()
        g, _ = run_memory_planner(g, config)

        for tid in ("t_mid", "t_reformatted", "t_mid2"):
            t = g.tensors[tid]
            assert t.l1_offset is not None, f"{tid} 应有 l1_offset"

    def test_post_validate_local_passes(self):
        """memory_planner post_validate 不应对 local tensor 报 hbm 缺失错误。"""
        g = _make_reformat_chain()
        g = run_idma(g, {})
        config = self._load_hw_config()
        g, _ = run_memory_planner(g, config)

        errors = mp_post_validate(g)
        assert errors == [], f"不应有错误: {errors}"


# ── 用例：matmul+bias → safe_softmax ────────────────────


def _make_matmul_bias_softmax() -> Graph:
    """HBM(fp16,nd) → matmul(cube,nz) → reformat(idma,nz→nd) → bias_add(vector) → softmax(vector,fp32) → HBM

    数据流：
        t_input(fp16,nd,hbm)  ─┐
        t_weight(fp16,nz,hbm) ─┤→ [cube_matmul] → t_mm_out(fp16,nz)
                                                        │ cube→idma
                                            [dma_reformat(nz→nd)] → t_reformatted(fp16,nd)
                                                                          │ idma→vector
        t_bias(fp16,nz,hbm) ──────────────────→ [vector_add] → t_add_out(fp16,nd)
                                                                       │ vector→vector
                                                          [safe_softmax(fp32)] → t_out(fp16,nd,hbm)

    allowed_pairs 边：
      - t_mm_out:      cube → idma     (matmul → reformat)
      - t_reformatted: idma → vector   (reformat → bias_add)
      - t_add_out:     vector → vector (bias_add → softmax)

    7 tensors × 4MB > 16MB L1 → 走 per-op 路径。
    """
    shape = [1, 1024, 2048]
    g = Graph()

    # ── 张量 ──
    g.add_tensor(Tensor(
        id="t_input", shape=shape, dtype="fp16", format="nd",
        is_model_input=True, consumer_node_ids=["node_matmul"],
    ))
    g.add_tensor(Tensor(
        id="t_weight", shape=shape, dtype="fp16", format="nz",
        is_weight=True, consumer_node_ids=["node_matmul"],
    ))
    # matmul 输出 nz → reformat 输入
    g.add_tensor(Tensor(
        id="t_mm_out", shape=shape, dtype="fp16", format="nz",
        producer_node_id="node_matmul", consumer_node_ids=["node_reformat"],
    ))
    # reformat 输出 nd → bias_add 输入
    g.add_tensor(Tensor(
        id="t_reformatted", shape=shape, dtype="fp16", format="nd",
        producer_node_id="node_reformat", consumer_node_ids=["node_bias_add"],
    ))
    g.add_tensor(Tensor(
        id="t_bias", shape=shape, dtype="fp16", format="nz",
        is_weight=True, consumer_node_ids=["node_bias_add"],
    ))
    g.add_tensor(Tensor(
        id="t_add_out", shape=shape, dtype="fp16", format="nd",
        producer_node_id="node_bias_add", consumer_node_ids=["node_softmax"],
    ))
    g.add_tensor(Tensor(
        id="t_out", shape=shape, dtype="fp16", format="nd",
        producer_node_id="node_softmax", is_model_output=True,
    ))

    # ── 节点 ──
    g.add_node(Node(
        id="node_matmul", op_type="cube_matmul",
        inputs=["t_input", "t_weight"], outputs=["t_mm_out"],
        compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
        format_annotation={
            "inputs":  [{"format": "nz", "dtype": "fp16"},
                        {"format": "nz", "dtype": "fp16"}],
            "outputs": [{"format": "nz", "dtype": "fp16"}],
        },
    ))
    g.add_node(Node(
        id="node_reformat", op_type="dma_reformat",
        inputs=["t_mm_out"], outputs=["t_reformatted"],
        compute_unit="idma", npu_op="dma_reformat", is_mapped=True,
        format_annotation={
            "inputs":  [{"format": "nz", "dtype": "fp16"}],
            "outputs": [{"format": "nd", "dtype": "fp16"}],
        },
    ))
    g.add_node(Node(
        id="node_bias_add", op_type="vector_add",
        inputs=["t_reformatted", "t_bias"], outputs=["t_add_out"],
        compute_unit="vector", npu_op="vector_add", is_mapped=True,
        format_annotation={
            "inputs":  [{"format": "nd", "dtype": "fp16"},
                        {"format": "nd", "dtype": "fp16"}],
            "outputs": [{"format": "nd", "dtype": "fp16"}],
        },
    ))
    g.add_node(Node(
        id="node_softmax", op_type="safe_softmax",
        inputs=["t_add_out"], outputs=["t_out"],
        params={"compute_dtype": "fp32"},
        compute_unit="vector", npu_op="safe_softmax", is_mapped=True,
        format_annotation={
            "inputs":  [{"format": "nd", "dtype": "fp16"}],
            "outputs": [{"format": "nd", "dtype": "fp16"}],
        },
    ))

    g.execution_order = ["node_matmul", "node_reformat", "node_bias_add", "node_softmax"]
    return g


class TestMatmulBiasSoftmax:
    """用例：matmul(cube) → reformat(idma) → bias_add(vector) → softmax(vector)"""

    def _load_hw_config(self) -> dict:
        return load_config(HARDWARE_CONFIG_PATH)

    # ── idma 标记 ──

    def test_storage_assignment(self):
        """中间 tensor 根据 allowed_pairs 标 local；输入/权重/输出留 hbm。"""
        g = _make_matmul_bias_softmax()
        g = run_idma(g, {})

        # cube→idma: t_mm_out 可 local
        assert g.tensors["t_mm_out"].storage == "local"
        # idma→vector: t_reformatted 可 local
        assert g.tensors["t_reformatted"].storage == "local"
        # vector→vector: t_add_out 可 local
        assert g.tensors["t_add_out"].storage == "local"
        # 外部输入 / 权重 / 模型输出 不能 local
        assert g.tensors["t_input"].storage == "hbm"
        assert g.tensors["t_weight"].storage == "hbm"
        assert g.tensors["t_bias"].storage == "hbm"
        assert g.tensors["t_out"].storage == "hbm"

    def test_block_idma_pair_forces_hbm(self):
        """禁止 cube→idma 时，t_mm_out 回落 hbm，但 idma→vector 仍然 local。"""
        g = _make_matmul_bias_softmax()
        g = run_idma(g, {"allowed_pairs": [["idma", "vector"], ["vector", "vector"]]})

        assert g.tensors["t_mm_out"].storage == "hbm"         # cube→idma 被阻
        assert g.tensors["t_reformatted"].storage == "local"   # idma→vector 允许
        assert g.tensors["t_add_out"].storage == "local"       # vector→vector 允许

    # ── HBM 分配 ──

    def test_local_tensors_no_hbm(self):
        """local tensor 不分配 HBM。"""
        g = _make_matmul_bias_softmax()
        g = run_idma(g, {})
        g, _ = run_memory_planner(g, self._load_hw_config())

        for tid in ("t_mm_out", "t_reformatted", "t_add_out"):
            t = g.tensors[tid]
            assert t.hbm_offset is None, f"{tid} 不应有 hbm_offset"
            assert t.hbm_size is None, f"{tid} 不应有 hbm_size"
            assert t.l1_offset is not None, f"{tid} 应有 l1_offset"

    def test_hbm_tensors_allocated(self):
        """hbm tensor 必须分配 HBM。"""
        g = _make_matmul_bias_softmax()
        g = run_idma(g, {})
        g, _ = run_memory_planner(g, self._load_hw_config())

        for tid in ("t_input", "t_weight", "t_bias", "t_out"):
            t = g.tensors[tid]
            assert t.hbm_offset is not None, f"{tid} 应有 hbm_offset"

    # ── DMA 计划 ──

    def test_matmul_dma(self):
        """matmul: load input(nd→nz) + weight(nz)；不 store t_mm_out（local, cube→idma）。"""
        g = _make_matmul_bias_softmax()
        g = run_idma(g, {})
        g, plans = run_memory_planner(g, self._load_hw_config())

        matmul_plan = [p for p in plans if p.node_id == "node_matmul"][0]

        load_tids = {ld.tensor_id for ld in matmul_plan.loads}
        assert "t_input" in load_tids
        assert "t_weight" in load_tids

        # t_input 的 DMA load 做 nd→nz reformat
        input_load = [ld for ld in matmul_plan.loads if ld.tensor_id == "t_input"][0]
        assert input_load.src_format == "nd"
        assert input_load.dst_format == "nz"

        # t_mm_out 是 local → 不 store
        store_tids = {s.tensor_id for s in matmul_plan.stores}
        assert "t_mm_out" not in store_tids

    def test_reformat_dma(self):
        """reformat(idma): 不 load t_mm_out（local）；不 store t_reformatted（local）。"""
        g = _make_matmul_bias_softmax()
        g = run_idma(g, {})
        g, plans = run_memory_planner(g, self._load_hw_config())

        reformat_plan = [p for p in plans if p.node_id == "node_reformat"][0]

        load_tids = {ld.tensor_id for ld in reformat_plan.loads}
        assert "t_mm_out" not in load_tids

        store_tids = {s.tensor_id for s in reformat_plan.stores}
        assert "t_reformatted" not in store_tids

    def test_bias_add_dma(self):
        """bias_add: 不 load t_reformatted（local）；从 HBM load t_bias；不 store t_add_out（local）。"""
        g = _make_matmul_bias_softmax()
        g = run_idma(g, {})
        g, plans = run_memory_planner(g, self._load_hw_config())

        add_plan = [p for p in plans if p.node_id == "node_bias_add"][0]

        load_tids = {ld.tensor_id for ld in add_plan.loads}
        assert "t_reformatted" not in load_tids
        assert "t_bias" in load_tids

        store_tids = {s.tensor_id for s in add_plan.stores}
        assert "t_add_out" not in store_tids

    def test_softmax_dma(self):
        """softmax: 不 load t_add_out（local）；store t_out 到 HBM。"""
        g = _make_matmul_bias_softmax()
        g = run_idma(g, {})
        g, plans = run_memory_planner(g, self._load_hw_config())

        sm_plan = [p for p in plans if p.node_id == "node_softmax"][0]

        load_tids = {ld.tensor_id for ld in sm_plan.loads}
        assert "t_add_out" not in load_tids

        store_tids = {s.tensor_id for s in sm_plan.stores}
        assert "t_out" in store_tids

    # ── 校验 ──

    def test_post_validate_passes(self):
        """整个流程后 memory_planner 校验应通过。"""
        g = _make_matmul_bias_softmax()
        g = run_idma(g, {})
        g, _ = run_memory_planner(g, self._load_hw_config())

        errors = mp_post_validate(g)
        assert errors == [], f"不应有错误: {errors}"

    def test_softmax_compute_dtype_preserved(self):
        """safe_softmax 的 compute_dtype=fp32 参数应保留在 node.params 中。"""
        g = _make_matmul_bias_softmax()
        g = run_idma(g, {})
        node = g.nodes["node_softmax"]
        assert node.params["compute_dtype"] == "fp32"
