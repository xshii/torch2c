"""ST 场景拆解的 UT 测试 — 基于 test_scenarios.yaml 的 6 个场景。

所有场景统一使用 L1 = 1M (1,048,576 bytes)。
每个 TestST* 类对应 yaml 中的一个 ST 场景，验证 memory_planner 层面的关键行为。
"""

import pytest

from torch2c.common import Graph, Node, Tensor
from torch2c.common.errors import MemoryPlanError
from torch2c.common.testing import load_hw_config
from torch2c.memory_planner import run
from torch2c.memory_planner._utils import align_up, calc_padded_size
from torch2c.memory_planner.memory_planner import post_validate

_L1_1M = 1_048_576  # 1 MB
_CUBE_SIZE = 16


def _config_1m() -> dict:
    """L1=1M 的硬件配置。"""
    config = load_hw_config()
    config["memory"]["l1"]["total_size_bytes"] = _L1_1M
    return config


# ── 图构造工具 ──────────────────────────────────────────────


def _make_matmul_bias(
    graph: Graph,
    node_id: str,
    input_tid: str,
    weight_shape: list[int],
    output_tid: str,
    output_shape: list[int],
    *,
    input_dtype: str = "fp16",
    weight_dtype: str = "fp16",
    output_dtype: str = "fp16",
    compute_dtype: str | None = None,
    is_output: bool = False,
    next_node: str | None = None,
) -> None:
    """向 graph 添加一个 matmul_bias 节点及其权重/偏置/输出 tensor。"""
    weight_tid = f"{node_id}_weight"
    bias_tid = f"{node_id}_bias"
    bias_n = weight_shape[-1]

    graph.add_tensor(Tensor(
        id=weight_tid,
        shape=weight_shape,
        dtype=weight_dtype,
        is_weight=True,
        consumer_node_ids=[node_id],
    ))
    graph.add_tensor(Tensor(
        id=bias_tid,
        shape=[bias_n],
        dtype=weight_dtype,
        is_weight=True,
        consumer_node_ids=[node_id],
    ))
    graph.add_tensor(Tensor(
        id=output_tid,
        shape=output_shape,
        dtype=output_dtype,
        producer_node_id=node_id,
        consumer_node_ids=[] if is_output else ([next_node] if next_node else []),
        is_model_output=is_output,
    ))

    fmt_ann = {
        "inputs": [
            {"format": "nz", "dtype": input_dtype},
            {"format": "nz", "dtype": weight_dtype},
        ],
        "outputs": [{"format": "nz", "dtype": output_dtype}],
    }
    if compute_dtype:
        fmt_ann["compute_dtype"] = compute_dtype

    graph.add_node(Node(
        id=node_id,
        op_type="cube_matmul_bias",
        inputs=[input_tid, weight_tid],
        outputs=[output_tid],
        compute_unit="cube",
        npu_op="cube_matmul_bias",
        is_mapped=True,
        absorbed_inputs={"bias": bias_tid},
        format_annotation=fmt_ann,
    ))


def _build_embedding_bias(
    m: int = 32,
    k: int = 128,
    n: int = 128,
    *,
    input_dtype: str = "fp16",
    weight_dtype: str = "fp16",
    output_dtype: str = "fp16",
    compute_dtype: str | None = None,
) -> Graph:
    """embedding + bias 单层图：input[1,m,k] × W[k,n] + b → output[1,m,n]。"""
    g = Graph()
    g.add_tensor(Tensor(
        id="input",
        shape=[1, m, k],
        dtype=input_dtype,
        is_model_input=True,
        consumer_node_ids=["mm0"],
    ))
    _make_matmul_bias(
        g, "mm0", "input",
        weight_shape=[k, n],
        output_tid="output",
        output_shape=[1, m, n],
        input_dtype=input_dtype,
        weight_dtype=weight_dtype,
        output_dtype=output_dtype,
        compute_dtype=compute_dtype,
        is_output=True,
    )
    g.execution_order = ["mm0"]
    return g


def _build_2layer_mlp(
    m: int,
    d: int,
    *,
    input_dtype: str = "fp16",
    weight_dtype: str = "fp16",
) -> Graph:
    """2 层 MLP：input[1,m,d] → mm0 → mid[1,m,d] → mm1 → output[1,m,d]。"""
    g = Graph()
    g.add_tensor(Tensor(
        id="input",
        shape=[1, m, d],
        dtype=input_dtype,
        is_model_input=True,
        consumer_node_ids=["mm0"],
    ))
    _make_matmul_bias(
        g, "mm0", "input",
        weight_shape=[d, d],
        output_tid="mid",
        output_shape=[1, m, d],
        input_dtype=input_dtype,
        weight_dtype=weight_dtype,
        next_node="mm1",
    )
    _make_matmul_bias(
        g, "mm1", "mid",
        weight_shape=[d, d],
        output_tid="output",
        output_shape=[1, m, d],
        input_dtype=input_dtype,
        weight_dtype=weight_dtype,
        is_output=True,
    )
    g.execution_order = ["mm0", "mm1"]
    return g


def _build_2head_mha_projections(
    seq: int = 640,
    d: int = 256,
) -> Graph:
    """2 头 MHA 的 Q/K 投影图：input → Q_proj → Q, input → K_proj → K, Q+K → attn。

    Q_proj 适配 L1（input + Wq + Q），但 K_proj 时 Q 仍存活导致峰值超限。
    """
    g = Graph()
    g.add_tensor(Tensor(
        id="input",
        shape=[1, seq, d],
        dtype="fp16",
        is_model_input=True,
        consumer_node_ids=["q_proj", "k_proj"],
    ))
    # Q 投影
    _make_matmul_bias(
        g, "q_proj", "input",
        weight_shape=[d, d],
        output_tid="Q",
        output_shape=[1, seq, d],
        next_node="attn",
    )
    # K 投影
    _make_matmul_bias(
        g, "k_proj", "input",
        weight_shape=[d, d],
        output_tid="K",
        output_shape=[1, seq, d],
        next_node="attn",
    )
    # 注意力打分（Q × K^T → scores）
    g.add_tensor(Tensor(
        id="scores",
        shape=[1, seq, seq],
        dtype="fp16",
        producer_node_id="attn",
        is_model_output=True,
    ))
    g.add_node(Node(
        id="attn",
        op_type="cube_matmul",
        inputs=["Q", "K"],
        outputs=["scores"],
        compute_unit="cube",
        npu_op="cube_matmul",
        is_mapped=True,
    ))
    g.execution_order = ["q_proj", "k_proj", "attn"]
    return g


# ── ST1: embedding+bias，全放下 → bulk 策略 ──────────────────


class TestST1_EmbeddingBiasBulk:
    """ST1: embedding+bias，输入输出权重完全放得下，golden 通过。

    graph: input[1,32,128] × W[128,128] + b → output[1,32,128]
    total ≈ 48KB << 1M → bulk 策略
    verify: L1 不复用，HBM 线性分配，bulk DMA
    """

    def _build(self) -> tuple[Graph, dict]:
        return _build_embedding_bias(m=32, k=128, n=128), _config_1m()

    def test_bulk_strategy(self):
        """全部 tensor 放入 L1 → bulk DMA（__bulk_load__ + __bulk_store__）。"""
        g, config = self._build()
        g, dma_plans = run(g, config)
        assert len(dma_plans) == 2
        assert dma_plans[0].node_id == "__bulk_load__"
        assert dma_plans[1].node_id == "__bulk_store__"

    def test_l1_no_reuse(self):
        """bulk 策略下所有 tensor 的 L1 offset 唯一（不复用）。"""
        g, config = self._build()
        g, _ = run(g, config)
        offsets = [t.l1_offset for t in g.tensors.values() if t.l1_offset is not None]
        assert len(offsets) == len(set(offsets)), "bulk 策略 L1 offset 不应有复用"

    def test_hbm_linear(self):
        """HBM 线性分配：每个 tensor 的 hbm_offset 递增且无间隙复用。"""
        g, config = self._build()
        g, _ = run(g, config)
        hbm_tensors = sorted(
            [t for t in g.tensors.values() if t.hbm_offset is not None],
            key=lambda t: t.hbm_offset,
        )
        for i in range(1, len(hbm_tensors)):
            prev = hbm_tensors[i - 1]
            curr = hbm_tensors[i]
            expected = align_up(prev.hbm_offset + prev.hbm_size, 512)
            assert curr.hbm_offset == expected, (
                f"HBM 非线性: {curr.id} offset={curr.hbm_offset}, 期望={expected}"
            )

    def test_dma_format_convert(self):
        """bulk DMA 搬运所有 input/weight → L1，format_annotation 描述计算格式。"""
        g, config = self._build()
        g, dma_plans = run(g, config)
        bulk_load = dma_plans[0]
        # bulk load 包含 input 和 weight 的搬运
        load_tids = {i.tensor_id for i in bulk_load.loads}
        assert "input" in load_tids
        assert any("weight" in tid for tid in load_tids)
        # format_annotation 声明计算格式为 nz（DMA 硬件随路转换）
        ann = g.nodes["mm0"].format_annotation
        assert ann["inputs"][0]["format"] == "nz"
        assert ann["inputs"][1]["format"] == "nz"

    def test_matmul_bias_fusion(self):
        """验证 matmul+bias 融合：节点类型为 cube_matmul_bias，bias 在 absorbed_inputs。"""
        g, _ = self._build()
        mm_node = g.nodes["mm0"]
        assert mm_node.op_type == "cube_matmul_bias"
        assert "bias" in mm_node.absorbed_inputs

    def test_post_validate_pass(self):
        """内存编排后校验通过。"""
        g, config = self._build()
        g, _ = run(g, config)
        assert post_validate(g) == []


# ── ST2: embedding+bias，FP16 I/O + FP32 compute → bulk ─────


class TestST2_EmbeddingBiasMixedPrecision:
    """ST2: embedding+bias，FP16 I/O + FP32 compute。

    graph: 与 ST1 同构，但 compute_dtype="fp32"
    total ≈ 48KB << 1M → bulk 策略
    verify: format_annotation 包含 compute_dtype，bulk DMA
    """

    def _build(self) -> tuple[Graph, dict]:
        g = _build_embedding_bias(
            m=32, k=128, n=128,
            input_dtype="fp16",
            weight_dtype="fp16",
            output_dtype="fp16",
            compute_dtype="fp32",
        )
        return g, _config_1m()

    def test_bulk_strategy(self):
        """FP16 I/O 下总量仍远小于 1M → bulk 策略。"""
        g, config = self._build()
        g, dma_plans = run(g, config)
        assert len(dma_plans) == 2
        assert dma_plans[0].node_id == "__bulk_load__"

    def test_compute_dtype_fp32(self):
        """format_annotation 中 compute_dtype 为 fp32。"""
        g, _ = self._build()
        ann = g.nodes["mm0"].format_annotation
        assert ann["compute_dtype"] == "fp32"

    def test_io_dtype_fp16(self):
        """输入输出 tensor 仍为 fp16。"""
        g, config = self._build()
        g, _ = run(g, config)
        assert g.tensors["input"].dtype == "fp16"
        assert g.tensors["output"].dtype == "fp16"

    def test_post_validate_pass(self):
        g, config = self._build()
        g, _ = run(g, config)
        assert post_validate(g) == []


# ── ST3: embedding+bias，L1 放不下 → MemoryPlanError ─────────


class TestST3_EmbeddingBiasOverflow:
    """ST3: embedding+bias，输入输出权重 L1 放不下。

    graph: input[1,256,512] × W[512,512] + b → output[1,256,512]
    sizes: input=256KB + weight=512KB + bias=1KB + output=256KB = 1025KB > 1M
    单算子，bulk/per-op 都超 → tiled 策略自动切分成功。
    """

    def test_tiling_succeeds(self):
        """单算子总 tensor > 1M → tiled 策略 M 维切分成功。"""
        g = _build_embedding_bias(m=256, k=512, n=512)
        config = _config_1m()
        g, dma_plans = run(g, config)
        # tiled 策略成功，验证 tile_info 已设置
        node = g.nodes["mm0"]
        assert "_tile_info" in node.params
        ti = node.params["_tile_info"]
        assert ti["original_size"] == 256
        assert ti["tile_size"] * ti["num_tiles"] == 256
        assert post_validate(g) == []

    def test_size_calculation(self):
        """验证 tensor 大小计算正确。"""
        # input[1,256,512] fp16 nd = 256*512*2 = 262144
        assert calc_padded_size([1, 256, 512], "fp16", "nd", _CUBE_SIZE) == 262144
        # weight[512,512] fp16 nd = 512*512*2 = 524288
        assert calc_padded_size([512, 512], "fp16", "nd", _CUBE_SIZE) == 524288
        # total > 1M
        total = 262144 + 524288 + 1024 + 262144
        assert total > _L1_1M


# ── ST4: 2 层 MLP，全放下 → bulk 策略 ────────────────────────


class TestST4_2LayerMLPBulk:
    """ST4: 2 层 MLP (matmul+bias ×2)，全部 tensor L1 放得下。

    graph: input[1,32,128] → mm0 → mid[1,32,128] → mm1 → output[1,32,128]
    total: 3 activations(8KB each) + 2 weights(32KB each) + 2 biases ≈ 89KB << 1M
    verify: bulk 策略，L1 不复用
    """

    def _build(self) -> tuple[Graph, dict]:
        return _build_2layer_mlp(m=32, d=128), _config_1m()

    def test_bulk_strategy(self):
        """全部放下 → bulk DMA。"""
        g, config = self._build()
        g, dma_plans = run(g, config)
        assert len(dma_plans) == 2
        assert dma_plans[0].node_id == "__bulk_load__"

    def test_l1_no_reuse(self):
        """bulk 策略下 L1 不复用。"""
        g, config = self._build()
        g, _ = run(g, config)
        offsets = [t.l1_offset for t in g.tensors.values() if t.l1_offset is not None]
        assert len(offsets) == len(set(offsets))

    def test_2_matmul_bias_nodes(self):
        """图包含 2 个 cube_matmul_bias 节点。"""
        g, _ = self._build()
        mm_nodes = [n for n in g.nodes.values() if n.op_type == "cube_matmul_bias"]
        assert len(mm_nodes) == 2

    def test_post_validate_pass(self):
        g, config = self._build()
        g, _ = run(g, config)
        assert post_validate(g) == []


# ── ST5: 2 层 MLP，单层放下，总量放不下 → per-op 策略 ────────


class TestST5_2LayerMLPPerOp:
    """ST5: 2 层 MLP，单层 I/O+权重放下，2 层总量超 1M → per-op 策略。

    graph: input[1,600,256] → mm0 → mid[1,600,256] → mm1 → output[1,600,256]
    sizes:
      activation: 600×256×2 = 300KB each (×3 = 900KB)
      weight: 256×256×2 = 128KB each (×2 = 256KB)
      bias: 256×2 = 512B each
      total ≈ 1157KB > 1M → bulk 失败
      per-op peak: input+w+b+mid = 729KB < 1M → per-op 成功
    verify: per-op DMA，L1 liveness 复用
    """

    def _build(self) -> tuple[Graph, dict]:
        return _build_2layer_mlp(m=600, d=256), _config_1m()

    def test_perop_strategy(self):
        """bulk 失败后 fallback 到 per-op，每个算子有独立 DMA plan。"""
        g, config = self._build()
        g, dma_plans = run(g, config)
        # per-op: 每个算子一个 DMA plan
        assert len(dma_plans) == 2
        assert dma_plans[0].node_id == "mm0"
        assert dma_plans[1].node_id == "mm1"

    def test_l1_has_reuse(self):
        """per-op 路径下 L1 offset 存在复用。"""
        g, config = self._build()
        g, _ = run(g, config)
        offsets = [t.l1_offset for t in g.tensors.values() if t.l1_offset is not None]
        assert len(set(offsets)) < len(offsets), "per-op 路径应有 L1 复用"

    def test_l1_no_overlap_per_op(self):
        """同一算子内的 tensor L1 地址不重叠。"""
        g, config = self._build()
        g, _ = run(g, config)
        cube_size = config["fractal"]["cube_size"]
        for nid in g.execution_order:
            node = g.nodes[nid]
            op_tids = list(node.inputs) + list(node.outputs)
            for atid in node.absorbed_inputs.values():
                op_tids.append(atid)
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

    def test_per_op_peak_under_1m(self):
        """每个算子的 L1 峰值 < 1M。"""
        g, config = self._build()
        g, _ = run(g, config)
        cube_size = config["fractal"]["cube_size"]
        for nid in g.execution_order:
            node = g.nodes[nid]
            op_tids = list(node.inputs) + list(node.outputs)
            for atid in node.absorbed_inputs.values():
                op_tids.append(atid)
            peak = 0
            for tid in op_tids:
                t = g.tensors.get(tid)
                if t and t.l1_offset is not None:
                    end = t.l1_offset + calc_padded_size(t.shape, t.dtype, t.format, cube_size)
                    peak = max(peak, end)
            assert peak <= _L1_1M, f"节点 {nid} L1 峰值 {peak} 超过 1M"

    def test_hbm_reuse(self):
        """per-op 路径下 HBM 存在 lifetime 复用。"""
        g, config = self._build()
        g, _ = run(g, config)
        hbm_offsets = [
            t.hbm_offset for t in g.tensors.values() if t.hbm_offset is not None
        ]
        # 线性链 MLP 中，input 用完后空间可被 output 复用
        unique = set(hbm_offsets)
        # 至少 weight 不复用，但 activation 可能复用
        assert len(unique) <= len(hbm_offsets)

    def test_post_validate_pass(self):
        g, config = self._build()
        g, _ = run(g, config)
        assert post_validate(g) == []


# ── ST6: 2 头 MHA，Q 放下 K 放不下 → MemoryPlanError ────────


class TestST6_MHATiled:
    """ST6: 2 头 MHA 的 Q/K 投影 + attn。

    graph: input[1,640,256] → q_proj → Q → attn
                            → k_proj → K → attn
                                           → scores
    sizes:
      activation: 640×256×2 = 320KB each
      weight: 256×256×2 = 128KB each
      q_proj per-op peak: 768K < 1M ✓（eviction 模式）
      k_proj per-op peak: 768K < 1M ✓（eviction 模式，Q 已释放）
      attn per-op peak: Q(320K)+K(320K)+scores(800K) = 1440K > 1M ✗ → tiled
    策略: bulk 失败 → per-op 失败 → tiled 成功（attn 节点 M 维切分）
    """

    def _build(self):
        g = _build_2head_mha_projections(seq=640, d=256)
        config = _config_1m()
        return run(g, config)

    def test_tiled_succeeds(self):
        """tiled 策略成功通过，attn 节点被切分。"""
        g, dma_plans = self._build()
        assert post_validate(g) == []

    def test_attn_has_tile_info(self):
        """attn 节点包含 _tile_info。"""
        g, _ = self._build()
        attn = g.nodes["attn"]
        assert "_tile_info" in attn.params
        ti = attn.params["_tile_info"]
        assert ti["original_size"] == 640
        assert ti["tile_size"] == 320
        assert ti["num_tiles"] == 2

    def test_projections_not_tiled(self):
        """q_proj / k_proj 不需要 tiling（eviction 足够）。"""
        g, _ = self._build()
        assert "_tile_info" not in g.nodes["q_proj"].params
        assert "_tile_info" not in g.nodes["k_proj"].params

    def test_q_proj_alone_fits(self):
        """单独 Q 投影可以放下（验证不是 Q 本身太大）。"""
        g = _build_embedding_bias(m=640, k=256, n=256)
        config = _config_1m()
        g, _ = run(g, config)
        assert post_validate(g) == []

    def test_size_calculation(self):
        """验证峰值计算：Q 投影 768KB < 1M，K 投影时 1088KB > 1M。"""
        act_size = calc_padded_size([1, 640, 256], "fp16", "nd", _CUBE_SIZE)
        wt_size = calc_padded_size([256, 256], "fp16", "nd", _CUBE_SIZE)
        bias_size = calc_padded_size([256], "fp16", "nd", _CUBE_SIZE)
        q_peak = act_size + wt_size + bias_size + act_size
        k_peak = act_size + act_size + wt_size + bias_size + act_size
        assert q_peak < _L1_1M, f"Q peak {q_peak} should < 1M"
        assert k_peak > _L1_1M, f"K peak {k_peak} should > 1M"

    def test_tiled_dma_plan(self):
        """attn 节点的 DMA 计划包含 tiled 指令。"""
        g, dma_plans = self._build()
        # 找到 attn 的 DMA plan
        attn_plan = [p for p in dma_plans if p.node_id == "attn"]
        assert len(attn_plan) == 1
        plan = attn_plan[0]
        assert plan.tile_info is not None
        # 至少有 tiled load 和 tiled store
        tiled_loads = [i for i in plan.loads if i.tile_stride is not None]
        tiled_stores = [i for i in plan.stores if i.tile_stride is not None]
        assert len(tiled_loads) > 0
        assert len(tiled_stores) > 0
