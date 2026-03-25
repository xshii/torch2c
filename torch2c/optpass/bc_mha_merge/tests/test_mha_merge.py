"""mha_merge pass 单元测试。"""

from __future__ import annotations

import copy

from torch2c.common import Graph, Node, Tensor
from torch2c.optpass.bc_mha_merge.mha_merge import (
    ForwardChain,
    MhaGroup,
    _apply_split,
    _compute_costs,
    _detect_mha_groups,
    _match_forward_chain,
    post_validate,
    run,
)


# ---- helpers ----


def _make_mha_chain(
    graph: Graph,
    prefix: str,
    B: int, S: int, H: int, Dh: int,
    *,
    has_bias: bool = True,
) -> str:
    """构建一条完整的前向链（proj → view → transpose → reshape），返回 proj node id。"""
    D = H * Dh
    npu_op = "cube_matmul_bias" if has_bias else "cube_matmul"

    # Tensors
    act_tid = f"{prefix}_act"
    w_tid = f"{prefix}_weight"
    proj_out_tid = f"{prefix}_proj_out"
    view_out_tid = f"{prefix}_view_out"
    tr_out_tid = f"{prefix}_tr_out"
    rs_out_tid = f"{prefix}_rs_out"

    if act_tid not in graph.tensors:
        graph.add_tensor(Tensor(id=act_tid, shape=[B, S, D], dtype="fp16",
                                is_model_input=True))
    graph.add_tensor(Tensor(id=w_tid, shape=[D, D], dtype="fp16",
                            is_weight=True, name=f"{prefix}.weight"))
    graph.add_tensor(Tensor(id=proj_out_tid, shape=[B, S, D], dtype="fp16"))
    graph.add_tensor(Tensor(id=view_out_tid, shape=[B, S, H, Dh], dtype="fp16"))
    graph.add_tensor(Tensor(id=tr_out_tid, shape=[B, H, S, Dh], dtype="fp16"))
    graph.add_tensor(Tensor(id=rs_out_tid, shape=[B * H, S, Dh], dtype="fp16"))

    # Nodes
    proj_nid = f"{prefix}_proj"
    view_nid = f"{prefix}_view"
    tr_nid = f"{prefix}_tr"
    rs_nid = f"{prefix}_rs"

    inputs = [act_tid, w_tid]
    if has_bias:
        bias_tid = f"{prefix}_bias"
        graph.add_tensor(Tensor(id=bias_tid, shape=[D], dtype="fp16",
                                is_weight=True, name=f"{prefix}.bias"))
        inputs.append(bias_tid)

    graph.add_node(Node(id=proj_nid, op_type="aten.mm.default", inputs=inputs,
                        outputs=[proj_out_tid], npu_op=npu_op, is_mapped=True,
                        compute_unit="cube"))
    graph.add_node(Node(id=view_nid, op_type="aten.view.default",
                        inputs=[proj_out_tid], outputs=[view_out_tid],
                        npu_op="idma_reshape", is_mapped=True, compute_unit="idma"))
    graph.add_node(Node(id=tr_nid, op_type="aten.transpose.int",
                        inputs=[view_out_tid], outputs=[tr_out_tid],
                        npu_op="vector_transpose", is_mapped=True,
                        compute_unit="vector",
                        params={"dim0": 1, "dim1": 2}))
    graph.add_node(Node(id=rs_nid, op_type="aten.reshape.default",
                        inputs=[tr_out_tid], outputs=[rs_out_tid],
                        npu_op="idma_reshape", is_mapped=True, compute_unit="idma"))

    # Wire producer/consumer
    graph.tensors[proj_out_tid].producer_node_id = proj_nid
    graph.tensors[proj_out_tid].consumer_node_ids = [view_nid]
    graph.tensors[view_out_tid].producer_node_id = view_nid
    graph.tensors[view_out_tid].consumer_node_ids = [tr_nid]
    graph.tensors[tr_out_tid].producer_node_id = tr_nid
    graph.tensors[tr_out_tid].consumer_node_ids = [rs_nid]
    graph.tensors[rs_out_tid].producer_node_id = rs_nid

    return proj_nid


def _make_mha_graph(
    B: int = 1, S: int = 32, H: int = 4, Dh: int = 64,
    *,
    has_bias: bool = True,
) -> Graph:
    """构建一个完整的 MHA 图（Q/K/V 三条链）。"""
    g = Graph()
    for role in ("q", "k", "v"):
        _make_mha_chain(g, role, B, S, H, Dh, has_bias=has_bias)
    return g


def _default_config(**overrides) -> dict:
    cfg = {
        "prefer_merged_threshold": 0.9,
        "max_batch_for_split": 1,
        "hardware": {
            "last_dim_align": 16,
            "l1_size_bytes": 16 * 1024 * 1024,
            "dma_bytes_per_cycle": 256,
            "matmul_launch_cycles": 100,
        },
    }
    cfg.update(overrides)
    return cfg


# ---- T1: pattern detection ----


class TestPatternDetection:
    def test_detect_single_chain(self):
        g = Graph()
        proj_nid = _make_mha_chain(g, "q", 1, 32, 4, 64)
        chain = _match_forward_chain(g, proj_nid)
        assert chain is not None
        assert chain.B == 1
        assert chain.S == 32
        assert chain.H == 4
        assert chain.Dh == 64

    def test_detect_mha_groups(self):
        g = _make_mha_graph()
        groups = _detect_mha_groups(g)
        assert len(groups) == 1
        grp = groups[0]
        assert grp.q_chain.H == 4
        assert grp.q_chain.Dh == 64
        assert grp.D == 256


# ---- T2: broken chain ----


class TestBrokenChain:
    def test_fanout_breaks_chain(self):
        """链中某步有 fan-out 应导致检测失败。"""
        g = Graph()
        _make_mha_chain(g, "q", 1, 32, 4, 64)
        # 给 view 输出增加第二个消费者
        extra_nid = "extra_consumer"
        g.add_node(Node(id=extra_nid, op_type="nop",
                        inputs=["q_proj_out"], outputs=[]))
        g.tensors["q_proj_out"].consumer_node_ids.append(extra_nid)

        chain = _match_forward_chain(g, "q_proj")
        assert chain is None

    def test_wrong_transpose_dims(self):
        """transpose dims 不匹配应检测失败。"""
        g = Graph()
        _make_mha_chain(g, "q", 1, 32, 4, 64)
        g.nodes["q_tr"].params = {"dim0": 0, "dim1": 2}

        chain = _match_forward_chain(g, "q_proj")
        assert chain is None

    def test_no_groups_fewer_than_3(self):
        """不足 3 条链无法组成 group。"""
        g = Graph()
        _make_mha_chain(g, "q", 1, 32, 4, 64)
        _make_mha_chain(g, "k", 1, 32, 4, 64)
        groups = _detect_mha_groups(g)
        assert groups == []


# ---- T3: cost aligned Dh ----


class TestCostAlignedDh:
    def test_aligned_dh_merged_wins(self):
        """Dh=64, H=4 已对齐，merged wins（无 padding 差异 + 省 pipeline overhead）。"""
        g = _make_mha_graph(B=1, S=32, H=4, Dh=64)
        groups = _detect_mha_groups(g)
        assert len(groups) == 1
        cost_a, cost_b = _compute_costs(groups[0], _default_config())
        # merged 更便宜或相当
        assert cost_a * 0.9 < cost_b


# ---- T4: cost misaligned Dh ----


class TestCostMisalignedDh:
    def test_misaligned_dh_merged_wins(self):
        """Dh=33, H=4, 大 L1: merged padding 省 5x，无 tiling。"""
        g = _make_mha_graph(B=1, S=32, H=4, Dh=33)
        groups = _detect_mha_groups(g)
        cost_a, cost_b = _compute_costs(groups[0], _default_config())
        # merged padding: ceil(132/16)*16 - 132 = 144-132 = 12 元素
        # split padding: 4*(48-33) = 60 元素 → split 更贵
        assert cost_a * 0.9 < cost_b


# ---- T5: cost tiling trigger ----


class TestCostTilingTrigger:
    def test_large_scores_split_wins(self):
        """Dh=64, S=384, L1=1MB → scores > L1, split wins。"""
        config = _default_config()
        config["hardware"]["l1_size_bytes"] = 1 * 1024 * 1024  # 1 MB
        g = _make_mha_graph(B=1, S=384, H=4, Dh=64)
        groups = _detect_mha_groups(g)
        cost_a, cost_b = _compute_costs(groups[0], config)
        # scores = 4*384*384*2 = 1,179,648 > 1MB → tiling penalty
        assert cost_b < cost_a


# ---- T6: split transform B=1 ----


class TestSplitTransform:
    def test_split_creates_correct_nodes(self):
        """执行 split 后：4 旧节点删除，H 个 matmul + 1 concat 创建。"""
        g = _make_mha_graph(B=1, S=32, H=4, Dh=64)
        groups = _detect_mha_groups(g)
        old_node_count = len(g.nodes)

        _apply_split(g, groups[0])

        # 3 chains × 4 old = 12 deleted
        # 3 chains × (4 heads + 1 concat) = 15 created
        expected = old_node_count - 12 + 15
        assert len(g.nodes) == expected

        # 确认 concat 节点存在
        concat_nodes = [n for n in g.nodes.values() if n.npu_op == "idma_concat"]
        assert len(concat_nodes) == 3  # Q, K, V 各一个

        # 每个 concat 有 4 个输入
        for cn in concat_nodes:
            assert len(cn.inputs) == 4

        # 旧节点已删除
        for role in ("q", "k", "v"):
            assert f"{role}_proj" not in g.nodes
            assert f"{role}_view" not in g.nodes
            assert f"{role}_tr" not in g.nodes
            assert f"{role}_rs" not in g.nodes

    def test_output_tensor_preserved(self):
        """split 后原 reshape 输出 tensor id 保留。"""
        g = _make_mha_graph(B=1, S=32, H=4, Dh=64)
        groups = _detect_mha_groups(g)
        orig_out_tids = [
            g.nodes[groups[0].q_chain.reshape_node_id].outputs[0],
            g.nodes[groups[0].k_chain.reshape_node_id].outputs[0],
            g.nodes[groups[0].v_chain.reshape_node_id].outputs[0],
        ]

        _apply_split(g, groups[0])

        for tid in orig_out_tids:
            assert tid in g.tensors


# ---- T7: skip B>1 ----


class TestSkipBatchGt1:
    def test_b2_stays_merged(self):
        """B=2 时保持 merged，无变换。"""
        g = _make_mha_graph(B=2, S=32, H=4, Dh=64)
        node_count_before = len(g.nodes)
        run(g, _default_config())
        assert len(g.nodes) == node_count_before


# ---- T8: weight slice annotation ----


class TestWeightSliceAnnotation:
    def test_weight_slices_correct(self):
        """per-head matmul 有正确的 _weight_slices 标注。"""
        g = _make_mha_graph(B=1, S=32, H=4, Dh=64)
        groups = _detect_mha_groups(g)
        _apply_split(g, groups[0])

        # 检查 q 链的 head 0
        h0 = g.get_node("q_proj_q_h0")
        assert h0 is not None
        ws = h0.params.get("_weight_slices")
        assert ws is not None
        assert ws["source_name"] == "q.weight"
        assert ws["dim"] == 0
        assert ws["start"] == 0
        assert ws["end"] == 64

        # 检查 head 3
        h3 = g.get_node("q_proj_q_h3")
        assert h3 is not None
        ws3 = h3.params["_weight_slices"]
        assert ws3["start"] == 192
        assert ws3["end"] == 256


# ---- T9: post_validate clean ----


class TestPostValidate:
    def test_clean_after_split(self):
        """split 后 post_validate 返回空列表。"""
        g = _make_mha_graph(B=1, S=32, H=4, Dh=64)
        groups = _detect_mha_groups(g)
        _apply_split(g, groups[0])
        errors = post_validate(g)
        assert errors == []


# ---- T10: idempotent ----


class TestIdempotent:
    def test_run_twice_same_result(self):
        """run 两次结果相同（第二次无 MHA 链可检测）。"""
        # 使用 tiling 场景强制 split
        config = _default_config()
        config["hardware"]["l1_size_bytes"] = 1 * 1024 * 1024
        g = _make_mha_graph(B=1, S=384, H=4, Dh=64)

        run(g, config)
        snapshot1 = g.to_dict()

        run(g, config)
        snapshot2 = g.to_dict()

        assert snapshot1 == snapshot2


# ---- T11: run() full pass ----


class TestRunPass:
    def test_run_with_split_decision(self):
        """完整 run() + L1=1MB 场景触发 split。"""
        config = _default_config()
        config["hardware"]["l1_size_bytes"] = 1 * 1024 * 1024
        g = _make_mha_graph(B=1, S=384, H=4, Dh=64)

        result = run(g, config)
        assert result is g  # in-place

        # 应有 concat 节点
        concat_nodes = [n for n in g.nodes.values() if n.npu_op == "idma_concat"]
        assert len(concat_nodes) == 3

        # post_validate 通过
        assert post_validate(g) == []

    def test_run_no_transform_when_merged_wins(self):
        """merged 方案更优时不变换。"""
        g = _make_mha_graph(B=1, S=32, H=4, Dh=64)
        node_ids_before = set(g.nodes.keys())
        run(g, _default_config())
        # 节点集合不变
        assert set(g.nodes.keys()) == node_ids_before

    def test_run_no_mha_chains(self):
        """无 MHA 链时安全跳过。"""
        g = Graph()
        g.add_tensor(Tensor(id="t1", shape=[1, 32, 64], dtype="fp16"))
        g.add_node(Node(id="n1", op_type="relu", inputs=["t1"], outputs=["t1"],
                        npu_op="vector_relu"))
        run(g, _default_config())
        assert len(g.nodes) == 1
