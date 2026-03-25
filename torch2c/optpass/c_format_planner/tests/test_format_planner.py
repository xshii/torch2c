"""format_planner 单元测试。"""

from __future__ import annotations

import pytest

from torch2c.common import Graph, Node, Tensor, NpuSpec
from torch2c.optpass.c_format_planner import run as run_format_planner
from torch2c.optpass.c_format_planner.format_planner import post_validate
from torch2c.optpass.c_format_planner._capabilities import (
    FormatCapabilities, UnitCapability, parse_capabilities,
)


# ---- capabilities 解析测试 ----


class TestParseCapabilities:
    def test_positional_src(self):
        raw = {"cube": {"src0": "nd", "src1": "nz", "dst": ["nd", "nz"]}}
        caps = parse_capabilities(raw)
        cap = caps.get("cube")
        assert cap is not None
        assert cap.get_src_formats(0) == frozenset({"nd"})
        assert cap.get_src_formats(1) == frozenset({"nz"})
        assert cap.dst == frozenset({"nd", "nz"})

    def test_unified_src(self):
        raw = {"vector": {"src": "nd", "dst": ["nd", "nz"]}}
        caps = parse_capabilities(raw)
        cap = caps.get("vector")
        assert cap is not None
        assert cap.get_src_formats(0) == frozenset({"nd"})
        assert cap.get_src_formats(1) == frozenset({"nd"})
        assert cap.get_src_formats(5) == frozenset({"nd"})

    def test_list_src(self):
        raw = {"idma": {"src": ["nd", "nz"], "dst": ["nd", "nz"]}}
        caps = parse_capabilities(raw)
        cap = caps.get("idma")
        assert cap is not None
        assert cap.get_src_formats(0) == frozenset({"nd", "nz"})

    def test_no_constraint_fallback(self):
        """无 src 约束时返回默认 {nd, nz}。"""
        cap = UnitCapability(src_ports={}, dst=frozenset({"nd"}))
        assert cap.get_src_formats(0) == frozenset({"nd", "nz"})

    def test_unconfigured_unit_defaults_no_constraint(self):
        """未配置的计算单元默认无约束。"""
        raw = {"cube": {"src0": "nd", "dst": "nz"}}
        caps = parse_capabilities(raw)
        # "scalar" 没有配置 → 默认无约束
        cap = caps.get("scalar")
        assert cap.get_src_formats(0) == frozenset({"nd", "nz"})
        assert cap.dst == frozenset({"nd", "nz"})

    def test_by_op_override(self):
        """by_op 算子级别规则覆盖计算单元级别。"""
        raw = {
            "vector": {"src": "nd", "dst": ["nd", "nz"]},
            "by_op": {
                "vector_ln": {"src": ["nd", "nz"], "dst": "nd"},
            },
        }
        caps = parse_capabilities(raw)
        # 计算单元级别
        unit_cap = caps.get("vector")
        assert unit_cap is not None
        assert unit_cap.get_src_formats(0) == frozenset({"nd"})
        # 算子级别覆盖
        op_cap = caps.get("vector", "vector_ln")
        assert op_cap is not None
        assert op_cap.get_src_formats(0) == frozenset({"nd", "nz"})
        assert op_cap.dst == frozenset({"nd"})
        # 无匹配算子 → 回退计算单元
        fallback = caps.get("vector", "vector_add")
        assert fallback is unit_cap


# ---- 测试用图构建辅助 ----


_CAPS_CONFIG = {
    "format_capabilities": {
        "cube": {"src0": "nd", "src1": "nz", "dst": ["nd", "nz"]},
        "vector": {"src": "nd", "dst": ["nd", "nz"]},
        "idma": {"src": ["nd", "nz"], "dst": ["nd", "nz"]},
    },
}

_SHAPE = [1, 32, 64]


def _make_cube_vector_chain() -> Graph:
    """cube_matmul(dst:[nd,nz]) → vector_add(src:nd)"""
    g = Graph()
    g.add_tensor(Tensor(
        id="t_in", shape=_SHAPE, dtype="fp16", format="nd",
        is_model_input=True, consumer_node_ids=["n_mm"],
    ))
    g.add_tensor(Tensor(
        id="t_w", shape=_SHAPE, dtype="fp16", format="nz",
        is_weight=True, consumer_node_ids=["n_mm"],
    ))
    g.add_tensor(Tensor(
        id="t_mm_out", shape=_SHAPE, dtype="fp16", format="nd",
        producer_node_id="n_mm", consumer_node_ids=["n_add"],
    ))
    g.add_tensor(Tensor(
        id="t_out", shape=_SHAPE, dtype="fp16", format="nd",
        producer_node_id="n_add", is_model_output=True,
    ))
    g.add_node(Node(
        id="n_mm", op_type="cube_matmul",
        inputs=["t_in", "t_w"], outputs=["t_mm_out"],
        compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
    ))
    g.add_node(Node(
        id="n_add", op_type="vector_add",
        inputs=["t_mm_out"], outputs=["t_out"],
        compute_unit="vector", npu_op="vector_add", is_mapped=True,
    ))
    g.execution_order = ["n_mm", "n_add"]
    return g


# ---- T1: cube→vector 链，零 reformat ----


class TestT1CubeVectorZeroReformat:
    def test_cube_output_nd(self):
        """cube 输出选 nd（匹配 vector src=nd），无需 reformat。"""
        g = _make_cube_vector_chain()
        g = run_format_planner(g, _CAPS_CONFIG)
        assert g.tensors["t_mm_out"].format == "nd"

    def test_annotation_filled(self):
        g = _make_cube_vector_chain()
        g = run_format_planner(g, _CAPS_CONFIG)
        ann = g.nodes["n_mm"].format_annotation
        assert ann is not None
        assert ann["outputs"][0]["format"] == "nd"


# ---- T2: cube→cube 链 ----


class TestT2CubeCubeChain:
    def test_cube_to_cube_output_nd(self):
        """cube→cube: 第一个 cube 输出选 nd（匹配 src0=nd）。"""
        g = Graph()
        g.add_tensor(Tensor(
            id="t_in", shape=_SHAPE, dtype="fp16", format="nd",
            is_model_input=True, consumer_node_ids=["n_mm1"],
        ))
        g.add_tensor(Tensor(
            id="t_w1", shape=_SHAPE, dtype="fp16", format="nz",
            is_weight=True, consumer_node_ids=["n_mm1"],
        ))
        g.add_tensor(Tensor(
            id="t_mid", shape=_SHAPE, dtype="fp16", format="nd",
            producer_node_id="n_mm1", consumer_node_ids=["n_mm2"],
        ))
        g.add_tensor(Tensor(
            id="t_w2", shape=_SHAPE, dtype="fp16", format="nz",
            is_weight=True, consumer_node_ids=["n_mm2"],
        ))
        g.add_tensor(Tensor(
            id="t_out", shape=_SHAPE, dtype="fp16", format="nd",
            producer_node_id="n_mm2", is_model_output=True,
        ))
        g.add_node(Node(
            id="n_mm1", op_type="cube_matmul",
            inputs=["t_in", "t_w1"], outputs=["t_mid"],
            compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
        ))
        g.add_node(Node(
            id="n_mm2", op_type="cube_matmul",
            inputs=["t_mid", "t_w2"], outputs=["t_out"],
            compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
        ))
        g.execution_order = ["n_mm1", "n_mm2"]

        g = run_format_planner(g, _CAPS_CONFIG)
        # 第一个 cube 输出 t_mid → 消费者 n_mm2 的 src0 要求 nd
        assert g.tensors["t_mid"].format == "nd"


# ---- T3: fan-out，一个 tensor 多消费者 ----


class TestT3FanOut:
    def test_fanout_all_nd(self):
        """cube→(vector + cube): 选 nd 满足两个消费者。"""
        g = Graph()
        g.add_tensor(Tensor(
            id="t_in", shape=_SHAPE, dtype="fp16", format="nd",
            is_model_input=True, consumer_node_ids=["n_mm"],
        ))
        g.add_tensor(Tensor(
            id="t_w", shape=_SHAPE, dtype="fp16", format="nz",
            is_weight=True, consumer_node_ids=["n_mm"],
        ))
        g.add_tensor(Tensor(
            id="t_mid", shape=_SHAPE, dtype="fp16", format="nd",
            producer_node_id="n_mm",
            consumer_node_ids=["n_add", "n_mm2"],
        ))
        g.add_tensor(Tensor(
            id="t_out1", shape=_SHAPE, dtype="fp16", format="nd",
            producer_node_id="n_add", is_model_output=True,
        ))
        g.add_tensor(Tensor(
            id="t_w2", shape=_SHAPE, dtype="fp16", format="nz",
            is_weight=True, consumer_node_ids=["n_mm2"],
        ))
        g.add_tensor(Tensor(
            id="t_out2", shape=_SHAPE, dtype="fp16", format="nd",
            producer_node_id="n_mm2", is_model_output=True,
        ))

        g.add_node(Node(
            id="n_mm", op_type="cube_matmul",
            inputs=["t_in", "t_w"], outputs=["t_mid"],
            compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
        ))
        g.add_node(Node(
            id="n_add", op_type="vector_add",
            inputs=["t_mid"], outputs=["t_out1"],
            compute_unit="vector", npu_op="vector_add", is_mapped=True,
        ))
        g.add_node(Node(
            id="n_mm2", op_type="cube_matmul",
            inputs=["t_mid", "t_w2"], outputs=["t_out2"],
            compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
        ))
        g.execution_order = ["n_mm", "n_add", "n_mm2"]

        g = run_format_planner(g, _CAPS_CONFIG)
        # vector 要求 nd，cube src0 要求 nd → 选 nd（满足 2/2）
        assert g.tensors["t_mid"].format == "nd"


# ---- T4: fan-out 冲突，必须 reformat ----


class TestT4FanOutConflict:
    def test_forced_nd_when_only_nd_available(self):
        """dst 只有 [nd] → 选 nd，即使某些消费者需要 nz。"""
        caps_config = {
            "format_capabilities": {
                "cube": {"src0": "nd", "src1": "nz", "dst": "nd"},
                "vector": {"src": "nz", "dst": "nd"},
            },
        }
        g = Graph()
        g.add_tensor(Tensor(
            id="t_in", shape=_SHAPE, dtype="fp16", format="nd",
            is_model_input=True, consumer_node_ids=["n_mm"],
        ))
        g.add_tensor(Tensor(
            id="t_w", shape=_SHAPE, dtype="fp16", format="nz",
            is_weight=True, consumer_node_ids=["n_mm"],
        ))
        g.add_tensor(Tensor(
            id="t_mid", shape=_SHAPE, dtype="fp16", format="nd",
            producer_node_id="n_mm",
            consumer_node_ids=["n_add"],
        ))
        g.add_tensor(Tensor(
            id="t_out", shape=_SHAPE, dtype="fp16", format="nd",
            producer_node_id="n_add", is_model_output=True,
        ))
        g.add_node(Node(
            id="n_mm", op_type="cube_matmul",
            inputs=["t_in", "t_w"], outputs=["t_mid"],
            compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
        ))
        g.add_node(Node(
            id="n_add", op_type="vector_add",
            inputs=["t_mid"], outputs=["t_out"],
            compute_unit="vector", npu_op="vector_add", is_mapped=True,
        ))
        g.execution_order = ["n_mm", "n_add"]

        g = run_format_planner(g, caps_config)
        # cube dst 只有 nd → 选 nd
        assert g.tensors["t_mid"].format == "nd"
        # vector src 要求 nz，但 tensor 是 nd → annotation 记录 nz
        ann = g.nodes["n_add"].format_annotation
        assert ann["inputs"][0]["format"] == "nz"


# ---- T5: 用户 override 告警 ----


class TestT5UserOverride:
    def test_npu_hint_overrides(self):
        """用户通过 npu() 指定 output format，应覆盖最优选择。"""
        g = _make_cube_vector_chain()
        # 设置 npu() hint: output=NpuSpec("fp16", "nz")
        g.nodes["n_mm"].params["_npu"] = {
            "output": NpuSpec(dtype="fp16", format="nz"),
        }

        g = run_format_planner(g, _CAPS_CONFIG)
        # 最优选 nd，但用户指定 nz → 使用 nz
        assert g.tensors["t_mm_out"].format == "nz"


# ---- T6: 向后兼容（无 format_capabilities）----


class TestT6BackwardCompat:
    def test_no_capabilities_noop(self):
        """无 format_capabilities 配置时跳过，不修改 graph。"""
        g = _make_cube_vector_chain()
        original_format = g.tensors["t_mm_out"].format

        g = run_format_planner(g, {})
        assert g.tensors["t_mm_out"].format == original_format

    def test_empty_capabilities_noop(self):
        g = _make_cube_vector_chain()
        original_format = g.tensors["t_mm_out"].format

        g = run_format_planner(g, {"format_capabilities": {}})
        assert g.tensors["t_mm_out"].format == original_format


# ---- T7: idma 万能转换 ----


class TestT7Idma:
    def test_idma_accepts_any(self):
        """idma 接受 nd 和 nz，输出选最优格式。"""
        g = Graph()
        g.add_tensor(Tensor(
            id="t_in", shape=_SHAPE, dtype="fp16", format="nd",
            is_model_input=True, consumer_node_ids=["n_vec"],
        ))
        g.add_tensor(Tensor(
            id="t_vec_out", shape=_SHAPE, dtype="fp16", format="nd",
            producer_node_id="n_vec", consumer_node_ids=["n_idma"],
        ))
        g.add_tensor(Tensor(
            id="t_idma_out", shape=_SHAPE, dtype="fp16", format="nd",
            producer_node_id="n_idma", consumer_node_ids=["n_mm"],
        ))
        g.add_tensor(Tensor(
            id="t_w", shape=_SHAPE, dtype="fp16", format="nz",
            is_weight=True, consumer_node_ids=["n_mm"],
        ))
        g.add_tensor(Tensor(
            id="t_out", shape=_SHAPE, dtype="fp16", format="nd",
            producer_node_id="n_mm", is_model_output=True,
        ))

        g.add_node(Node(
            id="n_vec", op_type="vector_add",
            inputs=["t_in"], outputs=["t_vec_out"],
            compute_unit="vector", npu_op="vector_add", is_mapped=True,
        ))
        g.add_node(Node(
            id="n_idma", op_type="dma_reformat",
            inputs=["t_vec_out"], outputs=["t_idma_out"],
            compute_unit="idma", npu_op="dma_reformat", is_mapped=True,
        ))
        g.add_node(Node(
            id="n_mm", op_type="cube_matmul",
            inputs=["t_idma_out", "t_w"], outputs=["t_out"],
            compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
        ))
        g.execution_order = ["n_vec", "n_idma", "n_mm"]

        g = run_format_planner(g, _CAPS_CONFIG)
        # idma 的消费者 n_mm 的 src0 要求 nd → idma 输出选 nd
        assert g.tensors["t_idma_out"].format == "nd"


# ---- T8: 位置敏感的 src（cube src0 vs src1）----


class TestT8PositionalSrc:
    def test_positional_matching(self):
        """cube src0=nd, src1=nz — annotation 应正确反映位置要求。"""
        g = Graph()
        g.add_tensor(Tensor(
            id="t_act", shape=_SHAPE, dtype="fp16", format="nd",
            is_model_input=True, consumer_node_ids=["n_mm"],
        ))
        g.add_tensor(Tensor(
            id="t_w", shape=_SHAPE, dtype="fp16", format="nz",
            is_weight=True, consumer_node_ids=["n_mm"],
        ))
        g.add_tensor(Tensor(
            id="t_out", shape=_SHAPE, dtype="fp16", format="nd",
            producer_node_id="n_mm", is_model_output=True,
        ))
        g.add_node(Node(
            id="n_mm", op_type="cube_matmul",
            inputs=["t_act", "t_w"], outputs=["t_out"],
            compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
        ))
        g.execution_order = ["n_mm"]

        g = run_format_planner(g, _CAPS_CONFIG)
        ann = g.nodes["n_mm"].format_annotation
        # src0 (t_act) → nd, src1 (t_w) → nz
        assert ann["inputs"][0]["format"] == "nd"
        assert ann["inputs"][1]["format"] == "nz"


# ---- T9: 算子级别覆盖计算单元级别规则 ----


class TestPerOpOverride:
    def test_op_level_overrides_unit_level(self):
        """by_op 中定义的算子规则优先于 compute_unit 级别。

        vector_ln 的 by_op 规则: src=[nd,nz], dst=nd
        vector 的 unit 规则:     src=nd, dst=[nd,nz]

        cube → vector_ln: cube 输出选 nd 或 nz 都被 vector_ln 接受
        """
        caps_config = {
            "format_capabilities": {
                "cube": {"src0": "nd", "src1": "nz", "dst": ["nd", "nz"]},
                "vector": {"src": "nd", "dst": ["nd", "nz"]},
                "by_op": {
                    "vector_ln": {"src": ["nd", "nz"], "dst": "nd"},
                },
            },
        }
        g = Graph()
        g.add_tensor(Tensor(
            id="t_in", shape=_SHAPE, dtype="fp16", format="nd",
            is_model_input=True, consumer_node_ids=["n_mm"],
        ))
        g.add_tensor(Tensor(
            id="t_w", shape=_SHAPE, dtype="fp16", format="nz",
            is_weight=True, consumer_node_ids=["n_mm"],
        ))
        g.add_tensor(Tensor(
            id="t_mid", shape=_SHAPE, dtype="fp16", format="nd",
            producer_node_id="n_mm", consumer_node_ids=["n_ln"],
        ))
        g.add_tensor(Tensor(
            id="t_out", shape=_SHAPE, dtype="fp16", format="nd",
            producer_node_id="n_ln", is_model_output=True,
        ))
        g.add_node(Node(
            id="n_mm", op_type="cube_matmul",
            inputs=["t_in", "t_w"], outputs=["t_mid"],
            compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
        ))
        g.add_node(Node(
            id="n_ln", op_type="vector_ln",
            inputs=["t_mid"], outputs=["t_out"],
            compute_unit="vector", npu_op="vector_ln", is_mapped=True,
        ))
        g.execution_order = ["n_mm", "n_ln"]

        g = run_format_planner(g, caps_config)
        # vector_ln 的 by_op 规则 src=[nd,nz]，两者都兼容
        # cube dst=[nd,nz]，lookahead 看 vector_ln 接受 nd 和 nz → 两者得分相同
        # 但 nd 和 nz 都可以 → 不需要 reformat
        t_mid = g.tensors["t_mid"]
        assert t_mid.format in ("nd", "nz")  # 两者都合法

        # vector_ln 输出按 by_op 规则 dst=nd
        assert g.tensors["t_out"].format == "nd"

    def test_op_override_restricts_dst(self):
        """算子级别限制 dst 为单一格式。"""
        caps_config = {
            "format_capabilities": {
                "vector": {"src": "nd", "dst": ["nd", "nz"]},
                "by_op": {
                    "vector_softmax": {"src": "nd", "dst": "nd"},
                },
            },
        }
        g = Graph()
        g.add_tensor(Tensor(
            id="t_in", shape=_SHAPE, dtype="fp16", format="nd",
            is_model_input=True, consumer_node_ids=["n_sm"],
        ))
        g.add_tensor(Tensor(
            id="t_out", shape=_SHAPE, dtype="fp16", format="nd",
            producer_node_id="n_sm", is_model_output=True,
        ))
        g.add_node(Node(
            id="n_sm", op_type="vector_softmax",
            inputs=["t_in"], outputs=["t_out"],
            compute_unit="vector", npu_op="vector_softmax", is_mapped=True,
        ))
        g.execution_order = ["n_sm"]

        g = run_format_planner(g, caps_config)
        # by_op 限制 dst=nd
        assert g.tensors["t_out"].format == "nd"

    def test_no_op_override_falls_back(self):
        """无 by_op 匹配时回退到计算单元规则。"""
        caps_config = {
            "format_capabilities": {
                "vector": {"src": "nd", "dst": ["nd", "nz"]},
                "by_op": {
                    "vector_softmax": {"src": "nd", "dst": "nd"},
                },
            },
        }
        g = Graph()
        g.add_tensor(Tensor(
            id="t_in", shape=_SHAPE, dtype="fp16", format="nd",
            is_model_input=True, consumer_node_ids=["n_add"],
        ))
        g.add_tensor(Tensor(
            id="t_out", shape=_SHAPE, dtype="fp16", format="nd",
            producer_node_id="n_add", is_model_output=True,
        ))
        g.add_node(Node(
            id="n_add", op_type="vector_add",
            inputs=["t_in"], outputs=["t_out"],
            compute_unit="vector", npu_op="vector_add", is_mapped=True,
        ))
        g.execution_order = ["n_add"]

        g = run_format_planner(g, caps_config)
        # vector_add 没有 by_op，回退到 vector 的 dst=[nd,nz]
        # 无消费者 → 默认选 nd
        assert g.tensors["t_out"].format == "nd"


# ---- post_validate ----


class TestPostValidate:
    def test_valid(self):
        g = _make_cube_vector_chain()
        g = run_format_planner(g, _CAPS_CONFIG)
        assert post_validate(g) == []

    def test_missing_format(self):
        g = Graph()
        g.add_tensor(Tensor(
            id="t0", shape=[1], dtype="fp16", format="",
            producer_node_id="n0",
        ))
        g.add_node(Node(id="n0", op_type="op", outputs=["t0"]))
        errors = post_validate(g)
        assert any("format" in e for e in errors)
