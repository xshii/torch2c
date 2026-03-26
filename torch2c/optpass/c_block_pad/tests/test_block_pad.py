"""block_pad 单元测试 — format×dtype 二维对齐表。"""

from torch2c.common import Graph, Node, Tensor
from torch2c.optpass.c_block_pad.block_pad import (
    AlignRule, pad_shape, parse_alignment_table, get_align_rule,
    run, post_validate,
)


# ── pad_shape 单元测试 ──

# ND fp16: dim[-2] 不对齐(1), dim[-1] 对齐 16
_ND_FP16 = AlignRule(dim_neg2=1, dim_neg1=16, single_dim=256)
# NZ fp16: 两维都对齐 16
_NZ_FP16 = AlignRule(dim_neg2=16, dim_neg1=16, single_dim=256)
# ZZ int8: dim[-2] 对齐 16, dim[-1] 对齐 32
_ZZ_INT8 = AlignRule(dim_neg2=16, dim_neg1=32, single_dim=256)
# NZ int8: dim[-2] 对齐 32, dim[-1] 对齐 16
_NZ_INT8 = AlignRule(dim_neg2=32, dim_neg1=16, single_dim=256)


class TestPadShape:
    """pad_shape 核心逻辑测试。"""

    def test_already_aligned_nd(self):
        assert pad_shape([1, 16, 16], _ND_FP16) == [1, 16, 16]

    def test_nd_only_last_dim(self):
        # ND: dim[-2] 不对齐, dim[-1] 对齐到 16
        assert pad_shape([1, 3, 17], _ND_FP16) == [1, 3, 32]

    def test_nd_dim_neg2_unchanged(self):
        # ND fp16: dim[-2] 对齐=1，不改变
        assert pad_shape([1, 5, 16], _ND_FP16) == [1, 5, 16]

    def test_nz_both_dims(self):
        # NZ fp16: 两维都对齐 16
        result = pad_shape([1, 5, 17], _NZ_FP16)
        assert result == [1, 16, 32]

    def test_nz_already_aligned(self):
        assert pad_shape([1, 16, 16], _NZ_FP16) == [1, 16, 16]

    def test_zz_int8_asymmetric(self):
        # ZZ int8: dim[-2]=16, dim[-1]=32
        result = pad_shape([1, 5, 17], _ZZ_INT8)
        assert result == [1, 16, 32]

    def test_nz_int8_asymmetric(self):
        # NZ int8: dim[-2]=32, dim[-1]=16
        result = pad_shape([1, 5, 17], _NZ_INT8)
        assert result == [1, 32, 32]

    def test_1d_tensor_aligns_to_single_dim(self):
        assert pad_shape([17], _ND_FP16) == [256]

    def test_1d_already_aligned(self):
        assert pad_shape([256], _ND_FP16) == [256]
        assert pad_shape([512], _ND_FP16) == [512]

    def test_scalar_noop(self):
        assert pad_shape([], _ND_FP16) == []

    def test_2d_tensor_nd(self):
        result = pad_shape([3, 17], _ND_FP16)
        assert result == [3, 32]

    def test_2d_tensor_nz(self):
        result = pad_shape([3, 17], _NZ_FP16)
        assert result == [16, 32]

    def test_large_last_dim(self):
        assert pad_shape([1, 1, 256], _ND_FP16) == [1, 1, 256]

    def test_4d_tensor_nz(self):
        result = pad_shape([2, 3, 5, 17], _NZ_FP16)
        assert result[0] == 2
        assert result[1] == 3
        assert result[-2] == 16
        assert result[-1] == 32


# ── parse_alignment_table 测试 ──


class TestParseAlignmentTable:

    def test_empty_config(self):
        table, fallback = parse_alignment_table({})
        assert table == {}
        assert fallback.dim_neg2 == 16
        assert fallback.dim_neg1 == 16
        assert fallback.single_dim == 256

    def test_with_alignment(self):
        cfg = {
            "alignment": {
                "nd": {"fp16": [1, 16], "int8": [1, 32]},
                "nz": {"fp16": [16, 16], "int8": [32, 16]},
            },
            "fallback": [8, 8],
            "single_dim": 128,
        }
        table, fallback = parse_alignment_table(cfg)
        assert table[("nd", "fp16")].dim_neg2 == 1
        assert table[("nd", "fp16")].dim_neg1 == 16
        assert table[("nd", "int8")].dim_neg1 == 32
        assert table[("nz", "fp16")].dim_neg2 == 16
        assert table[("nz", "int8")].dim_neg2 == 32
        assert table[("nz", "int8")].dim_neg1 == 16
        assert fallback.dim_neg2 == 8
        assert fallback.dim_neg1 == 8
        assert fallback.single_dim == 128

    def test_fallback_used_for_missing(self):
        cfg = {
            "alignment": {"nd": {"fp16": [1, 16]}},
            "fallback": [16, 16],
        }
        table, fallback = parse_alignment_table(cfg)
        rule = get_align_rule(table, fallback, "zz", "fp16")
        assert rule.dim_neg2 == 16
        assert rule.dim_neg1 == 16


# ── run() 集成测试 ──


_FULL_CONFIG = {
    "alignment": {
        "nd": {
            "fp16": [1, 16],
            "int8": [1, 32],
        },
        "nz": {
            "fp16": [16, 16],
            "int8": [32, 16],
        },
        "zz": {
            "fp16": [16, 16],
            "int8": [16, 32],
        },
    },
    "fallback": [16, 16],
    "single_dim": 256,
}


def _make_graph(*tensor_specs: tuple[str, list[int], str, str]) -> Graph:
    """spec: (id, shape, dtype, format)。"""
    g = Graph()
    node = Node(id="n1", op_type="test_op")
    for tid, shape, dtype, fmt in tensor_specs:
        t = Tensor(id=tid, shape=shape, dtype=dtype, format=fmt)
        g.add_tensor(t)
        node.inputs.append(tid)
    g.add_node(node)
    return g


class TestRun:

    def test_nd_fp16_only_last_dim(self):
        g = _make_graph(("t1", [1, 3, 17], "fp16", "nd"))
        run(g, _FULL_CONFIG)
        t1 = g.tensors["t1"]
        assert t1.shape == [1, 3, 32]
        assert t1.original_shape == [1, 3, 17]

    def test_nz_fp16_both_dims(self):
        g = _make_graph(("t1", [1, 3, 17], "fp16", "nz"))
        run(g, _FULL_CONFIG)
        t1 = g.tensors["t1"]
        assert t1.shape == [1, 16, 32]
        assert t1.original_shape == [1, 3, 17]

    def test_zz_int8_asymmetric(self):
        g = _make_graph(("t1", [1, 5, 17], "int8", "zz"))
        run(g, _FULL_CONFIG)
        t1 = g.tensors["t1"]
        # ZZ int8: dim[-2] 对齐 16, dim[-1] 对齐 32
        assert t1.shape[-2] == 16
        assert t1.shape[-1] == 32

    def test_nz_int8_asymmetric(self):
        g = _make_graph(("t1", [1, 5, 17], "int8", "nz"))
        run(g, _FULL_CONFIG)
        t1 = g.tensors["t1"]
        # NZ int8: dim[-2] 对齐 32, dim[-1] 对齐 16
        assert t1.shape[-2] == 32
        assert t1.shape[-1] == 32  # 17 → 32 (ceil to 16)

    def test_no_pad_needed(self):
        g = _make_graph(("t1", [1, 16, 16], "fp16", "nz"))
        run(g, _FULL_CONFIG)
        assert g.tensors["t1"].original_shape is None

    def test_1d_tensor_pads_to_256(self):
        g = _make_graph(("t1", [17], "fp16", "nd"))
        run(g, _FULL_CONFIG)
        assert g.tensors["t1"].shape == [256]
        assert g.tensors["t1"].original_shape == [17]

    def test_scalar_skipped(self):
        g = _make_graph(("t1", [], "fp16", "nd"))
        run(g, _FULL_CONFIG)
        assert g.tensors["t1"].shape == []

    def test_mixed_formats(self):
        """同一个 graph 中不同 format 的 tensor 用不同规则对齐。"""
        g = _make_graph(
            ("t_nd", [1, 5, 17], "fp16", "nd"),
            ("t_nz", [1, 5, 17], "fp16", "nz"),
            ("t_zz", [1, 5, 17], "fp16", "zz"),
        )
        run(g, _FULL_CONFIG)
        # ND: dim[-2] 不变, dim[-1] → 32
        assert g.tensors["t_nd"].shape == [1, 5, 32]
        # NZ: dim[-2] → 16, dim[-1] → 32
        assert g.tensors["t_nz"].shape == [1, 16, 32]
        # ZZ: dim[-2] → 16, dim[-1] → 32
        assert g.tensors["t_zz"].shape == [1, 16, 32]

    def test_fallback_for_unknown_format(self):
        """未在表中的 format 使用 fallback 规则。"""
        g = _make_graph(("t1", [1, 5, 17], "fp16", "custom_fmt"))
        run(g, _FULL_CONFIG)
        t1 = g.tensors["t1"]
        # fallback [16, 16]
        assert t1.shape[-2] == 16
        assert t1.shape[-1] == 32

    def test_empty_config_uses_fallback(self):
        g = _make_graph(("t1", [1, 3, 17], "fp16", "nd"))
        run(g, {})
        t1 = g.tensors["t1"]
        # fallback [16, 16]: 两维都对齐
        assert t1.shape[-1] % 16 == 0
        assert t1.shape[-2] % 16 == 0


# ── post_validate 测试 ──


class TestPostValidate:

    def test_valid_after_run(self):
        g = _make_graph(
            ("t1", [1, 3, 17], "fp16", "nd"),
            ("t2", [2, 5, 64], "fp16", "nz"),
        )
        run(g, _FULL_CONFIG)
        assert post_validate(g, _FULL_CONFIG) == []

    def test_valid_1d_after_run(self):
        g = _make_graph(("t1", [17], "fp16", "nd"))
        run(g, _FULL_CONFIG)
        assert post_validate(g, _FULL_CONFIG) == []

    def test_detects_bad_last_dim(self):
        g = _make_graph(("t1", [1, 16, 17], "fp16", "nz"))
        errors = post_validate(g, _FULL_CONFIG)
        assert any("17" in e for e in errors)

    def test_detects_bad_neg2_dim(self):
        g = _make_graph(("t1", [1, 5, 16], "fp16", "nz"))
        errors = post_validate(g, _FULL_CONFIG)
        # NZ fp16: dim[-2] 必须是 16 的倍数，5 不是
        assert any("dim[-2]" in e for e in errors)

    def test_nd_allows_unaligned_neg2(self):
        """ND 格式 dim[-2] 对齐=1，不报错。"""
        g = _make_graph(("t1", [1, 5, 16], "fp16", "nd"))
        errors = post_validate(g, _FULL_CONFIG)
        assert errors == []

    def test_detects_bad_1d(self):
        g = Graph()
        g.add_tensor(Tensor(id="t1", shape=[17], dtype="fp16", format="nd"))
        errors = post_validate(g, _FULL_CONFIG)
        assert any("1D" in e for e in errors)

    def test_no_config_uses_defaults(self):
        g = _make_graph(("t1", [1, 16, 17], "fp16", "nz"))
        errors = post_validate(g)
        assert any("17" in e for e in errors)
