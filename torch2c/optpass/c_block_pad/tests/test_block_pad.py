"""block_pad 单元测试。"""

from torch2c.common import Graph, Node, Tensor
from torch2c.optpass.c_block_pad.block_pad import (
    PadRule, _pad_shape, _parse_pad_rules, run, post_validate,
)


# ── _pad_shape 单元测试 ──

_DEFAULT_RULE = PadRule(last_dim=16, last_two_product=256, single_dim=256)


class TestPadShape:
    """_pad_shape 核心逻辑测试。"""

    def test_already_aligned(self):
        assert _pad_shape([1, 16, 16], _DEFAULT_RULE) == [1, 16, 16]

    def test_last_dim_pad_to_16(self):
        assert _pad_shape([1, 16, 17], _DEFAULT_RULE) == [1, 16, 32]

    def test_product_pad_to_256(self):
        result = _pad_shape([1, 5, 16], _DEFAULT_RULE)
        assert result == [1, 16, 16]

    def test_both_dims_need_padding(self):
        result = _pad_shape([1, 3, 17], _DEFAULT_RULE)
        assert result[-1] == 32
        assert result[-2] == 8
        assert result[-2] * result[-1] == 256

    def test_1d_tensor_aligns_to_single_dim(self):
        assert _pad_shape([17], _DEFAULT_RULE) == [256]

    def test_1d_already_aligned(self):
        assert _pad_shape([256], _DEFAULT_RULE) == [256]
        assert _pad_shape([512], _DEFAULT_RULE) == [512]

    def test_scalar_noop(self):
        assert _pad_shape([], _DEFAULT_RULE) == []

    def test_2d_tensor(self):
        result = _pad_shape([3, 17], _DEFAULT_RULE)
        assert result[-1] == 32
        assert result[-2] * result[-1] % 256 == 0

    def test_large_last_dim(self):
        assert _pad_shape([1, 1, 256], _DEFAULT_RULE) == [1, 1, 256]

    def test_4d_tensor(self):
        result = _pad_shape([2, 3, 5, 17], _DEFAULT_RULE)
        assert result[0] == 2
        assert result[1] == 3
        assert result[-1] % 16 == 0
        assert result[-2] * result[-1] % 256 == 0

    def test_custom_rule(self):
        rule = PadRule(last_dim=32, last_two_product=1024, single_dim=1024)
        result = _pad_shape([1, 5, 17], rule)
        assert result[-1] % 32 == 0
        assert result[-2] * result[-1] % 1024 == 0

    def test_custom_single_dim(self):
        rule = PadRule(last_dim=16, last_two_product=256, single_dim=128)
        assert _pad_shape([17], rule) == [128]
        assert _pad_shape([129], rule) == [256]


# ── _parse_pad_rules 测试（config 即 block_pad section 内容） ──


class TestParseRules:

    def test_empty_config(self):
        default, by_dtype = _parse_pad_rules({})
        assert default.last_dim == 16
        assert default.last_two_product == 256
        assert default.single_dim == 256
        assert by_dtype == {}

    def test_default_only(self):
        cfg = {"default": {"last_dim": 32, "last_two_product": 512}}
        default, _ = _parse_pad_rules(cfg)
        assert default.last_dim == 32
        assert default.last_two_product == 512
        assert default.single_dim == 512

    def test_explicit_single_dim(self):
        cfg = {"default": {
            "last_dim": 16, "last_two_product": 256, "single_dim": 128,
        }}
        default, _ = _parse_pad_rules(cfg)
        assert default.single_dim == 128

    def test_by_dtype_overrides(self):
        cfg = {
            "default": {"last_dim": 16, "last_two_product": 256},
            "by_dtype": {
                "int8": {"last_dim": 32, "last_two_product": 1024},
            },
        }
        default, by_dtype = _parse_pad_rules(cfg)
        assert default.last_dim == 16
        assert by_dtype["int8"].last_dim == 32
        assert by_dtype["int8"].last_two_product == 1024
        assert by_dtype["int8"].single_dim == 1024

    def test_by_dtype_partial_override(self):
        cfg = {
            "default": {"last_dim": 16, "last_two_product": 256},
            "by_dtype": {"int8": {"last_dim": 32}},
        }
        _, by_dtype = _parse_pad_rules(cfg)
        assert by_dtype["int8"].last_dim == 32
        assert by_dtype["int8"].last_two_product == 256

    def test_by_dtype_explicit_single_dim(self):
        cfg = {
            "default": {"last_dim": 16, "last_two_product": 256},
            "by_dtype": {"int8": {"single_dim": 64}},
        }
        _, by_dtype = _parse_pad_rules(cfg)
        assert by_dtype["int8"].single_dim == 64
        assert by_dtype["int8"].last_two_product == 256


# ── run() 集成测试（config 即 block_pad section 内容） ──


def _make_graph(*tensor_specs: tuple[str, list[int], str]) -> Graph:
    """spec: (id, shape, dtype)。"""
    g = Graph()
    node = Node(id="n1", op_type="test_op")
    for tid, shape, dtype in tensor_specs:
        t = Tensor(id=tid, shape=shape, dtype=dtype)
        g.add_tensor(t)
        node.inputs.append(tid)
    g.add_node(node)
    return g


class TestRun:

    def test_pads_tensors_default(self):
        g = _make_graph(("t1", [1, 3, 17], "fp16"))
        run(g, {})
        t1 = g.tensors["t1"]
        assert t1.shape[-1] % 16 == 0
        assert t1.shape[-2] * t1.shape[-1] % 256 == 0
        assert t1.original_shape == [1, 3, 17]

    def test_no_pad_needed(self):
        g = _make_graph(("t1", [1, 16, 16], "fp16"))
        run(g, {})
        assert g.tensors["t1"].original_shape is None

    def test_1d_tensor_pads_to_256(self):
        g = _make_graph(("t1", [17], "fp16"))
        run(g, {})
        assert g.tensors["t1"].shape == [256]
        assert g.tensors["t1"].original_shape == [17]

    def test_scalar_skipped(self):
        g = _make_graph(("t1", [], "fp16"))
        run(g, {})
        assert g.tensors["t1"].shape == []

    def test_per_dtype_rule(self):
        cfg = {
            "default": {"last_dim": 16, "last_two_product": 256},
            "by_dtype": {
                "int8": {"last_dim": 32, "last_two_product": 1024},
            },
        }
        g = _make_graph(
            ("t_fp16", [1, 5, 17], "fp16"),
            ("t_int8", [1, 5, 17], "int8"),
        )
        run(g, cfg)
        assert g.tensors["t_fp16"].shape[-1] % 16 == 0
        assert g.tensors["t_fp16"].shape[-2] * g.tensors["t_fp16"].shape[-1] % 256 == 0
        assert g.tensors["t_int8"].shape[-1] % 32 == 0
        assert g.tensors["t_int8"].shape[-2] * g.tensors["t_int8"].shape[-1] % 1024 == 0

    def test_per_dtype_1d(self):
        cfg = {
            "default": {"last_dim": 16, "last_two_product": 256},
            "by_dtype": {"int8": {"single_dim": 64}},
        }
        g = _make_graph(("t_fp16", [17], "fp16"), ("t_int8", [17], "int8"))
        run(g, cfg)
        assert g.tensors["t_fp16"].shape == [256]
        assert g.tensors["t_int8"].shape == [64]


# ── post_validate 测试 ──


class TestPostValidate:

    def test_valid_after_run(self):
        g = _make_graph(("t1", [1, 3, 17], "fp16"), ("t2", [2, 5, 64], "fp16"))
        run(g, {})
        assert post_validate(g) == []

    def test_valid_1d_after_run(self):
        g = _make_graph(("t1", [17], "fp16"))
        run(g, {})
        assert post_validate(g) == []

    def test_detects_bad_last_dim(self):
        g = _make_graph(("t1", [1, 16, 17], "fp16"))
        errors = post_validate(g)
        assert any("17" in e for e in errors)

    def test_detects_bad_product(self):
        g = Graph()
        g.add_tensor(Tensor(id="t1", shape=[1, 5, 16], dtype="fp16"))
        errors = post_validate(g)
        assert any("乘积" in e for e in errors)

    def test_detects_bad_1d(self):
        g = Graph()
        g.add_tensor(Tensor(id="t1", shape=[17], dtype="fp16"))
        errors = post_validate(g)
        assert any("1D" in e for e in errors)
