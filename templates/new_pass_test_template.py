"""TODO_PASS_NAME 单元测试。

测试结构：
  - _make_graph: 构建测试用最小图
  - TestRun: run() 的功能测试
  - TestPostValidate: post_validate() 的校验测试

命名规则：
  test_basic_xxx     — 基本正确性
  test_no_op_xxx     — 不需要变换时什么也不做
  test_idempotent    — 幂等性（跑两次结果相同）
  test_edge_xxx      — 边界情况
"""

from torch2c.common import Graph, Node, Tensor

# TODO: 修改下面的 import 路径，指向你的 pass 模块
# 格式: torch2c.optpass.{prefix}_{pass_name}.{pass_name}
from torch2c.optpass.TODO_PREFIX_TODO_PASS_NAME.TODO_PASS_NAME import (
    run,
    post_validate,
    # TODO: 如果有需要单独测试的辅助函数，也在这里 import
    # 示例: _parse_config, _should_transform,
)


# ── 测试配置 ──────────────────────────────────────────────────────────
# TODO: 定义测试用的 config 字典（模拟 YAML 配置）

_TEST_CONFIG = {
    # TODO: 填写你的 pass 的配置参数
    # 示例:
    # "threshold": 16,
    # "mode": "strict",
}


# ── 辅助函数 ──────────────────────────────────────────────────────────


def _make_graph(*tensor_specs: tuple) -> Graph:
    """构建测试用最小图。

    TODO: 根据你的 pass 需要，设计 tensor_specs 的格式。

    常见模式 1 — 按 (id, shape, dtype, format) 构建：
        _make_graph(("t1", [1, 16, 16], "fp16", "nd"))

    常见模式 2 — 按 (id, shape, op_type) 构建：
        _make_graph(("t1", [1, 16, 16], "vector_add"))
    """
    g = Graph()

    # TODO: 根据测试需要构建 node 和 tensor
    # 下面是一个通用模板，按需修改

    node = Node(id="n1", op_type="test_op")

    for spec in tensor_specs:
        # TODO: 解构 spec 元组，创建 Tensor
        # 示例（按 id, shape, dtype, format 解构）：
        tid, shape, dtype, fmt = spec  # TODO: 根据实际 spec 格式修改
        t = Tensor(id=tid, shape=shape, dtype=dtype, format=fmt)
        g.add_tensor(t)
        node.inputs.append(tid)

    g.add_node(node)
    return g


# ── run() 测试 ────────────────────────────────────────────────────────


class TestRun:
    """run() 功能测试。"""

    def test_basic_transform(self):
        """基本变换正确性。

        TODO: 构建一个需要变换的图，验证变换后的结果。
        """
        g = _make_graph(
            # TODO: 填入需要变换的 tensor 规格
            ("t1", [1, 3, 17], "fp16", "nd"),
        )
        run(g, _TEST_CONFIG)

        # TODO: 断言变换结果
        # 示例:
        # t1 = g.tensors["t1"]
        # assert t1.some_field == expected_value

    def test_no_op_when_nothing_to_do(self):
        """不需要变换时，图保持不变。

        TODO: 构建一个已经满足条件的图，验证 run 不修改它。
        """
        g = _make_graph(
            # TODO: 填入不需要变换的 tensor 规格
            ("t1", [1, 16, 16], "fp16", "nd"),
        )
        run(g, _TEST_CONFIG)

        # TODO: 断言图未被修改
        # 示例:
        # assert g.tensors["t1"].shape == [1, 16, 16]

    def test_idempotent(self):
        """幂等性：跑两次 run 结果相同。

        所有 pass 都必须幂等。这个测试不需要大改，
        只需要调整 _make_graph 参数。
        """
        g = _make_graph(
            # TODO: 填入 tensor 规格
            ("t1", [1, 3, 17], "fp16", "nd"),
        )
        run(g, _TEST_CONFIG)

        # 拍快照
        snap1 = str(g)

        run(g, _TEST_CONFIG)
        snap2 = str(g)

        assert snap1 == snap2, "pass 不幂等！第二次 run 改变了结果"

    # TODO: 添加更多测试用例，覆盖：
    #   - 多种输入组合
    #   - 边界情况（空图、scalar tensor、1D tensor 等）
    #   - 不同配置参数
    #   - 多个 tensor/node 混合的情况

    # def test_edge_case_empty_graph(self):
    #     g = Graph()
    #     run(g, _TEST_CONFIG)
    #     assert len(g.tensors) == 0

    # def test_edge_case_scalar(self):
    #     g = _make_graph(("t1", [], "fp16", "nd"))
    #     run(g, _TEST_CONFIG)

    # def test_multiple_tensors(self):
    #     g = _make_graph(
    #         ("t1", [1, 3, 17], "fp16", "nd"),
    #         ("t2", [2, 5, 64], "fp16", "nz"),
    #     )
    #     run(g, _TEST_CONFIG)


# ── post_validate() 测试 ─────────────────────────────────────────────


class TestPostValidate:
    """post_validate() 校验测试。"""

    def test_clean_after_run(self):
        """run 之后 post_validate 应该通过（无错误）。

        TODO: 只需调整 _make_graph 参数。
        """
        g = _make_graph(
            # TODO: 填入 tensor 规格
            ("t1", [1, 3, 17], "fp16", "nd"),
        )
        run(g, _TEST_CONFIG)
        assert post_validate(g) == []

    def test_detects_violation(self):
        """手动构造违反不变量的图，验证能检测到。

        TODO: 构造一个「看起来像跑过 pass 但其实不满足约束」的图。
        """
        g = _make_graph(
            # TODO: 填入会导致 post_validate 报错的 tensor 规格
            ("t1", [1, 16, 17], "fp16", "nd"),
        )
        # 注意：不跑 run，直接校验
        errors = post_validate(g)
        # TODO: 断言检测到了错误
        # assert len(errors) > 0
        # assert any("关键字" in e for e in errors)
