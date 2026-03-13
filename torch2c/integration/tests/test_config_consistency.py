"""配置一致性测试 — 交叉校验各配置的算子集合匹配。

防范问题：
1. direct_mappings / decompositions 引用的 npu_op 不在 signatures 中
2. Python 硬编码的算子列表与 c_api_signatures 不一致（遗漏新算子 / 包含已删除算子）
"""

from __future__ import annotations

import re

import pytest

from torch2c.common import INTEGRATION_CONFIG_DIR, NPU_CPU_MOCK_DIR, load_config


@pytest.fixture(scope="module")
def configs() -> dict:
    return {
        "signatures": load_config(str(INTEGRATION_CONFIG_DIR / "c_api_signatures.yaml")),
        "mapping": load_config(str(INTEGRATION_CONFIG_DIR / "direct_mappings.yaml")),
        "decomposition": load_config(str(INTEGRATION_CONFIG_DIR / "decompositions.yaml")),
    }


def _signature_ops(configs: dict) -> set[str]:
    """从 c_api_signatures 提取所有算子名称（compute + dma + idma）。"""
    sigs = configs["signatures"]
    ops = set()
    for section in ("compute_ops", "dma_ops", "idma_ops"):
        ops.update(sigs.get(section, {}).keys())
    return ops


def _mapping_npu_ops(configs: dict) -> set[str]:
    """从 direct_mappings 提取所有目标 npu_op。"""
    return {v["npu_op"] for v in configs["mapping"].get("mappings", {}).values()}


def _decomposition_npu_ops(configs: dict) -> set[str]:
    """从 decompositions 提取所有裂解产生的 npu_op。"""
    ops = set()
    for rule in configs["decomposition"].get("decompositions", {}).values():
        for step in rule.get("steps", []):
            ops.add(step["npu_op"])
    return ops


# ---- 测试用例 ----


class TestMappingsVsSignatures:
    """direct_mappings / decompositions 引用的 npu_op 必须在 signatures 中。"""

    def test_direct_mapping_ops_in_signatures(self, configs):
        """direct_mappings 的目标 npu_op 必须在 signatures 中有定义。"""
        sig_ops = _signature_ops(configs)
        map_ops = _mapping_npu_ops(configs)
        missing = map_ops - sig_ops
        assert not missing, (
            f"direct_mappings 引用的 npu_op 在 c_api_signatures 中无定义: {sorted(missing)}\n"
            f"请在 c_api_signatures.yaml 中添加这些算子的参数签名。"
        )

    def test_decomposition_ops_in_signatures(self, configs):
        """decompositions 产生的 npu_op 必须在 signatures 中有定义。"""
        sig_ops = _signature_ops(configs)
        decomp_ops = _decomposition_npu_ops(configs)
        missing = decomp_ops - sig_ops
        assert not missing, (
            f"decompositions 引用的 npu_op 在 c_api_signatures 中无定义: {sorted(missing)}\n"
            f"请在 c_api_signatures.yaml 中添加这些算子的参数签名。"
        )


class TestSignatureConfigCompleteness:
    """c_api_signatures 条目结构完整性。"""

    def test_all_compute_ops_have_params(self, configs):
        """每个 compute_op 必须有 params 列表。"""
        ops = configs["signatures"].get("compute_ops", {})
        for op_name, entry in ops.items():
            assert "params" in entry, f"c_api_signatures 中 {op_name} 缺少 params"
            assert isinstance(entry["params"], list), (
                f"c_api_signatures 中 {op_name} 的 params 应为列表"
            )

    def test_param_entries_have_required_fields(self, configs):
        """每个 param 条目必须有 name、type、source。"""
        ops = configs["signatures"].get("compute_ops", {})
        for op_name, entry in ops.items():
            for i, param in enumerate(entry.get("params", [])):
                assert "name" in param, f"{op_name} param[{i}] 缺少 name"
                assert "type" in param, f"{op_name} param[{i}] 缺少 type"
                assert "source" in param, f"{op_name} param[{i}] 缺少 source"


# ---- C mock 一致性 ----

# npu_api.h 中的工具函数，不需要在 signatures 中定义
_C_MOCK_UTILITY_FUNCTIONS = {
    "npu_dtype_size",
    "npu_read_as_float",
    "npu_write_from_float",
    "npu_round_to_dtype",
}


def _c_mock_functions() -> set[str]:
    """从 npu_api.h 提取所有非 static/inline 导出函数名。"""
    header = NPU_CPU_MOCK_DIR / "include" / "npu_api.h"
    text = header.read_text()
    # 匹配行首 void/size_t/float func_name( 形式的函数声明
    pattern = r"^(?:void|size_t|float)\s+(\w+)\s*\("
    return set(re.findall(pattern, text, re.MULTILINE))


class TestSignaturesVsCMock:
    """c_api_signatures 与 npu_api.h C mock 的一致性。"""

    def test_signature_ops_in_c_mock(self, configs):
        """signatures 中的所有算子必须在 C mock header 中有声明。"""
        sig_ops = _signature_ops(configs)
        c_funcs = _c_mock_functions()
        missing = sig_ops - c_funcs
        assert not missing, (
            f"c_api_signatures 中的算子在 npu_api.h 中无声明: {sorted(missing)}\n"
            f"请在 npu_cpu_mock/include/npu_api.h 中添加这些函数。"
        )

    def test_c_mock_ops_in_signatures(self, configs):
        """C mock 中的算子函数应在 signatures 中有定义（工具函数除外）。"""
        sig_ops = _signature_ops(configs)
        c_funcs = _c_mock_functions() - _C_MOCK_UTILITY_FUNCTIONS
        missing = c_funcs - sig_ops
        if missing:
            import warnings
            warnings.warn(
                f"npu_api.h 中的函数在 c_api_signatures 中无定义: {sorted(missing)}",
                stacklevel=1,
            )


# ---- Python 硬编码算子列表 vs c_api_signatures ----
#
# 以下测试确保 Python 代码中散落的算子列表与 c_api_signatures 保持同步。
# 新增算子时如果遗漏了某处硬编码，对应测试会报错并给出修复指引。


def _sig_compute_ops(configs: dict) -> set[str]:
    """signatures 中的 compute_ops（cube + vector，不含 dma/idma）。"""
    return set(configs["signatures"].get("compute_ops", {}).keys())


def _sig_all_ops(configs: dict) -> set[str]:
    """signatures 中所有算子（compute + dma + idma）。"""
    return _signature_ops(configs)


class TestTilingCoverage:
    """memory_planner._tiling._TILEABLE_OPS 覆盖性校验。

    每个 signatures 中的算子必须出现在 _TILEABLE_OPS 中，
    否则该算子不会做 tiling，可能导致 L1 溢出。
    """

    # 已知不需要 tiling 的算子（显式排除，新算子不应随意加入此集合）
    _KNOWN_UNTILEABLE: set[str] = set()

    def test_all_sig_ops_tileable_or_excluded(self, configs):
        from torch2c.memory_planner._tiling import _TILEABLE_OPS

        sig_ops = _sig_all_ops(configs)
        covered = set(_TILEABLE_OPS.keys()) | self._KNOWN_UNTILEABLE
        missing = sig_ops - covered
        assert not missing, (
            f"以下算子未在 memory_planner/_tiling.py 的 _TILEABLE_OPS 中注册:\n"
            f"  {sorted(missing)}\n"
            f"请在 _TILEABLE_OPS 中添加这些算子及其切分维度偏移，"
            f"或将其加入 TestTilingCoverage._KNOWN_UNTILEABLE 显式排除。"
        )

    def test_tileable_ops_are_valid(self, configs):
        """_TILEABLE_OPS 中的算子必须在 signatures 中存在（防止拼写错误 / 旧算子残留）。"""
        from torch2c.memory_planner._tiling import _TILEABLE_OPS

        sig_ops = _sig_all_ops(configs)
        invalid = set(_TILEABLE_OPS.keys()) - sig_ops
        assert not invalid, (
            f"_TILEABLE_OPS 中包含 signatures 中不存在的算子:\n"
            f"  {sorted(invalid)}\n"
            f"请检查算子名是否拼写正确或是否已被删除。"
        )


class TestCodegenNamingCoverage:
    """codegen._naming._OP_SHORT 覆盖性校验。

    每个 signatures 中的算子应在 _OP_SHORT 中有简短 C 名称，
    否则生成的 C 代码变量名会使用冗长的原始 npu_op 名。
    """

    def test_all_sig_ops_have_short_name(self, configs):
        from torch2c.codegen._naming import _OP_SHORT

        sig_ops = _sig_all_ops(configs)
        missing = sig_ops - set(_OP_SHORT.keys())
        assert not missing, (
            f"以下算子未在 codegen/_naming.py 的 _OP_SHORT 中注册:\n"
            f"  {sorted(missing)}\n"
            f"请为这些算子添加简短的 C 变量名前缀。"
        )

    def test_short_names_are_valid(self, configs):
        """_OP_SHORT 中的算子必须在 signatures 中存在。"""
        from torch2c.codegen._naming import _OP_SHORT

        sig_ops = _sig_all_ops(configs)
        # 允许 scalar_ 前缀的变体（codegen 内部使用）
        _CODEGEN_INTERNAL = {k for k in _OP_SHORT if k.startswith("scalar_")}
        invalid = set(_OP_SHORT.keys()) - sig_ops - _CODEGEN_INTERNAL
        assert not invalid, (
            f"_OP_SHORT 中包含 signatures 中不存在的算子:\n"
            f"  {sorted(invalid)}\n"
            f"请检查算子名是否拼写正确或是否已被删除。"
        )


class TestCostModelCoverage:
    """viz.cost_model.COST_FORMULAS 覆盖性校验。

    每个 signatures 中的算子应在 COST_FORMULAS 中有估算公式，
    否则 pipeline schedule 可视化的 cycle 估算不准确。
    """

    def test_all_sig_ops_have_cost_formula(self, configs):
        from torch2c.viz.cost_model import COST_FORMULAS

        sig_ops = _sig_all_ops(configs)
        missing = sig_ops - set(COST_FORMULAS.keys())
        assert not missing, (
            f"以下算子未在 viz/cost_model.py 的 COST_FORMULAS 中注册:\n"
            f"  {sorted(missing)}\n"
            f"请为这些算子添加 cycle 估算公式（或复用 _vector_elementwise_cost / _dma_cost 等通用公式）。"
        )

    def test_cost_formulas_are_valid(self, configs):
        """COST_FORMULAS 中的算子必须在 signatures 中存在。"""
        from torch2c.viz.cost_model import COST_FORMULAS

        sig_ops = _sig_all_ops(configs)
        invalid = set(COST_FORMULAS.keys()) - sig_ops
        assert not invalid, (
            f"COST_FORMULAS 中包含 signatures 中不存在的算子:\n"
            f"  {sorted(invalid)}\n"
            f"请检查算子名是否拼写正确或是否已被删除。"
        )


class TestHardcodedOpSetsValidity:
    """各模块硬编码的算子子集有效性校验。

    这些集合不要求覆盖所有算子（语义子集），
    但每个条目必须是 signatures 中的合法算子名。
    """

    def test_absorption_matmul_ops_valid(self, configs):
        from torch2c.op_absorption.op_absorption import _MATMUL_OPS

        sig_ops = _sig_all_ops(configs)
        invalid = _MATMUL_OPS - sig_ops
        assert not invalid, (
            f"op_absorption._MATMUL_OPS 中包含无效算子: {sorted(invalid)}"
        )

    def test_decomposition_elementwise_ops_valid(self, configs):
        from torch2c.op_decomposition.op_decomposition import _ELEMENTWISE_OPS

        sig_ops = _sig_all_ops(configs)
        invalid = _ELEMENTWISE_OPS - sig_ops
        assert not invalid, (
            f"op_decomposition._ELEMENTWISE_OPS 中包含无效算子: {sorted(invalid)}"
        )

    def test_global_tiler_tileable_ops_valid(self, configs):
        from torch2c.global_tiler.global_tiler import _GLOBAL_TILEABLE_OPS

        sig_ops = _sig_all_ops(configs)
        invalid = _GLOBAL_TILEABLE_OPS - sig_ops
        assert not invalid, (
            f"global_tiler._GLOBAL_TILEABLE_OPS 中包含无效算子: {sorted(invalid)}"
        )

    def test_roofline_flops_multiplier_valid(self, configs):
        from torch2c.roofline.roofline_analyzer import _VECTOR_FLOPS_MULTIPLIER

        sig_ops = _sig_all_ops(configs)
        # 这里用前缀匹配（vector_softmax 匹配 vector_softmax_part1 等）
        for op_prefix in _VECTOR_FLOPS_MULTIPLIER:
            matches = [op for op in sig_ops if op.startswith(op_prefix)]
            assert matches, (
                f"roofline._VECTOR_FLOPS_MULTIPLIER 中的前缀 '{op_prefix}' "
                f"在 signatures 中无匹配算子"
            )
