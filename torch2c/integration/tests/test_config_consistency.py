"""配置一致性测试 — 交叉校验各 YAML 配置文件的算子集合匹配。

防范问题：
1. direct_mappings / decompositions 引用的 npu_op 不在 signatures 中
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
