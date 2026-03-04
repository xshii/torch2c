"""配置一致性测试 — 交叉校验各 YAML 配置文件的算子集合匹配。

防范问题：
1. c_api_signatures 中有算子但 type_format_config 缺少条目
2. direct_mappings / decompositions 引用的 npu_op 不在 signatures 中
3. type_format_config 有冗余条目（无对应 signature）
"""

from __future__ import annotations

import pathlib

import pytest

from npu_compiler.common import load_config

_CONFIG_DIR = pathlib.Path(__file__).resolve().parent.parent / "config"


@pytest.fixture(scope="module")
def configs() -> dict:
    return {
        "signatures": load_config(str(_CONFIG_DIR / "c_api_signatures.yaml")),
        "format": load_config(str(_CONFIG_DIR / "type_format_config.yaml")),
        "mapping": load_config(str(_CONFIG_DIR / "direct_mappings.yaml")),
        "decomposition": load_config(str(_CONFIG_DIR / "decompositions.yaml")),
    }


def _signature_ops(configs: dict) -> set[str]:
    """从 c_api_signatures 提取所有 compute_ops 名称。"""
    return set(configs["signatures"].get("compute_ops", {}).keys())


def _format_ops(configs: dict) -> set[str]:
    """从 type_format_config 提取所有算子名称。"""
    return set(configs["format"].get("op_format_requirements", {}).keys())


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


class TestSignaturesVsFormat:
    """c_api_signatures 与 type_format_config 的算子集合必须匹配。"""

    def test_all_signature_ops_have_format(self, configs):
        """每个 signature 中的 compute_op 都必须有 format 配置。"""
        sig_ops = _signature_ops(configs)
        fmt_ops = _format_ops(configs)
        missing = sig_ops - fmt_ops
        assert not missing, (
            f"算子在 c_api_signatures 中有定义但 type_format_config 缺失: {sorted(missing)}\n"
            f"请在 type_format_config.yaml 中为这些算子添加 format/dtype 条目。"
        )

    def test_all_format_ops_have_signature(self, configs):
        """每个 format 配置中的算子都必须有 signature。"""
        sig_ops = _signature_ops(configs)
        fmt_ops = _format_ops(configs)
        extra = fmt_ops - sig_ops
        assert not extra, (
            f"算子在 type_format_config 中有定义但 c_api_signatures 缺失: {sorted(extra)}\n"
            f"请在 c_api_signatures.yaml 中为这些算子添加参数签名。"
        )


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

    def test_direct_mapping_ops_have_format(self, configs):
        """direct_mappings 的目标 npu_op 必须有 format 配置。"""
        fmt_ops = _format_ops(configs)
        map_ops = _mapping_npu_ops(configs)
        missing = map_ops - fmt_ops
        assert not missing, (
            f"direct_mappings 引用的 npu_op 在 type_format_config 中缺失: {sorted(missing)}\n"
            f"请在 type_format_config.yaml 中为这些算子添加 format/dtype 条目。"
        )

    def test_decomposition_ops_have_format(self, configs):
        """decompositions 产生的 npu_op 必须有 format 配置。"""
        fmt_ops = _format_ops(configs)
        decomp_ops = _decomposition_npu_ops(configs)
        missing = decomp_ops - fmt_ops
        assert not missing, (
            f"decompositions 引用的 npu_op 在 type_format_config 中缺失: {sorted(missing)}\n"
            f"请在 type_format_config.yaml 中为这些算子添加 format/dtype 条目。"
        )


class TestFormatConfigCompleteness:
    """type_format_config 条目结构完整性。"""

    def test_all_entries_have_required_fields(self, configs):
        """每个 format 条目必须包含 inputs、outputs、compute_dtype。"""
        required_fields = {"inputs", "outputs", "compute_dtype"}
        reqs = configs["format"].get("op_format_requirements", {})
        for op_name, entry in reqs.items():
            missing = required_fields - set(entry.keys())
            assert not missing, (
                f"type_format_config 中 {op_name} 缺少必需字段: {sorted(missing)}"
            )

    def test_io_entries_have_format_and_dtype(self, configs):
        """每个 input/output 条目必须有 format 和 dtype。"""
        reqs = configs["format"].get("op_format_requirements", {})
        for op_name, entry in reqs.items():
            for i, inp in enumerate(entry.get("inputs", [])):
                assert "format" in inp, f"{op_name} input[{i}] 缺少 format"
                assert "dtype" in inp, f"{op_name} input[{i}] 缺少 dtype"
            for i, out in enumerate(entry.get("outputs", [])):
                assert "format" in out, f"{op_name} output[{i}] 缺少 format"
                assert "dtype" in out, f"{op_name} output[{i}] 缺少 dtype"


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
