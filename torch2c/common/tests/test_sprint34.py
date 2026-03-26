"""Sprint 3 + Sprint 4 新增 API 的测试。

T7:  PassDesc kind/requires/provides + dependency validation
T8:  Pass interface validation
T9:  Config schemas (MhaMergeConfig, FormatConfig, BlockPadConfig)
T10: graph_transaction context manager
T12: STAGE_CONTRACTS correctness
"""

import pytest

from torch2c.common.graph_ir import Graph, Node, Tensor, graph_transaction
from torch2c.common.config_schemas import (
    BlockPadConfig,
    FormatConfig,
    MhaMergeConfig,
)


# ═══════════════════════════════════════════════════════════════
# T10: graph_transaction
# ═══════════════════════════════════════════════════════════════


class TestGraphTransaction:
    def test_rollback_on_exception(self):
        """异常时 graph 回滚到 transaction 前状态。"""
        g = Graph()
        g.add_tensor(Tensor(id="t0", shape=[1], dtype="fp16"))
        g.add_node(Node(id="n0", op_type="test", inputs=["t0"], outputs=["t0"]))

        with pytest.raises(ValueError):
            with graph_transaction(g):
                g.add_node(Node(id="n_bad", op_type="bad"))
                g.add_tensor(Tensor(id="t_bad", shape=[2], dtype="fp16"))
                assert "n_bad" in g.nodes
                raise ValueError("simulated failure")

        # 回滚：n_bad 和 t_bad 应消失
        assert "n_bad" not in g.nodes
        assert "t_bad" not in g.tensors
        assert "n0" in g.nodes
        assert "t0" in g.tensors

    def test_no_rollback_on_success(self):
        """无异常时改动保留。"""
        g = Graph()
        g.add_tensor(Tensor(id="t0", shape=[1], dtype="fp16"))

        with graph_transaction(g):
            g.add_tensor(Tensor(id="t1", shape=[2], dtype="fp16"))

        assert "t1" in g.tensors


# ═══════════════════════════════════════════════════════════════
# T9: Config Schemas
# ═══════════════════════════════════════════════════════════════


class TestMhaMergeConfig:
    def test_defaults(self):
        cfg = MhaMergeConfig()
        assert cfg.prefer_merged_threshold == 0.9
        assert cfg.max_batch_for_split == 1
        assert cfg.l1_size_bytes == 16 * 1024 * 1024

    def test_from_raw_nested(self):
        """pipeline._load_configs 产出的嵌套格式。"""
        raw = {
            "prefer_merged_threshold": 0.8,
            "max_batch_for_split": 2,
            "hardware": {
                "last_dim_align": 32,
                "l1_size_bytes": 8 * 1024 * 1024,
                "dma_bytes_per_cycle": 512,
            },
            "cost_model": {"key": "value"},
        }
        cfg = MhaMergeConfig.from_raw(raw)
        assert cfg.prefer_merged_threshold == 0.8
        assert cfg.last_dim_align == 32
        assert cfg.l1_size_bytes == 8 * 1024 * 1024
        assert cfg.cost_model == {"key": "value"}

    def test_from_raw_empty(self):
        cfg = MhaMergeConfig.from_raw({})
        assert cfg.prefer_merged_threshold == 0.9

    def test_roundtrip(self):
        cfg = MhaMergeConfig(prefer_merged_threshold=0.7, last_dim_align=32)
        d = cfg.to_dict()
        cfg2 = MhaMergeConfig.from_raw(d)
        assert cfg2.prefer_merged_threshold == 0.7
        assert cfg2.last_dim_align == 32


class TestFormatConfig:
    def test_from_raw(self):
        raw = {"target_dtype": "fp16", "target_format": "nz"}
        cfg = FormatConfig.from_raw(raw)
        assert cfg.target_dtype == "fp16"
        assert cfg.target_format == "nz"
        assert cfg.compute_dtype is None

    def test_to_dict_omits_none(self):
        cfg = FormatConfig(target_dtype="fp16")
        d = cfg.to_dict()
        assert "target_dtype" in d
        assert "compute_dtype" not in d


class TestBlockPadConfig:
    def test_get_align(self):
        cfg = BlockPadConfig(
            alignment={"nd": {"fp16": [1, 16]}, "nz": {"fp16": [16, 16]}},
            fallback=[8, 8],
        )
        assert cfg.get_align("nd", "fp16") == [1, 16]
        assert cfg.get_align("zz", "fp16") == [8, 8]  # fallback

    def test_from_raw(self):
        cfg = BlockPadConfig.from_raw({"single_dim": 128})
        assert cfg.single_dim == 128
        assert cfg.fallback == [16, 16]


# ═══════════════════════════════════════════════════════════════
# T12: STAGE_CONTRACTS
# ═══════════════════════════════════════════════════════════════


class TestStageContracts:
    def test_op_decomposition_contract(self):
        """op_decomposition 后所有节点应有 npu_op + is_mapped。"""
        g = Graph()
        g.add_tensor(Tensor(id="t0", shape=[1], dtype="fp16"))
        g.add_node(Node(id="n0", op_type="test", inputs=["t0"], outputs=["t0"],
                        npu_op="vector_relu", compute_unit="vector", is_mapped=True))
        assert g.validate_stage("op_decomposition") == []

    def test_op_decomposition_contract_fails(self):
        g = Graph()
        g.add_tensor(Tensor(id="t0", shape=[1], dtype="fp16"))
        g.add_node(Node(id="n0", op_type="test", inputs=["t0"], outputs=["t0"]))
        errors = g.validate_stage("op_decomposition")
        assert len(errors) >= 2  # missing npu_op + compute_unit + is_mapped

    def test_no_contract_returns_empty(self):
        g = Graph()
        assert g.validate_stage("nonexistent_pass") == []

    def test_old_op_mapping_contract_removed(self):
        """旧 op_mapping 契约（含错误的 is_mapped）已被移除。"""
        assert "op_mapping" not in Graph.STAGE_CONTRACTS


# ═══════════════════════════════════════════════════════════════
# T7: Pass dependencies
# ═══════════════════════════════════════════════════════════════


class TestPassDependencies:
    def test_current_pipeline_valid(self):
        """当前 pipeline 的依赖拓扑应通过校验。"""
        from torch2c.integration.pipeline import (
            _ANNOTATION_PASSES,
            _LATE_PASSES,
            _OPTIMIZATION_PASSES,
            _validate_pass_dependencies,
        )
        errors = _validate_pass_dependencies(
            [_OPTIMIZATION_PASSES, _ANNOTATION_PASSES, _LATE_PASSES],
        )
        assert errors == [], f"Dependency validation failed: {errors}"

    def test_missing_dependency_detected(self):
        from torch2c.integration.pipeline import _PassDesc, _validate_pass_dependencies
        bad_passes = [[
            _PassDesc("late_pass", "①", lambda g, c: g, None,
                      requires=frozenset({"nonexistent"})),
        ]]
        errors = _validate_pass_dependencies(bad_passes)
        assert len(errors) == 1
        assert "nonexistent" in errors[0]


# ═══════════════════════════════════════════════════════════════
# T8: Pass interface validation
# ═══════════════════════════════════════════════════════════════


class TestPassInterface:
    def test_all_passes_have_run(self):
        """所有 pass 模块应导出可调用的 run 函数。"""
        from torch2c.integration.pipeline import (
            _ANNOTATION_PASSES,
            _LATE_PASSES,
            _OPTIMIZATION_PASSES,
        )
        all_passes = _OPTIMIZATION_PASSES + _ANNOTATION_PASSES + _LATE_PASSES
        for p in all_passes:
            assert callable(p.run_fn), f"Pass {p.name} run_fn is not callable"

    def test_all_validators_callable(self):
        """所有声明了 validate_fn 的 pass，其校验函数应可调用。"""
        from torch2c.integration.pipeline import (
            _ANNOTATION_PASSES,
            _LATE_PASSES,
            _OPTIMIZATION_PASSES,
        )
        all_passes = _OPTIMIZATION_PASSES + _ANNOTATION_PASSES + _LATE_PASSES
        for p in all_passes:
            if p.validate_fn is not None:
                assert callable(p.validate_fn), f"Pass {p.name} validate_fn not callable"

    def test_kind_values_valid(self):
        """所有 pass 的 kind 应为 'analysis' 或 'transform'。"""
        from torch2c.integration.pipeline import (
            _ANNOTATION_PASSES,
            _LATE_PASSES,
            _OPTIMIZATION_PASSES,
        )
        all_passes = _OPTIMIZATION_PASSES + _ANNOTATION_PASSES + _LATE_PASSES
        for p in all_passes:
            assert p.kind in ("analysis", "transform"), \
                f"Pass {p.name} has invalid kind={p.kind!r}"
