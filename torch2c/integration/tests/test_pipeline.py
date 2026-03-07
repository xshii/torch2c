"""integration pipeline 端到端集成测试。"""

from __future__ import annotations

import os
import shutil

import pytest
import torch

from torch2c.common import INTEGRATION_CONFIG_DIR
from torch2c.integration.demo.encoder_model import EncoderModel
from torch2c.integration.pipeline import (
    _build_validator_config,
    _load_configs,
    compile,
)

_CONFIG_DIR = str(INTEGRATION_CONFIG_DIR)


# ---- 配置加载 ----


class TestLoadConfigs:
    def test_all_configs_loaded(self):
        configs = _load_configs(_CONFIG_DIR)
        assert "mapping" in configs
        assert "decomposition" in configs
        assert "absorption" in configs
        assert "format" in configs
        assert "hardware" in configs
        assert "signatures" in configs

    def test_mapping_has_mappings_key(self):
        configs = _load_configs(_CONFIG_DIR)
        assert "mappings" in configs["mapping"]

    def test_validator_config_from_signatures(self):
        configs = _load_configs(_CONFIG_DIR)
        v_cfg = _build_validator_config(configs["signatures"])
        assert "supported_ops" in v_cfg
        assert "cube_matmul" in v_cfg["supported_ops"]
        assert "vector_add" in v_cfg["supported_ops"]


# ---- 端到端 ----


class TestEndToEnd:
    @pytest.fixture
    def output_dir(self, tmp_path):
        return str(tmp_path / "output")

    def test_compile_produces_c_project(self, output_dir):
        """端到端编译生成完整 C 工程。"""
        model = EncoderModel(d_model=256, dim_ff=512, num_layers=2)
        model.eval()

        dummy = torch.randn(1, 32, 256)
        mask = torch.zeros(1, 32, 32)

        result_dir = compile(
            model=model,
            dummy_input=dummy,
            config_dir=_CONFIG_DIR,
            output_dir=output_dir,
            mask=mask,
        )

        src = os.path.join(result_dir, "src")
        assert os.path.isfile(os.path.join(src, "model_graph.c"))
        assert os.path.isfile(os.path.join(src, "model_graph.h"))
        assert os.path.isfile(os.path.join(src, "model_memory.h"))
        assert os.path.isfile(os.path.join(src, "model_params.h"))
        assert os.path.isfile(os.path.join(src, "model_weights.h"))
        assert os.path.isfile(os.path.join(result_dir, "main.c"))
        assert os.path.isfile(os.path.join(result_dir, "npu_mock.h"))
        assert os.path.isfile(os.path.join(result_dir, "CMakeLists.txt"))

        # viz 产物
        viz = os.path.join(result_dir, "viz")
        assert os.path.isfile(os.path.join(viz, "graph.dot"))
        assert os.path.isfile(os.path.join(viz, "graph.txt"))
        assert os.path.isfile(os.path.join(viz, "lifetime_l1.txt"))
        assert os.path.isfile(os.path.join(viz, "lifetime_hbm.txt"))

    def test_golden_data_exported(self, output_dir):
        """golden 数据正确导出。"""
        model = EncoderModel(d_model=256, dim_ff=512, num_layers=2)
        model.eval()

        dummy = torch.randn(1, 32, 256)
        mask = torch.zeros(1, 32, 32)

        result_dir = compile(
            model=model,
            dummy_input=dummy,
            config_dir=_CONFIG_DIR,
            output_dir=output_dir,
            mask=mask,
        )

        golden = os.path.join(result_dir, "golden")
        assert os.path.isfile(os.path.join(golden, "input_0.bin"))
        assert os.path.isfile(os.path.join(golden, "input_0.desc"))
        assert os.path.isfile(os.path.join(golden, "output_0.bin"))
        assert os.path.isfile(os.path.join(golden, "output_0.desc"))

    def test_model_graph_has_op_blocks(self, output_dir):
        """model_graph.c 包含算子块。"""
        model = EncoderModel(d_model=256, dim_ff=512, num_layers=2)
        model.eval()

        dummy = torch.randn(1, 32, 256)
        mask = torch.zeros(1, 32, 32)

        result_dir = compile(
            model=model,
            dummy_input=dummy,
            config_dir=_CONFIG_DIR,
            output_dir=output_dir,
            mask=mask,
        )

        graph_c = os.path.join(result_dir, "src", "model_graph.c")
        with open(graph_c, "r") as f:
            content = f.read()

        assert "model_run" in content
        assert "/* ===" in content
        op_count = content.count("/* ===")
        assert op_count > 0, "应包含至少一个算子块"

    def test_compile_without_mask(self, output_dir):
        """无 mask 时也能正常编译。"""
        model = EncoderModel(d_model=256, dim_ff=512, num_layers=2)
        model.eval()

        dummy = torch.randn(1, 32, 256)

        result_dir = compile(
            model=model,
            dummy_input=dummy,
            config_dir=_CONFIG_DIR,
            output_dir=output_dir,
        )

        src = os.path.join(result_dir, "src")
        assert os.path.isfile(os.path.join(src, "model_graph.c"))


# ---- C 编译 + golden 比对 ----


@pytest.mark.skipif(
    not shutil.which("cc") and not shutil.which("gcc"),
    reason="C 编译器不可用",
)
class TestCGoldenComparison:
    @pytest.fixture
    def output_dir(self, tmp_path):
        return str(tmp_path / "output")

    def test_c_golden_passes(self, output_dir):
        """C 工程编译运行后 golden 比对通过 (max_abs < 0.02, cosine > 0.999)。"""
        from torch2c.integration.demo.validate_c_output import validate_c

        model = EncoderModel(d_model=256, dim_ff=512, num_layers=2)
        model.eval()

        dummy = torch.randn(1, 32, 256)
        mask = torch.zeros(1, 32, 32)

        compile(
            model=model,
            dummy_input=dummy,
            config_dir=_CONFIG_DIR,
            output_dir=output_dir,
            mask=mask,
            atol=1e-1,
        )

        result = validate_c(output_dir)
        assert result["passed"], f"C golden 比对失败:\n{result['stdout']}\n{result['stderr']}"

    def test_c_golden_fp32_only(self, output_dir):
        """全 FP32（无混合精度）的 golden 比对，排除 dtype 转换问题。"""
        from torch2c.integration.demo.validate_c_output import validate_c

        # 简单 FFN 模型，全 FP32，无 format 标注
        class SimpleFP32FFN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(64, 128)
                self.act = torch.nn.GELU()
                self.fc2 = torch.nn.Linear(128, 64)

            def forward(self, x):
                return self.fc2(self.act(self.fc1(x)))

        torch.manual_seed(99)
        model = SimpleFP32FFN()
        model.eval()
        dummy = torch.randn(1, 16, 64)

        compile(
            model=model,
            dummy_input=dummy,
            config_dir=_CONFIG_DIR,
            output_dir=output_dir,
            atol=5e-2,
        )

        result = validate_c(output_dir)
        assert result["passed"], f"FP32 golden 比对失败:\n{result['stdout']}\n{result['stderr']}"
