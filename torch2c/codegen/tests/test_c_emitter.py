"""test_c_emitter — c_emitter 单元测试。"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from torch2c.codegen import c_emitter, mock_emitter
from torch2c.common import INTEGRATION_CONFIG_DIR, CodegenError, load_config

_CONFIG_DIR = str(INTEGRATION_CONFIG_DIR)
_DEMO_PLAN = str(Path(__file__).resolve().parents[1] / "demo" / "demo_input_plan.json")


def _load_plan() -> dict:
    with open(_DEMO_PLAN) as f:
        return json.load(f)


def _load_sigs() -> dict:
    return load_config(
        os.path.join(_CONFIG_DIR, "c_api_signatures.yaml"),
        required_keys=["compute_ops"],
    )


# -- 辅助：构造最小 node/tensor 用于单算子测试 --


def _make_tensor(tid, shape, dtype="fp16", fmt="nd", hbm_offset=0, l1_offset=0, hbm_size=4096):
    return {
        "id": tid,
        "shape": shape,
        "dtype": dtype,
        "format": fmt,
        "hbm_offset": hbm_offset,
        "hbm_size": hbm_size,
        "l1_offset": l1_offset,
    }


def _make_node(
    nid, npu_op, inputs, outputs, params=None, compute_unit="Vector", absorbed_inputs=None
):
    return {
        "id": nid,
        "op_type": npu_op,
        "npu_op": npu_op,
        "inputs": inputs,
        "outputs": outputs,
        "params": params or {},
        "compute_unit": compute_unit,
        "absorbed_inputs": absorbed_inputs or {},
    }


def _empty_dma():
    return {"loads": [], "stores": []}


class TestOpBlockGeneration:
    """单个算子生成正确的三段式代码。"""

    def test_matmul_block_has_dma_and_call(self):
        plan = _load_plan()
        sigs = _load_sigs()
        node = plan["nodes"]["node_0"]
        dp = plan["dma_plans"][0]
        block = c_emitter.gen_op_block(node, plan["tensors"], dp, sigs)
        assert "npu_dma_load" in block
        assert "cube_matmul" in block
        assert "npu_dma_store" in block
        assert "npu_dma_barrier" in block

    def test_gelu_block(self):
        plan = _load_plan()
        sigs = _load_sigs()
        node = plan["nodes"]["node_1"]
        dp = plan["dma_plans"][1]
        block = c_emitter.gen_op_block(node, plan["tensors"], dp, sigs)
        assert "vector_gelu" in block
        assert "2048" in block  # elem_count = 1*32*64

    def test_add_block(self):
        plan = _load_plan()
        sigs = _load_sigs()
        node = plan["nodes"]["node_2"]
        dp = plan["dma_plans"][2]
        block = c_emitter.gen_op_block(node, plan["tensors"], dp, sigs)
        assert "vector_add" in block

    def test_no_loads_no_barrier_before_call(self):
        """无 DMA load 时，不生成 barrier。"""
        sigs = _load_sigs()
        tensors = {"in": _make_tensor("in", [1, 32, 64]), "out": _make_tensor("out", [1, 32, 64])}
        node = _make_node("n", "vector_gelu", ["in"], ["out"])
        block = c_emitter.gen_op_block(node, tensors, _empty_dma(), sigs)
        # 只有 op 调用，没有 DMA 部分
        assert "vector_gelu" in block
        assert "npu_dma_load" not in block

    def test_unknown_op_raises(self):
        sigs = _load_sigs()
        node = _make_node("n", "npu_nonexistent", ["in"], ["out"])
        with pytest.raises(CodegenError, match="签名未找到"):
            c_emitter.gen_op_block(node, {}, _empty_dma(), sigs)


class TestParamFilling:
    """参数根据 signature 正确填入。"""

    def test_matmul_shape_params(self):
        plan = _load_plan()
        sigs = _load_sigs()
        node = plan["nodes"]["node_0"]
        dp = plan["dma_plans"][0]
        block = c_emitter.gen_op_block(node, plan["tensors"], dp, sigs)
        assert "cube_matmul(" in block
        assert "NPU_DTYPE_FP16" in block

    def test_addr_params_use_offsets(self):
        plan = _load_plan()
        sigs = _load_sigs()
        node = plan["nodes"]["node_0"]
        dp = plan["dma_plans"][0]
        block = c_emitter.gen_op_block(node, plan["tensors"], dp, sigs)
        assert "(void*)(l1 + 0)" in block
        assert "(void*)(l1 + 4096)" in block

    def test_mul_scalar_float_param(self):
        """vector_mul_scalar 的 scalar 参数应为浮点数格式。"""
        sigs = _load_sigs()
        tensors = {
            "in": _make_tensor("in", [1, 32, 64]),
            "out": _make_tensor("out", [1, 32, 64], l1_offset=4096),
        }
        node = _make_node("n", "vector_mul_scalar", ["in"], ["out"], params={"scalar_value": 0.25})
        block = c_emitter.gen_op_block(node, tensors, _empty_dma(), sigs)
        assert "vector_mul_scalar(" in block
        assert "0.250000f" in block
        assert "2048" in block  # elem_count

    def test_layernorm_shape_negative_index(self):
        """shape.-1 和 shape.-2 负索引正确解析。"""
        sigs = _load_sigs()
        tensors = {
            "in": _make_tensor("in", [1, 32, 64]),
            "gamma": _make_tensor("gamma", [64], l1_offset=4096),
            "beta": _make_tensor("beta", [64], l1_offset=8192),
            "out": _make_tensor("out", [1, 32, 64], l1_offset=12288),
        }
        node = _make_node(
            "n", "vector_layernorm_part1", ["in", "gamma", "beta"], ["out"], params={"epsilon": 1e-5}
        )
        block = c_emitter.gen_op_block(node, tensors, _empty_dma(), sigs)
        assert "vector_layernorm_part1(" in block
        assert "64" in block  # hidden = shape[-1]
        assert "32" in block  # seq = shape[-2]

    def test_transpose_4d_with_int_array(self):
        """vector_transpose 4D 接口：ndim + dims(int_array) + dim0/dim1。"""
        sigs = _load_sigs()
        tensors = {
            "in": _make_tensor("in", [1, 4, 32, 16]),
            "out": _make_tensor("out", [1, 4, 16, 32], l1_offset=4096),
        }
        node = _make_node("n", "vector_transpose", ["in"], ["out"], params={"dim0": 2, "dim1": 3})
        block = c_emitter.gen_op_block(node, tensors, _empty_dma(), sigs)
        assert "vector_transpose(" in block
        assert "4" in block  # ndim
        assert "(const int[]){1, 4, 32, 16}" in block  # dims
        assert "2" in block  # dim0
        assert "3" in block  # dim1


class TestModelGraphGeneration:
    """完整 model_graph.c 生成测试。"""

    def test_full_generation(self):
        plan = _load_plan()
        sigs = _load_sigs()
        code = c_emitter.emit_model_graph_c(plan, sigs)
        assert "void model_run(" in code
        assert "cube_matmul" in code
        assert "vector_gelu" in code
        assert "vector_add" in code

    def test_execution_order_preserved(self):
        plan = _load_plan()
        sigs = _load_sigs()
        code = c_emitter.emit_model_graph_c(plan, sigs)
        matmul_pos = code.index("cube_matmul")
        gelu_pos = code.index("vector_gelu")
        add_pos = code.index("vector_add")
        assert matmul_pos < gelu_pos < add_pos

    def test_model_graph_h(self):
        header = c_emitter.emit_model_graph_h()
        assert "MODEL_GRAPH_H" in header
        assert "void model_run(" in header

    def test_model_memory_h(self):
        plan = _load_plan()
        header = c_emitter.emit_model_memory_h(plan)
        assert "T_INPUT_HBM_OFFSET" in header
        assert "T_WEIGHT_HBM_SIZE" in header

    def test_model_params_h_with_params(self):
        plan = _load_plan()
        # 添加一个有 params 的节点
        plan["nodes"]["node_1"]["params"]["test_key"] = 42
        header = c_emitter.emit_model_params_h(plan)
        assert "NODE_1_TEST_KEY" in header
        assert "42" in header


class TestDmaBlock:
    """DMA 代码生成细节测试。"""

    def test_load_format_conversion(self):
        """DMA load 支持格式转换参数。"""
        sigs = _load_sigs()
        tensors = {"in": _make_tensor("in", [1, 32, 64]), "out": _make_tensor("out", [1, 32, 64])}
        node = _make_node("n", "vector_gelu", ["in"], ["out"])
        dma = {
            "loads": [
                {
                    "op": "load",
                    "tensor_id": "in",
                    "hbm_offset": 0,
                    "l1_offset": 0,
                    "size_bytes": 4096,
                    "src_format": "nd",
                    "dst_format": "nz",
                }
            ],
            "stores": [],
        }
        block = c_emitter.gen_op_block(node, tensors, dma, sigs)
        assert "NPU_FORMAT_ND" in block
        assert "NPU_FORMAT_NZ" in block

    def test_store_uses_correct_offsets(self):
        sigs = _load_sigs()
        tensors = {"in": _make_tensor("in", [1, 32, 64]), "out": _make_tensor("out", [1, 32, 64])}
        node = _make_node("n", "vector_gelu", ["in"], ["out"])
        dma = {
            "loads": [],
            "stores": [
                {
                    "op": "store",
                    "tensor_id": "out",
                    "hbm_offset": 8192,
                    "l1_offset": 4096,
                    "size_bytes": 4096,
                    "src_format": "nd",
                    "dst_format": "nd",
                }
            ],
        }
        block = c_emitter.gen_op_block(node, tensors, dma, sigs)
        assert "npu_dma_store((void*)(hbm + 8192)" in block
        assert "(void*)(l1 + 4096)" in block


class TestSyntaxCheck:
    """生成的 C 代码通过 gcc 语法检查。"""

    def test_syntax_check(self):
        plan = _load_plan()
        with tempfile.TemporaryDirectory() as tmpdir:
            c_emitter.run(plan, tmpdir, _CONFIG_DIR)
            mock_emitter.run(tmpdir, _CONFIG_DIR)

            weights_h = os.path.join(tmpdir, "src", "model_weights.h")
            with open(weights_h, "w") as f:
                f.write(
                    "#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n"
                    "static inline void load_weights(unsigned char* hbm)"
                    " { (void)hbm; }\n#endif\n"
                )

            model_c = os.path.join(tmpdir, "src", "model_graph.c")
            mock_h = os.path.join(tmpdir, "npu_mock.h")
            cmd = [
                "gcc",
                "-fsyntax-only",
                "-std=c99",
                f"-include{mock_h}",
                f"-I{os.path.join(tmpdir, 'src')}",
                f"-I{tmpdir}",
                model_c,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0, f"gcc 语法检查失败:\n{result.stderr}"
