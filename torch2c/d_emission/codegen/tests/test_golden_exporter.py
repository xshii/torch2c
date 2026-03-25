"""golden_exporter 单元测试 — 覆盖 per-tensor dtype 导出。"""

from __future__ import annotations

import os

import numpy as np
import pytest

from torch2c.d_emission.codegen import golden_exporter


class TestExportGoldenPerTensorDtype:
    """per-tensor dtype 导出：输入输出可以有不同 dtype。"""

    def test_single_dtype_fp32(self, tmp_path):
        """全 FP32 导出，desc 中 dtype 应为 fp32。"""
        inp = [np.random.randn(1, 4, 8).astype(np.float32)]
        out = [np.random.randn(1, 4, 8).astype(np.float32)]
        golden_exporter.export_golden(inp, out, str(tmp_path), dtype="fp32")

        desc = (tmp_path / "output_0.desc").read_text()
        assert "dtype: fp32" in desc
        assert (tmp_path / "output_0.bin").stat().st_size == 1 * 4 * 8 * 4

    def test_single_dtype_fp16(self, tmp_path):
        """全 FP16 导出，desc 中 dtype 应为 fp16。"""
        inp = [np.random.randn(1, 4, 8).astype(np.float16)]
        out = [np.random.randn(1, 4, 8).astype(np.float16)]
        golden_exporter.export_golden(inp, out, str(tmp_path), dtype="fp16")

        desc = (tmp_path / "output_0.desc").read_text()
        assert "dtype: fp16" in desc
        assert (tmp_path / "output_0.bin").stat().st_size == 1 * 4 * 8 * 2

    def test_per_tensor_dtype_list(self, tmp_path):
        """per-tensor dtype: 输入 fp32, 输出 fp16。"""
        inp = [np.random.randn(1, 4, 8).astype(np.float32)]
        out = [np.random.randn(1, 4, 8).astype(np.float16)]
        golden_exporter.export_golden(
            inp, out, str(tmp_path),
            dtype=["fp32"],
            output_dtype=["fp16"],
        )

        in_desc = (tmp_path / "input_0.desc").read_text()
        assert "dtype: fp32" in in_desc
        assert (tmp_path / "input_0.bin").stat().st_size == 1 * 4 * 8 * 4

        out_desc = (tmp_path / "output_0.desc").read_text()
        assert "dtype: fp16" in out_desc
        assert (tmp_path / "output_0.bin").stat().st_size == 1 * 4 * 8 * 2

    def test_per_tensor_mixed_multiple_outputs(self, tmp_path):
        """多个输出各有不同 dtype。"""
        inp = [np.random.randn(2, 4).astype(np.float32)]
        out = [
            np.random.randn(2, 4).astype(np.float16),
            np.random.randn(2, 4).astype(np.float32),
        ]
        golden_exporter.export_golden(
            inp, out, str(tmp_path),
            dtype=["fp32"],
            output_dtype=["fp16", "fp32"],
        )

        assert "dtype: fp16" in (tmp_path / "output_0.desc").read_text()
        assert "dtype: fp32" in (tmp_path / "output_1.desc").read_text()
        assert (tmp_path / "output_0.bin").stat().st_size == 2 * 4 * 2
        assert (tmp_path / "output_1.bin").stat().st_size == 2 * 4 * 4


class TestInferIoDtypes:
    """测试 _infer_io_dtypes 从 graph 正确推断 I/O dtype。"""

    def _make_graph(self, input_dtype="fp32", output_dtype="fp16"):
        from torch2c.common import Graph, Node, Tensor
        g = Graph()
        g.add_tensor(Tensor(id="t_in", shape=[1, 4], dtype=input_dtype,
                            is_model_input=True))
        g.add_tensor(Tensor(id="t_out", shape=[1, 4], dtype=output_dtype,
                            is_model_output=True))
        g.add_node(Node(id="n0", op_type="matmul",
                        inputs=["t_in"], outputs=["t_out"]))
        return g

    def test_mixed_precision_io(self):
        from torch2c.integration._codegen_runner import _infer_io_dtypes
        g = self._make_graph(input_dtype="fp32", output_dtype="fp16")
        in_dtypes, out_dtypes = _infer_io_dtypes(g)
        assert in_dtypes == [np.float32]
        assert out_dtypes == [np.float16]

    def test_all_fp32(self):
        from torch2c.integration._codegen_runner import _infer_io_dtypes
        g = self._make_graph(input_dtype="fp32", output_dtype="fp32")
        in_dtypes, out_dtypes = _infer_io_dtypes(g)
        assert in_dtypes == [np.float32]
        assert out_dtypes == [np.float32]

    def test_all_fp16(self):
        from torch2c.integration._codegen_runner import _infer_io_dtypes
        g = self._make_graph(input_dtype="fp16", output_dtype="fp16")
        in_dtypes, out_dtypes = _infer_io_dtypes(g)
        assert in_dtypes == [np.float16]
        assert out_dtypes == [np.float16]
