"""test_utils_emitter — utils_emitter 单元测试。"""

from __future__ import annotations

import os
import subprocess
import tempfile

from npu_compiler.codegen import utils_emitter


class TestComparatorGeneration:
    """test_comparator_generation: comparator.c 语法正确。"""

    def test_comparator_c_content(self):
        code = utils_emitter.emit_comparator_c()
        assert "compare_tensors" in code
        assert "cosine_similarity" in code

    def test_comparator_h_content(self):
        header = utils_emitter.emit_comparator_h()
        assert "COMPARATOR_H" in header
        assert "compare_result_t" in header

    def test_comparator_syntax(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            utils_emitter.run(tmpdir)
            cmd = [
                "gcc",
                "-fsyntax-only",
                "-std=c99",
                f"-I{os.path.join(tmpdir, 'utils')}",
                os.path.join(tmpdir, "utils", "comparator.c"),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0, f"comparator.c 语法检查失败:\n{result.stderr}"


class TestDataLoaderGeneration:
    """test_data_loader_generation: data_loader.c 语法正确。"""

    def test_data_loader_c_content(self):
        code = utils_emitter.emit_data_loader_c()
        assert "parse_desc" in code
        assert "load_tensor" in code

    def test_data_loader_h_content(self):
        header = utils_emitter.emit_data_loader_h()
        assert "DATA_LOADER_H" in header
        assert "tensor_desc_t" in header

    def test_data_loader_syntax(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            utils_emitter.run(tmpdir)
            cmd = [
                "gcc",
                "-fsyntax-only",
                "-std=c99",
                f"-I{os.path.join(tmpdir, 'utils')}",
                os.path.join(tmpdir, "utils", "data_loader.c"),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0, f"data_loader.c 语法检查失败:\n{result.stderr}"


class TestDataDumperGeneration:
    def test_data_dumper_syntax(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            utils_emitter.run(tmpdir)
            cmd = [
                "gcc",
                "-fsyntax-only",
                "-std=c99",
                f"-I{os.path.join(tmpdir, 'utils')}",
                os.path.join(tmpdir, "utils", "data_dumper.c"),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0, f"data_dumper.c 语法检查失败:\n{result.stderr}"
