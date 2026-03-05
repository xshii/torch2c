"""单算子语义测试 — 对比 PyTorch 参考实现与端到端编译结果。

防范问题：
1. ATen → NPU 映射语义不匹配（如 linear 缺少 weight 转置）
2. 分解算子（softmax、layernorm）与 PyTorch 行为不一致
3. 新增算子时未正确处理隐含语义（broadcast、transpose 等）

每个测试：
1. 定义一个只含目标算子的最小 PyTorch 模型
2. 通过编译管线生成 C 工程
3. 编译运行 C 工程
4. 对比 C 输出与 PyTorch golden
"""

from __future__ import annotations

import os
import pathlib
import shutil

import numpy as np
import pytest
import torch
import torch.nn as nn

from npu_compiler.integration.pipeline import compile

_CONFIG_DIR = str(pathlib.Path(__file__).resolve().parent.parent / "config")
_HAS_CC = shutil.which("cc") is not None or shutil.which("gcc") is not None
pytestmark = pytest.mark.skipif(not _HAS_CC, reason="C 编译器不可用")


# ---- 工具函数 ----


def _validate_c(output_dir: str) -> dict:
    """编译运行 C 工程并返回结果。"""
    from npu_compiler.integration.demo.validate_c_output import validate_c

    return validate_c(output_dir)


def _read_golden_output(output_dir: str) -> np.ndarray:
    """读取 golden output 数据。"""
    golden_dir = os.path.join(output_dir, "golden")
    desc_path = os.path.join(golden_dir, "output_0.desc")
    bin_path = os.path.join(golden_dir, "output_0.bin")

    with open(desc_path) as f:
        lines = f.read().strip().split("\n")
    dtype_str = lines[0].split("=")[1].strip() if "=" in lines[0] else "fp16"
    shape_str = lines[1].split("=")[1].strip() if len(lines) > 1 and "=" in lines[1] else ""

    np_dtype = np.float16 if "16" in dtype_str else np.float32
    data = np.fromfile(bin_path, dtype=np_dtype)

    if shape_str:
        shape = [int(s) for s in shape_str.split(",") if s.strip()]
        data = data.reshape(shape)
    return data


def _compare(actual: np.ndarray, expected: np.ndarray, atol: float, cos_tol: float) -> dict:
    """比较两个数组的精度。"""
    diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    max_abs = float(diff.max())

    a = actual.astype(np.float64).flatten()
    e = expected.astype(np.float64).flatten()
    dot = np.dot(a, e)
    na, ne_ = np.linalg.norm(a), np.linalg.norm(e)
    cosine = float(dot / (na * ne_)) if na > 0 and ne_ > 0 else 0.0

    passed = max_abs <= atol and cosine >= cos_tol
    return {"passed": passed, "max_abs": max_abs, "cosine": cosine}


# ---- 最小模型定义 ----


class LinearOnly(nn.Module):
    """单个 nn.Linear，验证 linear = input @ weight.T + bias。"""

    def __init__(self, in_f: int = 64, out_f: int = 32):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f)

    def forward(self, x):
        return self.linear(x)


class MatmulOnly(nn.Module):
    """单个矩阵乘法 (mm)，验证 C = A @ B。"""

    def __init__(self, k: int = 64, n: int = 32):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(k, n))

    def forward(self, x):
        return torch.mm(x, self.weight)


class GeluOnly(nn.Module):
    """单个 GELU。"""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return torch.nn.functional.gelu(self.linear(x))


class SoftmaxOnly(nn.Module):
    """Linear + Softmax，验证 softmax 分解正确性。"""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return torch.softmax(self.linear(x), dim=-1)


class LayerNormOnly(nn.Module):
    """Linear + LayerNorm，验证 layernorm 分解正确性。"""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(self.linear(x))


class LinearChain(nn.Module):
    """两层 Linear，验证多次 weight transpose 处理。"""

    def __init__(self, dim: int = 64, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


# ---- 语义测试用例 ----


class TestLinearSemantics:
    """验证 aten.linear.default → cube_matmul_bias 的语义等价性。

    关键语义：linear(x, w, b) = x @ w.T + b
    历史 bug：曾忘记对 weight 做转置。
    """

    @pytest.fixture
    def output_dir(self, tmp_path):
        return str(tmp_path / "output")

    def test_linear_matches_pytorch(self, output_dir):
        torch.manual_seed(42)
        model = LinearOnly(in_f=64, out_f=32)
        model.eval()
        dummy = torch.randn(1, 16, 64)

        with torch.no_grad():
            expected = model(dummy).cpu().numpy().astype(np.float16)

        compile(
            model=model,
            dummy_input=dummy,
            config_dir=_CONFIG_DIR,
            output_dir=output_dir,
            atol=2e-2,
        )

        result = _validate_c(output_dir)
        assert result["passed"], (
            f"Linear 语义不匹配 (可能缺少 weight 转置):\n"
            f"{result['stdout']}\n{result['stderr']}"
        )


class TestMatmulSemantics:
    """验证 aten.mm.default → cube_matmul 的语义等价性。"""

    @pytest.fixture
    def output_dir(self, tmp_path):
        return str(tmp_path / "output")

    def test_mm_matches_pytorch(self, output_dir):
        torch.manual_seed(43)
        model = MatmulOnly(k=64, n=32)
        model.eval()
        dummy = torch.randn(16, 64)

        compile(
            model=model,
            dummy_input=dummy,
            config_dir=_CONFIG_DIR,
            output_dir=output_dir,
            atol=2e-2,
        )

        result = _validate_c(output_dir)
        assert result["passed"], (
            f"Matmul 语义不匹配:\n{result['stdout']}\n{result['stderr']}"
        )


class TestSoftmaxSemantics:
    """验证 softmax 分解（part1 + part2）与 PyTorch 语义一致。"""

    @pytest.fixture
    def output_dir(self, tmp_path):
        return str(tmp_path / "output")

    def test_softmax_matches_pytorch(self, output_dir):
        torch.manual_seed(44)
        model = SoftmaxOnly(dim=64)
        model.eval()
        dummy = torch.randn(1, 16, 64)

        compile(
            model=model,
            dummy_input=dummy,
            config_dir=_CONFIG_DIR,
            output_dir=output_dir,
            atol=2e-2,
        )

        result = _validate_c(output_dir)
        assert result["passed"], (
            f"Softmax 分解语义不匹配:\n{result['stdout']}\n{result['stderr']}"
        )


class TestLayerNormSemantics:
    """验证 layernorm 分解（part1 + part2）与 PyTorch 语义一致。"""

    @pytest.fixture
    def output_dir(self, tmp_path):
        return str(tmp_path / "output")

    def test_layernorm_matches_pytorch(self, output_dir):
        torch.manual_seed(45)
        model = LayerNormOnly(dim=64)
        model.eval()
        dummy = torch.randn(1, 16, 64)

        compile(
            model=model,
            dummy_input=dummy,
            config_dir=_CONFIG_DIR,
            output_dir=output_dir,
            atol=2e-2,
        )

        result = _validate_c(output_dir)
        assert result["passed"], (
            f"LayerNorm 分解语义不匹配:\n{result['stdout']}\n{result['stderr']}"
        )


class TestGeluSemantics:
    """验证 GELU 与 PyTorch 语义一致。"""

    @pytest.fixture
    def output_dir(self, tmp_path):
        return str(tmp_path / "output")

    def test_gelu_matches_pytorch(self, output_dir):
        torch.manual_seed(46)
        model = GeluOnly(dim=64)
        model.eval()
        dummy = torch.randn(1, 16, 64)

        compile(
            model=model,
            dummy_input=dummy,
            config_dir=_CONFIG_DIR,
            output_dir=output_dir,
            atol=2e-2,
        )

        result = _validate_c(output_dir)
        assert result["passed"], (
            f"GELU 语义不匹配:\n{result['stdout']}\n{result['stderr']}"
        )


class TestLinearChainSemantics:
    """验证多层 Linear 链路（多次 weight transpose）正确性。"""

    @pytest.fixture
    def output_dir(self, tmp_path):
        return str(tmp_path / "output")

    def test_two_linear_chain(self, output_dir):
        torch.manual_seed(47)
        model = LinearChain(dim=64, hidden=128)
        model.eval()
        dummy = torch.randn(1, 16, 64)

        compile(
            model=model,
            dummy_input=dummy,
            config_dir=_CONFIG_DIR,
            output_dir=output_dir,
            atol=2e-2,
        )

        result = _validate_c(output_dir)
        assert result["passed"], (
            f"Linear 链路语义不匹配:\n{result['stdout']}\n{result['stderr']}"
        )
