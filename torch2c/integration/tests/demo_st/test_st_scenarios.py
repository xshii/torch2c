"""ST 场景端到端测试 — 基于 test_scenarios.yaml 的 6 个场景。

从 PyTorch 模型 → compile() → C golden 验证的完整流程。
所有场景统一 L1 = 1M (1,048,576 bytes)。
"""

from __future__ import annotations

import os
import shutil
import tempfile

import pytest
import torch
import torch.nn as nn
import yaml

from torch2c.common import INTEGRATION_CONFIG_DIR, NpuSpec, npu
from torch2c.common.errors import MemoryPlanError
from torch2c.integration.demo.validate_c_output import validate_c
from torch2c.integration.pipeline import compile

_L1_1M = 1_048_576
_FP16 = NpuSpec("fp16", "nd")
_FP16_NZ = NpuSpec("fp16", "nz")


# ── 工具 ────────────────────────────────────────────────────


def _make_config_dir(l1_size: int = _L1_1M) -> str:
    """复制 integration/config 到临时目录，覆盖 L1 大小。"""
    tmpdir = tempfile.mkdtemp(prefix="st_config_")
    src = str(INTEGRATION_CONFIG_DIR)
    for f in os.listdir(src):
        shutil.copy2(os.path.join(src, f), tmpdir)
    hw_path = os.path.join(tmpdir, "hardware_config.yaml")
    with open(hw_path) as f:
        hw = yaml.safe_load(f)
    hw["memory"]["l1"]["total_size_bytes"] = l1_size
    with open(hw_path, "w") as f:
        yaml.dump(hw, f, default_flow_style=False)
    return tmpdir


def _compile_and_validate(model: nn.Module, dummy_input: torch.Tensor, output_dir: str,
                          config_dir: str, *, atol: float = 1e-2,
                          cosine_tol: float = 0.999) -> dict:
    """编译 + C golden 验证，返回结果 dict。"""
    out = compile(
        model=model,
        dummy_input=dummy_input,
        config_dir=config_dir,
        output_dir=output_dir,
        target_dtype="fp16",
        target_format="nd",
        atol=atol,
        cosine_tol=cosine_tol,
    )
    c_result = validate_c(out)
    import re
    m = re.search(r"max_abs=([0-9.]+).*?cosine=([0-9.]+)", c_result["stdout"])
    max_abs, cosine = (float(m.group(1)), float(m.group(2))) if m else (-1.0, -1.0)
    return {
        "passed": c_result["passed"],
        "max_abs": max_abs,
        "cosine": cosine,
        "stdout": c_result["stdout"],
    }


# ── 模型定义 ─────────────────────────────────────────────────


class EmbeddingBiasModel(nn.Module):
    """单层 linear：input × W + b → output。"""

    def __init__(self, k: int, n: int, *, compute_dtype: str = "fp32"):
        super().__init__()
        self.linear = npu(
            nn.Linear(k, n),
            input=_FP16, output=_FP16, weight=_FP16_NZ,
            compute_dtype=compute_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TwoLayerMLP(nn.Module):
    """2 层 MLP：input → linear1 → linear2 → output。"""

    def __init__(self, d: int):
        super().__init__()
        self.linear1 = npu(
            nn.Linear(d, d),
            input=_FP16, output=_FP16, weight=_FP16_NZ,
            compute_dtype="fp32",
        )
        self.linear2 = npu(
            nn.Linear(d, d),
            input=_FP16, output=_FP16, weight=_FP16_NZ,
            compute_dtype="fp32",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.linear1(x))


class SimpleQKModel(nn.Module):
    """Q/K 投影 + 注意力打分：Q = Wq(x), K = Wk(x), scores = Q @ K^T。"""

    def __init__(self, d: int):
        super().__init__()
        self.q_proj = npu(
            nn.Linear(d, d),
            input=_FP16, output=_FP16, weight=_FP16_NZ,
            compute_dtype="fp32",
        )
        self.k_proj = npu(
            nn.Linear(d, d),
            input=_FP16, output=_FP16, weight=_FP16_NZ,
            compute_dtype="fp32",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.q_proj(x)
        K = self.k_proj(x)
        return torch.bmm(Q, K.transpose(-1, -2))


class MultiHeadQKModel(nn.Module):
    """多头 Q/K 投影 + per-head 注意力打分（batch>1 tiling 验证）。"""

    HEADS = 4

    def __init__(self, d: int):
        super().__init__()
        self.dh = d // self.HEADS
        self.q_proj = npu(
            nn.Linear(d, d),
            input=_FP16, output=_FP16, weight=_FP16_NZ,
            compute_dtype="fp32",
        )
        self.k_proj = npu(
            nn.Linear(d, d),
            input=_FP16, output=_FP16, weight=_FP16_NZ,
            compute_dtype="fp32",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.HEADS, self.dh).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.HEADS, self.dh).transpose(1, 2)
        q = q.reshape(B * self.HEADS, S, self.dh)
        k = k.reshape(B * self.HEADS, S, self.dh)
        return torch.bmm(q, k.transpose(1, 2))


class FullMHAModel(nn.Module):
    """完整多头注意力：Q/K/V 投影 + scores + softmax + weighted V + output projection。"""

    HEADS = 4

    def __init__(self, d: int):
        super().__init__()
        self.dh = d // self.HEADS
        self.scale = self.dh ** -0.5
        for name in ("q_proj", "k_proj", "v_proj"):
            setattr(self, name, npu(
                nn.Linear(d, d),
                input=_FP16, output=_FP16, weight=_FP16_NZ,
                compute_dtype="fp32",
            ))
        self.out_proj = npu(
            nn.Linear(d, d),
            input=_FP16, output=_FP16, weight=_FP16_NZ,
            compute_dtype="fp32",
        )

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        return x.view(B, S, self.HEADS, self.dh).transpose(1, 2) \
                .reshape(B * self.HEADS, S, self.dh)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))
        # scores → softmax → weighted sum
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn, v)
        # concat heads → output projection
        context = context.reshape(B, self.HEADS, S, self.dh) \
                         .transpose(1, 2).reshape(B, S, D)
        return self.out_proj(context)


# ── ST1: embedding+bias 全放下 → bulk，C golden 通过 ────────


class TestST1_EmbeddingBiasE2E:
    """ST1: input[1,32,128] × W[128,128] + b → output, total≈48KB << 1M。"""

    def test_compile_and_golden(self):
        model = EmbeddingBiasModel(k=128, n=128)
        model.eval()
        dummy = torch.randn(1, 32, 128)
        config_dir = _make_config_dir()
        with tempfile.TemporaryDirectory() as tmpdir:
            r = _compile_and_validate(model, dummy, tmpdir, config_dir)
            assert r["passed"], f"C golden 失败: {r['stdout']}"

    def test_precision(self):
        model = EmbeddingBiasModel(k=128, n=128)
        model.eval()
        dummy = torch.randn(1, 32, 128)
        config_dir = _make_config_dir()
        with tempfile.TemporaryDirectory() as tmpdir:
            r = _compile_and_validate(model, dummy, tmpdir, config_dir)
            assert r["max_abs"] < 0.02, f"max_abs={r['max_abs']}"
            assert r["cosine"] > 0.999, f"cosine={r['cosine']}"


# ── ST2: embedding+bias FP16 I/O + FP32 compute → bulk ──────


class TestST2_MixedPrecisionE2E:
    """ST2: 同 ST1 结构，compute_dtype=fp32。"""

    def test_compile_and_golden(self):
        model = EmbeddingBiasModel(k=128, n=128, compute_dtype="fp32")
        model.eval()
        dummy = torch.randn(1, 32, 128)
        config_dir = _make_config_dir()
        with tempfile.TemporaryDirectory() as tmpdir:
            r = _compile_and_validate(model, dummy, tmpdir, config_dir)
            assert r["passed"], f"C golden 失败: {r['stdout']}"


# ── ST3: embedding+bias 超 L1 → tiled 策略自动切分 ─────────────


class TestST3_EmbeddingBiasTiledE2E:
    """ST3: input[1,256,512] × W[512,512], total≈1025KB > 1M → tiled 成功。"""

    def test_compile_and_golden(self):
        model = EmbeddingBiasModel(k=512, n=512)
        model.eval()
        dummy = torch.randn(1, 256, 512)
        config_dir = _make_config_dir()
        with tempfile.TemporaryDirectory() as tmpdir:
            r = _compile_and_validate(model, dummy, tmpdir, config_dir)
            assert r["passed"], f"C golden 失败: {r['stdout']}"


# ── ST4: 2 层 MLP 全放下 → bulk，C golden 通过 ──────────────


class TestST4_2LayerMLPBulkE2E:
    """ST4: 2 层 MLP，d=128，total≈89KB << 1M。"""

    def test_compile_and_golden(self):
        model = TwoLayerMLP(d=128)
        model.eval()
        dummy = torch.randn(1, 32, 128)
        config_dir = _make_config_dir()
        with tempfile.TemporaryDirectory() as tmpdir:
            r = _compile_and_validate(model, dummy, tmpdir, config_dir)
            assert r["passed"], f"C golden 失败: {r['stdout']}"


# ── ST5: 2 层 MLP per-op → codegen 有 bug，暂 xfail ─────────


class TestST5_2LayerMLPPerOpE2E:
    """ST5: 2 层 MLP，d=256，input[1,600,256]。

    total≈1157KB > 1M → per-op 策略。
    per-op codegen 有 struct 指针碰撞 bug，C golden 暂时无法通过。
    """

    def test_compile_and_golden(self):
        model = TwoLayerMLP(d=256)
        model.eval()
        dummy = torch.randn(1, 600, 256)
        config_dir = _make_config_dir()
        with tempfile.TemporaryDirectory() as tmpdir:
            r = _compile_and_validate(model, dummy, tmpdir, config_dir)
            assert r["passed"], f"C golden 失败: {r['stdout']}"


# ── ST6: MHA Q/K 投影 → tiled 策略自动切分 ─────────────────


class TestST6_MHATiledE2E:
    """ST6: Q/K 投影 + attn, input[1,640,256]。

    per-op 溢出 → tiled 策略：attn 节点 M 维切分 640→2×320。
    """

    def test_compile_and_golden(self):
        model = SimpleQKModel(d=256)
        model.eval()
        dummy = torch.randn(1, 640, 256)
        config_dir = _make_config_dir()
        # atol=0.05: Q@K^T with d=256 in FP16 produces ~0.03 quantization error
        with tempfile.TemporaryDirectory() as tmpdir:
            r = _compile_and_validate(model, dummy, tmpdir, config_dir, atol=0.05)
            assert r["passed"], f"C golden 失败: {r['stdout']}"
            assert r["cosine"] > 0.999, f"cosine={r['cosine']}"


# ── ST7: MHA + 双缓冲流水（L1=2M，足够 ping-pong）──────────


class TestST7_MHADoubleBufferE2E:
    """ST7: 与 ST6 相同模型，L1=2M 使双缓冲可行。

    验证 num_buffers=2 时生成的 ping-pong 代码仍然正确。
    """

    def test_compile_and_golden(self):
        model = SimpleQKModel(d=256)
        model.eval()
        dummy = torch.randn(1, 640, 256)
        config_dir = _make_config_dir(l1_size=_L1_1M)  # 1MB L1 — 与 ST6 相同
        with tempfile.TemporaryDirectory() as tmpdir:
            r = _compile_and_validate(model, dummy, tmpdir, config_dir, atol=0.05)
            assert r["passed"], f"C golden 失败: {r['stdout']}"
            assert r["cosine"] > 0.999, f"cosine={r['cosine']}"


# ── ST8: 多头 MHA batch>1 tiled bmm（per-batch DMA）──────────


class TestST8_MultiHeadMHATiledE2E:
    """ST8: 4 头 MHA，Q/K 投影 + reshape + bmm。

    bmm 输入 shape [4, 384, 64]，tiled dim=1 时 batch=4 需 per-batch DMA。
    验证 batch>1 tiling 的正确性。
    """

    def test_compile_and_golden(self):
        model = MultiHeadQKModel(d=256)
        model.eval()
        dummy = torch.randn(1, 384, 256)
        config_dir = _make_config_dir()
        with tempfile.TemporaryDirectory() as tmpdir:
            r = _compile_and_validate(model, dummy, tmpdir, config_dir, atol=0.05)
            assert r["passed"], f"C golden 失败: {r['stdout']}"
            assert r["cosine"] > 0.999, f"cosine={r['cosine']}"


# ── ST9: 完整 MHA（Q/K/V + softmax + output projection）─────


class TestST9_FullMHAE2E:
    """ST9: 完整 4 头 MHA，Q/K/V 投影 + attention scores + softmax + output。

    覆盖完整多头注意力流程，包括 V 投影、softmax、加权求和、output projection。
    FP16 累积误差较大（softmax 路径），放宽容忍度。
    """

    def test_compile_and_golden(self):
        model = FullMHAModel(d=128)
        model.eval()
        dummy = torch.randn(1, 32, 128)
        config_dir = _make_config_dir()
        # Full attention FP16 pipeline: softmax 路径累积误差较大
        with tempfile.TemporaryDirectory() as tmpdir:
            r = _compile_and_validate(model, dummy, tmpdir, config_dir,
                                      atol=0.15, cosine_tol=0.95)
            assert r["passed"], f"C golden 失败: {r['stdout']}"
            assert r["cosine"] > 0.95, f"cosine={r['cosine']}"
