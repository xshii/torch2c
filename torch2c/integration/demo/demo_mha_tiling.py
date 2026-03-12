"""4 头 MHA tiling 对比 demo。

4 个变体（L1 = 1MB, d=256, heads=4, seq=384）：
  1. 多头计算：标准 MHA，拆 4 头 bmm [4, 384, 64] @ [4, 64, 384] → [4, 384, 384]
  2. 手动 tiling 多头：用户在 PyTorch 里手动切 seq 维度（384 → 2×192）
  3. 合并计算：不拆头，Q[1,384,256] @ K^T[1,256,384] → [1,384,384]，再拆给 softmax
  4. 手动 tiling 合并：合并计算 + 手动切 seq

对比编译器对每种写法的 tiling 决策。
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile

import torch
import torch.nn as nn
import yaml

from torch2c.common import INTEGRATION_CONFIG_DIR, NpuSpec, npu
from torch2c.integration.demo.validate_c_output import validate_c
from torch2c.integration.pipeline import compile

_FP16 = NpuSpec("fp16", "nd")
_FP16_NZ = NpuSpec("fp16", "nz")

D, HEADS, DH = 256, 4, 64
SEQ = 384
L1 = 1_048_576


def _npu_linear(d_in: int, d_out: int) -> nn.Linear:
    return npu(nn.Linear(d_in, d_out),
               input=_FP16, output=_FP16, weight=_FP16_NZ, compute_dtype="fp32")


# ── 模型 1: 多头计算（标准 MHA）─────────────────────────────


class MHA_PerHead(nn.Module):
    """标准 4 头：view/transpose → [B*H, S, dh] bmm。"""

    def __init__(self):
        super().__init__()
        self.q_proj = _npu_linear(D, D)
        self.k_proj = _npu_linear(D, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, HEADS, DH).transpose(1, 2).reshape(B * HEADS, S, DH)
        k = self.k_proj(x).view(B, S, HEADS, DH).transpose(1, 2).reshape(B * HEADS, S, DH)
        scores = torch.bmm(q, k.transpose(1, 2))  # [4, 384, 384]
        return scores


# ── 模型 2: 手动 tiling 多头计算 ───────────────────────────


class MHA_PerHead_ManualTile(nn.Module):
    """手动切 seq=384 → 2×192，每个 tile 独立 bmm。

    Q_chunk[4,192,64] @ K_chunk^T[4,64,192] → [4,192,192] = 384KB << 1MB。
    注意：这是 local attention（chunk 内 attend），语义与全 attention 不同。
    """

    def __init__(self):
        super().__init__()
        self.q_proj = _npu_linear(D, D)
        self.k_proj = _npu_linear(D, D)
        self.tile_size = SEQ // 2  # 192

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, HEADS, DH).transpose(1, 2).reshape(B * HEADS, S, DH)
        k = self.k_proj(x).view(B, S, HEADS, DH).transpose(1, 2).reshape(B * HEADS, S, DH)
        ts = self.tile_size
        s0 = torch.bmm(q[:, :ts, :], k[:, :ts, :].transpose(1, 2))   # [4,192,192]
        s1 = torch.bmm(q[:, ts:, :], k[:, ts:, :].transpose(1, 2))   # [4,192,192]
        return torch.cat([s0, s1], dim=1)  # [4,384,192] — 拼回 seq 维


# ── 模型 3: 合并计算 ──────────────────────────────────────


class MHA_Merged(nn.Module):
    """不拆头，直接 Q[1,S,D] @ K^T[1,D,S] → [1,S,S]，再拆给 softmax。

    一次大 matmul 代替 4 头独立 bmm。
    output[1,384,384] = 288KB，total ≈ 672KB < 1MB → 无需 tiling。
    """

    def __init__(self):
        super().__init__()
        self.q_proj = _npu_linear(D, D)
        self.k_proj = _npu_linear(D, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)  # [1, 384, 256]
        k = self.k_proj(x)  # [1, 384, 256]
        scores = torch.bmm(q, k.transpose(1, 2))  # [1, 384, 384]
        # 实际 MHA 里这里会 view(B,H,S,S) 再 per-head softmax
        return scores


# ── 模型 4: 手动 tiling 合并计算 ─────────────────────────


class MHA_Merged_ManualTile(nn.Module):
    """合并 + 手动切 seq。

    Q_chunk[1,192,256] @ K^T[1,256,192] → [1,192,192] = 168KB。
    已经很小，纯属演示手动 tiling 的效果。
    """

    def __init__(self):
        super().__init__()
        self.q_proj = _npu_linear(D, D)
        self.k_proj = _npu_linear(D, D)
        self.tile_size = SEQ // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)  # [1, 384, 256]
        k = self.k_proj(x)  # [1, 384, 256]
        ts = self.tile_size
        s0 = torch.bmm(q[:, :ts, :], k[:, :ts, :].transpose(1, 2))  # [1,192,192]
        s1 = torch.bmm(q[:, ts:, :], k[:, ts:, :].transpose(1, 2))  # [1,192,192]
        return torch.cat([s0, s1], dim=1)


# ── 运行对比 ──────────────────────────────────────────────


def _make_config_dir() -> str:
    tmpdir = tempfile.mkdtemp(prefix="mha_config_")
    src = str(INTEGRATION_CONFIG_DIR)
    for f in os.listdir(src):
        shutil.copy2(os.path.join(src, f), tmpdir)
    hw_path = os.path.join(tmpdir, "hardware_config.yaml")
    with open(hw_path) as fh:
        hw = yaml.safe_load(fh)
    hw["memory"]["l1"]["total_size_bytes"] = L1
    with open(hw_path, "w") as fh:
        yaml.dump(hw, fh, default_flow_style=False)
    return tmpdir


def _compile_one(name: str, model: nn.Module, output_dir: str, config_dir: str) -> dict:
    model.eval()
    dummy = torch.randn(1, SEQ, D)
    out = compile(
        model=model, dummy_input=dummy, config_dir=config_dir,
        output_dir=output_dir, target_dtype="fp16", target_format="nd",
        atol=0.05,
    )
    # 收集 tiling 信息
    graph = None
    # 尝试从 pipeline 返回值获取 graph（返回的是 output_dir 字符串）
    # 需要重新读取序列化的 graph 或从 compile 内部获取
    # 简化：直接从 C golden 验证
    c_result = validate_c(out)
    m = re.search(r"max_abs=([0-9.]+).*?cosine=([0-9.]+)", c_result["stdout"])
    max_abs, cosine = (float(m.group(1)), float(m.group(2))) if m else (-1.0, -1.0)
    return {
        "name": name,
        "passed": c_result["passed"],
        "max_abs": max_abs,
        "cosine": cosine,
        "output_dir": out,
    }


def main():
    config_dir = _make_config_dir()
    base_out = os.path.join(os.path.dirname(__file__), "..", "..", "..", "output", "mha_tiling")
    os.makedirs(base_out, exist_ok=True)

    variants = [
        ("① 多头计算 (auto)", MHA_PerHead()),
        ("② 手动tiling多头", MHA_PerHead_ManualTile()),
        ("③ 合并计算 (auto)", MHA_Merged()),
        ("④ 手动tiling合并", MHA_Merged_ManualTile()),
    ]

    print(f"\n{'='*70}")
    print(f"  4 头 MHA tiling 对比  (d={D}, heads={HEADS}, seq={SEQ}, L1={L1//1024}KB)")
    print(f"{'='*70}\n")

    results = []
    for name, model in variants:
        out_dir = os.path.join(base_out, name.split()[0])
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        print(f"── {name} ──")
        try:
            r = _compile_one(name, model, out_dir, config_dir)
            results.append(r)
            status = "✓ PASS" if r["passed"] else "✗ FAIL"
            print(f"  {status}  max_abs={r['max_abs']:.4f}  cosine={r['cosine']:.6f}\n")
        except Exception as e:
            print(f"  ✗ ERROR: {e}\n")
            results.append({"name": name, "passed": False, "error": str(e)})

    # 汇总表格
    print(f"\n{'='*70}")
    print(f"  汇总")
    print(f"{'='*70}")
    print(f"{'变体':<25} {'结果':>6} {'max_abs':>10} {'cosine':>10}")
    print(f"{'-'*55}")
    for r in results:
        if "error" in r:
            print(f"{r['name']:<25} {'ERROR':>6} {r['error']}")
        else:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"{r['name']:<25} {status:>6} {r['max_abs']:>10.4f} {r['cosine']:>10.6f}")

    shutil.rmtree(config_dir)


if __name__ == "__main__":
    main()
