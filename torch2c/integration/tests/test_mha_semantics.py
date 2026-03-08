"""多头注意力渐进式语义测试 — 逐步验证 view/transpose/reshape/bmm 组合正确性。

测试策略：从最简单的单算子开始，逐步组合更多算子，
当某个测试失败时可以精确定位是哪个算子组合引入了误差。

测试层次：
1. 单头投影 + view/transpose/reshape（无 bmm）
2. 多头投影 + view/transpose/reshape（无 bmm）
3. 多头 Q@K^T（bmm）
4. 多头 Q@K^T + softmax + attn@V（完整注意力，无 mask）
5. 完整注意力 + mask
6. 完整 EncoderLayer（注意力 + FFN + 残差 + LayerNorm）
"""

from __future__ import annotations

import shutil

import pytest
import torch
import torch.nn as nn

from torch2c.common import INTEGRATION_CONFIG_DIR, NpuSpec, npu
from torch2c.integration.pipeline import compile

_CONFIG_DIR = str(INTEGRATION_CONFIG_DIR)
_HAS_CC = shutil.which("cc") is not None or shutil.which("gcc") is not None
pytestmark = pytest.mark.skipif(not _HAS_CC, reason="C 编译器不可用")

_FP16 = NpuSpec("fp16", "nd")
_FP16_NZ = NpuSpec("fp16", "nz")
_FP32 = NpuSpec("fp32", "nd")

D_MODEL, NUM_HEADS, HEAD_DIM = 192, 3, 64
DIM_FF = 384


# ---- 共享 fixture ----


@pytest.fixture
def output_dir(tmp_path):
    return str(tmp_path / "output")


def _validate_c(output_dir: str) -> dict:
    from torch2c.integration.demo.validate_c_output import validate_c
    return validate_c(output_dir)


def _npu_linear(d_in: int, d_out: int) -> nn.Linear:
    return npu(nn.Linear(d_in, d_out),
               input=_FP16, output=_FP16, weight=_FP16_NZ, compute_dtype="fp32")


# ---- 最小模型定义 ----


class SingleHeadProj(nn.Module):
    """单头投影：linear → view → transpose → reshape（1 head = 无实际 head 拆分）。"""

    def __init__(self):
        super().__init__()
        self.q_proj = _npu_linear(D_MODEL, D_MODEL)

    def forward(self, x):
        B, S, _ = x.shape
        return self.q_proj(x).view(B, S, 1, D_MODEL).transpose(1, 2).reshape(B, S, D_MODEL)


class MultiHeadProj(nn.Module):
    """多头投影：linear → view → transpose → reshape（3 heads）。"""

    def __init__(self):
        super().__init__()
        self.q_proj = _npu_linear(D_MODEL, D_MODEL)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, NUM_HEADS, HEAD_DIM)
        q = q.transpose(1, 2).reshape(B * NUM_HEADS, S, HEAD_DIM)
        return q


class MultiHeadBmm(nn.Module):
    """多头 Q@K^T：两路投影 + view/transpose/reshape + bmm。"""

    def __init__(self):
        super().__init__()
        self.q_proj = _npu_linear(D_MODEL, D_MODEL)
        self.k_proj = _npu_linear(D_MODEL, D_MODEL)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2).reshape(B * NUM_HEADS, S, HEAD_DIM)
        k = self.k_proj(x).view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2).reshape(B * NUM_HEADS, S, HEAD_DIM)
        return torch.bmm(q, k.transpose(1, 2))


class MultiHeadAttnNoMask(nn.Module):
    """完整多头注意力（无 mask）：Q/K/V 投影 + bmm + softmax + bmm + 输出投影。"""

    def __init__(self):
        super().__init__()
        self.q_proj = _npu_linear(D_MODEL, D_MODEL)
        self.k_proj = _npu_linear(D_MODEL, D_MODEL)
        self.v_proj = _npu_linear(D_MODEL, D_MODEL)
        self.o_proj = _npu_linear(D_MODEL, D_MODEL)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2).reshape(B * NUM_HEADS, S, HEAD_DIM)
        k = self.k_proj(x).view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2).reshape(B * NUM_HEADS, S, HEAD_DIM)
        v = self.v_proj(x).view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2).reshape(B * NUM_HEADS, S, HEAD_DIM)
        scores = torch.bmm(q, k.transpose(1, 2))
        attn = torch.softmax(scores, dim=-1)
        out = torch.bmm(attn, v)
        out = out.view(B, NUM_HEADS, S, HEAD_DIM).transpose(1, 2).reshape(B, S, D_MODEL)
        return self.o_proj(out)


class BroadcastMaskAdd(nn.Module):
    """最小广播加法：scores [B,H,S,S] + mask.unsqueeze(1) [B,1,S,S]。

    使用 bmm 生成 [B*H,S,S] 形状的 scores，再 view 为 [B,H,S,S] 做广播加法。
    """

    def __init__(self):
        super().__init__()
        self.q_proj = _npu_linear(D_MODEL, D_MODEL)
        self.k_proj = _npu_linear(D_MODEL, D_MODEL)

    def forward(self, x, mask=None):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2).reshape(B * NUM_HEADS, S, HEAD_DIM)
        k = self.k_proj(x).view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2).reshape(B * NUM_HEADS, S, HEAD_DIM)
        scores = torch.bmm(q, k.transpose(1, 2))
        if mask is not None:
            scores = scores.view(B, NUM_HEADS, S, S)
            scores = scores + mask.unsqueeze(1)  # broadcast [B,1,S,S] → [B,H,S,S]
            scores = scores.view(B * NUM_HEADS, S, S)
        return scores


class MultiHeadAttnWithMask(nn.Module):
    """完整多头注意力（含 mask）。"""

    def __init__(self):
        super().__init__()
        self.q_proj = _npu_linear(D_MODEL, D_MODEL)
        self.k_proj = _npu_linear(D_MODEL, D_MODEL)
        self.v_proj = _npu_linear(D_MODEL, D_MODEL)
        self.o_proj = _npu_linear(D_MODEL, D_MODEL)

    def forward(self, x, mask=None):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2).reshape(B * NUM_HEADS, S, HEAD_DIM)
        k = self.k_proj(x).view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2).reshape(B * NUM_HEADS, S, HEAD_DIM)
        v = self.v_proj(x).view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2).reshape(B * NUM_HEADS, S, HEAD_DIM)
        scores = torch.bmm(q, k.transpose(1, 2))
        if mask is not None:
            scores = scores.view(B, NUM_HEADS, S, S)
            scores = scores + mask.unsqueeze(1)
            scores = scores.view(B * NUM_HEADS, S, S)
        attn = torch.softmax(scores, dim=-1)
        out = torch.bmm(attn, v)
        out = out.view(B, NUM_HEADS, S, HEAD_DIM).transpose(1, 2).reshape(B, S, D_MODEL)
        return self.o_proj(out)


class EncoderLayerModel(nn.Module):
    """单层 Encoder：多头注意力 + 残差 + LayerNorm + FFN。"""

    def __init__(self):
        super().__init__()
        from torch2c.integration.demo.encoder_model import EncoderLayer
        self.layer = EncoderLayer(D_MODEL, DIM_FF, NUM_HEADS, precision="mixed")

    def forward(self, x, mask=None):
        return self.layer(x, mask)


# ---- 测试用例 ----


class TestSingleHeadProj:
    """单头投影：view/transpose/reshape 不改变数据布局（1 head）。"""

    def test_single_head_proj(self, output_dir):
        torch.manual_seed(100)
        model = SingleHeadProj()
        model.eval()
        x = torch.randn(1, 32, D_MODEL)

        compile(model=model, dummy_input=x, config_dir=_CONFIG_DIR,
                output_dir=output_dir, atol=5e-3)

        r = _validate_c(output_dir)
        assert r["passed"], f"单头投影失败:\n{r['stdout']}"


class TestMultiHeadProj:
    """多头投影：view [B,S,H,Dh] → transpose(1,2) → reshape [B*H,S,Dh]。"""

    def test_multi_head_proj(self, output_dir):
        torch.manual_seed(101)
        model = MultiHeadProj()
        model.eval()
        x = torch.randn(1, 32, D_MODEL)

        compile(model=model, dummy_input=x, config_dir=_CONFIG_DIR,
                output_dir=output_dir, atol=5e-3)

        r = _validate_c(output_dir)
        assert r["passed"], f"多头投影失败:\n{r['stdout']}"


class TestMultiHeadBmm:
    """多头 Q@K^T：bmm batch 维度 = num_heads。"""

    def test_multi_head_bmm(self, output_dir):
        torch.manual_seed(102)
        model = MultiHeadBmm()
        model.eval()
        x = torch.randn(1, 32, D_MODEL)

        # bmm 累积误差较大，放宽到 0.03
        compile(model=model, dummy_input=x, config_dir=_CONFIG_DIR,
                output_dir=output_dir, atol=3e-2)

        r = _validate_c(output_dir)
        assert r["passed"], f"多头 bmm 失败:\n{r['stdout']}"


class TestMultiHeadAttnNoMask:
    """完整多头注意力（无 mask）：投影 + bmm + softmax + bmm + 输出投影。"""

    def test_attn_no_mask(self, output_dir):
        torch.manual_seed(103)
        model = MultiHeadAttnNoMask()
        model.eval()
        x = torch.randn(1, 32, D_MODEL)

        compile(model=model, dummy_input=x, config_dir=_CONFIG_DIR,
                output_dir=output_dir, atol=5e-2)

        r = _validate_c(output_dir)
        assert r["passed"], f"多头注意力（无 mask）失败:\n{r['stdout']}"


class TestBroadcastMaskAdd:
    """广播 mask 加法：scores [B,H,S,S] + mask.unsqueeze(1) [B,1,S,S]。

    根因：aten.add.Tensor 输入 shape 不同时需要广播，
    但 vector_add 只做同 shape 逐元素加法。
    """

    def test_broadcast_mask_add(self, output_dir):
        torch.manual_seed(106)
        model = BroadcastMaskAdd()
        model.eval()
        x = torch.randn(1, 32, D_MODEL)
        mask = torch.zeros(1, 32, 32)

        compile(model=model, dummy_input=x, config_dir=_CONFIG_DIR,
                output_dir=output_dir, mask=mask, atol=5e-2)

        r = _validate_c(output_dir)
        assert r["passed"], f"广播 mask 加法失败:\n{r['stdout']}"


class TestMultiHeadAttnWithMask:
    """完整多头注意力（含 mask）：mask unsqueeze + add + softmax。"""

    def test_attn_with_mask(self, output_dir):
        torch.manual_seed(104)
        model = MultiHeadAttnWithMask()
        model.eval()
        x = torch.randn(1, 32, D_MODEL)
        mask = torch.zeros(1, 32, 32)

        compile(model=model, dummy_input=x, config_dir=_CONFIG_DIR,
                output_dir=output_dir, mask=mask, atol=5e-2)

        r = _validate_c(output_dir)
        assert r["passed"], f"多头注意力（含 mask）失败:\n{r['stdout']}"


class TestEncoderLayer:
    """单层 Encoder：多头注意力 + 残差 + LN + FFN + 残差 + LN。"""

    def test_encoder_layer(self, output_dir):
        torch.manual_seed(105)
        model = EncoderLayerModel()
        model.eval()
        x = torch.randn(1, 32, D_MODEL)
        mask = torch.zeros(1, 32, 32)

        compile(model=model, dummy_input=x, config_dir=_CONFIG_DIR,
                output_dir=output_dir, mask=mask, atol=5e-2)

        r = _validate_c(output_dir)
        assert r["passed"], f"EncoderLayer 失败:\n{r['stdout']}"
