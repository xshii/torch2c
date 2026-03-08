"""2 层 Encoder Transformer 模型定义（demo 用）。

使用单头注意力 + LayerNorm + GELU FFN，所有算子均在已支持的 NPU 算子集合内。

支持两种精度模式：
- mixed：linear1/norm1 为 fp32 全精度，其余 fp16 IO + fp32 计算
- fp16：全部 fp16 IO + fp16 权重 + fp32 计算
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torch2c.common import NpuSpec, npu, npu_input, torch2c_config  # noqa: F401

# 常用 spec 缩写
_FP32 = NpuSpec("fp32", "nd")
_FP16 = NpuSpec("fp16", "nd")
_FP16_NZ = NpuSpec("fp16", "nz")


def _linear(d_in: int, d_out: int, *, precision: str) -> nn.Linear:
    """按精度模式包装 nn.Linear。"""
    m = nn.Linear(d_in, d_out)
    if precision == "fp16":
        return npu(m, input=_FP16, output=_FP16, weight=_FP16_NZ, compute_dtype="fp32")
    return m  # mixed 模式由调用者单独标注


class SelfAttention(nn.Module):
    """单头自注意力。"""

    def __init__(self, d_model: int, *, precision: str = "mixed"):
        super().__init__()
        self.q_proj = _linear(d_model, d_model, precision=precision)
        self.k_proj = _linear(d_model, d_model, precision=precision)
        self.v_proj = _linear(d_model, d_model, precision=precision)
        self.o_proj = _linear(d_model, d_model, precision=precision)
        if precision == "mixed":
            for proj in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
                npu(proj, input=_FP16, output=_FP16, weight=_FP16_NZ, compute_dtype="fp32")

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.bmm(q, k.transpose(1, 2))
        if mask is not None:
            scores = scores + mask
        attn = torch.softmax(scores, dim=-1)
        out = torch.bmm(attn, v)
        return self.o_proj(out)


class EncoderLayer(nn.Module):
    """单层 Encoder：SelfAttention + FFN，各带残差连接和 LayerNorm。"""

    def __init__(self, d_model: int, dim_ff: int, *, precision: str = "mixed"):
        super().__init__()
        self.self_attn = SelfAttention(d_model, precision=precision)
        if precision == "mixed":
            # linear1 + norm1: fp32 全精度（高精度路径）
            self.linear1 = npu(nn.Linear(d_model, dim_ff),
                               input=_FP32, output=_FP32, weight=_FP32, compute_dtype="fp32")
            self.norm1 = npu(nn.LayerNorm(d_model),
                             input=_FP32, output=_FP32, weight=_FP32, compute_dtype="fp32")
        else:
            # fp16: 全部 fp16
            self.linear1 = npu(nn.Linear(d_model, dim_ff),
                               input=_FP16, output=_FP16, weight=_FP16_NZ, compute_dtype="fp32")
            self.norm1 = npu(nn.LayerNorm(d_model),
                             input=_FP16, output=_FP16, weight=_FP16, compute_dtype="fp32")
        # linear2 + norm2 + activation: 两种模式都用 fp16
        self.linear2 = npu(nn.Linear(dim_ff, d_model),
                           input=_FP16, output=_FP16, weight=_FP16_NZ, compute_dtype="fp32")
        self.norm2 = npu(nn.LayerNorm(d_model),
                         input=_FP16, output=_FP16, weight=_FP16, compute_dtype="fp32")
        self.activation = npu(nn.GELU(),
                              input=_FP16, output=_FP16, compute_dtype="fp16")

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.self_attn(x, mask)
        x = self.norm1(x + attn_out)
        ff_out = self.linear2(self.activation(self.linear1(x)))
        x = self.norm2(x + ff_out)
        return x


class EncoderModel(nn.Module):
    """2 层 Encoder Transformer 模型。

    Args:
        precision: "mixed"（混合精度）或 "fp16"（全 fp16）。
    """

    def __init__(
        self, d_model: int = 256, dim_ff: int = 512,
        num_layers: int = 2, *, precision: str = "mixed",
    ):
        super().__init__()
        self.precision = precision
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, dim_ff, precision=precision) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


def make_demo_inputs(
    d_model: int = 256,
    seq_len: int = 32,
    batch: int = 1,
    *,
    with_mask: bool = True,
    weights_path: str | None = None,
) -> dict:
    """创建带标注的 demo 输入（激活 + mask + 权重路径）。

    NPU 视角的三类外部输入：
        x:            激活输入，npu_input() 标注 dtype/format
        mask:         attention mask，npu_input() 标注 dtype/format
        weights_path: 权重文件路径（.pth），权重的 dtype/format
                      由模型定义中 npu(module, weight=NpuSpec(...)) 指定

    权重数据流：
        pth 文件 → host 加载 → HBM (dtype 转换) → DMA 搬移到 L1 (format 转换) → 计算
    """
    x = npu_input(torch.randn(batch, seq_len, d_model), dtype="fp16", format="nd")
    mask = None
    if with_mask:
        mask = npu_input(torch.zeros(batch, seq_len, seq_len), dtype="fp16", format="nd")
    return {"x": x, "mask": mask, "weights_path": weights_path}


if __name__ == "__main__":
    import sys

    from torch2c.integration import compile, inspect
    from torch2c.common import INTEGRATION_CONFIG_DIR

    model = EncoderModel(d_model=256, dim_ff=512, num_layers=2)
    inputs = make_demo_inputs(weights_path="encoder_weights.pth")

    inspect(model, inputs["x"], mask=inputs["mask"])

    if "--compile" in sys.argv:
        compile(
            model, inputs["x"],
            mask=inputs["mask"],
            config_dir=str(INTEGRATION_CONFIG_DIR),
            output_dir="output_demo",
        )
