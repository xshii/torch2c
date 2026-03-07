"""npu — 就近标注 nn.Module / Tensor 的编译属性。

用法：
    from torch2c.common import npu, npu_input, NpuSpec

    # 标注子模块
    self.linear1 = npu(nn.Linear(d, ff),
        input=NpuSpec("fp32", "nd"),
        output=NpuSpec("fp32", "nd"),
        weight=NpuSpec("fp32", "nd"),
        compute_dtype="fp32",
    )
    self.linear2 = npu(nn.Linear(ff, d),
        input=NpuSpec("fp16", "nd"),
        output=NpuSpec("fp16", "nd"),
        weight=NpuSpec("fp16", "nz"),
        compute_dtype="fp32",
    )
    self.ffn = nn.Linear(d, ff)  # 不标注，继承全局默认

    # 标注模型输入
    dummy = npu_input(torch.randn(1, 32, 256), dtype="fp16", format="nd")

graph_capture 通过 nn_module_stack 匹配 FX 节点到模块路径，
将标注写入 Node.params["_npu"]，供 format_annotator 等下游 Pass 使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


_ATTR_NAME = "_torch2c"


@dataclass(frozen=True)
class NpuSpec:
    """Tensor 的 NPU 编译属性：dtype + format 配对。"""

    dtype: str = "fp16"
    format: str = "nd"

    def __str__(self) -> str:
        return f"{self.dtype}/{self.format}"


def npu(
    module: nn.Module,
    *,
    input: NpuSpec | None = None,
    output: NpuSpec | None = None,
    weight: NpuSpec | None = None,
    compute_dtype: str | None = None,
) -> nn.Module:
    """为 nn.Module 实例附加编译标注。

    Args:
        input:  激活输入的 dtype/format。
        output: 输出的 dtype/format。
        weight: 权重的 dtype/format。
        compute_dtype: 计算精度（如 "fp32"）。

    标注通过 nn_module_stack 传播到该模块下的所有算子节点。
    """
    ann: dict[str, Any] = {}
    if input is not None:
        ann["input"] = input
    if output is not None:
        ann["output"] = output
    if weight is not None:
        ann["weight"] = weight
    if compute_dtype is not None:
        ann["compute_dtype"] = compute_dtype
    setattr(module, _ATTR_NAME, ann)
    return module


def npu_input(tensor: torch.Tensor, dtype: str = "fp16", format: str = "nd") -> torch.Tensor:
    """为输入 Tensor 附加编译标注（不影响 PyTorch 运行）。"""
    setattr(tensor, _ATTR_NAME, NpuSpec(dtype=dtype, format=format))
    return tensor


def get_npu_annotations(model: nn.Module) -> dict[str, dict[str, Any]]:
    """收集模型内所有带 _torch2c 标注的子模块，返回 {module_path: annotations}。"""
    result: dict[str, dict[str, Any]] = {}
    for name, mod in model.named_modules():
        ann = getattr(mod, _ATTR_NAME, None)
        if ann:
            result[name] = ann
    return result


def get_input_annotation(tensor: torch.Tensor) -> NpuSpec | None:
    """读取 Tensor 上的编译标注，返回 None 若未标注。"""
    return getattr(tensor, _ATTR_NAME, None)


# ---- 格式化 helper ----


def format_annotation(ann: dict[str, Any]) -> str:
    """将单个模块的标注 dict 格式化为可读字符串。

    >>> format_annotation({"input": NpuSpec("fp16", "nd"), "compute_dtype": "fp32"})
    'input=fp16/nd  compute=fp32'
    """
    parts = []
    for key in ("input", "output", "weight"):
        spec = ann.get(key)
        if spec is not None:
            parts.append(f"{key}={spec}")
    cd = ann.get("compute_dtype")
    if cd:
        parts.append(f"compute={cd}")
    return "  ".join(parts)


def format_model_annotations(model: nn.Module, inputs: list[torch.Tensor] | None = None) -> str:
    """格式化模型所有标注，返回多行字符串。"""
    lines: list[str] = []
    for path, ann in get_npu_annotations(model).items():
        lines.append(f"  {path:45s} {format_annotation(ann)}")
    if inputs:
        for i, t in enumerate(inputs):
            spec = get_input_annotation(t)
            if spec is not None:
                lines.append(f"  {'input_' + str(i):45s} {spec}")
    return "\n".join(lines)
