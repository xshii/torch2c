"""graph_capture 内部常量和映射表。"""

from __future__ import annotations

from typing import Any

import torch

# graph_capture 产出的 positional param 名 → codegen 期望的 param 名
PARAM_RENAMES: dict[str, dict[str, str]] = {
    "aten.transpose.int": {"p0": "dim0", "p1": "dim1"},
    "aten._softmax.default": {"p0": "dim"},
    "aten.native_layer_norm.default": {"p1": "epsilon", "eps": "epsilon"},
}

DTYPE_MAP: dict[torch.dtype, str] = {
    torch.float16: "fp16",
    torch.float32: "fp32",
    torch.float64: "fp64",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.bool: "bool",
    torch.bfloat16: "bf16",
}

# addmm(bias, mat1, mat2) → NPU 期望 (mat1, mat2, bias)
ADDMM_REORDER: list[int] = [1, 2, 0]

# 需要将 dim 索引转换为维度大小的算子
DIM_TO_SIZE_OPS: set[str] = {"aten._softmax.default"}


def dtype_str(dt: torch.dtype) -> str:
    """将 torch dtype 转换为字符串标识。"""
    return DTYPE_MAP.get(dt, str(dt).replace("torch.", ""))


def op_name(target: Any) -> str:
    """提取 ATen 算子全称，如 'aten.mm.default'。"""
    s = str(target)
    return s[len("torch.ops."):] if s.startswith("torch.ops.") else s


def is_tensor_overload(op_name_str: str) -> bool:
    """判断算子重载是否期望全部 Tensor 参数（如 aten.mul.Tensor）。"""
    parts = op_name_str.rsplit(".", 1)
    return len(parts) >= 2 and parts[-1] == "Tensor"
