"""graph_capture 内部常量和映射表。

算子规则从 integration/config/capture_rules.yaml 加载，dtype 映射为 torch 专用常量。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from torch2c.common import load_config

# ── YAML 规则加载 ──────────────────────────────────────

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "integration" / "config" / "capture_rules.yaml"
_RULES = load_config(str(_CONFIG_PATH))

PARAM_RENAMES: dict[str, dict[str, str]] = _RULES.get("param_renames", {})
INPUT_REORDER: dict[str, list[int]] = _RULES.get("input_reorder", {})
DIM_TO_SIZE_OPS: set[str] = set(_RULES.get("dim_to_size_ops", []))
PARAM_DEFAULTS: dict[str, dict[str, Any]] = _RULES.get("param_defaults", {})
WEIGHT_TRANSPOSE_INPUT: dict[str, int] = _RULES.get("weight_transpose_input", {})

# ── dtype 映射 (torch 对象，不能放 YAML) ──────────────

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
