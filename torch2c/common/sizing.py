"""sizing — 通用内存大小与对齐工具函数。

被 memory_planner、codegen、viz 等多模块使用的基础设施。
"""

from __future__ import annotations

import math

from torch2c.common.dtypes import dtype_bytes

_FRACTAL_MIN_NDIM = 2  # 分形格式 (nz/zz/nn) 最少需要 2 维

# 默认对齐表: format → dtype → (dim[-2], dim[-1])
# 与 hardware_config.yaml block_pad.alignment 保持一致
_DEFAULT_ALIGNMENT: dict[str, dict[str, tuple[int, int]]] = {
    "nd":  {"fp16": (1, 16), "fp32": (1, 16), "bf16": (1, 16),
            "int8": (1, 32), "int32": (1, 16)},
    "nz":  {"fp16": (16, 16), "fp32": (16, 16), "bf16": (16, 16),
            "int8": (32, 16), "int32": (16, 16)},
    "zz":  {"fp16": (16, 16), "fp32": (16, 16), "bf16": (16, 16),
            "int8": (16, 32), "int32": (16, 16)},
    "nn":  {"fp16": (16, 16), "fp32": (16, 16), "bf16": (16, 16),
            "int8": (32, 16), "int32": (16, 16)},
}
_FALLBACK_ALIGN = (16, 16)


def get_dim_align(fmt: str, dtype: str) -> tuple[int, int]:
    """根据 format 和 dtype 返回 (dim[-2]对齐, dim[-1]对齐)。"""
    return _DEFAULT_ALIGNMENT.get(fmt, {}).get(dtype, _FALLBACK_ALIGN)


def align_up(offset: int, alignment: int) -> int:
    """向上对齐到 alignment 的整数倍。"""
    return ((offset + alignment - 1) // alignment) * alignment


def calc_padded_size(
    shape: list[int],
    dtype: str,
    fmt: str,
    dim_align: tuple[int, int],
) -> int:
    """计算 padding 后的字节数。

    Args:
        shape: tensor shape。
        dtype: 数据类型。
        fmt: 存储格式 (nd/nz/zz/nn)。
        dim_align: (dim[-2]对齐, dim[-1]对齐)。
                   分形格式两维均按对应值对齐；
                   ND 格式对齐值为 1 时跳过该维。
    """
    elem_bytes = dtype_bytes(dtype)

    if fmt in ("nz", "zz", "nn") and len(shape) >= _FRACTAL_MIN_NDIM:
        padded = list(shape)
        padded[-2] = math.ceil(shape[-2] / dim_align[0]) * dim_align[0]
        padded[-1] = math.ceil(shape[-1] / dim_align[1]) * dim_align[1]
        num_elem = 1
        for d in padded:
            num_elem *= d
        return num_elem * elem_bytes

    # ND 或 ndim < 2：按 dim_align 对齐（值为 1 时跳过）
    if len(shape) >= _FRACTAL_MIN_NDIM:
        padded = list(shape)
        if dim_align[0] > 1:
            padded[-2] = math.ceil(shape[-2] / dim_align[0]) * dim_align[0]
        if dim_align[1] > 1:
            padded[-1] = math.ceil(shape[-1] / dim_align[1]) * dim_align[1]
        num_elem = 1
        for d in padded:
            num_elem *= d
        return num_elem * elem_bytes

    num_elem = 1
    for d in shape:
        num_elem *= d
    return num_elem * elem_bytes
