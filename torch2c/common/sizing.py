"""sizing — 通用内存大小与对齐工具函数。

被 memory_planner、codegen、viz 等多模块使用的基础设施。
"""

from __future__ import annotations

import math

from torch2c.common.dtypes import dtype_bytes

_FRACTAL_MIN_NDIM = 2  # 分形格式 (nz/zz/nn) 最少需要 2 维


def align_up(offset: int, alignment: int) -> int:
    """向上对齐到 alignment 的整数倍。"""
    return ((offset + alignment - 1) // alignment) * alignment


def calc_padded_size(
    shape: list[int],
    dtype: str,
    fmt: str,
    cube_size: int,
    *,
    dim_align: tuple[int, int] | None = None,
) -> int:
    """计算 padding 后的字节数。

    Args:
        shape: tensor shape。
        dtype: 数据类型。
        fmt: 存储格式 (nd/nz/zz/nn)。
        cube_size: 分形块尺寸（兜底值，dim_align 为 None 时使用）。
        dim_align: 可选 (dim[-2]对齐, dim[-1]对齐)。
                   提供时优先于 cube_size，支持 format×dtype 精确对齐。
                   未提供时分形格式两维均按 cube_size 对齐（向后兼容）。
    """
    elem_bytes = dtype_bytes(dtype)

    if fmt in ("nz", "zz", "nn") and len(shape) >= _FRACTAL_MIN_NDIM:
        padded = list(shape)
        if dim_align is not None:
            padded[-2] = math.ceil(shape[-2] / dim_align[0]) * dim_align[0]
            padded[-1] = math.ceil(shape[-1] / dim_align[1]) * dim_align[1]
        else:
            padded[-1] = math.ceil(shape[-1] / cube_size) * cube_size
            padded[-2] = math.ceil(shape[-2] / cube_size) * cube_size
        num_elem = 1
        for d in padded:
            num_elem *= d
        return num_elem * elem_bytes

    # ND 或 ndim < 2：不做分形对齐
    if dim_align is not None and len(shape) >= _FRACTAL_MIN_NDIM:
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
