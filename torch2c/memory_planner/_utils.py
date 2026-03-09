"""memory_planner 内部工具函数。"""

from __future__ import annotations

import math

from torch2c.common import dtype_bytes

_FRACTAL_MIN_NDIM = 2  # 分形格式 (nz/nc1hwc0) 最少需要 2 维


def align_up(offset: int, alignment: int) -> int:
    """向上对齐到 alignment 的整数倍。"""
    return ((offset + alignment - 1) // alignment) * alignment


def best_fit_alloc(free_blocks: list[list[int]], aligned_size: int) -> int | None:
    """从 free_blocks 找最优空闲块，返回 offset 或 None。"""
    best_idx = -1
    best_fit_size = float("inf")
    for i, (_, sz) in enumerate(free_blocks):
        if sz >= aligned_size and sz < best_fit_size:
            best_idx = i
            best_fit_size = sz

    if best_idx < 0:
        return None

    blk_off, blk_sz = free_blocks[best_idx]
    remaining = blk_sz - aligned_size
    if remaining > 0:
        free_blocks[best_idx] = [blk_off + aligned_size, remaining]
    else:
        free_blocks.pop(best_idx)
    return blk_off


def calc_padded_size(shape: list[int], dtype: str, fmt: str, cube_size: int) -> int:
    """计算 padding 后的字节数。

    分形格式 (nz/nc1hwc0) 将最后两维分别对齐到 cube_size 的整数倍，
    nd 格式直接按原始 shape 计算。
    """
    elem_bytes = dtype_bytes(dtype)
    if fmt in ("nz", "nc1hwc0") and len(shape) >= _FRACTAL_MIN_NDIM:
        padded = list(shape)
        padded[-1] = math.ceil(shape[-1] / cube_size) * cube_size
        padded[-2] = math.ceil(shape[-2] / cube_size) * cube_size
        num_elem = 1
        for d in padded:
            num_elem *= d
        return num_elem * elem_bytes
    num_elem = 1
    for d in shape:
        num_elem *= d
    return num_elem * elem_bytes
