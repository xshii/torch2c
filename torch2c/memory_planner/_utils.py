"""memory_planner 内部工具函数。"""

from __future__ import annotations

# align_up / calc_padded_size 已提升到 common.sizing，此处 re-export 保持兼容
from torch2c.common.sizing import align_up, calc_padded_size  # noqa: F401


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
