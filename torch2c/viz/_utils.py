"""viz 共享工具：颜色常量 + 格式化函数。"""

from __future__ import annotations

import os

# ── 颜色 ─────────────────────────────────────────────────

# compute_unit → 十六进制填充色
CU_COLOR = {
    "cube": "#E8D0A9",
    "vector": "#B8D4E3",
    "scalar": "#D4B8E3",
    "dma": "#E3B8B8",
    "idma": "#B8E3D4",
}

# storage → 颜色
STORAGE_COLOR = {
    "pipe": "#2E8B57",
    "local": "#4169E1",
    "hbm": "#999999",
}


# ── 格式化 ────────────────────────────────────────────────


def human_size(n: int) -> str:
    """字节数转人类可读字符串。"""
    if n >= 1024 * 1024:
        return f"{n / 1024 / 1024:.1f}M"
    if n >= 1024:
        return f"{n / 1024:.1f}K"
    return f"{n}B"


def shape_str(shape: list[int]) -> str:
    """形状列表转 1x128x512 格式字符串。"""
    return "x".join(str(d) for d in shape)


def ensure_viz_dir(output_dir: str) -> str:
    """确保 output_dir/viz/ 存在，返回 viz 目录路径。"""
    viz_dir = os.path.join(output_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)
    return viz_dir
