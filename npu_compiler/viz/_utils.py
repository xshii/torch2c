"""viz 共享工具：ANSI 颜色常量 + 格式化函数。"""

from __future__ import annotations

# ── ANSI 颜色 ─────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
GREEN_BG = "\033[42m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RED = "\033[31m"
BLUE = "\033[34m"
WHITE = "\033[37m"
GRAY = "\033[90m"

# 背景色列表（用于 lifetime 图）
BG_COLORS = [
    "\033[41m", "\033[42m", "\033[43m", "\033[44m",
    "\033[45m", "\033[46m", "\033[47m", "\033[100m",
    "\033[101m", "\033[102m", "\033[103m", "\033[104m",
]

# compute_unit → 前景色
CU_COLOR = {"cube": RED, "vector": BLUE, "scalar": MAGENTA, "idma": CYAN}

# compute_unit → DOT 填充色
CU_DOT_COLOR = {
    "cube": "#E8D0A9",
    "vector": "#B8D4E3",
    "scalar": "#D4B8E3",
    "idma": "#B8E3D4",
}

# storage → DOT 边样式
STORAGE_EDGE = {
    "pipe":  {"color": "#2E8B57", "penwidth": "3.0", "style": "bold"},
    "local": {"color": "#4169E1", "penwidth": "2.0", "style": "dashed"},
    "hbm":   {"color": "#666666", "penwidth": "1.0", "style": "solid"},
}

# storage → ASCII 填充符
STORAGE_SYMBOL = {"hbm": "#", "local": "=", "pipe": "~"}


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
