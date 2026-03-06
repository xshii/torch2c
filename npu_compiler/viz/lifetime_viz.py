"""lifetime_viz — L1 / HBM 内存生命周期图。

横轴时间（execution_order），纵轴内存地址。
绑定到 memory_planner（⑦）之后自动生成。
"""

from __future__ import annotations

import math
import os
import re

from npu_compiler.common import Graph, get_logger
from npu_compiler.memory_planner._utils import align_up, calc_padded_size
from npu_compiler.memory_planner.memory_planner import _analyze_l1_lifetimes, _analyze_lifetimes

logger = get_logger(__name__)

# ── ANSI ──────────────────────────────────────────────────

_COLORS = [
    "\033[41m", "\033[42m", "\033[43m", "\033[44m",
    "\033[45m", "\033[46m", "\033[47m", "\033[100m",
    "\033[101m", "\033[102m", "\033[103m", "\033[104m",
]
_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_PIPE_MARK = "\033[36m~\033[0m"

STORAGE_SYMBOLS = {"hbm": "#", "local": "=", "pipe": "~"}


def _human_size(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / 1024 / 1024:.1f}M"
    if n >= 1024:
        return f"{n / 1024:.1f}K"
    return str(n)


# ── 核心渲染 ──────────────────────────────────────────────


def render_ascii(
    graph: Graph,
    cube_size: int,
    mode: str = "l1",
    col_width: int = 14,
    max_rows: int = 40,
) -> str:
    """渲染 ASCII 内存生命周期图（含 ANSI 颜色）。"""
    order = graph.execution_order
    n_ops = len(order)

    if mode == "l1":
        lifetimes = _analyze_l1_lifetimes(graph)
    else:
        lifetimes = _analyze_lifetimes(graph)

    blocks: list[dict] = []
    for tid, (first, last) in lifetimes.items():
        t = graph.tensors[tid]
        if mode == "l1":
            offset = t.l1_offset
            size = calc_padded_size(t.shape, t.dtype, t.format, cube_size)
        else:
            offset = t.hbm_offset
            size = t.hbm_size
        if offset is None:
            continue
        blocks.append({
            "tid": tid, "offset": offset, "size": size,
            "first": first, "last": last, "storage": t.storage,
        })

    pipe_tensors = [tid for tid, t in graph.tensors.items() if t.storage == "pipe"]

    if not blocks:
        return "(无可显示的内存块)"

    blocks.sort(key=lambda b: b["offset"])

    max_addr = max(b["offset"] + b["size"] for b in blocks)
    row_height = max(1, math.ceil(max_addr / max_rows))
    row_height = align_up(row_height, 1024)
    n_rows = math.ceil(max_addr / row_height)

    canvas = [[None] * n_ops for _ in range(n_rows)]

    color_map = {}
    for i, b in enumerate(blocks):
        color_map[b["tid"]] = i % len(_COLORS)

    for b in blocks:
        r_start = b["offset"] // row_height
        r_end = min(n_rows - 1, (b["offset"] + b["size"] - 1) // row_height)
        ci = color_map[b["tid"]]
        sym = STORAGE_SYMBOLS.get(b["storage"], "#")
        for r in range(r_start, r_end + 1):
            for c in range(b["first"], b["last"] + 1):
                canvas[r][c] = (ci, sym, b["tid"])

    lines: list[str] = []
    mem_label = "L1" if mode == "l1" else "HBM"

    lines.append(f"{_BOLD}{mem_label} Memory Lifetime (row={_human_size(row_height)}/row){_RESET}")
    lines.append("")

    header = " " * 12
    for nid in order:
        node = graph.nodes[nid]
        cu = (node.compute_unit or "?")[0].upper()
        short = nid.replace("n_", "")
        header += f"{cu}:{short}".center(col_width)
    lines.append(header)

    idx_line = " " * 12
    for i in range(n_ops):
        idx_line += f"t={i}".center(col_width)
    lines.append(idx_line)
    lines.append(" " * 12 + "-" * (col_width * n_ops))

    for r in range(n_rows - 1, -1, -1):
        addr = r * row_height
        row_str = f"{_human_size(addr):>10s} |"
        for c in range(n_ops):
            cell = canvas[r][c]
            if cell is None:
                row_str += _DIM + "." * col_width + _RESET
            else:
                ci, sym, tid = cell
                display = tid[:col_width - 2]
                padded = display.center(col_width, sym)
                row_str += _COLORS[ci] + padded + _RESET
        lines.append(row_str)

    lines.append(" " * 12 + "-" * (col_width * n_ops))

    lines.append("")
    lines.append(f"{_BOLD}Legend:{_RESET}")
    for b in blocks:
        ci = color_map[b["tid"]]
        t = graph.tensors[b["tid"]]
        tags = []
        if t.is_weight:
            tags.append("weight")
        if t.is_model_input:
            tags.append("input")
        if t.is_model_output:
            tags.append("output")
        if b["tid"] in {v for n in graph.nodes.values() for v in n.absorbed_inputs.values()}:
            tags.append("absorbed")
        tag_str = f" ({', '.join(tags)})" if tags else ""
        sz = _human_size(b["size"])
        lines.append(
            f"  {_COLORS[ci]} {b['tid']:^12s} {_RESET}"
            f"  {mem_label}={_human_size(b['offset'])}  size={sz}"
            f"  life=[{b['first']},{b['last']}]"
            f"  storage={b['storage']}{tag_str}"
        )

    if pipe_tensors:
        lines.append("")
        lines.append(f"  {_PIPE_MARK} pipe tensors (硬件直连, 不占 {mem_label}):")
        for tid in pipe_tensors:
            t = graph.tensors[tid]
            prod = t.producer_node_id or "?"
            cons = ", ".join(t.consumer_node_ids) if t.consumer_node_ids else "?"
            prod_cu = graph.nodes[prod].compute_unit if prod in graph.nodes else "?"
            cons_cid = t.consumer_node_ids[0] if t.consumer_node_ids else None
            cons_cu = graph.nodes[cons_cid].compute_unit if cons_cid and cons_cid in graph.nodes else "?"
            lines.append(f"    {tid}: {prod}({prod_cu}) -> {cons}({cons_cu})")

    return "\n".join(lines)


# ── 对外接口（pipeline 调用）──────────────────────────────


def emit_lifetime_ascii(graph: Graph, output_dir: str, cube_size: int) -> str:
    """生成 L1 + HBM 生命周期图到 output_dir/viz/，返回目录路径。"""
    viz_dir = os.path.join(output_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    for mode in ("l1", "hbm"):
        text = render_ascii(graph, cube_size, mode=mode)
        clean = re.sub(r"\033\[[0-9;]*m", "", text)
        path = os.path.join(viz_dir, f"lifetime_{mode}.txt")
        with open(path, "w") as f:
            f.write(clean)
        logger.info("内存生命周期 %s 已写入: %s", mode.upper(), path)

    return viz_dir
