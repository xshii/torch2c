"""可视化 L1 / HBM 内存生命周期：横轴时间（execution_order），纵轴地址。

用法:
    python -m npu_compiler.memory_planner.demo.viz_lifetime [--hbm] [--json graph.json]

默认使用内置的 matmul+bias 吸收算子示例图。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from npu_compiler.common import Graph, Node, Tensor, load_config
from npu_compiler.idma import run as run_idma
from npu_compiler.memory_planner import run as run_memory_planner
from npu_compiler.memory_planner._utils import align_up, calc_padded_size
from npu_compiler.memory_planner.memory_planner import _analyze_l1_lifetimes, _analyze_lifetimes

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "hardware_config.yaml")

# ── 颜色 ──────────────────────────────────────────────────

_COLORS = [
    "\033[41m",   # 红底
    "\033[42m",   # 绿底
    "\033[43m",   # 黄底
    "\033[44m",   # 蓝底
    "\033[45m",   # 紫底
    "\033[46m",   # 青底
    "\033[47m",   # 白底（黑字）
    "\033[100m",  # 亮灰底
    "\033[101m",  # 亮红底
    "\033[102m",  # 亮绿底
    "\033[103m",  # 亮黄底
    "\033[104m",  # 亮蓝底
]
_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"

# pipe 用虚线标识
_PIPE_MARK = "\033[36m" + "~" + _RESET  # 青色波浪

STORAGE_SYMBOLS = {"hbm": "#", "local": "=", "pipe": "~"}


def _human_size(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / 1024 / 1024:.1f}M"
    if n >= 1024:
        return f"{n / 1024:.1f}K"
    return str(n)


# ── 示例图构建 ────────────────────────────────────────────

def _build_demo_graph() -> tuple[Graph, list]:
    """构建 matmul+bias(absorbed) → add → gelu → add2 示例并运行 pipeline。"""
    shape = [1, 128, 2048]
    g = Graph()
    g.add_tensor(Tensor(id="t_input", shape=shape, dtype="fp16", format="nd",
                        is_model_input=True, consumer_node_ids=["n_matmul"]))
    g.add_tensor(Tensor(id="t_weight", shape=shape, dtype="fp16", format="nz",
                        is_weight=True, consumer_node_ids=["n_matmul"]))
    g.add_tensor(Tensor(id="t_bias", shape=[1, 1, shape[-1]], dtype="fp16", format="nd",
                        is_weight=True, consumer_node_ids=["n_matmul"]))
    g.add_tensor(Tensor(id="t_mm_out", shape=shape, dtype="fp16", format="nd",
                        producer_node_id="n_matmul", consumer_node_ids=["n_add"]))
    g.add_tensor(Tensor(id="t_mid", shape=shape, dtype="fp16", format="nd",
                        producer_node_id="n_add", consumer_node_ids=["n_gelu"]))
    g.add_tensor(Tensor(id="t_mid2", shape=shape, dtype="fp16", format="nd",
                        producer_node_id="n_gelu", consumer_node_ids=["n_add2"]))
    g.add_tensor(Tensor(id="t_out", shape=shape, dtype="fp16", format="nd",
                        producer_node_id="n_add2", is_model_output=True))
    g.add_node(Node(id="n_matmul", op_type="cube_matmul",
                    inputs=["t_input", "t_weight"], outputs=["t_mm_out"],
                    compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
                    absorbed_inputs={"bias": "t_bias"}))
    g.add_node(Node(id="n_add", op_type="vector_add",
                    inputs=["t_mm_out"], outputs=["t_mid"],
                    compute_unit="vector", npu_op="vector_add", is_mapped=True))
    g.add_node(Node(id="n_gelu", op_type="vector_gelu",
                    inputs=["t_mid"], outputs=["t_mid2"],
                    compute_unit="vector", npu_op="vector_gelu", is_mapped=True))
    g.add_node(Node(id="n_add2", op_type="vector_add",
                    inputs=["t_mid2"], outputs=["t_out"],
                    compute_unit="vector", npu_op="vector_add", is_mapped=True))
    g.execution_order = ["n_matmul", "n_add", "n_gelu", "n_add2"]

    # 运行 idma + memory_planner
    g = run_idma(g, {"pipe_pairs": [["cube", "vector"]]})
    config = load_config(CONFIG_PATH)
    g, plans = run_memory_planner(g, config)
    return g, plans


# ── ASCII 渲染 ────────────────────────────────────────────

def _render_ascii(
    graph: Graph,
    mode: str = "l1",
    col_width: int = 14,
    max_rows: int = 40,
) -> str:
    """渲染 ASCII 内存生命周期图。

    横轴 = execution_order 中的算子索引
    纵轴 = 内存地址（从低到高）
    """
    config = load_config(CONFIG_PATH)
    cube_size = config["fractal"]["cube_size"]

    order = graph.execution_order
    n_ops = len(order)

    # 收集需要展示的 tensor 及其 lifetime
    if mode == "l1":
        lifetimes = _analyze_l1_lifetimes(graph)
    else:
        lifetimes = _analyze_lifetimes(graph)

    # 构建 tensor 信息: (tid, offset, size, first, last, storage)
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

    # pipe tensors（不在 lifetime 中，但需要标注）
    pipe_tensors = []
    for tid, t in graph.tensors.items():
        if t.storage == "pipe":
            pipe_tensors.append(tid)

    if not blocks:
        return "(无可显示的内存块)"

    # 排序：按 offset
    blocks.sort(key=lambda b: b["offset"])

    # 计算地址范围
    max_addr = max(b["offset"] + b["size"] for b in blocks)
    row_height = max(1, math.ceil(max_addr / max_rows))
    # 对齐到好看的数字
    row_height = align_up(row_height, 1024)
    n_rows = math.ceil(max_addr / row_height)

    # 构建画布: rows x cols，每个 cell 是 (color_idx, char)
    canvas = [[None] * n_ops for _ in range(n_rows)]

    # 为每个 tensor 分配颜色
    color_map = {}
    for i, b in enumerate(blocks):
        color_map[b["tid"]] = i % len(_COLORS)

    # 填充画布
    for b in blocks:
        r_start = b["offset"] // row_height
        r_end = min(n_rows - 1, (b["offset"] + b["size"] - 1) // row_height)
        c_start = b["first"]
        c_end = b["last"]
        ci = color_map[b["tid"]]
        sym = STORAGE_SYMBOLS.get(b["storage"], "#")
        for r in range(r_start, r_end + 1):
            for c in range(c_start, c_end + 1):
                canvas[r][c] = (ci, sym, b["tid"])

    # 渲染
    lines: list[str] = []
    mem_label = "L1" if mode == "l1" else "HBM"

    # 标题
    title = f"{_BOLD}{mem_label} Memory Lifetime (row={_human_size(row_height)}/row){_RESET}"
    lines.append(title)
    lines.append("")

    # 列头（算子名称）
    header = " " * 12  # Y 轴标签宽度
    for i, nid in enumerate(order):
        node = graph.nodes[nid]
        # 取 compute_unit 首字母 + 算子短名
        cu = (node.compute_unit or "?")[0].upper()
        short = nid.replace("n_", "")
        label = f"{cu}:{short}"
        header += label.center(col_width)
    lines.append(header)

    # 时间轴索引
    idx_line = " " * 12
    for i in range(n_ops):
        idx_line += f"t={i}".center(col_width)
    lines.append(idx_line)
    lines.append(" " * 12 + "-" * (col_width * n_ops))

    # 行（从高地址到低地址）
    for r in range(n_rows - 1, -1, -1):
        addr = r * row_height
        y_label = f"{_human_size(addr):>10s} |"
        row_str = y_label
        for c in range(n_ops):
            cell = canvas[r][c]
            if cell is None:
                row_str += _DIM + "." * col_width + _RESET
            else:
                ci, sym, tid = cell
                # 居中显示 tensor 名（截断）
                display = tid[:col_width - 2]
                padded = display.center(col_width, sym)
                row_str += _COLORS[ci] + padded + _RESET
        lines.append(row_str)

    # 底部地址轴
    lines.append(" " * 12 + "-" * (col_width * n_ops))

    # 图例
    lines.append("")
    lines.append(f"{_BOLD}Legend:{_RESET}")
    legend_items = []
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
        if b["tid"] in [v for n in graph.nodes.values() for v in n.absorbed_inputs.values()]:
            tags.append("absorbed")
        tag_str = f" ({', '.join(tags)})" if tags else ""
        sz = _human_size(b["size"])
        entry = (
            f"  {_COLORS[ci]} {b['tid']:^12s} {_RESET}"
            f"  {mem_label}={_human_size(b['offset'])}  size={sz}"
            f"  life=[{b['first']},{b['last']}]"
            f"  storage={b['storage']}{tag_str}"
        )
        legend_items.append(entry)
    lines.extend(legend_items)

    # pipe tensors 补充说明
    if pipe_tensors:
        lines.append("")
        lines.append(f"  {_PIPE_MARK} pipe tensors (硬件直连, 不占 {mem_label}):")
        for tid in pipe_tensors:
            t = graph.tensors[tid]
            prod = t.producer_node_id or "?"
            cons = ", ".join(t.consumer_node_ids) if t.consumer_node_ids else "?"
            prod_cu = graph.nodes[prod].compute_unit if prod in graph.nodes else "?"
            cons_cu = graph.nodes[t.consumer_node_ids[0]].compute_unit if t.consumer_node_ids and t.consumer_node_ids[0] in graph.nodes else "?"
            lines.append(f"    {tid}: {prod}({prod_cu}) -> {cons}({cons_cu})")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="可视化内存生命周期")
    parser.add_argument("--hbm", action="store_true", help="显示 HBM（默认显示 L1）")
    parser.add_argument("--json", type=str, default=None, help="输入 graph JSON 文件")
    parser.add_argument("--width", type=int, default=14, help="每列宽度")
    args = parser.parse_args()

    if args.json:
        with open(args.json) as f:
            data = json.load(f)
        graph = Graph.from_dict(data)
        config = load_config(CONFIG_PATH)
        graph, _ = run_memory_planner(graph, config)
    else:
        graph, _ = _build_demo_graph()

    mode = "hbm" if args.hbm else "l1"
    print(_render_ascii(graph, mode=mode, col_width=args.width))

    # 也打印另一个视图
    other = "l1" if mode == "hbm" else "hbm"
    print("\n" + "=" * 80 + "\n")
    print(_render_ascii(graph, mode=other, col_width=args.width))


if __name__ == "__main__":
    main()
