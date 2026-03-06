"""graph_viz — 算子依赖关系图（DOT + ASCII）。

节点使用 C API 算子名（npu_op），pipe 边绿色高亮。
绑定到 idma（⑤b）之后自动生成。
"""

from __future__ import annotations

import os

from npu_compiler.common import Graph, get_logger
from npu_compiler.memory_planner._utils import calc_padded_size

logger = get_logger(__name__)

# ── 工具 ──────────────────────────────────────────────────


def _human_size(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / 1024 / 1024:.1f}M"
    if n >= 1024:
        return f"{n / 1024:.1f}K"
    return f"{n}B"


def _shape_str(shape: list[int]) -> str:
    return "x".join(str(d) for d in shape)


# ── ANSI ──────────────────────────────────────────────────

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_GREEN_BG = "\033[42m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"
_RED = "\033[31m"
_BLUE = "\033[34m"
_WHITE = "\033[37m"
_GRAY = "\033[90m"

_CU_COLOR = {"cube": _RED, "vector": _BLUE, "scalar": _MAGENTA, "idma": _CYAN}

# ── DOT ───────────────────────────────────────────────────

_CU_DOT_COLOR = {
    "cube": "#E8D0A9",
    "vector": "#B8D4E3",
    "scalar": "#D4B8E3",
    "idma": "#B8E3D4",
}

_STORAGE_EDGE = {
    "pipe":  {"color": "#2E8B57", "penwidth": "3.0", "style": "bold"},
    "local": {"color": "#4169E1", "penwidth": "2.0", "style": "dashed"},
    "hbm":   {"color": "#666666", "penwidth": "1.0", "style": "solid"},
}


def render_dot(graph: Graph, cube_size: int) -> str:
    """生成 DOT 格式的算子依赖图。"""
    lines = [
        'digraph NPUGraph {',
        '  rankdir=TB;',
        '  fontname="Helvetica";',
        '  node [fontname="Helvetica", fontsize=11];',
        '  edge [fontname="Helvetica", fontsize=9];',
        '  bgcolor="#FAFAFA";',
        '',
    ]

    input_tids = [tid for tid, t in graph.tensors.items() if t.is_model_input]
    absorbed_set = {v for n in graph.nodes.values() for v in n.absorbed_inputs.values()}
    weight_tids = [tid for tid, t in graph.tensors.items()
                   if t.is_weight and tid not in absorbed_set]
    output_tids = [tid for tid, t in graph.tensors.items() if t.is_model_output]

    for tid in input_tids:
        t = graph.tensors[tid]
        label = f"{tid}\\n{_shape_str(t.shape)} {t.dtype}"
        lines.append(f'  "{tid}" [label="{label}", shape=ellipse, '
                     f'style=filled, fillcolor="#C8E6C9"];')

    for tid in weight_tids:
        t = graph.tensors[tid]
        label = f"{tid}\\n{_shape_str(t.shape)} {t.dtype} {t.format}"
        lines.append(f'  "{tid}" [label="{label}", shape=ellipse, '
                     f'style=filled, fillcolor="#FFF9C4"];')

    lines.append('')

    for nid in graph.execution_order:
        node = graph.nodes[nid]
        cu = node.compute_unit or "?"
        op_name = node.npu_op or node.op_type
        fill = _CU_DOT_COLOR.get(cu, "#EEEEEE")

        absorbed_labels = []
        for param, atid in sorted(node.absorbed_inputs.items()):
            at = graph.tensors.get(atid)
            if at:
                absorbed_labels.append(f"+{param}: {_shape_str(at.shape)}")

        label = f"{op_name}\\n[{cu}]"
        if absorbed_labels:
            label += "\\n" + "\\n".join(absorbed_labels)

        border_color = "#2E8B57" if any(
            graph.tensors.get(tid) and graph.tensors[tid].storage == "pipe"
            for tid in node.outputs
        ) else "#333333"

        lines.append(
            f'  "{nid}" [label="{label}", shape=box, style="filled,rounded", '
            f'fillcolor="{fill}", color="{border_color}", penwidth=2];'
        )

    lines.append('')

    for tid in output_tids:
        t = graph.tensors[tid]
        label = f"{tid}\\n{_shape_str(t.shape)} {t.dtype}"
        lines.append(f'  "{tid}" [label="{label}", shape=ellipse, '
                     f'style=filled, fillcolor="#FFCDD2"];')

    lines.append('')

    for tid, t in graph.tensors.items():
        storage = t.storage or "hbm"
        edge_attrs = _STORAGE_EDGE.get(storage, _STORAGE_EDGE["hbm"])
        size = calc_padded_size(t.shape, t.dtype, t.format, cube_size)

        label_parts = [f"{_shape_str(t.shape)}"]
        if storage != "hbm":
            label_parts.append(storage.upper())
        label_parts.append(_human_size(size))
        edge_label = " ".join(label_parts)

        attr_str = ", ".join(f'{k}="{v}"' for k, v in edge_attrs.items())
        attr_str += f', label="{edge_label}"'
        if storage == "pipe":
            attr_str += ', fontcolor="#2E8B57", fontsize=10'

        src = t.producer_node_id if t.producer_node_id else tid
        for cid in t.consumer_node_ids:
            if t.is_model_input or t.is_weight:
                lines.append(f'  "{tid}" -> "{cid}" [{attr_str}];')
            else:
                lines.append(f'  "{src}" -> "{cid}" [{attr_str}];')

        if t.is_model_output and t.producer_node_id:
            lines.append(f'  "{t.producer_node_id}" -> "{tid}" [{attr_str}];')

    lines.append('')

    lines.append('  subgraph cluster_legend {')
    lines.append('    label="Legend"; fontsize=12; style=rounded; color="#999999";')
    lines.append('    bgcolor="#F5F5F5";')
    lines.append('    node [shape=plaintext, fontsize=10];')
    lines.append('    legend [label=<')
    lines.append('      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">')
    lines.append('        <TR><TD BGCOLOR="#E8D0A9">Cube</TD>'
                 '<TD BGCOLOR="#B8D4E3">Vector</TD>'
                 '<TD BGCOLOR="#D4B8E3">Scalar</TD></TR>')
    lines.append('        <TR><TD COLSPAN="3">')
    lines.append('          <FONT COLOR="#2E8B57"><B>━━ Pipe</B></FONT>  '
                 '<FONT COLOR="#4169E1">╌╌ Local</FONT>  '
                 '<FONT COLOR="#666666">── HBM</FONT>')
    lines.append('        </TD></TR>')
    lines.append('        <TR><TD BGCOLOR="#C8E6C9">Input</TD>'
                 '<TD BGCOLOR="#FFF9C4">Weight</TD>'
                 '<TD BGCOLOR="#FFCDD2">Output</TD></TR>')
    lines.append('      </TABLE>>];')
    lines.append('  }')

    lines.append('}')
    return "\n".join(lines)


def render_ascii(graph: Graph, cube_size: int) -> str:
    """纯终端 ASCII 依赖图渲染。"""
    lines: list[str] = []
    lines.append(f"{_BOLD}NPU Graph — C API Op Dependency{_RESET}")
    lines.append("")

    node_inputs: dict[str, list] = {}
    node_outputs: dict[str, list] = {}

    for nid in graph.execution_order:
        node = graph.nodes[nid]
        ins = []
        for tid in node.inputs:
            t = graph.tensors.get(tid)
            if not t:
                continue
            storage = t.storage or "hbm"
            src = t.producer_node_id
            src_label = graph.nodes[src].npu_op if src and src in graph.nodes else tid
            ins.append((tid, src_label, storage, t))
        for param, atid in sorted(node.absorbed_inputs.items()):
            at = graph.tensors.get(atid)
            if at:
                ins.append((atid, f"[absorbed:{param}]", at.storage or "hbm", at))
        node_inputs[nid] = ins

        outs = []
        for tid in node.outputs:
            t = graph.tensors.get(tid)
            if t:
                outs.append((tid, t.storage or "hbm", t))
        node_outputs[nid] = outs

    for idx, nid in enumerate(graph.execution_order):
        node = graph.nodes[nid]
        cu = node.compute_unit or "?"
        op = node.npu_op or node.op_type
        cu_color = _CU_COLOR.get(cu, _WHITE)

        for tid, src_label, storage, t in node_inputs[nid]:
            size = calc_padded_size(t.shape, t.dtype, t.format, cube_size)
            shape = _shape_str(t.shape)
            if storage == "pipe":
                lines.append(f"  {_GREEN}{_BOLD}{'':>4s}┃ {src_label}{_RESET}")
                lines.append(f"{_GREEN}  ║ pipe  {shape} {t.dtype} ({_human_size(size)}){_RESET}")
            elif storage == "local":
                lines.append(f"  {_BLUE}{'':>4s}┃ {src_label}{_RESET}")
                lines.append(f"{_BLUE}  ┆ local {shape} {t.dtype} ({_human_size(size)}){_RESET}")
            else:
                src_tag = ""
                if t.is_model_input:
                    src_tag = " [INPUT]"
                elif t.is_weight:
                    src_tag = " [WEIGHT]"
                lines.append(f"  {_GRAY}{'':>4s}│ {src_label}{src_tag}{_RESET}")
                lines.append(f"{_GRAY}  │ hbm   {shape} {t.dtype} {t.format} ({_human_size(size)}){_RESET}")

        absorbed_str = ""
        if node.absorbed_inputs:
            parts = []
            for p, atid in sorted(node.absorbed_inputs.items()):
                at = graph.tensors.get(atid)
                parts.append(f"+{p}:{_shape_str(at.shape)}" if at else f"+{p}")
            absorbed_str = f" {_DIM}({', '.join(parts)}){_RESET}"

        has_pipe_out = any(s == "pipe" for _, s, _ in node_outputs[nid])
        pipe_badge = f" {_GREEN_BG}{_BOLD} PIPE {_RESET}" if has_pipe_out else ""

        lines.append(f"  {'':>4s}┌{'─' * 50}┐")
        lines.append(f"  {_BOLD}t={idx:<3d}{_RESET}│ {cu_color}{_BOLD}{op:^30s}{_RESET} [{cu:^6s}]{absorbed_str}{pipe_badge}")
        lines.append(f"  {'':>4s}└{'─' * 50}┘")

        for tid, storage, t in node_outputs[nid]:
            size = calc_padded_size(t.shape, t.dtype, t.format, cube_size)
            shape = _shape_str(t.shape)
            out_tag = " [OUTPUT]" if t.is_model_output else ""
            if storage == "pipe":
                lines.append(f"  {_GREEN}{_BOLD}{'':>4s}┃ {tid} pipe {shape} ({_human_size(size)}){_RESET}")
            elif storage == "local":
                lines.append(f"  {_BLUE}{'':>4s}┆ {tid} local {shape} ({_human_size(size)}){_RESET}")
            else:
                lines.append(f"  {_GRAY}{'':>4s}│ {tid} hbm {shape} ({_human_size(size)}){out_tag}{_RESET}")

        if idx < len(graph.execution_order) - 1:
            lines.append("")

    lines.append("")
    lines.append(f"{_BOLD}Summary:{_RESET}")
    n_pipe = sum(1 for t in graph.tensors.values() if t.storage == "pipe")
    n_local = sum(1 for t in graph.tensors.values() if t.storage == "local")
    n_hbm = sum(1 for t in graph.tensors.values() if t.storage == "hbm")
    lines.append(f"  Nodes: {len(graph.nodes)}  Tensors: {len(graph.tensors)}")
    lines.append(f"  {_GREEN}pipe: {n_pipe}{_RESET}  "
                 f"{_BLUE}local: {n_local}{_RESET}  "
                 f"{_GRAY}hbm: {n_hbm}{_RESET}")

    saved = sum(
        calc_padded_size(t.shape, t.dtype, t.format, cube_size)
        for t in graph.tensors.values() if t.storage in ("pipe", "local")
    )
    lines.append(f"  HBM bandwidth saved: ~{_human_size(saved)} "
                 f"(pipe+local skip DMA round-trip)")

    return "\n".join(lines)


# ── 对外接口（pipeline 调用）──────────────────────────────


def emit_graph_dot(graph: Graph, output_dir: str, cube_size: int) -> str:
    """生成 DOT 文件到 output_dir/viz/graph.dot，返回文件路径。"""
    viz_dir = os.path.join(output_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)
    path = os.path.join(viz_dir, "graph.dot")
    dot = render_dot(graph, cube_size)
    with open(path, "w") as f:
        f.write(dot)
    logger.info("依赖图 DOT 已写入: %s", path)
    return path


def emit_graph_ascii(graph: Graph, output_dir: str, cube_size: int) -> str:
    """生成 ASCII 依赖图到 output_dir/viz/graph.txt，返回文件路径。"""
    viz_dir = os.path.join(output_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)
    path = os.path.join(viz_dir, "graph.txt")
    # ASCII 文件去除 ANSI 转义
    import re
    text = render_ascii(graph, cube_size)
    clean = re.sub(r"\033\[[0-9;]*m", "", text)
    with open(path, "w") as f:
        f.write(clean)
    logger.info("依赖图 ASCII 已写入: %s", path)
    return path
