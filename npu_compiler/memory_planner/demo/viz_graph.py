"""生成类似 ONNX 的算子依赖关系图（DOT 格式 + ASCII fallback）。

节点使用 C API 算子名（npu_op），pipe 边用绿色高亮。

用法:
    # 输出 DOT 文件（可用 Graphviz 渲染为 SVG/PNG）
    python -m npu_compiler.memory_planner.demo.viz_graph --dot graph.dot

    # 终端 ASCII 渲染（默认）
    python -m npu_compiler.memory_planner.demo.viz_graph

    # 自定义输入
    python -m npu_compiler.memory_planner.demo.viz_graph --json graph.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from npu_compiler.common import Graph, Node, Tensor, load_config
from npu_compiler.idma import run as run_idma
from npu_compiler.memory_planner._utils import calc_padded_size

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "hardware_config.yaml")

# ── ANSI 颜色 ─────────────────────────────────────────────

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_GREEN_BG = "\033[42m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_MAGENTA = "\033[35m"
_RED = "\033[31m"
_BLUE = "\033[34m"
_WHITE = "\033[37m"
_GRAY = "\033[90m"

_CU_COLOR = {
    "cube": _RED,
    "vector": _BLUE,
    "scalar": _MAGENTA,
    "idma": _CYAN,
}


def _human_size(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / 1024 / 1024:.1f}M"
    if n >= 1024:
        return f"{n / 1024:.1f}K"
    return f"{n}B"


def _shape_str(shape: list[int]) -> str:
    return "x".join(str(d) for d in shape)


# ── 示例图构建（Self-Attention 块）────────────────────────

def _build_attention_graph() -> Graph:
    """构建简化的 Self-Attention 块，展示 pipe/local/hbm 混合场景。

    结构（吸收后）:
        hidden ──┬── [cube_matmul+bias] ─→ Q
                 ├── [cube_matmul+bias] ─→ K
                 └── [cube_matmul+bias] ─→ V
                          │
                 Q,K → [cube_matmul] → scores
                          │
              scores,mask → [vector_add] → masked  (mask 被 softmax 吸收 in real case)
                          │
                   [vector_softmax_part1] → attn_p1
                          │
                   [vector_softmax_part2] → attn_weights
                          │
             attn_weights,V → [cube_matmul] → context
                          │
                   [vector_add] → output
    """
    B, H, S, D = 1, 8, 128, 64
    hidden_shape = [B, S, H * D]       # [1, 128, 512]
    qkv_shape = [B, H, S, D]          # [1, 8, 128, 64]
    score_shape = [B, H, S, S]        # [1, 8, 128, 128]
    bias_shape = [1, 1, H * D]
    mask_shape = [1, 1, S, S]

    g = Graph()

    # ── 外部输入/权重 ──
    def add_t(tid, shape, **kw):
        fmt = kw.pop("format", "nd")
        g.add_tensor(Tensor(id=tid, shape=shape, dtype="fp16", format=fmt, **kw))

    add_t("t_hidden", hidden_shape, is_model_input=True,
          consumer_node_ids=["n_qproj", "n_kproj", "n_vproj"])
    for name in ("q", "k", "v"):
        add_t(f"t_w{name}", hidden_shape, format="nz", is_weight=True,
              consumer_node_ids=[f"n_{name}proj"])
        add_t(f"t_b{name}", bias_shape, is_weight=True,
              consumer_node_ids=[f"n_{name}proj"])

    add_t("t_mask", mask_shape, is_weight=True, consumer_node_ids=["n_mask_add"])

    # ── Q/K/V projection (cube_matmul + absorbed bias) ──
    for name in ("q", "k", "v"):
        add_t(f"t_{name}", qkv_shape,
              producer_node_id=f"n_{name}proj",
              consumer_node_ids=["n_score_mm" if name in ("q", "k") else "n_ctx_mm"])
        g.add_node(Node(
            id=f"n_{name}proj", op_type="cube_matmul",
            inputs=["t_hidden", f"t_w{name}"], outputs=[f"t_{name}"],
            compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
            absorbed_inputs={"bias": f"t_b{name}"},
        ))

    # ── scores = Q @ K^T ──
    add_t("t_scores", score_shape,
          producer_node_id="n_score_mm", consumer_node_ids=["n_mask_add"])
    g.add_node(Node(
        id="n_score_mm", op_type="cube_matmul",
        inputs=["t_q", "t_k"], outputs=["t_scores"],
        compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
    ))

    # ── masked = scores + mask ──
    add_t("t_masked", score_shape,
          producer_node_id="n_mask_add", consumer_node_ids=["n_softmax_p1"])
    g.add_node(Node(
        id="n_mask_add", op_type="vector_add",
        inputs=["t_scores", "t_mask"], outputs=["t_masked"],
        compute_unit="vector", npu_op="vector_add", is_mapped=True,
    ))

    # ── softmax part1 → part2 ──
    add_t("t_sm_p1", score_shape,
          producer_node_id="n_softmax_p1", consumer_node_ids=["n_softmax_p2"])
    g.add_node(Node(
        id="n_softmax_p1", op_type="vector_softmax_part1",
        inputs=["t_masked"], outputs=["t_sm_p1"],
        compute_unit="vector", npu_op="vector_softmax_part1", is_mapped=True,
    ))

    add_t("t_attn_w", score_shape,
          producer_node_id="n_softmax_p2", consumer_node_ids=["n_ctx_mm"])
    g.add_node(Node(
        id="n_softmax_p2", op_type="vector_softmax_part2",
        inputs=["t_sm_p1"], outputs=["t_attn_w"],
        compute_unit="vector", npu_op="vector_softmax_part2", is_mapped=True,
    ))

    # ── context = attn_weights @ V ──
    add_t("t_context", qkv_shape,
          producer_node_id="n_ctx_mm", consumer_node_ids=["n_out_add"])
    g.add_node(Node(
        id="n_ctx_mm", op_type="cube_matmul",
        inputs=["t_attn_w", "t_v"], outputs=["t_context"],
        compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
    ))

    # ── output = context + residual (简化为 add) ──
    add_t("t_output", hidden_shape,
          producer_node_id="n_out_add", is_model_output=True)
    g.add_node(Node(
        id="n_out_add", op_type="vector_add",
        inputs=["t_context"], outputs=["t_output"],
        compute_unit="vector", npu_op="vector_add", is_mapped=True,
    ))

    g.execution_order = [
        "n_qproj", "n_kproj", "n_vproj",
        "n_score_mm", "n_mask_add",
        "n_softmax_p1", "n_softmax_p2",
        "n_ctx_mm", "n_out_add",
    ]

    # 运行 idma 标记 pipe/local
    g = run_idma(g, {"pipe_pairs": [["cube", "vector"]]})
    return g


# ── DOT 生成 ──────────────────────────────────────────────

_CU_DOT_COLOR = {
    "cube": "#E8D0A9",       # warm beige
    "vector": "#B8D4E3",     # light blue
    "scalar": "#D4B8E3",     # light purple
    "idma": "#B8E3D4",       # light teal
}

_STORAGE_EDGE = {
    "pipe":  {"color": "#2E8B57", "penwidth": "3.0", "style": "bold"},
    "local": {"color": "#4169E1", "penwidth": "2.0", "style": "dashed"},
    "hbm":   {"color": "#666666", "penwidth": "1.0", "style": "solid"},
}


def generate_dot(graph: Graph) -> str:
    """生成 DOT 格式的算子依赖图。"""
    config = load_config(CONFIG_PATH)
    cube_size = config["fractal"]["cube_size"]

    lines = [
        'digraph NPUGraph {',
        '  rankdir=TB;',
        '  fontname="Helvetica";',
        '  node [fontname="Helvetica", fontsize=11];',
        '  edge [fontname="Helvetica", fontsize=9];',
        '  bgcolor="#FAFAFA";',
        '',
    ]

    # ── 输入节点（虚拟）──
    input_tids = [tid for tid, t in graph.tensors.items() if t.is_model_input]
    weight_tids = [tid for tid, t in graph.tensors.items()
                   if t.is_weight and tid not in
                   {v for n in graph.nodes.values() for v in n.absorbed_inputs.values()}]
    output_tids = [tid for tid, t in graph.tensors.items() if t.is_model_output]

    # 输入椭圆
    for tid in input_tids:
        t = graph.tensors[tid]
        label = f"{tid}\\n{_shape_str(t.shape)} {t.dtype}"
        lines.append(f'  "{tid}" [label="{label}", shape=ellipse, '
                     f'style=filled, fillcolor="#C8E6C9"];')

    # 权重椭圆（不含被吸收的）
    for tid in weight_tids:
        t = graph.tensors[tid]
        label = f"{tid}\\n{_shape_str(t.shape)} {t.dtype} {t.format}"
        lines.append(f'  "{tid}" [label="{label}", shape=ellipse, '
                     f'style=filled, fillcolor="#FFF9C4"];')

    lines.append('')

    # ── 计算节点 ──
    for nid in graph.execution_order:
        node = graph.nodes[nid]
        cu = node.compute_unit or "?"
        op_name = node.npu_op or node.op_type
        fill = _CU_DOT_COLOR.get(cu, "#EEEEEE")

        # absorbed inputs 标注
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

    # ── 输出椭圆 ──
    for tid in output_tids:
        t = graph.tensors[tid]
        label = f"{tid}\\n{_shape_str(t.shape)} {t.dtype}"
        lines.append(f'  "{tid}" [label="{label}", shape=ellipse, '
                     f'style=filled, fillcolor="#FFCDD2"];')

    lines.append('')

    # ── 边 ──
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

        # 来源 → 算子
        src = t.producer_node_id if t.producer_node_id else tid
        # 目标 → 消费算子
        for cid in t.consumer_node_ids:
            if t.is_model_input or t.is_weight:
                lines.append(f'  "{tid}" -> "{cid}" [{attr_str}];')
            else:
                lines.append(f'  "{src}" -> "{cid}" [{attr_str}];')

        # 到输出椭圆
        if t.is_model_output and t.producer_node_id:
            lines.append(f'  "{t.producer_node_id}" -> "{tid}" [{attr_str}];')

    lines.append('')

    # ── 图例 ──
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


# ── ASCII 渲染 ────────────────────────────────────────────

def _render_ascii(graph: Graph) -> str:
    """纯终端 ASCII 依赖图渲染。"""
    config = load_config(CONFIG_PATH)
    cube_size = config["fractal"]["cube_size"]

    lines: list[str] = []
    lines.append(f"{_BOLD}NPU Graph — C API Op Dependency{_RESET}")
    lines.append("")

    # 按 execution_order 逐层渲染
    # 先收集每个节点的入边和出边信息
    node_inputs: dict[str, list[str]] = {}   # nid -> [tensor descriptions]
    node_outputs: dict[str, list[str]] = {}

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
        # absorbed
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

    # 渲染
    for idx, nid in enumerate(graph.execution_order):
        node = graph.nodes[nid]
        cu = node.compute_unit or "?"
        op = node.npu_op or node.op_type
        cu_color = _CU_COLOR.get(cu, _WHITE)

        # 入边
        for tid, src_label, storage, t in node_inputs[nid]:
            size = calc_padded_size(t.shape, t.dtype, t.format, cube_size)
            shape = _shape_str(t.shape)
            if storage == "pipe":
                arrow = f"  {_GREEN}{_BOLD}{'':>4s}┃ {src_label}{_RESET}"
                detail = f"{_GREEN}  ║ pipe  {shape} {t.dtype} ({_human_size(size)}){_RESET}"
            elif storage == "local":
                arrow = f"  {_BLUE}{'':>4s}┃ {src_label}{_RESET}"
                detail = f"{_BLUE}  ┆ local {shape} {t.dtype} ({_human_size(size)}){_RESET}"
            else:
                src_tag = ""
                if t.is_model_input:
                    src_tag = " [INPUT]"
                elif t.is_weight:
                    src_tag = " [WEIGHT]"
                arrow = f"  {_GRAY}{'':>4s}│ {src_label}{src_tag}{_RESET}"
                detail = f"{_GRAY}  │ hbm   {shape} {t.dtype} {t.format} ({_human_size(size)}){_RESET}"
            lines.append(arrow)
            lines.append(detail)

        # 节点框
        absorbed_str = ""
        if node.absorbed_inputs:
            parts = []
            for p, atid in sorted(node.absorbed_inputs.items()):
                at = graph.tensors.get(atid)
                parts.append(f"+{p}:{_shape_str(at.shape)}" if at else f"+{p}")
            absorbed_str = f" {_DIM}({', '.join(parts)}){_RESET}"

        # 检查输出是否有 pipe
        has_pipe_out = any(s == "pipe" for _, s, _ in node_outputs[nid])
        pipe_badge = f" {_GREEN_BG}{_BOLD} PIPE {_RESET}" if has_pipe_out else ""

        box_top = f"  {'':>4s}{'┌' + '─' * 50 + '┐'}"
        box_mid = f"  {_BOLD}t={idx:<3d}{_RESET}│ {cu_color}{_BOLD}{op:^30s}{_RESET} [{cu:^6s}]{absorbed_str}{pipe_badge}"
        box_bot = f"  {'':>4s}{'└' + '─' * 50 + '┘'}"
        lines.append(box_top)
        lines.append(box_mid)
        lines.append(box_bot)

        # 出边
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

    # 统计
    lines.append("")
    lines.append(f"{_BOLD}Summary:{_RESET}")
    n_pipe = sum(1 for t in graph.tensors.values() if t.storage == "pipe")
    n_local = sum(1 for t in graph.tensors.values() if t.storage == "local")
    n_hbm = sum(1 for t in graph.tensors.values() if t.storage == "hbm")
    lines.append(f"  Nodes: {len(graph.nodes)}  Tensors: {len(graph.tensors)}")
    lines.append(f"  {_GREEN}pipe: {n_pipe}{_RESET}  "
                 f"{_BLUE}local: {n_local}{_RESET}  "
                 f"{_GRAY}hbm: {n_hbm}{_RESET}")

    # pipe 节省
    saved = sum(
        calc_padded_size(t.shape, t.dtype, t.format, cube_size)
        for t in graph.tensors.values() if t.storage in ("pipe", "local")
    )
    lines.append(f"  HBM bandwidth saved: ~{_human_size(saved)} "
                 f"(pipe+local skip DMA round-trip)")

    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="生成算子依赖图")
    parser.add_argument("--dot", type=str, default=None,
                        help="输出 DOT 文件路径（不指定则 ASCII 输出）")
    parser.add_argument("--json", type=str, default=None,
                        help="输入 graph JSON 文件（默认用 attention 示例）")
    parser.add_argument("--pipe-pairs", type=str, default=None,
                        help='pipe 对，如 "cube:vector,vector:vector"')
    args = parser.parse_args()

    if args.json:
        with open(args.json) as f:
            data = json.load(f)
        graph = Graph.from_dict(data)
        # 解析 pipe_pairs
        pp = []
        if args.pipe_pairs:
            for pair in args.pipe_pairs.split(","):
                a, b = pair.strip().split(":")
                pp.append([a.strip(), b.strip()])
        graph = run_idma(graph, {"pipe_pairs": pp})
    else:
        graph = _build_attention_graph()

    if args.dot:
        dot = generate_dot(graph)
        with open(args.dot, "w") as f:
            f.write(dot)
        print(f"DOT file written to: {args.dot}")
        print(f"Render with: dot -Tsvg {args.dot} -o graph.svg")
        print(f"         or: dot -Tpng {args.dot} -o graph.png")
    else:
        print(_render_ascii(graph))


if __name__ == "__main__":
    main()
