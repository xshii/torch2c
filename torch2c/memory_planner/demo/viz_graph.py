"""生成算子依赖关系图（pyecharts 交互式 HTML）。

核心渲染逻辑位于 torch2c.viz.graph_viz，本文件为独立运行入口。

用法:
    python -m torch2c.memory_planner.demo.viz_graph [-o graph.html]
    python -m torch2c.memory_planner.demo.viz_graph --json graph.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from torch2c.common import HARDWARE_CONFIG_PATH, Graph, Node, Tensor, load_config
from torch2c.storage_assigner import run as run_idma
from torch2c.viz.graph_viz import render_graph

CONFIG_PATH = str(HARDWARE_CONFIG_PATH)


def _build_attention_graph() -> Graph:
    """构建简化的 Self-Attention 块，展示 pipe/local/hbm 混合场景。"""
    B, H, S, D = 1, 8, 128, 64
    hidden_shape = [B, S, H * D]
    qkv_shape = [B, H, S, D]
    score_shape = [B, H, S, S]
    bias_shape = [1, 1, H * D]
    mask_shape = [1, 1, S, S]

    g = Graph()

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

    add_t("t_scores", score_shape,
          producer_node_id="n_score_mm", consumer_node_ids=["n_mask_add"])
    g.add_node(Node(
        id="n_score_mm", op_type="cube_matmul",
        inputs=["t_q", "t_k"], outputs=["t_scores"],
        compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
    ))

    add_t("t_masked", score_shape,
          producer_node_id="n_mask_add", consumer_node_ids=["n_softmax_p1"])
    g.add_node(Node(
        id="n_mask_add", op_type="vector_add",
        inputs=["t_scores", "t_mask"], outputs=["t_masked"],
        compute_unit="vector", npu_op="vector_add", is_mapped=True,
    ))

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

    add_t("t_context", qkv_shape,
          producer_node_id="n_ctx_mm", consumer_node_ids=["n_out_add"])
    g.add_node(Node(
        id="n_ctx_mm", op_type="cube_matmul",
        inputs=["t_attn_w", "t_v"], outputs=["t_context"],
        compute_unit="cube", npu_op="cube_matmul", is_mapped=True,
    ))

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

    g = run_idma(g, {"pipe_pairs": [["cube", "vector"]]})
    return g


def main() -> None:
    parser = argparse.ArgumentParser(description="生成算子依赖图")
    parser.add_argument("-o", "--output", type=str, default="graph.html",
                        help="输出 HTML 文件路径（默认 graph.html）")
    parser.add_argument("--json", type=str, default=None,
                        help="输入 graph JSON 文件（默认用 attention 示例）")
    parser.add_argument("--pipe-pairs", type=str, default=None,
                        help='pipe 对，如 "cube:vector,vector:vector"')
    args = parser.parse_args()

    config = load_config(CONFIG_PATH)
    cube_size = config["fractal"]["cube_size"]

    if args.json:
        with open(args.json) as f:
            data = json.load(f)
        graph = Graph.from_dict(data)
        pp = []
        if args.pipe_pairs:
            for pair in args.pipe_pairs.split(","):
                a, b = pair.strip().split(":")
                pp.append([a.strip(), b.strip()])
        graph = run_idma(graph, {"pipe_pairs": pp})
    else:
        graph = _build_attention_graph()

    if args.output == "graph.html":
        out_dir = pathlib.Path(__file__).resolve().parents[3] / "output" / "viz_demo"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / "graph.html")
    else:
        output_path = args.output
    chart = render_graph(graph, cube_size)
    chart.render(output_path)
    print(f"HTML 已生成: {output_path}")
    print(f"用浏览器打开: open {output_path}")


if __name__ == "__main__":
    main()
