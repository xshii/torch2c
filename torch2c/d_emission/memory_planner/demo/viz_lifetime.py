"""可视化 L1 / HBM 内存生命周期（pyecharts 交互式 HTML）。

核心渲染逻辑位于 torch2c.viz.lifetime_viz，本文件为独立运行入口。

用法:
    python -m torch2c.d_emission.memory_planner.demo.viz_lifetime [-o lifetime.html]
    python -m torch2c.d_emission.memory_planner.demo.viz_lifetime --json graph.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from torch2c.common import HARDWARE_CONFIG_PATH, Graph, Node, Tensor, load_config
from torch2c.optpass.c_storage_assigner import run as run_idma
from torch2c.d_emission.memory_planner import run as run_memory_planner
from torch2c.viz.lifetime_viz import render_lifetime as _render_lifetime

CONFIG_PATH = str(HARDWARE_CONFIG_PATH)


def _build_demo_graph() -> Graph:
    """构建 matmul+bias(absorbed) → add → gelu → add2 示例。"""
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

    g = run_idma(g, {"pipe_pairs": [["cube", "vector"]]})
    config = load_config(CONFIG_PATH)
    g = run_memory_planner(g, config)
    return g


def main() -> None:
    parser = argparse.ArgumentParser(description="可视化内存生命周期")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="输出 HTML 文件前缀（生成 _l1.html 和 _hbm.html）")
    parser.add_argument("--json", type=str, default=None, help="输入 graph JSON 文件")
    args = parser.parse_args()

    config = load_config(CONFIG_PATH)
    cube_size = config["fractal"]["cube_size"]

    if args.json:
        with open(args.json) as f:
            data = json.load(f)
        graph = Graph.from_dict(data)
        graph = run_memory_planner(graph, config)
    else:
        graph = _build_demo_graph()

    if args.output:
        path = args.output
    else:
        out_dir = pathlib.Path(__file__).resolve().parents[3] / "output" / "viz_demo"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = str(out_dir / "lifetime.html")
    html = _render_lifetime(graph, cube_size, hw_config=config)
    with open(path, "w") as f:
        f.write(html)
    print(f"HTML 已生成: {path}")
    print(f"用浏览器打开: open {path}")


if __name__ == "__main__":
    main()
