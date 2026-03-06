"""共享测试工具：graph builder 和 config 加载。"""

from __future__ import annotations

import os

from torch2c.common import Graph, Node, Tensor, load_config

# ---- 配置路径 ----

_MODULE_ROOT = os.path.dirname(os.path.dirname(__file__))
HARDWARE_CONFIG_PATH = os.path.join(_MODULE_ROOT, "memory_planner", "config", "hardware_config.yaml")


def load_hw_config() -> dict:
    """加载 memory_planner 的 hardware_config.yaml 测试配置。"""
    return load_config(HARDWARE_CONFIG_PATH)


# ---- Graph builder ----


def make_linear_chain(
    n_ops: int = 5,
    shape: list[int] | None = None,
    *,
    ops: list[tuple[str, str, str]] | None = None,
) -> Graph:
    """创建线性链测试图。

    Args:
        n_ops: 算子数量（当 ops 为 None 时使用）。
        shape: tensor 形状，默认 [1, 32, 64]。
        ops: 显式算子列表 [(node_id, op_type, compute_unit), ...]。
             若提供则忽略 n_ops。
    """
    if shape is None:
        shape = [1, 32, 64]
    if ops is None:
        ops = [
            (f"node_{i}", "vector_add", "vector")
            for i in range(n_ops)
        ]

    g = Graph()
    t_in = Tensor(
        id="tensor_input",
        shape=shape,
        dtype="fp16",
        is_model_input=True,
        consumer_node_ids=[ops[0][0]],
    )
    g.add_tensor(t_in)

    prev_tid = "tensor_input"
    for i, (nid, op_type, unit) in enumerate(ops):
        is_last = i == len(ops) - 1
        out_tid = f"tensor_{i}" if not is_last else "tensor_output"
        node = Node(
            id=nid,
            op_type=op_type,
            inputs=[prev_tid],
            outputs=[out_tid],
            compute_unit=unit,
            npu_op=op_type,
            is_mapped=True,
        )
        g.add_node(node)
        t = Tensor(
            id=out_tid,
            shape=shape,
            dtype="fp16",
            producer_node_id=nid,
            consumer_node_ids=[] if is_last else [ops[i + 1][0]],
            is_model_output=is_last,
        )
        g.add_tensor(t)
        prev_tid = out_tid

    g.execution_order = [nid for nid, _, _ in ops]
    return g
