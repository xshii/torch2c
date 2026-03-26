"""test_utils — 测试辅助工具。

GraphBuilder: Fluent API 构建测试 Graph，自动维护 producer/consumer 引用。
LAST: 哨兵值，引用上一个 op 的输出 tensor。
"""

from __future__ import annotations

from .graph_ir import Graph, Node, Tensor

# 哨兵值：在 op() 的 inputs 中使用，引用上一个 op 的输出
LAST = "_LAST_OUTPUT_"


class GraphBuilder:
    """Fluent API 构建测试 Graph，自动维护 producer/consumer 引用。

    用法::

        b = GraphBuilder()
        x = b.input([1, 32, 64], "X")
        w = b.weight([64, 64], "W")
        mm = b.op("cube_matmul", [x, w], [1, 32, 64])
        relu = b.op("vector_relu", [LAST], [1, 32, 64], compute_unit="vector")
        b.mark_output()
        g = b.build()
    """

    def __init__(self) -> None:
        self._graph = Graph()
        self._counter = 0
        self._last_output: str | None = None

    def _auto_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"

    # ---- Tensor 创建 ----

    def weight(self, shape: list[int], tid: str | None = None,
               dtype: str = "fp16", name: str | None = None) -> str:
        """添加权重 tensor，返回 tensor id。"""
        tid = tid or self._auto_id("w")
        self._graph.add_tensor(Tensor(
            id=tid, shape=list(shape), dtype=dtype,
            is_weight=True, name=name or tid,
        ))
        self._last_output = tid
        return tid

    def input(self, shape: list[int], tid: str | None = None,
              dtype: str = "fp16") -> str:
        """添加模型输入 tensor，返回 tensor id。"""
        tid = tid or self._auto_id("inp")
        self._graph.add_tensor(Tensor(
            id=tid, shape=list(shape), dtype=dtype, is_model_input=True,
        ))
        self._last_output = tid
        return tid

    def tensor(self, shape: list[int], tid: str | None = None,
               dtype: str = "fp16", **kwargs) -> str:
        """添加自由 tensor（非权重、非输入），返回 tensor id。"""
        tid = tid or self._auto_id("t")
        self._graph.add_tensor(Tensor(
            id=tid, shape=list(shape), dtype=dtype, **kwargs,
        ))
        self._last_output = tid
        return tid

    # ---- 节点创建 ----

    def op(self, npu_op: str, inputs: list[str], output_shape: list[int],
           compute_unit: str = "cube", *, nid: str | None = None,
           out_tid: str | None = None, dtype: str = "fp16",
           is_mapped: bool = True, **params) -> str:
        """添加算子节点，自动创建输出 tensor 并接线。返回输出 tensor id。

        inputs 中可使用 LAST 哨兵值引用上一个 op 的输出。
        """
        nid = nid or self._auto_id("n")
        out_tid = out_tid or f"{nid}_out"

        # 解析 LAST 哨兵
        resolved: list[str] = []
        for inp in inputs:
            if inp is LAST or inp == LAST:
                assert self._last_output is not None, "LAST used but no previous output"
                resolved.append(self._last_output)
            else:
                resolved.append(inp)

        # 创建输出 tensor
        self._graph.add_tensor(Tensor(
            id=out_tid, shape=list(output_shape), dtype=dtype,
            producer_node_id=nid,
        ))

        # 更新输入 tensor 的 consumer 列表
        for inp_tid in resolved:
            t = self._graph.get_tensor(inp_tid)
            if t is not None and nid not in t.consumer_node_ids:
                t.consumer_node_ids.append(nid)

        # 创建节点
        self._graph.add_node(Node(
            id=nid, op_type=npu_op,
            inputs=resolved, outputs=[out_tid],
            npu_op=npu_op, compute_unit=compute_unit,
            is_mapped=is_mapped, params=dict(params),
        ))

        self._last_output = out_tid
        return out_tid

    # ---- 标记 ----

    def mark_output(self, tid: str | None = None) -> None:
        """标记 tensor 为模型输出。默认标记上一个 op 的输出。"""
        tid = tid or self._last_output
        if tid is not None:
            t = self._graph.get_tensor(tid)
            if t is not None:
                t.is_model_output = True

    # ---- 构建 ----

    @property
    def last(self) -> str:
        """上一个 op 的输出 tensor id。"""
        assert self._last_output is not None, "No output yet"
        return self._last_output

    @property
    def graph(self) -> Graph:
        """直接访问正在构建的 Graph（不拷贝）。"""
        return self._graph

    def build(self) -> Graph:
        """返回构建好的 Graph。"""
        return self._graph
