"""Graph IR 核心数据结构：Graph / Node / Tensor。

Field Ownership Table
=====================
每个字段由哪个 Pass 写入 (W)、谁读取 (R)。

Node 字段:
  id              W: graph_capture        R: all
  op_type         W: graph_capture        R: op_mapping, op_decomposition
  inputs          W: graph_capture/decomp R: all
  outputs         W: graph_capture/decomp R: all
  params          W: graph_capture        R: codegen
  compute_unit    W: op_mapping/decomp    R: scheduler, codegen
  npu_op          W: op_mapping/decomp    R: format_annotator, validator, codegen
  is_mapped       W: op_mapping/decomp    R: op_decomposition, validator
  format_annotation W: format_annotator   R: memory_planner, codegen
  schedule_order  W: scheduler            R: codegen
  dependencies    W: scheduler            R: codegen
  absorbed_inputs W: op_absorption        R: memory_planner, codegen

Tensor 字段:
  id              W: graph_capture        R: all
  shape           W: graph_capture        R: memory_planner, codegen
  dtype           W: graph_capture/fmt_ann R: memory_planner, codegen
  format          W: format_annotator     R: memory_planner, codegen
  hbm_offset      W: memory_planner       R: codegen
  hbm_size        W: memory_planner       R: codegen
  l1_offset       W: memory_planner       R: codegen
  is_weight       W: graph_capture        R: memory_planner, codegen
  is_model_input  W: graph_capture        R: memory_planner, codegen
  is_model_output W: graph_capture        R: memory_planner, codegen
  name            W: graph_capture        R: codegen (weight export)
  producer_node_id W: graph_capture/decomp R: memory_planner
  consumer_node_ids W: graph_capture/decomp R: memory_planner
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field


@dataclass
class Tensor:
    """张量描述。"""

    id: str
    shape: list[int]
    dtype: str
    format: str = "nd"
    hbm_offset: int | None = None
    hbm_size: int | None = None
    l1_offset: int | None = None
    is_weight: bool = False
    is_model_input: bool = False
    is_model_output: bool = False
    name: str | None = None
    producer_node_id: str | None = None
    consumer_node_ids: list[str] = field(default_factory=list)


@dataclass
class Node:
    """计算节点描述。"""

    id: str
    op_type: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    compute_unit: str | None = None
    npu_op: str | None = None
    is_mapped: bool = False
    format_annotation: dict | None = None
    schedule_order: int | None = None
    dependencies: list[str] = field(default_factory=list)
    absorbed_inputs: dict = field(default_factory=dict)


@dataclass
class Graph:
    """计算图，包含节点与张量的容器。"""

    nodes: dict[str, Node] = field(default_factory=dict)
    tensors: dict[str, Tensor] = field(default_factory=dict)
    execution_order: list[str] = field(default_factory=list)

    # ---- 节点操作 ----

    def add_node(self, node: Node) -> None:
        """添加节点到图中。"""
        self.nodes[node.id] = node
        if node.id not in self.execution_order:
            self.execution_order.append(node.id)

    def remove_node(self, node_id: str) -> None:
        """移除指定节点。"""
        self.nodes.pop(node_id, None)
        if node_id in self.execution_order:
            self.execution_order.remove(node_id)

    # ---- 张量操作 ----

    def add_tensor(self, tensor: Tensor) -> None:
        """添加张量到图中。"""
        self.tensors[tensor.id] = tensor

    def remove_tensor(self, tensor_id: str) -> None:
        """移除指定张量。"""
        self.tensors.pop(tensor_id, None)

    # ---- 查询 ----

    def get_node(self, node_id: str) -> Node | None:
        return self.nodes.get(node_id)

    def get_tensor(self, tensor_id: str) -> Tensor | None:
        return self.tensors.get(tensor_id)

    # ---- 拓扑排序 ----

    def topo_sort(self) -> list[str]:
        """基于张量依赖的拓扑排序，返回节点 ID 列表。"""
        # 构建 producer → consumers 的邻接表
        adj: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        in_degree: dict[str, int] = dict.fromkeys(self.nodes, 0)

        tensor_producer: dict[str, str] = {}
        for nid, node in self.nodes.items():
            for tid in node.outputs:
                tensor_producer[tid] = nid

        for nid, node in self.nodes.items():
            preds: set[str] = set()
            for tid in node.inputs:
                pred = tensor_producer.get(tid)
                if pred and pred != nid and pred not in preds:
                    preds.add(pred)
                    adj[pred].append(nid)
                    in_degree[nid] += 1

        queue = deque(nid for nid, d in in_degree.items() if d == 0)
        result: list[str] = []
        while queue:
            nid = queue.popleft()
            result.append(nid)
            for succ in adj[nid]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        return result

    # ---- 校验 ----

    def validate(self) -> list[str]:
        """校验图的完整性，返回错误列表。"""
        errors: list[str] = []
        for nid, node in self.nodes.items():
            for tid in node.inputs:
                if tid not in self.tensors:
                    errors.append(f"节点 {nid} 引用了不存在的输入张量 {tid}")
            for tid in node.outputs:
                if tid not in self.tensors:
                    errors.append(f"节点 {nid} 引用了不存在的输出张量 {tid}")
        return errors

    # ---- 序列化 ----

    def to_dict(self) -> dict:
        """将图序列化为普通字典。"""
        return {
            "nodes": {nid: asdict(n) for nid, n in self.nodes.items()},
            "tensors": {tid: asdict(t) for tid, t in self.tensors.items()},
            "execution_order": list(self.execution_order),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Graph:
        """从字典反序列化为 Graph。"""
        g = cls()
        for tid, td in data.get("tensors", {}).items():
            g.tensors[tid] = Tensor(**td)
        for nid, nd in data.get("nodes", {}).items():
            g.nodes[nid] = Node(**nd)
        g.execution_order = list(data.get("execution_order", []))
        return g

    # ---- 摘要 ----

    def summary(self) -> str:
        """返回图的文本摘要。"""
        op_counts: dict[str, int] = {}
        for node in self.nodes.values():
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        lines = [
            f"Graph: {len(self.nodes)} nodes, {len(self.tensors)} tensors",
        ]
        for op, cnt in sorted(op_counts.items()):
            lines.append(f"  {op}: {cnt}")
        return "\n".join(lines)
