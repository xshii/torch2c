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
  storage         W: idma                 R: memory_planner, codegen
  name            W: graph_capture        R: codegen (weight export)
  producer_node_id W: graph_capture/decomp R: memory_planner
  consumer_node_ids W: graph_capture/decomp R: memory_planner
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field


@dataclass
class Tensor:
    """张量描述。

    dtype/format 表示 NPU 目标精度/格式（经 npu() 标注或 format_annotator 设置）。
    src_dtype 保留原始精度（来自 .pth 文件 / torch.export），
    用于 codegen 生成加载时的 dtype 转换代码。
    format 是纯 NPU 概念，PyTorch 侧始终为 nd，无需保留源值。
    """

    id: str
    shape: list[int]
    dtype: str
    format: str = "nd"
    src_dtype: str | None = None  # 原始精度（来自 .pth / torch.export）
    hbm_offset: int | None = None
    hbm_size: int | None = None
    l1_offset: int | None = None
    is_weight: bool = False
    is_model_input: bool = False
    is_model_output: bool = False
    name: str | None = None
    storage: str = "hbm"  # "hbm" | "local" | "pipe"
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
    task_id: int = 0
    dependencies: list[str] = field(default_factory=list)
    absorbed_inputs: dict = field(default_factory=dict)
    module_path: str | None = None


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
        """移除指定节点，并清理相关 tensor 的 consumer/producer 引用。"""
        node = self.nodes.pop(node_id, None)
        if node_id in self.execution_order:
            self.execution_order.remove(node_id)
        if node is None:
            return
        # 清理输入 tensor 的 consumer 引用（安全移除，可能已被上层手动清理）
        for tid in node.inputs:
            t = self.tensors.get(tid)
            if t:
                try:
                    t.consumer_node_ids.remove(node_id)
                except ValueError:
                    pass
        # 清理输出 tensor 的 producer 引用
        for tid in node.outputs:
            t = self.tensors.get(tid)
            if t and t.producer_node_id == node_id:
                t.producer_node_id = None

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

        queue = deque(sorted(nid for nid, d in in_degree.items() if d == 0))
        result: list[str] = []
        while queue:
            nid = queue.popleft()
            result.append(nid)
            for succ in sorted(adj[nid]):
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

    def format_npu_annotations(self) -> str:
        """格式化 graph 中所有 npu 标注，按 pattern 分组汇总。"""
        from .npu_annotate import format_annotation

        sections: list[str] = []

        # ---- 节点标注：按 annotation pattern 分组 ----
        annotated = 0
        pattern_groups: dict[str, list[str]] = {}  # pattern → [op_types]
        for nid in self.execution_order:
            node = self.nodes[nid]
            npu_ann = node.params.get("_npu")
            if not npu_ann:
                continue
            annotated += 1
            key = format_annotation(npu_ann)
            pattern_groups.setdefault(key, []).append(node.op_type)

        if pattern_groups:
            lines = [f"[Node annotations] ({annotated}/{len(self.nodes)} nodes)"]
            for pattern, ops in pattern_groups.items():
                # 统计各 op 出现次数
                op_counts: dict[str, int] = {}
                for op in ops:
                    short = op.rsplit(".", 1)[0] if "." in op else op
                    op_counts[short] = op_counts.get(short, 0) + 1
                op_summary = ", ".join(f"{op}×{c}" if c > 1 else op for op, c in op_counts.items())
                lines.append(f"  {len(ops):3d} nodes  {pattern}")
                lines.append(f"           ({op_summary})")
            sections.append("\n".join(lines))

        # ---- 权重标注：按 src → target 转换分组 ----
        weight_groups: dict[str, list[str]] = {}
        for t in self.tensors.values():
            if t.is_weight and t.name:
                tgt = f"{t.dtype}/{t.format}"
                key = f"{t.src_dtype} → {tgt}" if t.src_dtype and t.src_dtype != t.dtype else tgt
                weight_groups.setdefault(key, []).append(t.name)

        if weight_groups:
            total = sum(len(v) for v in weight_groups.values())
            lines = [f"[Weight tensors] ({total} total)"]
            for spec, names in weight_groups.items():
                short_names = [n.rsplit(".", 1)[-1] for n in names]
                unique = sorted(set(short_names))
                lines.append(f"  {len(names):3d} weights  {spec:20s} ({', '.join(unique)})")
            sections.append("\n".join(lines))

        # ---- 输入标注 ----
        input_lines = ["[Input tensors]"]
        for t in self.tensors.values():
            if t.is_model_input:
                tgt = f"{t.dtype}/{t.format}"
                conv = f"{t.src_dtype} → {tgt}" if t.src_dtype and t.src_dtype != t.dtype else tgt
                input_lines.append(f"  {t.id:10s} {conv}")
        if len(input_lines) > 1:
            sections.append("\n".join(input_lines))

        return "\n\n".join(sections)


# ---- Graph diff ----


def graph_diff(before: dict, after: dict) -> dict:
    """比较两个 graph.to_dict() 快照，返回差异。

    Returns:
        {"nodes_added": [...], "nodes_removed": [...],
         "nodes_changed": {nid: {field: (old, new)}},
         "tensors_added": [...], "tensors_removed": [...],
         "tensors_changed": {tid: {field: (old, new)}}}
    """

    def _diff_section(before_items: dict, after_items: dict) -> tuple[list, list, dict]:
        before_keys = set(before_items)
        after_keys = set(after_items)
        added = sorted(after_keys - before_keys)
        removed = sorted(before_keys - after_keys)
        changed: dict[str, dict] = {}
        for key in before_keys & after_keys:
            old, new = before_items[key], after_items[key]
            field_diffs = {}
            all_fields = set(old) | set(new)
            for f in all_fields:
                oval, nval = old.get(f), new.get(f)
                if oval != nval:
                    field_diffs[f] = (oval, nval)
            if field_diffs:
                changed[key] = field_diffs
        return added, removed, changed

    n_added, n_removed, n_changed = _diff_section(
        before.get("nodes", {}), after.get("nodes", {}),
    )
    t_added, t_removed, t_changed = _diff_section(
        before.get("tensors", {}), after.get("tensors", {}),
    )
    return {
        "nodes_added": n_added,
        "nodes_removed": n_removed,
        "nodes_changed": n_changed,
        "tensors_added": t_added,
        "tensors_removed": t_removed,
        "tensors_changed": t_changed,
    }
