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
from typing import ClassVar
from dataclasses import asdict, dataclass, field


@dataclass
class Tensor:
    """张量描述。

    dtype/format 表示 NPU 目标精度/格式（经 npu() 标注或 format_annotator 设置）。
    src_dtype 保留原始精度（来自 .pth 文件 / torch.export），
    用于 codegen 生成加载时的 dtype 转换代码。
    format 是纯 NPU 概念，PyTorch 侧始终为 nd，无需保留源值。
    """

    # ── graph_capture 阶段 ──
    id: str
    shape: list[int]
    dtype: str
    is_weight: bool = False
    is_model_input: bool = False
    is_model_output: bool = False
    name: str | None = None
    producer_node_id: str | None = None
    consumer_node_ids: list[str] = field(default_factory=list)
    # ── format_annotator 阶段 ──
    format: str = "nd"
    src_dtype: str | None = None
    # ── storage_assigner 阶段 ──
    storage: str = "hbm"  # "hbm" | "local" | "pipe"
    # ── memory_planner 阶段 ──
    hbm_offset: int | None = None
    hbm_size: int | None = None
    l1_offset: int | None = None


@dataclass
class Node:
    """计算节点描述。"""

    # ── graph_capture 阶段 ──
    id: str
    op_type: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    module_path: str | None = None
    # ── op_mapping / op_decomposition 阶段 ──
    compute_unit: str | None = None
    npu_op: str | None = None
    is_mapped: bool = False
    # ── op_absorption 阶段 ──
    absorbed_inputs: dict = field(default_factory=dict)
    # ── format_annotator 阶段 ──
    format_annotation: dict | None = None
    # ── scheduler 阶段 ──
    schedule_order: int | None = None
    task_id: int = 0
    dependencies: list[str] = field(default_factory=list)


@dataclass
class DmaInstruction:
    """单条 DMA 搬运指令。"""

    op: str  # "load" | "store"
    tensor_id: str
    hbm_offset: int
    l1_offset: int
    size_bytes: int
    src_format: str
    dst_format: str
    dtype: str = "fp16"
    tile_stride: int | None = None
    batch_count: int | None = None
    hbm_batch_stride: int | None = None
    l1_batch_stride: int | None = None


@dataclass
class DmaPlan:
    """单个算子的 DMA 计划。"""

    node_id: str
    loads: list[DmaInstruction] = field(default_factory=list)
    stores: list[DmaInstruction] = field(default_factory=list)
    tile_info: dict | None = None
    l1_layout: dict[str, int] | None = None


@dataclass
class Graph:
    """计算图，包含节点与张量的容器。"""

    nodes: dict[str, Node] = field(default_factory=dict)
    tensors: dict[str, Tensor] = field(default_factory=dict)
    execution_order: list[str] = field(default_factory=list)
    dma_plans: list[DmaPlan] = field(default_factory=list)

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

    # ── Pass 阶段契约校验 ──

    STAGE_CONTRACTS: ClassVar[dict[str, dict]] = {
        "graph_capture": {
            "node_required": ["id", "op_type"],
            "tensor_required": ["id", "shape", "dtype"],
        },
        "op_mapping": {
            "node_required": ["compute_unit", "npu_op"],
            "node_bool_true": ["is_mapped"],
        },
        "format_annotator": {
            "node_required": ["format_annotation"],
            "tensor_required": ["format"],
        },
        "scheduler": {
            "node_required": ["schedule_order"],
        },
        "memory_planner": {
            "tensor_hbm_required": ["hbm_offset", "hbm_size", "l1_offset"],
        },
    }

    def validate_stage(self, stage: str) -> list[str]:
        """校验指定 pass 完成后的字段契约。

        Args:
            stage: pass 名称（如 "op_mapping", "memory_planner"）。

        Returns:
            违反契约的错误信息列表（空 = 通过）。
        """
        contract = self.STAGE_CONTRACTS.get(stage)
        if not contract:
            return []

        errors: list[str] = []

        node_req = contract.get("node_required", [])
        node_bool = contract.get("node_bool_true", [])
        for nid, node in self.nodes.items():
            for fld in node_req:
                if getattr(node, fld, None) is None:
                    errors.append(f"[{stage}] node {nid}: {fld} 未设置")
            for fld in node_bool:
                if not getattr(node, fld, False):
                    errors.append(f"[{stage}] node {nid}: {fld} 应为 True")

        tensor_req = contract.get("tensor_required", [])
        tensor_hbm_req = contract.get("tensor_hbm_required", [])
        for tid, t in self.tensors.items():
            for fld in tensor_req:
                if getattr(t, fld, None) is None:
                    errors.append(f"[{stage}] tensor {tid}: {fld} 未设置")
            if tensor_hbm_req and t.storage not in ("local", "pipe"):
                needs_mem = t.consumer_node_ids or t.is_model_output
                if needs_mem:
                    for fld in tensor_hbm_req:
                        if getattr(t, fld, None) is None:
                            errors.append(f"[{stage}] tensor {tid}: {fld} 未设置")

        return errors

    # ---- 序列化 ----

    def to_dict(self) -> dict:
        """将图序列化为普通字典。"""
        d: dict = {
            "nodes": {nid: asdict(n) for nid, n in self.nodes.items()},
            "tensors": {tid: asdict(t) for tid, t in self.tensors.items()},
            "execution_order": list(self.execution_order),
        }
        if self.dma_plans:
            d["dma_plans"] = [asdict(dp) for dp in self.dma_plans]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Graph:
        """从字典反序列化为 Graph。"""
        g = cls()
        for tid, td in data.get("tensors", {}).items():
            g.tensors[tid] = Tensor(**td)
        for nid, nd in data.get("nodes", {}).items():
            g.nodes[nid] = Node(**nd)
        g.execution_order = list(data.get("execution_order", []))
        for dp_dict in data.get("dma_plans", []):
            loads = [DmaInstruction(**ld) for ld in dp_dict.get("loads", [])]
            stores = [DmaInstruction(**st) for st in dp_dict.get("stores", [])]
            rest = {k: v for k, v in dp_dict.items() if k not in ("loads", "stores")}
            g.dma_plans.append(DmaPlan(**rest, loads=loads, stores=stores))
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
