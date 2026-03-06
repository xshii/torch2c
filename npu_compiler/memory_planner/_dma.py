"""DMA 计划生成：per-op DMA 和全局 L1 bulk DMA。"""

from __future__ import annotations

from dataclasses import dataclass, field

from npu_compiler.common import Graph, get_logger

from ._utils import align_up, calc_padded_size

logger = get_logger("memory_planner.dma")


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


@dataclass
class DmaPlan:
    """单个算子的 DMA 计划。"""

    node_id: str
    loads: list[DmaInstruction] = field(default_factory=list)
    stores: list[DmaInstruction] = field(default_factory=list)


def _get_dst_format(node, tensor_id: str, tensor) -> str:
    """根据 format_annotation 确定 DMA load 的目标格式。"""
    if node.format_annotation:
        for i, tid in enumerate(node.inputs):
            if tid == tensor_id and "inputs" in node.format_annotation:
                annots = node.format_annotation["inputs"]
                if i < len(annots) and "format" in annots[i]:
                    return annots[i]["format"]
        if tensor_id in node.absorbed_inputs.values():
            pass  # 使用 tensor 自身 format
    return tensor.format


def build_dma_plan(
    graph: Graph,
    node_id: str,
    l1_layout: dict[str, int],
    cube_size: int,
) -> DmaPlan:
    """为单个算子生成 DMA 计划。"""
    node = graph.nodes[node_id]
    plan = DmaPlan(node_id=node_id)

    load_tids = list(node.inputs)
    for _, atid in sorted(node.absorbed_inputs.items()):
        if atid not in load_tids:
            load_tids.append(atid)

    for tid in load_tids:
        t = graph.tensors.get(tid)
        # storage=local 的 tensor 已在 L1 中（由 producer 写入），跳过 DMA load
        # storage=pipe 的 tensor 走硬件直连，跳过 DMA load
        if t and t.storage in ("local", "pipe"):
            continue
        if t and t.hbm_offset is not None and tid in l1_layout:
            dst_fmt = _get_dst_format(node, tid, t)
            plan.loads.append(
                DmaInstruction(
                    op="load",
                    tensor_id=tid,
                    hbm_offset=t.hbm_offset,
                    l1_offset=l1_layout[tid],
                    size_bytes=calc_padded_size(t.shape, t.dtype, t.format, cube_size),
                    src_format=t.format,
                    dst_format=dst_fmt,
                )
            )

    for tid in node.outputs:
        t = graph.tensors.get(tid)
        # storage=local 的输出 tensor 不写回 HBM，留在 L1
        # storage=pipe 的输出 tensor 走硬件直连，不写 HBM
        if t and t.storage in ("local", "pipe"):
            continue
        if t and t.hbm_offset is not None and tid in l1_layout:
            l1_fmt = t.format
            if node.format_annotation and "outputs" in node.format_annotation:
                annots = node.format_annotation["outputs"]
                out_idx = node.outputs.index(tid)
                if out_idx < len(annots) and "format" in annots[out_idx]:
                    l1_fmt = annots[out_idx]["format"]
            plan.stores.append(
                DmaInstruction(
                    op="store",
                    tensor_id=tid,
                    hbm_offset=t.hbm_offset,
                    l1_offset=l1_layout[tid],
                    size_bytes=calc_padded_size(t.shape, t.dtype, t.format, cube_size),
                    src_format=l1_fmt,
                    dst_format=t.format,
                )
            )
    return plan


def try_global_l1_layout(
    graph: Graph,
    l1_alignment: int,
    l1_capacity: int,
    cube_size: int,
    hbm_alignment: int,
) -> bool:
    """尝试将所有 tensor 同时放入 L1。成功则填充偏移并返回 True。"""
    offset = 0
    layout: dict[str, int] = {}
    for tid, t in graph.tensors.items():
        # pipe tensor 走硬件直连，不占 L1
        if t.storage == "pipe":
            continue
        offset = align_up(offset, l1_alignment)
        layout[tid] = offset
        offset += calc_padded_size(t.shape, t.dtype, t.format, cube_size)

    if offset > l1_capacity:
        return False

    logger.info("所有张量适配 L1（%d / %d 字节），使用全局布局", offset, l1_capacity)

    hbm_offset = 0
    for tid, l1_off in layout.items():
        t = graph.tensors[tid]
        t.l1_offset = l1_off
        size = calc_padded_size(t.shape, t.dtype, t.format, cube_size)
        # storage=local/pipe 的 tensor 不分配 HBM
        if t.storage in ("local", "pipe"):
            continue
        t.hbm_size = size
        t.hbm_offset = hbm_offset
        hbm_offset = align_up(hbm_offset + size, hbm_alignment)
    return True


def build_bulk_dma(graph: Graph, cube_size: int) -> list[DmaPlan]:
    """为全局 L1 布局生成 bulk load/store DMA 计划。"""
    bulk_load = DmaPlan(node_id="__bulk_load__")
    bulk_store = DmaPlan(node_id="__bulk_store__")

    for t in graph.tensors.values():
        if t.hbm_offset is None or t.l1_offset is None:
            continue
        size = calc_padded_size(t.shape, t.dtype, t.format, cube_size)
        if t.is_model_input or t.is_weight:
            bulk_load.loads.append(
                DmaInstruction(
                    op="load",
                    tensor_id=t.id,
                    hbm_offset=t.hbm_offset,
                    l1_offset=t.l1_offset,
                    size_bytes=size,
                    src_format=t.format,
                    dst_format=t.format,
                )
            )
        if t.is_model_output:
            bulk_store.stores.append(
                DmaInstruction(
                    op="store",
                    tensor_id=t.id,
                    hbm_offset=t.hbm_offset,
                    l1_offset=t.l1_offset,
                    size_bytes=size,
                    src_format=t.format,
                    dst_format=t.format,
                )
            )

    return [bulk_load, bulk_store]
