"""DMA 计划生成：per-op DMA 和全局 L1 bulk DMA。"""

from __future__ import annotations

from torch2c.common import Graph, get_logger
from torch2c.common.graph_ir import DmaInstruction, DmaPlan

from ._utils import align_up, calc_padded_size
from torch2c.common.sizing import get_dim_align

logger = get_logger("memory_planner.dma")


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
                    size_bytes=calc_padded_size(t.shape, t.dtype, t.format, get_dim_align(t.format, t.dtype)),
                    src_format=t.format,
                    dst_format=dst_fmt,
                    dtype=t.dtype,
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
                    size_bytes=calc_padded_size(t.shape, t.dtype, t.format, get_dim_align(t.format, t.dtype)),
                    src_format=l1_fmt,
                    dst_format=t.format,
                    dtype=t.dtype,
                )
            )
    return plan


def build_dma_plan_residency(
    graph: Graph,
    node_id: str,
    l1_layout: dict[str, int],
    cube_size: int,
    l1_resident: set[str],
    l1_lifetimes: dict[str, tuple[int, int]],
    op_idx: int,
) -> DmaPlan:
    """L1 驻留感知的 DMA 计划生成。

    与 build_dma_plan 的区别：
      - Skip Load: 已在 L1 中的 tensor 不重新从 HBM 加载
      - Lazy Store: 所有 consumer 都在 L1 lifetime 内的中间 tensor 不写回 HBM

    Args:
        l1_resident: 当前已在 L1 中的 tensor id 集合（调用方维护）
        l1_lifetimes: tensor → (first_op_idx, last_op_idx) L1 生命周期
        op_idx: 当前算子在 execution_order 中的索引
    """
    node = graph.nodes[node_id]
    plan = DmaPlan(node_id=node_id)

    # ── Loads: 只加载不在 L1 中的 tensor ──
    load_tids = list(node.inputs)
    for _, atid in sorted(node.absorbed_inputs.items()):
        if atid not in load_tids:
            load_tids.append(atid)

    skipped_loads = 0
    for tid in load_tids:
        t = graph.tensors.get(tid)
        if t and t.storage in ("local", "pipe"):
            continue
        if tid in l1_resident:
            skipped_loads += 1
            continue  # 已驻留 L1，跳过
        if t and t.hbm_offset is not None and tid in l1_layout:
            dst_fmt = _get_dst_format(node, tid, t)
            plan.loads.append(
                DmaInstruction(
                    op="load",
                    tensor_id=tid,
                    hbm_offset=t.hbm_offset,
                    l1_offset=l1_layout[tid],
                    size_bytes=calc_padded_size(t.shape, t.dtype, t.format, get_dim_align(t.format, t.dtype)),
                    src_format=t.format,
                    dst_format=dst_fmt,
                    dtype=t.dtype,
                )
            )
            l1_resident.add(tid)

    # ── Stores: 只写回必须的 tensor ──
    skipped_stores = 0
    for tid in node.outputs:
        t = graph.tensors.get(tid)
        if t and t.storage in ("local", "pipe"):
            continue
        if t and t.hbm_offset is not None and tid in l1_layout:
            # 判断是否可以跳过 HBM 写回
            if not _must_writeback(t, tid, l1_lifetimes):
                skipped_stores += 1
                l1_resident.add(tid)
                continue  # lazy store — 留在 L1

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
                    size_bytes=calc_padded_size(t.shape, t.dtype, t.format, get_dim_align(t.format, t.dtype)),
                    src_format=l1_fmt,
                    dst_format=t.format,
                    dtype=t.dtype,
                )
            )
            l1_resident.add(tid)

    if skipped_loads or skipped_stores:
        logger.debug(
            "node %s: skipped %d loads, %d stores (L1 resident)",
            node_id, skipped_loads, skipped_stores,
        )
    return plan


def _must_writeback(tensor, tid: str, l1_lifetimes: dict[str, tuple[int, int]]) -> bool:
    """判断 tensor 是否必须写回 HBM。

    可跳过写回的条件：
      - 不是模型输出
      - L1 lifetime 覆盖所有 consumer（即所有 consumer 都能从 L1 访问）
    """
    if tensor.is_model_output:
        return True  # 模型输出必须在 HBM
    # 没有 L1 lifetime 信息 → 安全起见写回
    if tid not in l1_lifetimes:
        return True
    # L1 lifetime 的 last_op 就是 max(所有 consumer)
    # 如果 tensor 有 L1 lifetime，说明 L1 分配覆盖了所有使用它的 op → 可跳过
    return False


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
        offset += calc_padded_size(t.shape, t.dtype, t.format, get_dim_align(t.format, t.dtype))

    if offset > l1_capacity:
        return False

    logger.info("所有张量适配 L1（%d / %d 字节），使用全局布局", offset, l1_capacity)

    hbm_offset = 0
    for tid, l1_off in layout.items():
        t = graph.tensors[tid]
        t.l1_offset = l1_off
        size = calc_padded_size(t.shape, t.dtype, t.format, get_dim_align(t.format, t.dtype))
        # storage=local/pipe 的 tensor 不分配 HBM
        if t.storage in ("local", "pipe"):
            continue
        t.hbm_size = size
        t.hbm_offset = hbm_offset
        hbm_offset = align_up(hbm_offset + size, hbm_alignment)
    return True


def _bulk_load_dst_format(graph: Graph, tensor) -> str:
    """为 bulk load 确定 dst_format：查第一个消费者的 format_annotation。"""
    for cid in tensor.consumer_node_ids:
        consumer = graph.nodes.get(cid)
        if not consumer or not consumer.format_annotation:
            continue
        annots = consumer.format_annotation.get("inputs", [])
        for i, tid in enumerate(consumer.inputs):
            if tid == tensor.id and i < len(annots) and "format" in annots[i]:
                return annots[i]["format"]
    return tensor.format


def _bulk_store_src_format(graph: Graph, tensor) -> str:
    """为 bulk store 确定 src_format：查 producer 的 format_annotation。"""
    producer = graph.nodes.get(tensor.producer_node_id) if tensor.producer_node_id else None
    if producer and producer.format_annotation:
        annots = producer.format_annotation.get("outputs", [])
        try:
            out_idx = producer.outputs.index(tensor.id)
            if out_idx < len(annots) and "format" in annots[out_idx]:
                return annots[out_idx]["format"]
        except ValueError:
            pass
    return tensor.format


def build_bulk_dma(graph: Graph, cube_size: int) -> list[DmaPlan]:
    """为全局 L1 布局生成 bulk load/store DMA 计划。

    load: src_format = tensor.format (HBM 存储)，dst_format = 消费者 format_annotation
    store: src_format = producer format_annotation，dst_format = tensor.format (HBM 存储)
    """
    bulk_load = DmaPlan(node_id="__bulk_load__")
    bulk_store = DmaPlan(node_id="__bulk_store__")

    for t in graph.tensors.values():
        if t.hbm_offset is None or t.l1_offset is None:
            continue
        size = calc_padded_size(t.shape, t.dtype, t.format, get_dim_align(t.format, t.dtype))
        if t.is_model_input or t.is_weight:
            dst_fmt = _bulk_load_dst_format(graph, t)
            bulk_load.loads.append(
                DmaInstruction(
                    op="load",
                    tensor_id=t.id,
                    hbm_offset=t.hbm_offset,
                    l1_offset=t.l1_offset,
                    size_bytes=size,
                    src_format=t.format,
                    dst_format=dst_fmt,
                    dtype=t.dtype,
                )
            )
        if t.is_model_output:
            src_fmt = _bulk_store_src_format(graph, t)
            bulk_store.stores.append(
                DmaInstruction(
                    op="store",
                    tensor_id=t.id,
                    hbm_offset=t.hbm_offset,
                    l1_offset=t.l1_offset,
                    size_bytes=size,
                    src_format=src_fmt,
                    dst_format=t.format,
                    dtype=t.dtype,
                )
            )

    return [bulk_load, bulk_store]
