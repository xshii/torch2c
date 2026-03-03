"""memory_planner — Pass⑦：HBM 全局规划 + L1 局部排列 + DMA 计划生成。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from npu_compiler.common import Graph, MemoryPlanError, get_logger

logger = get_logger("memory_planner")

DTYPE_BYTES = {"fp16": 2, "fp32": 4, "bf16": 2, "int8": 1, "int32": 4}


# ── 数据结构 ──────────────────────────────────────────────


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


# ── 工具函数 ──────────────────────────────────────────────


def align_up(offset: int, alignment: int) -> int:
    """向上对齐到 alignment 的整数倍。"""
    return ((offset + alignment - 1) // alignment) * alignment


def calc_padded_size(
    shape: list[int], dtype: str, fmt: str, cube_size: int
) -> int:
    """计算 padding 后的字节数。

    分形格式 (nz/nc1hwc0) 将最后两维分别对齐到 cube_size 的整数倍，
    nd 格式直接按原始 shape 计算。
    """
    elem_bytes = DTYPE_BYTES.get(dtype, 2)
    if fmt in ("nz", "nc1hwc0") and len(shape) >= 2:
        padded = list(shape)
        padded[-1] = math.ceil(shape[-1] / cube_size) * cube_size
        padded[-2] = math.ceil(shape[-2] / cube_size) * cube_size
        num_elem = 1
        for d in padded:
            num_elem *= d
        return num_elem * elem_bytes
    num_elem = 1
    for d in shape:
        num_elem *= d
    return num_elem * elem_bytes


# ── HBM 分配 ─────────────────────────────────────────────


def _analyze_lifetimes(
    graph: Graph,
) -> dict[str, tuple[int, int]]:
    """分析每个 tensor 的生命周期 (first_use, last_use)。

    对于 weight/model_input，first_use=0；对于 model_output，last_use=max_order。
    无消费者且非 model_output 的 tensor 不分配 HBM（如 layernorm mean/rstd）。
    返回 {tensor_id: (first_use, last_use)}，跳过不需要分配的 tensor。
    """
    order_map: dict[str, int] = {}
    for idx, nid in enumerate(graph.execution_order):
        order_map[nid] = idx
    max_order = len(graph.execution_order) - 1

    lifetimes: dict[str, tuple[int, int]] = {}
    for tid, t in graph.tensors.items():
        # 确定 first_use
        if t.is_weight or t.is_model_input:
            first_use = 0
        elif t.producer_node_id and t.producer_node_id in order_map:
            first_use = order_map[t.producer_node_id]
        else:
            first_use = 0

        # 确定 last_use
        if t.is_model_output:
            last_use = max_order
        elif t.consumer_node_ids:
            consumer_orders = [
                order_map[c] for c in t.consumer_node_ids if c in order_map
            ]
            if consumer_orders:
                last_use = max(consumer_orders)
            else:
                continue  # 无有效消费者，跳过
        elif t.is_weight or t.is_model_input:
            # weight/input 被所有使用它的算子消费
            last_use = max_order
        else:
            continue  # 无消费者且非 model_output，不分配

        lifetimes[tid] = (first_use, last_use)
    return lifetimes


def _allocate_hbm(
    graph: Graph,
    lifetimes: dict[str, tuple[int, int]],
    hbm_alignment: int,
    cube_size: int,
) -> None:
    """Best-fit HBM 分配，支持空间复用。"""
    # 按 first_use 排序
    sorted_tids = sorted(lifetimes.keys(), key=lambda t: lifetimes[t][0])

    # 空闲块列表: [(offset, size)]
    free_blocks: list[list[int]] = []

    for tid in sorted_tids:
        t = graph.tensors[tid]
        size = calc_padded_size(t.shape, t.dtype, t.format, cube_size)
        t.hbm_size = size
        aligned_size = align_up(size, hbm_alignment)

        # 释放已过期的 tensor（last_use < 当前 tensor 的 first_use）
        current_first = lifetimes[tid][0]
        for other_tid in sorted_tids:
            if other_tid == tid:
                continue
            other_t = graph.tensors[other_tid]
            if other_t.hbm_offset is None:
                continue
            other_last = lifetimes[other_tid][1]
            if other_last < current_first:
                blk = [other_t.hbm_offset, align_up(other_t.hbm_size, hbm_alignment)]
                if blk not in free_blocks:
                    free_blocks.append(blk)

        # Best-fit: 找最小的满足 aligned_size 的空闲块
        best_idx = -1
        best_fit_size = float("inf")
        for i, (off, sz) in enumerate(free_blocks):
            if sz >= aligned_size and sz < best_fit_size:
                best_idx = i
                best_fit_size = sz

        if best_idx >= 0:
            blk_off, blk_sz = free_blocks[best_idx]
            t.hbm_offset = blk_off
            remaining = blk_sz - aligned_size
            if remaining > 0:
                free_blocks[best_idx] = [blk_off + aligned_size, remaining]
            else:
                free_blocks.pop(best_idx)
            logger.debug("HBM 复用: %s -> offset=%d, size=%d", tid, blk_off, size)
        else:
            # 分配新空间：找已分配的最大末尾
            max_end = 0
            for other_tid2 in sorted_tids:
                ot = graph.tensors[other_tid2]
                if ot.hbm_offset is not None and ot.hbm_size is not None:
                    end = ot.hbm_offset + align_up(ot.hbm_size, hbm_alignment)
                    max_end = max(max_end, end)
            t.hbm_offset = align_up(max_end, hbm_alignment)
            logger.debug("HBM 新分配: %s -> offset=%d, size=%d", tid, t.hbm_offset, size)


# ── L1 布局 ──────────────────────────────────────────────


def _plan_l1_layout(
    graph: Graph,
    node_id: str,
    l1_alignment: int,
    l1_capacity: int,
    cube_size: int,
) -> dict[str, int]:
    """为单个算子规划 L1 布局，返回 {tensor_id: l1_offset}。"""
    node = graph.nodes[node_id]
    offset = 0
    layout: dict[str, int] = {}

    # 输入 tensor（非 weight）
    for tid in node.inputs:
        t = graph.tensors.get(tid)
        if t and not t.is_weight:
            offset = align_up(offset, l1_alignment)
            layout[tid] = offset
            offset += calc_padded_size(t.shape, t.dtype, t.format, cube_size)

    # 权重 tensor
    for tid in node.inputs:
        t = graph.tensors.get(tid)
        if t and t.is_weight:
            offset = align_up(offset, l1_alignment)
            layout[tid] = offset
            offset += calc_padded_size(t.shape, t.dtype, t.format, cube_size)

    # absorbed_inputs 中的 tensor
    for _, atid in sorted(node.absorbed_inputs.items()):
        t = graph.tensors.get(atid)
        if t:
            offset = align_up(offset, l1_alignment)
            layout[atid] = offset
            offset += calc_padded_size(t.shape, t.dtype, t.format, cube_size)

    # 输出 tensor
    for tid in node.outputs:
        t = graph.tensors.get(tid)
        if t:
            offset = align_up(offset, l1_alignment)
            layout[tid] = offset
            offset += calc_padded_size(t.shape, t.dtype, t.format, cube_size)

    total = align_up(offset, l1_alignment)
    if total > l1_capacity:
        raise MemoryPlanError(
            f"L1 溢出: 节点 {node_id} 需要 {total} 字节, 容量 {l1_capacity} 字节"
        )
    return layout


# ── DMA 计划 ─────────────────────────────────────────────


def _get_dst_format(node, tensor_id: str, tensor) -> str:
    """根据 format_annotation 确定 DMA load 的目标格式。"""
    if node.format_annotation:
        # 检查 inputs 标注
        for i, tid in enumerate(node.inputs):
            if tid == tensor_id and "inputs" in node.format_annotation:
                annots = node.format_annotation["inputs"]
                if i < len(annots) and "format" in annots[i]:
                    return annots[i]["format"]
        # 检查 absorbed_inputs
        if tensor_id in node.absorbed_inputs.values():
            pass  # 使用 tensor 自身 format
    return tensor.format


def _build_dma_plan(
    graph: Graph,
    node_id: str,
    l1_layout: dict[str, int],
    cube_size: int,
) -> DmaPlan:
    """为单个算子生成 DMA 计划。"""
    node = graph.nodes[node_id]
    plan = DmaPlan(node_id=node_id)

    # Load: 所有输入 + 权重 + absorbed_inputs
    load_tids = list(node.inputs)
    for _, atid in sorted(node.absorbed_inputs.items()):
        if atid not in load_tids:
            load_tids.append(atid)

    for tid in load_tids:
        t = graph.tensors.get(tid)
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

    # Store: 所有输出
    for tid in node.outputs:
        t = graph.tensors.get(tid)
        if t and t.hbm_offset is not None and tid in l1_layout:
            plan.stores.append(
                DmaInstruction(
                    op="store",
                    tensor_id=tid,
                    hbm_offset=t.hbm_offset,
                    l1_offset=l1_layout[tid],
                    size_bytes=calc_padded_size(t.shape, t.dtype, t.format, cube_size),
                    src_format=t.format,
                    dst_format=t.format,
                )
            )
    return plan


# ── 主入口 ───────────────────────────────────────────────


def run(graph: Graph, config: dict) -> tuple[Graph, list[DmaPlan]]:
    """内存编排主函数。

    Args:
        graph: 校验通过的 Graph IR。
        config: hardware_config.yaml 解析后的字典。

    Returns:
        (编排后的 Graph, DMA 计划列表)
    """
    logger.info("Pass 开始，输入图: %d 个节点, %d 条张量", len(graph.nodes), len(graph.tensors))

    mem = config["memory"]
    hbm_align = mem["hbm"]["alignment_bytes"]
    l1_align = mem["l1"]["alignment_bytes"]
    l1_cap = mem["l1"]["total_size_bytes"]
    cube_size = config["fractal"]["cube_size"]

    # 确保 execution_order 存在
    if not graph.execution_order:
        graph.execution_order = graph.topo_sort()

    # 1. HBM 全局分配
    lifetimes = _analyze_lifetimes(graph)
    _allocate_hbm(graph, lifetimes, hbm_align, cube_size)

    # 2. L1 布局 + DMA 计划
    dma_plans: list[DmaPlan] = []
    for nid in graph.execution_order:
        l1_layout = _plan_l1_layout(graph, nid, l1_align, l1_cap, cube_size)
        # 将 l1_offset 写回 tensor（取最后一次规划的值）
        for tid, off in l1_layout.items():
            t = graph.tensors.get(tid)
            if t:
                t.l1_offset = off
        plan = _build_dma_plan(graph, nid, l1_layout, cube_size)
        dma_plans.append(plan)
        logger.debug("节点 %s: %d loads, %d stores", nid, len(plan.loads), len(plan.stores))

    # 统计
    allocated = sum(1 for t in graph.tensors.values() if t.hbm_offset is not None)
    reused = sum(
        1 for tid in lifetimes
        if any(
            graph.tensors[tid].hbm_offset == graph.tensors[other].hbm_offset
            and tid != other
            for other in lifetimes
            if graph.tensors[other].hbm_offset is not None
        )
    )
    logger.info(
        "Pass 完成。HBM 分配: %d 个张量, 复用: %d, DMA 计划: %d 个算子",
        allocated, reused, len(dma_plans),
    )
    return graph, dma_plans
