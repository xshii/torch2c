"""memory_planner — Pass⑦：HBM 全局规划 + L1 全局 liveness best-fit 分配。

HBM 分配逻辑见 _hbm_alloc.py，L1 分配逻辑见 _l1_alloc.py，
DMA 计划生成逻辑见 _dma.py，工具函数见 _utils.py。
"""

from __future__ import annotations

from torch2c.common import Graph, get_logger, memory_layout_enabled

from ._dma import DmaPlan, build_bulk_dma, build_dma_plan, try_global_l1_layout
from ._hbm_alloc import allocate_hbm, analyze_lifetimes
from ._l1_alloc import allocate_l1_global, build_per_op_l1_layouts

logger = get_logger("memory_planner")

def _dump_memory_layout(graph: Graph) -> None:
    """输出内存布局汇总日志：每个 tensor 的 HBM/L1 地址、大小、dtype、用途。"""
    if not memory_layout_enabled():
        return

    # 按 HBM offset 排序
    tensors = sorted(
        graph.tensors.values(),
        key=lambda t: (t.hbm_offset if t.hbm_offset is not None else 999999999,
                       t.l1_offset if t.l1_offset is not None else 999999999),
    )

    # 统计
    hbm_max = 0
    l1_max = 0
    for t in tensors:
        if t.hbm_offset is not None and t.hbm_size is not None:
            hbm_max = max(hbm_max, t.hbm_offset + t.hbm_size)
        if t.l1_offset is not None and t.hbm_size is not None:
            l1_max = max(l1_max, t.l1_offset + t.hbm_size)

    lines = [
        f"===== Memory Layout ({len(tensors)} tensors, "
        f"HBM peak={hbm_max} bytes, L1 peak={l1_max} bytes) =====",
        f"{'tensor_id':20s} {'dtype':6s} {'shape':20s} {'role':8s} "
        f"{'hbm_off':>10s} {'l1_off':>10s} {'size':>10s} {'storage':7s}",
        "-" * 100,
    ]

    for t in tensors:
        role = ""
        if t.is_weight:
            role = "weight"
        elif t.is_model_input:
            role = "input"
        elif t.is_model_output:
            role = "output"

        shape_str = str(t.shape) if t.shape else "[]"
        hbm_str = str(t.hbm_offset) if t.hbm_offset is not None else "-"
        l1_str = str(t.l1_offset) if t.l1_offset is not None else "-"
        size_str = str(t.hbm_size) if t.hbm_size is not None else "-"

        lines.append(
            f"{t.id:20s} {t.dtype:6s} {shape_str:20s} {role:8s} "
            f"{hbm_str:>10s} {l1_str:>10s} {size_str:>10s} {t.storage:7s}"
        )

    lines.append("=" * 100)
    logger.info("\n".join(lines))


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

    if not graph.execution_order:
        graph.execution_order = graph.topo_sort()

    # 尝试全局 L1 布局：如果所有 tensor 同时放得下，跳过 per-op DMA
    if try_global_l1_layout(graph, l1_align, l1_cap, cube_size, hbm_align):
        dma_plans = build_bulk_dma(graph, cube_size)
        allocated = sum(1 for t in graph.tensors.values() if t.hbm_offset is not None)
        logger.info(
            "Pass 完成（L1 全局布局）。HBM 分配: %d 个张量, DMA: bulk load/store",
            allocated,
        )
        _dump_memory_layout(graph)
        return graph, dma_plans

    # 常规路径：HBM 全局分配 + L1 全局 liveness best-fit 分配
    lifetimes = analyze_lifetimes(graph)
    reuse_count = allocate_hbm(graph, lifetimes, hbm_align, cube_size)

    # L1 全局分配
    global_l1 = allocate_l1_global(graph, l1_align, l1_cap, cube_size)
    for tid, off in global_l1.items():
        t = graph.tensors.get(tid)
        if t:
            t.l1_offset = off

    # 生成 per-op DMA 计划
    per_op_layouts = build_per_op_l1_layouts(graph, global_l1)
    dma_plans = []
    for nid, l1_layout in zip(graph.execution_order, per_op_layouts):
        plan = build_dma_plan(graph, nid, l1_layout, cube_size)
        dma_plans.append(plan)
        logger.debug("节点 %s: %d loads, %d stores", nid, len(plan.loads), len(plan.stores))

    allocated = sum(1 for t in graph.tensors.values() if t.hbm_offset is not None)
    logger.info(
        "Pass 完成。HBM 分配: %d 个张量, 复用: %d, DMA 计划: %d 个算子",
        allocated,
        reuse_count,
        len(dma_plans),
    )
    _dump_memory_layout(graph)
    return graph, dma_plans


def post_validate(graph: Graph) -> list[str]:
    """memory_planner 后的校验：有消费者或是输出的 tensor 必须有内存偏移。

    storage=local 的 tensor 不需要 HBM 偏移，只需要 L1 偏移。
    """
    errors: list[str] = []
    for t in graph.tensors.values():
        needs_mem = t.consumer_node_ids or t.is_model_output
        if not needs_mem:
            continue
        if t.storage == "pipe":
            continue
        if t.storage == "local":
            if t.l1_offset is None:
                errors.append(f"tensor {t.id} (storage=local) 缺少 l1_offset")
            continue
        if t.hbm_offset is None:
            errors.append(f"tensor {t.id} 缺少 hbm_offset")
        if t.hbm_size is None:
            errors.append(f"tensor {t.id} 缺少 hbm_size")
        if t.l1_offset is None:
            errors.append(f"tensor {t.id} 缺少 l1_offset")
    return errors
