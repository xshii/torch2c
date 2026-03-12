"""memory_planner — Pass⑧：内存编排。

策略选择（按优先级）：
  1. strategy_bulk  — 所有 tensor 同时放 L1，bulk DMA
  2. strategy_perop — L1 liveness 复用 + per-op DMA
  3. strategy_spill — selective eviction + 按需 tiling
  4. strategy_tiled — 全量 eviction + M 维切分（最后手段）

策略实现见 _strategy.py，HBM/L1 分配见 _hbm_alloc.py / _l1_alloc.py，
DMA 计划见 _dma.py，工具函数见 _utils.py。
"""

from __future__ import annotations

from torch2c.common import Graph, MemoryPlanError, get_logger, memory_layout_enabled

from ._dma import DmaPlan
from ._strategy import strategy_bulk, strategy_perop, strategy_spill, strategy_tiled
from ._utils import calc_padded_size

logger = get_logger("memory_planner")

def _dump_memory_layout(graph: Graph, cube_size: int) -> None:
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
        if t.l1_offset is not None:
            l1_size = calc_padded_size(t.shape, t.dtype, t.format, cube_size)
            l1_max = max(l1_max, t.l1_offset + l1_size)

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
    tile_override = config.get("tile_override")

    if not graph.execution_order:
        graph.execution_order = graph.topo_sort()

    def _reset_offsets() -> None:
        for t in graph.tensors.values():
            t.hbm_offset = None
            t.hbm_size = None
            t.l1_offset = None
        for n in graph.nodes.values():
            n.params.pop("_tile_info", None)

    # 依次尝试策略，第一个成功的生效
    ok, dma_plans = strategy_bulk(graph, l1_align, l1_cap, hbm_align, cube_size)
    if not ok:
        try:
            ok, dma_plans = strategy_perop(graph, l1_align, l1_cap, hbm_align, cube_size)
        except MemoryPlanError as exc:
            logger.warning("strategy_perop 失败: %s，降级到 strategy_spill", exc)
            _reset_offsets()
            try:
                ok, dma_plans = strategy_spill(graph, l1_align, l1_cap, hbm_align, cube_size, tile_override)
            except MemoryPlanError as exc:
                logger.warning("strategy_spill 失败: %s，降级到 strategy_tiled", exc)
                _reset_offsets()
                ok, dma_plans = strategy_tiled(graph, l1_align, l1_cap, hbm_align, cube_size, tile_override)

    allocated = sum(1 for t in graph.tensors.values() if t.hbm_offset is not None)
    logger.info("Pass 完成。HBM 分配: %d 个张量, DMA 计划: %d 条", allocated, len(dma_plans))
    _dump_memory_layout(graph, cube_size)
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
