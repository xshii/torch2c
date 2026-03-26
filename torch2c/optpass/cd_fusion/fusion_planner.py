"""fusion_planner — 激进 L1 驻留 + 线性融合链分析。

两阶段策略：
  Phase 1（激进 L1 驻留）：
    所有中间 tensor（非 model_input / model_output / weight，有 producer）
    默认标记 storage="local"，留在 L1 不回写 HBM。
    下游 memory_planner._spill_strategy 会在 L1 不够时自动降级回 hbm。

  Phase 2（线性链融合分析）：
    在 Phase 1 的基础上，识别可融合的线性算子链，标注融合组信息供 codegen 使用。
    融合条件：
    1. producer -> consumer 1:1 连接（中间 tensor 无 fan-out）
    2. consumer 除了 producer 的输出外，其他输入都是 weight/常量
    3. 组内至少有 2 个算子

输出：
- tensor.storage = "local" （所有合格中间 tensor）
- node.params["_fusion_group"] = group_id (str)
- node.params["_fusion_role"] = "head" | "middle" | "tail"
- graph 级别的 _fusion_groups metadata
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch2c.common import Graph, get_logger, calc_padded_size, get_dim_align
from torch2c.common.opt_log import log_opt

logger = get_logger(__name__)

# 不参与融合的计算单元类型（DMA 本身就是数据搬运，不应融合）
_UNFUSIBLE_UNITS = frozenset({"dma", "idma"})


@dataclass
class FusionGroup:
    """一组可融合的算子链。"""

    id: str                        # e.g. "fg_0"
    node_ids: list[str]            # ordered by topology
    internal_tensors: list[str]    # intermediate tensors (stay in L1)
    external_inputs: list[str]     # inputs from outside the group
    external_outputs: list[str]    # outputs leaving the group
    tile_dim: int = -2             # shared tiling dimension (M dim)
    estimated_dma_savings: int = 0 # bytes saved by fusion


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_tensor_fanout(graph: Graph) -> dict[str, list[str]]:
    """Map tensor_id -> list of consumer node_ids."""
    fanout: dict[str, list[str]] = {}
    for tid, t in graph.tensors.items():
        fanout[tid] = list(t.consumer_node_ids)
    return fanout


def _is_fusible_pair(
    producer_id: str,
    consumer_id: str,
    graph: Graph,
    fanout: dict[str, list[str]],
) -> bool:
    """判断 producer -> consumer 是否可融合。"""
    producer = graph.get_node(producer_id)
    consumer = graph.get_node(consumer_id)
    if producer is None or consumer is None:
        return False

    # DMA 节点不参与融合
    if (producer.compute_unit or "") in _UNFUSIBLE_UNITS:
        return False
    if (consumer.compute_unit or "") in _UNFUSIBLE_UNITS:
        return False

    # producer 必须恰好 1 个输出
    if len(producer.outputs) != 1:
        return False

    link_tid = producer.outputs[0]
    link_tensor = graph.get_tensor(link_tid)
    if link_tensor is None:
        return False

    # 中间 tensor 不能是模型输出（必须存回 HBM）
    if link_tensor.is_model_output:
        return False

    # 中间 tensor 必须仅被 consumer 消费（fan-out == 1）
    consumers = fanout.get(link_tid, [])
    if len(consumers) != 1 or consumers[0] != consumer_id:
        return False

    # consumer 的其他输入必须是 weight / model_input / absorbed
    absorbed_ids = set(consumer.absorbed_inputs.values())
    for inp_tid in consumer.inputs:
        if inp_tid == link_tid:
            continue
        inp_t = graph.get_tensor(inp_tid)
        if inp_t is None:
            return False
        if not (inp_t.is_weight or inp_t.is_model_input or inp_tid in absorbed_ids):
            return False

    # 不跨 storage 类型融合
    if link_tensor.storage not in ("hbm", "local", "pipe"):
        return False

    return True


def _find_linear_chains(
    graph: Graph, fanout: dict[str, list[str]],
) -> list[list[str]]:
    """沿 execution_order 贪心延伸线性融合链。"""
    chains: list[list[str]] = []
    in_chain: set[str] = set()

    for nid in graph.execution_order:
        if nid in in_chain:
            continue
        node = graph.get_node(nid)
        if node is None:
            continue
        if (node.compute_unit or "") in _UNFUSIBLE_UNITS:
            continue

        # 尝试从 nid 开始构建一条链
        chain = [nid]
        current = nid
        while True:
            extended = _try_extend_chain(current, graph, fanout, in_chain)
            if extended is None:
                break
            chain.append(extended)
            current = extended

        if len(chain) >= 2:
            chains.append(chain)
            in_chain.update(chain)

    return chains


def _try_extend_chain(
    current_id: str,
    graph: Graph,
    fanout: dict[str, list[str]],
    in_chain: set[str],
) -> str | None:
    """尝试找到 current 的下一个可融合节点。"""
    current = graph.get_node(current_id)
    if current is None or len(current.outputs) != 1:
        return None

    out_tid = current.outputs[0]
    consumers = fanout.get(out_tid, [])
    if len(consumers) != 1:
        return None

    candidate = consumers[0]
    if candidate in in_chain:
        return None
    if not _is_fusible_pair(current_id, candidate, graph, fanout):
        return None
    return candidate


def _compute_savings(chain: list[str], graph: Graph) -> int:
    """估算融合后节省的 DMA 字节数（中间 tensor 免 load+store）。"""
    savings = 0
    for nid in chain[:-1]:
        node = graph.get_node(nid)
        if node is None:
            continue
        for tid in node.outputs:
            t = graph.get_tensor(tid)
            if t is None:
                continue
            size = calc_padded_size(t.shape, t.dtype, t.format, get_dim_align(t.format, t.dtype))
            savings += size * 2  # saved load + store
    return savings


def _collect_internal_tensors(chain: list[str], graph: Graph) -> list[str]:
    """收集链内的中间 tensor（不含首尾的外部输入输出）。"""
    chain_set = set(chain)
    internal: list[str] = []
    for nid in chain[:-1]:
        node = graph.get_node(nid)
        if node is None:
            continue
        for tid in node.outputs:
            t = graph.get_tensor(tid)
            if t is None:
                continue
            # 所有消费者都在链内 → internal
            if all(c in chain_set for c in t.consumer_node_ids):
                internal.append(tid)
    return internal


def _collect_external_io(
    chain: list[str], internal_set: set[str], graph: Graph,
) -> tuple[list[str], list[str]]:
    """收集链的外部输入和外部输出。"""
    chain_set = set(chain)
    ext_inputs: list[str] = []
    seen_in: set[str] = set()
    for nid in chain:
        node = graph.get_node(nid)
        if node is None:
            continue
        for tid in node.inputs:
            if tid not in internal_set and tid not in seen_in:
                ext_inputs.append(tid)
                seen_in.add(tid)

    ext_outputs: list[str] = []
    tail = graph.get_node(chain[-1])
    if tail is not None:
        ext_outputs = list(tail.outputs)
    return ext_inputs, ext_outputs


def _annotate_chain(chain: list[str], group_id: str, graph: Graph) -> None:
    """为链中每个节点写入 _fusion_group 和 _fusion_role。"""
    for i, nid in enumerate(chain):
        node = graph.get_node(nid)
        if node is None:
            continue
        node.params["_fusion_group"] = group_id
        if i == 0:
            node.params["_fusion_role"] = "head"
        elif i == len(chain) - 1:
            node.params["_fusion_role"] = "tail"
        else:
            node.params["_fusion_role"] = "middle"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _promote_to_local(graph: Graph) -> int:
    """Phase 1：将 storage_assigner 未处理的中间 tensor 提升为 local。

    storage_assigner（⑤c）已经根据硬件 allowed_pairs 约束将合格的
    单消费者 tensor 标记为 local/pipe。它刻意留在 hbm 的 tensor 有
    硬件原因（计算单元对不支持 bypass），不应被覆盖。

    Phase 1 只提升以下 tensor：
      - fan-out > 1 的中间 tensor（storage_assigner 不处理多消费者）
      - 无消费者的中间 tensor（storage_assigner 不处理）
    这些 tensor storage_assigner 跳过了，但如果生命周期短，留在 L1
    仍有收益。下游 _spill_strategy 会在 L1 不够时自动降级回 hbm。

    Returns:
        标记为 local 的 tensor 数量。
    """
    count = 0
    for t in graph.tensors.values():
        if t.is_model_input or t.is_model_output or t.is_weight:
            continue
        if t.producer_node_id is None:
            continue
        if t.storage in ("pipe", "local"):
            continue  # 已有更优存储（storage_assigner 决策）
        # 只提升 fan-out > 1 的 tensor（storage_assigner 只处理 fan-out == 1）
        if len(t.consumer_node_ids) <= 1:
            continue  # fan-out ≤ 1: storage_assigner 已做决策，尊重其结果
        t.storage = "local"
        count += 1
    return count


def run(graph: Graph, config: dict) -> Graph:
    """执行融合 pass：激进 L1 驻留 + 线性链分析。

    Args:
        graph: 经过 storage_assigner 的 Graph IR。
        config: 配置字典。可选键：
            aggressive_local: bool — 是否激进标记 local，默认 True。

    Returns:
        同一 Graph 对象（原地修改）。
    """
    # Phase 1: 激进 L1 驻留
    aggressive = config.get("aggressive_local", True)
    if aggressive:
        promoted = _promote_to_local(graph)
        logger.info("fusion_planner phase1: %d 个中间 tensor 提升为 local", promoted)

    # Phase 2: 线性链融合分析
    fanout = _build_tensor_fanout(graph)
    chains = _find_linear_chains(graph, fanout)

    groups: list[FusionGroup] = []
    total_savings = 0

    for idx, chain in enumerate(chains):
        group_id = f"fg_{idx}"
        internal = _collect_internal_tensors(chain, graph)
        ext_in, ext_out = _collect_external_io(chain, set(internal), graph)
        savings = _compute_savings(chain, graph)

        group = FusionGroup(
            id=group_id,
            node_ids=chain,
            internal_tensors=internal,
            external_inputs=ext_in,
            external_outputs=ext_out,
            estimated_dma_savings=savings,
        )
        groups.append(group)
        total_savings += savings
        _annotate_chain(chain, group_id, graph)

        # 为融合组中每个节点记录 opt_log
        node_ops = []
        for nid in chain:
            n = graph.get_node(nid)
            if n:
                node_ops.append(f"{nid}({n.npu_op or n.op_type})")
        for nid in chain:
            n = graph.get_node(nid)
            if n:
                log_opt(
                    n, "fusion_planner", "算子融合",
                    f"与 {', '.join(node_ops)} 融合，"
                    f"{len(internal)} 个中间 tensor 留在 L1 不回写 HBM，"
                    f"省去 {savings} bytes DMA 搬运（约 {savings//256} cycles）",
                )

    # 在 graph.metadata 上附加融合组信息
    if groups:
        graph.metadata["fusion_groups"] = [
            {
                "id": g.id,
                "node_ids": g.node_ids,
                "internal_tensors": g.internal_tensors,
                "estimated_dma_savings": g.estimated_dma_savings,
            }
            for g in groups
        ]

    logger.info(
        "融合分析完成：%d 个融合组，预计节省 DMA %d bytes",
        len(groups), total_savings,
    )
    return graph
