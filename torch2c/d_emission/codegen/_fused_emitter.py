"""融合组代码生成 — 对同一 fusion_group 的节点生成共享代码块。

融合组内 storage=local 的 tensor 不生成 DMA load/store，
数据留在 L1 中由连续的算子依次消费。

当组内节点有 _tile_config 且共享 tile_size 时，生成 tile 循环包裹所有算子。
"""

from __future__ import annotations

from torch2c.common import Node, Tensor


def gen_fused_block(
    node_ids: list[str],
    nodes: dict[str, Node],
    tensors: dict[str, Tensor],
    dma_plans: dict,
    signatures: dict,
    c_names: dict | None = None,
    struct_prefix: str = "",
    dim_replace: dict | None = None,
    gen_dma_line_fn=None,
    gen_op_call_fn=None,
) -> str:
    """为融合组生成共享代码块。

    逻辑：
    1. 收集组内所有 DMA 指令
    2. 分类：external load / internal skip / external store
    3. 生成：external loads → 所有 compute ops → external stores

    如果组内有共享 tile_config，生成 tile 循环包裹 compute 部分。
    """
    if gen_dma_line_fn is None or gen_op_call_fn is None:
        raise ValueError("gen_dma_line_fn and gen_op_call_fn are required")

    indent = "    "
    group_id = nodes[node_ids[0]].params.get("_fusion_group", "?")

    # 收集组内节点的 npu_op 列表（用于注释）
    op_names = []
    for nid in node_ids:
        n = nodes[nid]
        op_names.append(f"{n.npu_op or n.op_type}")

    # 收集组内所有 tensor id（internal = storage 为 local 且 producer/consumer 都在组内）
    node_set = set(node_ids)
    internal_tids = set()
    for nid in node_ids:
        n = nodes[nid]
        for tid in n.outputs:
            t = tensors.get(tid)
            if t and t.storage in ("local", "pipe"):
                # 检查所有消费者是否在组内
                if all(cid in node_set for cid in t.consumer_node_ids):
                    internal_tids.add(tid)

    # 收集 DMA 指令，过滤内部 tensor
    external_loads: list[dict] = []
    external_stores: list[dict] = []
    for nid in node_ids:
        plan = dma_plans.get(nid, {})
        for instr in plan.get("loads", []):
            if instr.get("tensor_id") not in internal_tids:
                external_loads.append(instr)
        for instr in plan.get("stores", []):
            if instr.get("tensor_id") not in internal_tids:
                external_stores.append(instr)

    # 生成代码
    lines = [f"{indent}/* === Fusion Group {group_id}: {' → '.join(op_names)} === */"]

    # External loads
    for instr in external_loads:
        lines.append(f"{indent}{gen_dma_line_fn(instr)}")

    # Compute ops（组内 tensor 的 DMA 已被过滤）
    for nid in node_ids:
        n = nodes[nid]
        sig = _find_sig(signatures, n.npu_op or n.op_type)
        if sig is None:
            continue
        l1_layout = dma_plans.get(nid, {}).get("l1_layout")
        op_call = gen_op_call_fn(
            n.npu_op or n.op_type, sig, n, tensors,
            c_names, struct_prefix, dim_replace, l1_layout,
        )
        lines.append(f"{indent}/* {nid}: {n.npu_op} */")
        lines.append(f"{indent}{op_call};")

    # External stores
    for instr in external_stores:
        lines.append(f"{indent}{gen_dma_line_fn(instr)}")

    return "\n".join(lines)


def segment_by_fusion(
    order: list[str],
    nodes: dict[str, Node],
) -> list[tuple[bool, list[str]]]:
    """将 execution_order 按 fusion_group 分段。

    返回 [(is_fused, [node_ids]), ...] 列表。
    连续的同组节点合并为一个 fused 段，非融合节点各自独立。
    """
    segments: list[tuple[bool, list[str]]] = []
    current_group: str | None = None
    current_nids: list[str] = []

    for nid in order:
        node = nodes.get(nid)
        if node is None:
            continue
        fg = node.params.get("_fusion_group")

        if fg is not None and fg == current_group:
            # 同组，追加
            current_nids.append(nid)
        else:
            # 切换段
            if current_nids:
                is_fused = current_group is not None and len(current_nids) >= 2
                segments.append((is_fused, current_nids))
            current_group = fg
            current_nids = [nid]

    if current_nids:
        is_fused = current_group is not None and len(current_nids) >= 2
        segments.append((is_fused, current_nids))

    return segments


def _find_sig(signatures: dict, npu_op: str) -> dict | None:
    """在 signatures 的所有 section 中查找算子签名。"""
    for section in ("compute_ops", "dma_ops", "idma_ops"):
        sigs = signatures.get(section, {})
        if npu_op in sigs:
            return sigs[npu_op]
    return None
