"""mha_merge — MHA 投影层合并/拆分优化 Pass。

检测 Q/K/V 投影链（cube_matmul → view → transpose → reshape），
通过成本模型决定保持 merged 或拆分为 per-head 投影。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from torch2c.common import Graph, Node, Tensor, get_logger
from torch2c.common.opt_log import log_opt

logger = get_logger(__name__)

# ---- 数据结构 ----


@dataclass
class ForwardChain:
    """单条 Q/K/V 前向链。"""

    proj_node_id: str       # cube_matmul / cube_matmul_bias
    view_node_id: str       # idma_reshape (view)
    transpose_node_id: str  # idma_transpose
    reshape_node_id: str    # idma_reshape (reshape)
    B: int
    S: int
    H: int
    Dh: int


@dataclass
class MhaGroup:
    """一组完整的 MHA 前向链（Q/K/V）。"""

    id: str
    q_chain: ForwardChain
    k_chain: ForwardChain
    v_chain: ForwardChain
    reverse_chain_node_ids: list[str] = field(default_factory=list)
    D: int = 0  # H * Dh


# ---- Pattern Detection ----


def _single_consumer(graph: Graph, node_id: str) -> str | None:
    """返回 node 唯一输出 tensor 的唯一消费者 node id，否则 None。"""
    node = graph.get_node(node_id)
    if not node or len(node.outputs) != 1:
        return None
    t = graph.get_tensor(node.outputs[0])
    if not t or len(t.consumer_node_ids) != 1:
        return None
    return t.consumer_node_ids[0]


def _match_forward_chain(graph: Graph, proj_nid: str) -> ForwardChain | None:
    """从 proj 节点出发，匹配 4 节点前向链。"""
    proj = graph.get_node(proj_nid)
    if not proj or proj.npu_op not in ("cube_matmul", "cube_matmul_bias"):
        return None

    # proj → view (idma_reshape)
    view_nid = _single_consumer(graph, proj_nid)
    if not view_nid:
        return None
    view = graph.get_node(view_nid)
    if not view or view.npu_op != "idma_reshape":
        return None

    # view → transpose (idma_transpose)
    tr_nid = _single_consumer(graph, view_nid)
    if not tr_nid:
        return None
    tr = graph.get_node(tr_nid)
    if not tr or tr.npu_op != "idma_transpose":
        return None
    # 检查 transpose dims: dim0=1, dim1=2
    if tr.params.get("dim0") != 1 or tr.params.get("dim1") != 2:
        return None

    # transpose → reshape (idma_reshape)
    rs_nid = _single_consumer(graph, tr_nid)
    if not rs_nid:
        return None
    rs = graph.get_node(rs_nid)
    if not rs or rs.npu_op != "idma_reshape":
        return None

    # 提取 shape 信息
    proj_out = graph.get_tensor(proj.outputs[0])
    view_out = graph.get_tensor(view.outputs[0])
    rs_out = graph.get_tensor(rs.outputs[0])
    if not proj_out or not view_out or not rs_out:
        return None

    # proj_out: [B, S, D], view_out: [B, S, H, Dh], rs_out: [B*H, S, Dh]
    if len(proj_out.shape) != 3 or len(view_out.shape) != 4 or len(rs_out.shape) != 3:
        return None

    B, S, D = proj_out.shape
    _, _, H, Dh = view_out.shape
    if D != H * Dh:
        return None
    if rs_out.shape[0] != B * H:
        return None

    return ForwardChain(
        proj_node_id=proj_nid,
        view_node_id=view_nid,
        transpose_node_id=tr_nid,
        reshape_node_id=rs_nid,
        B=B, S=S, H=H, Dh=Dh,
    )


def _detect_mha_groups(graph: Graph) -> list[MhaGroup]:
    """扫描全图，找出所有 MHA 前向链组。

    按 (B, S, H, Dh) 分组，按 execution_order 稳定排序后每 3 条组成一个 group。
    """
    chains: list[ForwardChain] = []
    for nid, node in graph.nodes.items():
        if node.npu_op in ("cube_matmul", "cube_matmul_bias"):
            chain = _match_forward_chain(graph, nid)
            if chain:
                chains.append(chain)

    if len(chains) < 3:
        return []

    shape_groups: dict[tuple, list[ForwardChain]] = {}
    for c in chains:
        key = (c.B, c.S, c.H, c.Dh)
        shape_groups.setdefault(key, []).append(c)

    groups: list[MhaGroup] = []
    gid = 0
    for _key, cs in shape_groups.items():
        # 按 execution_order 中的位置排序，确保稳定配对
        order_map = {nid: i for i, nid in enumerate(graph.execution_order)}
        cs.sort(key=lambda c: order_map.get(c.proj_node_id, 0))

        for i in range(0, len(cs) - 2, 3):
            group = MhaGroup(
                id=f"mha_{gid}",
                q_chain=cs[i],
                k_chain=cs[i + 1],
                v_chain=cs[i + 2],
                D=cs[i].H * cs[i].Dh,
            )
            groups.append(group)
            gid += 1

    return groups


# ---- Cost Model ----


def _compute_costs(group: MhaGroup, config: dict) -> tuple[float, float]:
    """计算 merged/split 方案的总开销，返回 (cost_A, cost_B)。"""
    chain = group.q_chain  # 所有链 shape 一致
    B, S, H, Dh, D = chain.B, chain.S, chain.H, chain.Dh, group.D
    elem_bytes = 2  # fp16

    # 硬件参数
    hw = config.get("hardware", {})
    ld = hw.get("last_dim_align", 16)
    l1_bytes = hw.get("l1_size_bytes", 16 * 1024 * 1024)
    dma_bw = hw.get("dma_bytes_per_cycle", 256)
    matmul_launch = hw.get("matmul_launch_cycles", 100)

    # 方案 A: Merged
    pad_waste_merged = (math.ceil(D / ld) * ld - D) * B * S * elem_bytes * 3
    # reshape/transpose DMA: 3 chains × 3 nodes × tensor_bytes
    tensor_bytes = B * S * D * elem_bytes
    reshape_dma_cost = 3 * 3 * tensor_bytes / dma_bw
    # tiling: attention scores [B*H, S, S]
    scores_bytes = B * H * S * S * elem_bytes
    tiling_cost_merged = scores_bytes if scores_bytes > l1_bytes else 0
    cost_a = pad_waste_merged + reshape_dma_cost + tiling_cost_merged

    # 方案 B: Split
    pad_waste_split = H * (math.ceil(Dh / ld) * ld - Dh) * B * S * elem_bytes * 3
    pipeline_overhead = (H - 1) * 3 * matmul_launch
    concat_cost = H * B * S * Dh * elem_bytes / dma_bw
    cost_b = pad_waste_split + pipeline_overhead + concat_cost

    return cost_a, cost_b


def _annotate_group(
    graph: Graph, group: MhaGroup, decision: str,
    cost_a: float, cost_b: float,
) -> None:
    """在投影节点上写入 _mha_analysis metadata。"""
    analysis = {
        "group_id": group.id,
        "decision": decision,
        "cost_merged": cost_a,
        "cost_split": cost_b,
        "H": group.q_chain.H,
        "Dh": group.q_chain.Dh,
    }
    for chain in (group.q_chain, group.k_chain, group.v_chain):
        node = graph.get_node(chain.proj_node_id)
        if node:
            node.params["_mha_analysis"] = dict(analysis)


# ---- Graph Transformation ----


def _create_per_head_proj(
    graph: Graph, chain: ForwardChain, h: int, proj_role: str,
) -> str:
    """创建第 h 个 per-head matmul 节点，返回其输出 tensor id。"""
    H, Dh = chain.H, chain.Dh
    orig_proj = graph.get_node(chain.proj_node_id)
    if not orig_proj:
        raise ValueError(f"proj node {chain.proj_node_id} not found")

    suffix = f"_{proj_role}_h{h}"
    new_nid = f"{chain.proj_node_id}{suffix}"
    out_tid = f"{orig_proj.outputs[0]}{suffix}"

    # 权重 tensor：切片
    weight_input_idx = 1
    orig_weight_tid = orig_proj.inputs[weight_input_idx]
    orig_weight = graph.get_tensor(orig_weight_tid)
    w_tid = f"{orig_weight_tid}{suffix}"
    w_shape = list(orig_weight.shape) if orig_weight else [Dh, chain.B * chain.S]
    w_shape[0] = Dh  # output dim 切片
    graph.add_tensor(Tensor(
        id=w_tid, shape=w_shape, dtype=orig_weight.dtype if orig_weight else "fp16",
        is_weight=True,
        name=f"{orig_weight.name}{suffix}" if orig_weight and orig_weight.name else w_tid,
    ))

    # 新节点的输入列表
    new_inputs = [orig_proj.inputs[0], w_tid]  # activation + weight

    # bias 切片（如果有）
    bias_tid = None
    if orig_proj.npu_op == "cube_matmul_bias" and len(orig_proj.inputs) >= 3:
        orig_bias_tid = orig_proj.inputs[2]
        orig_bias = graph.get_tensor(orig_bias_tid)
        bias_tid = f"{orig_bias_tid}{suffix}"
        graph.add_tensor(Tensor(
            id=bias_tid, shape=[Dh],
            dtype=orig_bias.dtype if orig_bias else "fp16",
            is_weight=True,
            name=(f"{orig_bias.name}{suffix}"
                  if orig_bias and orig_bias.name else bias_tid),
        ))
        new_inputs.append(bias_tid)

    # 输出 tensor
    orig_out = graph.get_tensor(orig_proj.outputs[0])
    graph.add_tensor(Tensor(
        id=out_tid, shape=[chain.B, chain.S, Dh],
        dtype=orig_out.dtype if orig_out else "fp16",
        producer_node_id=new_nid,
    ))

    # 创建节点
    new_node = Node(
        id=new_nid,
        op_type=orig_proj.op_type,
        inputs=new_inputs,
        outputs=[out_tid],
        params={
            **{k: v for k, v in orig_proj.params.items()
               if not k.startswith("_mha")},
        },
        compute_unit=orig_proj.compute_unit,
        npu_op=orig_proj.npu_op,
        is_mapped=orig_proj.is_mapped,
    )

    # 权重切片标注
    source_name = orig_weight.name if orig_weight and orig_weight.name else orig_weight_tid
    new_node.params["_weight_slices"] = {
        "weight_input_index": 1,
        "source_name": source_name,
        "dim": 0,
        "start": h * Dh,
        "end": (h + 1) * Dh,
    }
    if bias_tid:
        orig_bias_t = graph.get_tensor(orig_proj.inputs[2])
        bias_source = (orig_bias_t.name if orig_bias_t and orig_bias_t.name
                       else orig_proj.inputs[2])
        new_node.params["_bias_slices"] = {
            "weight_input_index": 2,
            "source_name": bias_source,
            "dim": 0,
            "start": h * Dh,
            "end": (h + 1) * Dh,
        }

    graph.add_node(new_node)
    # 更新 activation tensor 的 consumer
    act_t = graph.get_tensor(orig_proj.inputs[0])
    if act_t and new_nid not in act_t.consumer_node_ids:
        act_t.consumer_node_ids.append(new_nid)
    # 更新 weight tensor 的 consumer
    wt = graph.get_tensor(w_tid)
    if wt:
        wt.consumer_node_ids.append(new_nid)
    if bias_tid:
        bt = graph.get_tensor(bias_tid)
        if bt:
            bt.consumer_node_ids.append(new_nid)

    return out_tid


def _apply_split(graph: Graph, group: MhaGroup) -> None:
    """对 MhaGroup 执行 split 变换。"""
    roles = [("q", group.q_chain), ("k", group.k_chain), ("v", group.v_chain)]

    for role, chain in roles:
        H = chain.H
        head_out_tids: list[str] = []

        # 创建 H 个 per-head matmul
        for h in range(H):
            out_tid = _create_per_head_proj(graph, chain, h, role)
            head_out_tids.append(out_tid)

        # 创建 idma_concat 节点
        final_out_tid = graph.get_node(chain.reshape_node_id).outputs[0]
        concat_nid = f"{chain.proj_node_id}_concat_{role}"
        concat_node = Node(
            id=concat_nid,
            op_type="idma_concat",
            inputs=head_out_tids,
            outputs=[final_out_tid],
            params={"axis": 0},
            compute_unit="idma",
            npu_op="idma_concat",
            is_mapped=True,
        )
        graph.add_node(concat_node)

        # 更新 final output tensor
        final_t = graph.get_tensor(final_out_tid)
        if final_t:
            final_t.producer_node_id = concat_nid
        # 更新 head outputs 的 consumer
        for tid in head_out_tids:
            t = graph.get_tensor(tid)
            if t:
                t.consumer_node_ids.append(concat_nid)

        # 更新 execution_order：在原 proj 位置插入新节点
        old_ids_set = {
            chain.proj_node_id, chain.view_node_id,
            chain.transpose_node_id, chain.reshape_node_id,
        }
        new_ids = [f"{chain.proj_node_id}_{role}_h{h}" for h in range(H)]
        new_ids.append(concat_nid)
        new_ids_set = set(new_ids)
        remove_set = old_ids_set | new_ids_set

        # 找到 proj 在过滤后序列中的插入位置，一次遍历完成
        filtered: list[str] = []
        insert_idx = len(filtered)
        for nid in graph.execution_order:
            if nid in remove_set:
                if nid == chain.proj_node_id:
                    insert_idx = len(filtered)
                continue
            filtered.append(nid)
        filtered[insert_idx:insert_idx] = new_ids
        graph.execution_order = filtered

        # 删除旧的中间 tensor（view/transpose 输出，不含 proj 输入和 reshape 输出）
        mid_tids = set()
        for old_nid in [chain.view_node_id, chain.transpose_node_id]:
            old_node = graph.get_node(old_nid)
            if old_node:
                mid_tids.update(old_node.outputs)
        # proj 输出也删除（已被 per-head 替代）
        proj_node = graph.get_node(chain.proj_node_id)
        if proj_node:
            mid_tids.update(proj_node.outputs)

        # 删除旧节点（remove_node 会清理 tensor 引用）
        for old_nid in old_ids_set:
            graph.remove_node(old_nid)

        # 删除中间 tensor
        for tid in mid_tids:
            graph.remove_tensor(tid)


# ---- Entry Points ----


def run(graph: Graph, config: dict) -> Graph:
    """MHA merge pass 入口。"""
    threshold = config.get("prefer_merged_threshold", 0.9)
    max_batch = config.get("max_batch_for_split", 1)

    groups = _detect_mha_groups(graph)
    if not groups:
        logger.info("mha_merge: 未检测到 MHA 前向链")
        return graph

    split_count = 0
    for group in groups:
        cost_a, cost_b = _compute_costs(group, config)

        # B > max_batch → 保持 merged
        if group.q_chain.B > max_batch:
            _annotate_group(graph, group, "merged_batch_limit", cost_a, cost_b)
            for chain in (group.q_chain, group.k_chain, group.v_chain):
                proj_node = graph.get_node(chain.proj_node_id)
                if proj_node:
                    log_opt(
                        proj_node, "mha_merge", "保持 merged（batch 限制）",
                        f"MHA 组 {group.id}: B={group.q_chain.B} > max_batch={max_batch}，"
                        f"强制保持 merged 方案。成本对比: merged={cost_a:.0f}, split={cost_b:.0f}，"
                        f"H={group.q_chain.H}, Dh={group.q_chain.Dh}, D={group.D}",
                    )
            logger.info("mha_merge %s: B=%d > %d, 保持 merged",
                        group.id, group.q_chain.B, max_batch)
            continue

        # 成本比较
        if cost_a * threshold < cost_b:
            _annotate_group(graph, group, "merged", cost_a, cost_b)
            for chain in (group.q_chain, group.k_chain, group.v_chain):
                proj_node = graph.get_node(chain.proj_node_id)
                if proj_node:
                    log_opt(
                        proj_node, "mha_merge", "保持 merged",
                        f"合并投影 [S,{group.D}] 比 {chain.H} 个 per-head [S,{chain.Dh}] 更优："
                        f"padding 浪费更少（D={group.D} vs {chain.H}×Dh={chain.Dh}），"
                        f"且省去 {chain.H}-1 次 matmul 启动开销",
                    )
            logger.info("mha_merge %s: merged wins (%.0f vs %.0f)",
                        group.id, cost_a, cost_b)
        else:
            _annotate_group(graph, group, "split", cost_a, cost_b)
            for chain in (group.q_chain, group.k_chain, group.v_chain):
                proj_node = graph.get_node(chain.proj_node_id)
                if proj_node:
                    log_opt(
                        proj_node, "mha_merge", "拆分 per-head",
                        f"拆为 {chain.H} 个独立 [S,{chain.Dh}] 投影："
                        f"注意力分数 [{chain.H},{chain.S},{chain.S}] 超出 L1 需要 tiling，"
                        f"拆分后每个 head 独立计算，免 tiling 且省去 view+transpose+reshape 链",
                    )
            _apply_split(graph, group)
            split_count += 1
            logger.info("mha_merge %s: split applied (%.0f vs %.0f)",
                        group.id, cost_a, cost_b)

    logger.info("mha_merge 完成: %d groups, %d split", len(groups), split_count)
    return graph


def post_validate(graph: Graph) -> list[str]:
    """校验 mha_merge 变换后的图完整性。"""
    errors = graph.validate()

    # mha_merge 特有：per-head matmul 必须有 _weight_slices 标注
    for nid, node in graph.nodes.items():
        if node.npu_op == "idma_concat":
            for tid in node.inputs:
                producer = graph.get_tensor(tid)
                if producer and producer.producer_node_id:
                    pn = graph.get_node(producer.producer_node_id)
                    if pn and pn.npu_op in ("cube_matmul", "cube_matmul_bias"):
                        if "_weight_slices" not in pn.params:
                            errors.append(
                                f"per-head matmul {pn.id} 缺少 _weight_slices 标注")

    return errors
