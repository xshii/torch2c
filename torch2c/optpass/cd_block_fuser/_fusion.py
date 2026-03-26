"""贪心融合算法 — 按 HBM 搬运收益降序融合，L1 容量约束。

与 fusion_planner 的线性链检测不同：
  - 支持 DAG（fan-out tensor 的消费者可以在同一组）
  - 融合决策内嵌 L1 容量检查（不会融合后 spill）
  - 按收益排序（优先消除大 tensor 的搬运）
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch2c.common import get_logger
from torch2c.optpass.cd_block_fuser._block_graph import BlockGraph, DataBlock

logger = get_logger(__name__)


@dataclass
class FusionGroup:
    """一组共享 L1 的算子。"""

    id: str
    node_ids: list[str]                # 拓扑序排列
    internal_block_ids: set[str]       # 组内 tensor（留 L1）
    external_input_ids: set[str]       # 外部输入（DMA load）
    external_output_ids: set[str]      # 外部输出（DMA store）
    total_benefit: int = 0             # 消除的 HBM 搬运 bytes


class _UnionFind:
    """并查集，用于跟踪节点所在的融合组。"""

    def __init__(self):
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


def fuse_blocks(
    block_graph: BlockGraph,
    l1_capacity: int,
) -> list[FusionGroup]:
    """贪心融合：按 elimination_benefit 降序，每步检查 L1 约束。

    Returns:
        融合组列表（不含单节点组）。
    """
    uf = _UnionFind()
    # 每个根节点的组成员
    group_members: dict[str, list[str]] = {
        nid: [nid] for nid in block_graph.topo_order
    }
    # 每个根节点已融合的 internal blocks
    group_internals: dict[str, set[str]] = {
        nid: set() for nid in block_graph.topo_order
    }

    edges = block_graph.get_fusible_edges()

    for db in edges:
        producer_root = uf.find(db.producer_id)

        for consumer_id in db.consumer_ids:
            consumer_root = uf.find(consumer_id)
            if producer_root == consumer_root:
                continue  # 已在同一组

            # 模拟合并后的 L1 峰值
            merged_internals = (
                group_internals[producer_root]
                | group_internals[consumer_root]
                | {db.tensor_id}
            )
            peak = _estimate_l1_peak(merged_internals, block_graph)
            if peak > l1_capacity:
                logger.debug(
                    "跳过融合 %s→%s: L1 peak %d > cap %d",
                    db.producer_id, consumer_id, peak, l1_capacity,
                )
                continue

            # 提交合并
            uf.union(producer_root, consumer_root)
            new_root = uf.find(producer_root)

            # 合并组成员
            old_root = consumer_root if new_root == producer_root else producer_root
            members = group_members.pop(old_root, [])
            group_members.setdefault(new_root, []).extend(members)

            # 合并 internals
            old_internals = group_internals.pop(old_root, set())
            group_internals.setdefault(new_root, set()).update(old_internals)
            group_internals[new_root].add(db.tensor_id)

    # 构建 FusionGroup（过滤单节点组）
    topo_idx = {nid: i for i, nid in enumerate(block_graph.topo_order)}
    groups: list[FusionGroup] = []
    gid = 0

    for root, members in group_members.items():
        if len(members) < 2:
            continue
        # 按拓扑序排列
        members.sort(key=lambda nid: topo_idx.get(nid, 0))
        internals = group_internals.get(root, set())
        ext_in, ext_out = _collect_external_io(members, internals, block_graph)
        benefit = sum(
            block_graph.data_blocks[tid].elimination_benefit
            for tid in internals
            if tid in block_graph.data_blocks
        )
        groups.append(FusionGroup(
            id=f"fg_{gid}",
            node_ids=members,
            internal_block_ids=internals,
            external_input_ids=ext_in,
            external_output_ids=ext_out,
            total_benefit=benefit,
        ))
        gid += 1

    logger.info(
        "block_fuser fusion: %d groups, eliminated %d bytes HBM traffic",
        len(groups), sum(g.total_benefit for g in groups),
    )
    return groups


def _estimate_l1_peak(internal_tids: set[str], bg: BlockGraph) -> int:
    """估算融合组在未 tile 时的 L1 峰值占用。

    简化模型：所有 internal tensor 同时活跃。
    """
    return sum(
        bg.data_blocks[tid].l1_pressure
        for tid in internal_tids
        if tid in bg.data_blocks
    )


def _collect_external_io(
    node_ids: list[str],
    internal_tids: set[str],
    bg: BlockGraph,
) -> tuple[set[str], set[str]]:
    """收集融合组的外部输入和输出。"""
    node_set = set(node_ids)
    ext_inputs: set[str] = set()
    ext_outputs: set[str] = set()

    for nid in node_ids:
        cb = bg.compute_blocks.get(nid)
        if cb is None:
            continue
        for tid in cb.input_block_ids:
            if tid not in internal_tids:
                ext_inputs.add(tid)
        for tid in cb.output_block_ids:
            db = bg.data_blocks.get(tid)
            if db is None:
                continue
            # 有消费者在组外 → 外部输出
            if any(cid not in node_set for cid in db.consumer_ids):
                ext_outputs.add(tid)
            # 无消费者（模型输出）→ 外部输出
            if not db.consumer_ids:
                ext_outputs.add(tid)

    return ext_inputs, ext_outputs
