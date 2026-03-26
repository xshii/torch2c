"""block_fuser — 块级数据流融合 pass。

替换 fusion_planner(⑥c) + global_tiler(⑦b)，联合决策 fusion + tiling + storage。

算法：
  1. 从 Graph IR 构建 BlockGraph（块级数据流视图）
  2. 贪心融合：按 HBM 搬运收益降序，每步检查 L1 约束
  3. 组内联合 tile 决策：共享 tile_size，二分搜索最大可行值
  4. 写回 Graph IR：tensor.storage / _tile_config / fusion_groups

输出接口与 fusion_planner + global_tiler 完全兼容，
下游 scheduler / memory_planner / codegen 无需改动。
"""

from __future__ import annotations

from torch2c.common import Graph, get_logger
from torch2c.common.opt_log import log_opt
from torch2c.optpass.cd_block_fuser._block_graph import BlockGraph
from torch2c.optpass.cd_block_fuser._fusion import FusionGroup, fuse_blocks
from torch2c.optpass.cd_block_fuser._tiling import assign_tile_sizes
from torch2c.optpass.cd_roofline.roofline_analyzer import (
    RooflineHwParams, parse_cost_model,
)

logger = get_logger(__name__)


def run(graph: Graph, config: dict) -> Graph:
    """块级融合 pass 入口。

    Args:
        graph: 经过 roofline_analyzer 的 Graph IR。
        config: 完整 configs dict，读取 hardware / cost_model。
    """
    hw_config = config.get("hardware", None)
    hw = RooflineHwParams.from_config(hw_config)
    cost_model = parse_cost_model(config.get("cost_model"))

    # L1 容量
    l1_cap = 16 * 1024 * 1024  # 默认 16MB
    if hw_config and "memory" in hw_config:
        l1_cap = hw_config["memory"].get("l1", {}).get("total_size_bytes", l1_cap)

    cube_size = 16
    if hw_config and "fractal" in hw_config:
        cube_size = hw_config["fractal"].get("cube_size", cube_size)

    # Step 1: 构建块级数据流图
    block_graph = BlockGraph.from_graph(graph, hw, cost_model)

    # Step 2: 贪心融合
    groups = fuse_blocks(block_graph, l1_cap)

    # Step 3: 组内联合 tile 决策
    for group in groups:
        tile_config = assign_tile_sizes(group, block_graph, l1_cap, cube_size)
        if tile_config:
            group_tile = tile_config
        else:
            group_tile = None
        # 暂存 tile 信息到 group
        object.__setattr__(group, "_tile_config", group_tile)

    # Step 4: 写回 Graph IR
    _apply_decisions(graph, groups, block_graph)

    total_benefit = sum(g.total_benefit for g in groups)
    logger.info(
        "block_fuser 完成: %d 个融合组，消除 %d bytes HBM 搬运",
        len(groups), total_benefit,
    )
    return graph


def _apply_decisions(
    graph: Graph,
    groups: list[FusionGroup],
    block_graph: BlockGraph,
) -> None:
    """将块级决策写回 Graph IR 的标准字段。"""
    fusion_meta = []

    for group in groups:
        # storage: 组内 tensor → local
        for tid in group.internal_block_ids:
            t = graph.tensors.get(tid)
            if t and t.storage not in ("pipe",):
                t.storage = "local"

        # tile_config
        tile_config = getattr(group, "_tile_config", None)
        if tile_config:
            for nid in group.node_ids:
                cb = block_graph.compute_blocks.get(nid)
                if cb and cb.tileable:
                    graph.nodes[nid].params["_tile_config"] = dict(tile_config)

        # fusion_group / fusion_role annotations
        for i, nid in enumerate(group.node_ids):
            node = graph.nodes.get(nid)
            if node is None:
                continue
            node.params["_fusion_group"] = group.id
            if i == 0:
                node.params["_fusion_role"] = "head"
            elif i == len(group.node_ids) - 1:
                node.params["_fusion_role"] = "tail"
            else:
                node.params["_fusion_role"] = "middle"

            # opt_log
            log_opt(
                node, "block_fuser", "块级融合",
                f"加入融合组 {group.id}（{len(group.node_ids)} 个算子），"
                f"{len(group.internal_block_ids)} 个中间 tensor 留 L1，"
                f"消除 {group.total_benefit} bytes HBM 搬运"
                + (f"，tile_size={tile_config['tile_size']}" if tile_config else ""),
            )

        fusion_meta.append({
            "id": group.id,
            "node_ids": group.node_ids,
            "internal_tensors": sorted(group.internal_block_ids),
            "estimated_dma_savings": group.total_benefit,
        })

    if fusion_meta:
        graph.metadata["fusion_groups"] = fusion_meta


def post_validate(graph: Graph) -> list[str]:
    """块级融合后校验。"""
    errors = graph.validate()
    # 检查 fusion_group 内节点一致性
    for nid, node in graph.nodes.items():
        fg = node.params.get("_fusion_group")
        if fg and "_fusion_role" not in node.params:
            errors.append(f"节点 {nid} 有 _fusion_group 但缺少 _fusion_role")
    return errors
