"""block_fuser — 块级数据流融合 pass。

替换 fusion_planner(⑥c) + global_tiler(⑦b)，联合决策 fusion + tiling + storage。
"""

from __future__ import annotations

from torch2c.common import Graph, get_logger

logger = get_logger(__name__)


def run(graph: Graph, config: dict) -> Graph:
    """块级融合 pass 入口。"""
    # Phase 1a/1b 实现后补充
    logger.info("block_fuser: placeholder — not yet implemented")
    return graph


def post_validate(graph: Graph) -> list[str]:
    """块级融合后校验。"""
    return graph.validate()
