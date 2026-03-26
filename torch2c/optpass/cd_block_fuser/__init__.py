"""block_fuser — 块级数据流融合（替换 fusion_planner + global_tiler）。"""

from .block_fuser import post_validate, run

__all__ = ["post_validate", "run"]
