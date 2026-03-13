"""策略函数 — 从 strategies/ 子包 re-export。"""

from .strategies import strategy_bulk, strategy_perop, strategy_spill, strategy_tiled

__all__ = ["strategy_bulk", "strategy_perop", "strategy_spill", "strategy_tiled"]
