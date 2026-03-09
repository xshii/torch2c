"""viz — 编译期可视化：Pipeline Schedule 甘特图 + 内存生命周期甘特图（ECharts HTML）。

绑定点（均在 memory_planner⑧ 之后调用，含 DMA 时间线）:
  - emit_graph_html    → Pipeline Schedule 甘特图
  - emit_lifetime_html → 统一 HBM+L1 内存生命周期图 + DMA 搬移箭头
"""

from torch2c.viz.graph_viz import emit_graph_html
from torch2c.viz.lifetime_viz import emit_lifetime_html

__all__ = ["emit_graph_html", "emit_lifetime_html"]
