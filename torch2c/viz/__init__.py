"""viz — 编译期可视化（ECharts 交互式 HTML）。

绑定点:
  - emit_graph_html     → Pipeline Schedule 甘特图 (⑧ memory_planner)
  - emit_lifetime_html  → HBM+L1 内存生命周期图 (⑧ memory_planner)
  - emit_roofline_html  → Roofline 算力分析图 (⑥b roofline)
  - emit_fusion_html    → 融合组拓扑图 (⑥c fusion)
  - emit_tid_html       → TID 调度甘特图 (⑧b tid_assign)
"""

from torch2c.viz.graph_viz import emit_graph_html
from torch2c.viz.lifetime_viz import emit_lifetime_html
from torch2c.viz.roofline_viz import emit_roofline_html
from torch2c.viz.fusion_viz import emit_fusion_html
from torch2c.viz.tid_viz import emit_tid_html
from torch2c.viz.unified_viz import emit_unified_html

__all__ = [
    "emit_graph_html",
    "emit_lifetime_html",
    "emit_roofline_html",
    "emit_fusion_html",
    "emit_tid_html",
    "emit_unified_html",
]
