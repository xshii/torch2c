"""viz — 编译期可视化：算子依赖图 + 内存生命周期图。

绑定点:
  - emit_graph_dot / emit_graph_ascii → idma（⑤b）后调用
  - emit_lifetime_ascii              → memory_planner（⑦）后调用
"""

from npu_compiler.viz.graph_viz import emit_graph_ascii, emit_graph_dot
from npu_compiler.viz.lifetime_viz import emit_lifetime_ascii

__all__ = ["emit_graph_ascii", "emit_graph_dot", "emit_lifetime_ascii"]
