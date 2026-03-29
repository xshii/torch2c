"""fusion_viz — 融合组可视化（节点按 fusion group 着色，ECharts 力导向图）。

每个节点一个圆，颜色 = fusion group，未融合节点灰色。
边 = 数据依赖（tensor 流向）。
内部 tensor（storage=local）标注为虚线。

绑定到 fusion_planner / block_fuser (⑥c) 之后自动生成。
"""

from __future__ import annotations

import json
import os

from torch2c.common import Graph, get_logger
from torch2c.viz._utils import CU_COLOR, ensure_viz_dir

logger = get_logger("viz.fusion")

# 融合组配色
_GROUP_COLORS = [
    "#ef4444", "#f97316", "#eab308", "#22c55e", "#06b6d4",
    "#3b82f6", "#8b5cf6", "#ec4899", "#14b8a6", "#f59e0b",
]


def build_fusion_data(graph: Graph) -> dict:
    """构建 fusion 数据（纯数据，不写文件）。返回 {nodes, edges, stats}。"""
    group_ids: dict[str, int] = {}
    next_color = 0

    nodes = []
    for nid in (graph.execution_order or list(graph.nodes)):
        node = graph.nodes.get(nid)
        if not node:
            continue
        fg = node.params.get("_fusion_group", "")
        role = node.params.get("_fusion_role", "")
        cu = (node.compute_unit or "vector").lower()

        if fg and fg not in group_ids:
            group_ids[fg] = next_color % len(_GROUP_COLORS)
            next_color += 1

        color = _GROUP_COLORS[group_ids[fg]] if fg else "#64748b"
        nodes.append({
            "id": nid, "_nid": nid,
            "name": f"{nid}\n{node.npu_op or node.op_type}",
            "category": fg or "ungrouped",
            "symbolSize": 30 if fg else 20,
            "itemStyle": {"color": color},
            "label": {"show": True, "fontSize": 9, "color": "#e2e8f0"},
            "_unit": cu,
            "_group": fg,
            "_role": role,
        })

    edges = []
    for nid in (graph.execution_order or list(graph.nodes)):
        node = graph.nodes.get(nid)
        if not node:
            continue
        for tid in node.outputs:
            t = graph.tensors.get(tid)
            if not t:
                continue
            for cid in t.consumer_node_ids:
                if cid in graph.nodes:
                    is_local = t.storage in ("local", "pipe")
                    edges.append({
                        "source": nid,
                        "target": cid,
                        "lineStyle": {
                            "type": "dashed" if is_local else "solid",
                            "color": "#4ade80" if is_local else "#475569",
                            "width": 2 if is_local else 1,
                        },
                        "_tensor": tid,
                        "_local": is_local,
                    })

    n_groups = len(group_ids)
    n_fused = sum(1 for n in nodes if n["_group"])
    n_local = sum(1 for e in edges if e["_local"])

    return {"nodes": nodes, "edges": edges, "stats": {
        "groups": n_groups, "fused_nodes": n_fused, "local_edges": n_local,
    }}


def emit_fusion_html(
    graph: Graph,
    output_dir: str,
    cube_size: int = 16,
    hw_config: dict | None = None,
) -> str:
    """生成 Fusion Group HTML，返回文件路径。"""
    viz_dir = ensure_viz_dir(output_dir)
    path = os.path.join(viz_dir, "fusion.html")

    raw = build_fusion_data(graph)
    data_json = json.dumps(raw, ensure_ascii=False)

    html = _TEMPLATE.replace("__DATA__", data_json)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    stats = raw["stats"]
    logger.info("Fusion HTML: %s (%d groups, %d fused nodes)", path, stats["groups"], stats["fused_nodes"])
    return path


_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Fusion Groups</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
  body { margin: 0; background: #0f172a; font-family: -apple-system, sans-serif; }
  #chart { width: 100vw; height: 100vh; }
  .stats { position: fixed; top: 10px; right: 10px; color: #94a3b8; font-size: 12px;
    background: rgba(15,23,42,0.8); padding: 8px 12px; border-radius: 6px; }
</style>
</head><body>
<div id="chart"></div>
<div class="stats" id="stats"></div>
<script>
const raw = __DATA__;
const chart = echarts.init(document.getElementById('chart'), 'dark');

document.getElementById('stats').innerHTML =
  `${raw.stats.groups} fusion groups | ${raw.stats.fused_nodes} fused nodes | ${raw.stats.local_edges} L1-resident edges`;

chart.setOption({
  title: { text: 'Fusion Groups', left: 'center', textStyle: { color: '#e2e8f0' } },
  tooltip: {
    formatter: p => {
      if (p.dataType === 'node') {
        const d = p.data;
        return `<b>${d.id}</b><br/>Unit: ${d._unit}<br/>Group: ${d._group || 'none'}<br/>Role: ${d._role || '-'}`;
      }
      if (p.dataType === 'edge') {
        const d = p.data;
        return `${d.source} → ${d.target}<br/>Tensor: ${d._tensor}<br/>${d._local ? 'L1 resident' : 'HBM'}`;
      }
    }
  },
  series: [{
    type: 'graph',
    layout: 'force',
    data: raw.nodes,
    links: raw.edges,
    roam: true,
    draggable: true,
    force: { repulsion: 200, edgeLength: [80, 200], gravity: 0.1 },
    edgeSymbol: ['none', 'arrow'],
    edgeSymbolSize: 8,
    emphasis: { focus: 'adjacency', lineStyle: { width: 3 } },
  }],
});
window.addEventListener('resize', () => chart.resize());
</script>
</body></html>
"""
