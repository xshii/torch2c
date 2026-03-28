"""tid_viz — TID 调度甘特图（主路径高亮 + 支路缩进 + DMA 穿插）。

横轴：TID（提交顺序），纵轴：4 泳道（cube / vector / dma / idma）。
主路径算子用实色，支路用半透明。DMA load/store 显示在 DMA 泳道。
依赖箭头连接相邻指令。

绑定到 tid_assign (⑧b) 之后自动生成。
"""

from __future__ import annotations

import json
import os

from torch2c.common import Graph, get_logger
from torch2c.viz._utils import CU_COLOR, ensure_viz_dir

logger = get_logger("viz.tid")


def emit_tid_html(
    graph: Graph,
    output_dir: str,
    cube_size: int = 16,
    hw_config: dict | None = None,
) -> str:
    """生成 TID 调度甘特图 HTML，返回文件路径。"""
    viz_dir = ensure_viz_dir(output_dir)
    path = os.path.join(viz_dir, "tid_schedule.html")

    plan_map = {p.node_id: p for p in (graph.dma_plans or [])}

    # 主路径检测
    from torch2c.d_emission.tid_assign.tid_assign import _find_critical_path, _classify_branches
    main_path = _find_critical_path(graph)
    branch_map = _classify_branches(graph, main_path)

    # 收集所有指令 (tid, unit, label, is_main, type, deps)
    items = []
    for nid in (graph.execution_order or []):
        node = graph.nodes.get(nid)
        if not node:
            continue
        plan = plan_map.get(nid)
        is_main = nid in main_path
        branch = branch_map.get(nid, 0)

        # DMA loads
        if plan:
            for ld in plan.loads:
                items.append({
                    "tid": ld.task_id,
                    "unit": "dma",
                    "label": f"load {ld.tensor_id}",
                    "main": is_main,
                    "branch": branch,
                    "type": "dma_load",
                    "deps": _dma_deps(ld),
                })

        # Compute
        items.append({
            "tid": node.task_id,
            "unit": (node.compute_unit or "vector").lower(),
            "label": f"{nid} ({node.npu_op or '?'})",
            "main": is_main,
            "branch": branch,
            "type": "compute",
            "deps": _node_deps(node),
        })

        # DMA stores
        if plan:
            for st in plan.stores:
                items.append({
                    "tid": st.task_id,
                    "unit": "dma",
                    "label": f"store {st.tensor_id}",
                    "main": is_main,
                    "branch": branch,
                    "type": "dma_store",
                    "deps": _dma_deps(st),
                })

    # Bulk DMA
    for key in ("__bulk_load__", "__bulk_store__"):
        plan = plan_map.get(key)
        if not plan:
            continue
        for instr in plan.loads + plan.stores:
            items.append({
                "tid": instr.task_id,
                "unit": "dma",
                "label": f"{'load' if instr.op == 'load' else 'store'} {instr.tensor_id}",
                "main": True,
                "branch": 0,
                "type": f"dma_{instr.op}",
                "deps": _dma_deps(instr),
            })

    items.sort(key=lambda x: x["tid"])
    data_json = json.dumps(items, ensure_ascii=False)
    n_main = sum(1 for it in items if it["main"])

    html = _TEMPLATE.replace("__DATA__", data_json)
    html = html.replace("__N_MAIN__", str(n_main))
    html = html.replace("__N_TOTAL__", str(len(items)))

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("TID Schedule HTML: %s (%d instructions, %d main path)", path, len(items), n_main)
    return path


def _dma_deps(instr) -> dict[str, int]:
    return {
        "cube": getattr(instr, "dep_cube_tid", 0),
        "vector": getattr(instr, "dep_vector_tid", 0),
        "dma": getattr(instr, "dep_dma_tid", 0),
        "idma": getattr(instr, "dep_idma_tid", 0),
    }


def _node_deps(node) -> dict[str, int]:
    return {
        "cube": node.params.get("_tid_dep_cube", 0),
        "vector": node.params.get("_tid_dep_vector", 0),
        "dma": node.params.get("_tid_dep_dma", 0),
        "idma": node.params.get("_tid_dep_idma", 0),
    }


_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>TID Schedule</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
  body { margin: 0; background: #0f172a; font-family: -apple-system, sans-serif; }
  #chart { width: 100vw; height: 100vh; }
  .stats { position: fixed; top: 10px; right: 10px; color: #94a3b8; font-size: 12px;
    background: rgba(15,23,42,0.8); padding: 8px 12px; border-radius: 6px; }
</style>
</head><body>
<div id="chart"></div>
<div class="stats">__N_MAIN__ main path / __N_TOTAL__ total instructions</div>
<script>
const data = __DATA__;
const chart = echarts.init(document.getElementById('chart'), 'dark');

const lanes = ['cube', 'vector', 'idma', 'dma'];
const laneColors = { cube: '#E8D0A9', vector: '#B8D4E3', idma: '#B8E3D4', dma: '#E3B8B8' };
const typeColors = { dma_load: '#4ade80', dma_store: '#f87171', compute: null };

// Bar data: each item becomes a bar at x=tid, y=lane
const barData = data.map(d => ({
  value: [d.tid, lanes.indexOf(d.unit)],
  itemStyle: {
    color: typeColors[d.type] || laneColors[d.unit] || '#999',
    opacity: d.main ? 1.0 : 0.4,
    borderColor: d.main ? '#fff' : 'transparent',
    borderWidth: d.main ? 1 : 0,
  },
  _d: d,
}));

// Dependency arrows
const arrows = [];
data.forEach(d => {
  Object.entries(d.deps).forEach(([unit, depTid]) => {
    if (depTid > 0) {
      const src = data.find(x => x.tid === depTid);
      if (src) {
        arrows.push({
          coords: [
            [src.tid, lanes.indexOf(src.unit)],
            [d.tid, lanes.indexOf(d.unit)],
          ],
          lineStyle: {
            color: d.main ? '#38bdf8' : '#475569',
            width: d.main ? 1.5 : 0.8,
            type: d.main ? 'solid' : 'dashed',
          },
        });
      }
    }
  });
});

chart.setOption({
  title: { text: 'TID Schedule (Main Path Highlighted)', left: 'center', textStyle: { color: '#e2e8f0' } },
  tooltip: {
    formatter: p => {
      const d = p.data?._d;
      if (!d) return '';
      const deps = Object.entries(d.deps).filter(([,v]) => v > 0).map(([k,v]) => `${k}:${v}`).join(', ');
      return `<b>TID ${d.tid}</b> — ${d.label}<br/>Unit: ${d.unit}<br/>` +
        `${d.main ? '<b>MAIN PATH</b>' : 'Branch ' + d.branch}<br/>` +
        `Deps: ${deps || 'none'}`;
    }
  },
  grid: { left: 80, right: 30, top: 60, bottom: 40 },
  xAxis: { type: 'value', name: 'Task ID (submission order)', min: 0,
    nameTextStyle: { color: '#94a3b8' }, axisLabel: { color: '#94a3b8' } },
  yAxis: { type: 'category', data: lanes,
    axisLabel: { color: '#94a3b8', fontSize: 12 } },
  series: [
    {
      type: 'scatter', data: barData, symbolSize: [16, 24], symbol: 'roundRect',
      z: 2,
    },
    {
      type: 'lines', coordinateSystem: 'cartesian2d', data: arrows,
      effect: { show: false },
      lineStyle: { curveness: 0.2 },
      z: 1,
    },
  ],
});
window.addEventListener('resize', () => chart.resize());
</script>
</body></html>
"""
