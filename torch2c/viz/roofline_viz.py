"""roofline_viz — Roofline 图（经典 log-log 坐标，ECharts 交互式 HTML）。

横轴：算术强度 OI (FLOP/Byte)，纵轴：算力 (FLOP/cycle)。
硬件天花板为两条线：peak compute 和 memory bandwidth。
每个算子一个点，颜色区分 compute_unit，大小映射 FLOP 量。

绑定到 roofline_analyzer (⑥b) 之后自动生成。
"""

from __future__ import annotations

import json
import os

from torch2c.common import Graph, get_logger
from torch2c.viz._utils import CU_COLOR, ensure_viz_dir

logger = get_logger("viz.roofline")


def build_roofline_data(
    graph: Graph,
    hw_config: dict | None = None,
) -> dict:
    """构建 roofline 数据（纯数据，不写文件）。返回 {points, cube_peak, vec_peak, bw}。"""
    compute = (hw_config or {}).get("compute", {})
    cube_peak = compute.get("cube_ops_per_cycle", 4096)
    vec_peak = compute.get("vector_ops_per_cycle", 128)
    dma_bw = compute.get("dma_bytes_per_cycle", 256)
    bw = dma_bw * 2  # load + store 双向

    points = []
    for nid in (graph.execution_order or []):
        node = graph.nodes.get(nid)
        if not node:
            continue
        rf = node.params.get("_roofline")
        if not rf:
            continue
        oi = rf.get("oi", 0)
        flops = rf.get("flops", 0)
        cu = (node.compute_unit or "vector").lower()
        nc = rf.get("node_cycles", 1)
        perf = flops / nc if nc > 0 else 0
        peak = cube_peak if cu == "cube" else vec_peak
        ridge = peak / bw if bw > 0 else 0
        points.append({
            "nid": nid,
            "name": f"{nid} ({node.npu_op})",
            "oi": oi,
            "perf": round(perf, 2),
            "flops": flops,
            "bytes": rf.get("bytes", 0),
            "dma_bytes": rf.get("dma_bytes", 0),
            "unit": cu,
            "bottleneck": rf.get("bottleneck", "?"),
            "cycles": nc,
            "comp_cy": rf.get("compute_cycles", 0),
            "dma_cy": rf.get("dma_cycles", 0),
            "ratio": rf.get("achievable_ratio", 0),
            "peak": peak,
            "ridge": round(ridge, 2),
            "color": CU_COLOR.get(cu, "#999"),
        })

    return {"points": points, "cube_peak": cube_peak, "vec_peak": vec_peak, "bw": bw}


def emit_roofline_html(
    graph: Graph,
    output_dir: str,
    cube_size: int,
    hw_config: dict | None = None,
) -> str:
    """生成 Roofline HTML，返回文件路径。"""
    viz_dir = ensure_viz_dir(output_dir)
    path = os.path.join(viz_dir, "roofline.html")

    data = build_roofline_data(graph, hw_config)
    points = data["points"]
    cube_peak = data["cube_peak"]
    vec_peak = data["vec_peak"]
    bw = data["bw"]
    cube_ridge = cube_peak / bw if bw > 0 else 1
    vec_ridge = vec_peak / bw if bw > 0 else 1

    data_json = json.dumps(points, ensure_ascii=False)

    html = _TEMPLATE.replace("__DATA__", data_json)
    html = html.replace("__CUBE_PEAK__", str(cube_peak))
    html = html.replace("__VEC_PEAK__", str(vec_peak))
    html = html.replace("__BW__", str(bw))
    html = html.replace("__CUBE_RIDGE__", str(round(cube_ridge, 4)))
    html = html.replace("__VEC_RIDGE__", str(round(vec_ridge, 4)))

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Roofline HTML: %s (%d ops)", path, len(points))
    return path


_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Roofline Analysis</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
  body { margin: 0; background: #0f172a; font-family: -apple-system, sans-serif; }
  #chart { width: 100vw; height: 100vh; }
</style>
</head><body>
<div id="chart"></div>
<script>
const data = __DATA__;
const cubePeak = __CUBE_PEAK__;
const vecPeak = __VEC_PEAK__;
const bw = __BW__;
const cubeRidge = __CUBE_RIDGE__;
const vecRidge = __VEC_RIDGE__;

const chart = echarts.init(document.getElementById('chart'), 'dark');

// Roofline ceiling lines
const oiMin = 0.01, oiMax = 1000;
function roofline(peak, ridge) {
  const pts = [];
  for (let oi = oiMin; oi <= oiMax; oi *= 1.1) {
    pts.push([oi, Math.min(oi * bw, peak)]);
  }
  return pts;
}

const cubeRoof = roofline(cubePeak, cubeRidge);
const vecRoof = roofline(vecPeak, vecRidge);

// Scatter data by unit
const units = {};
data.forEach(d => {
  if (!units[d.unit]) units[d.unit] = [];
  units[d.unit].push({
    value: [d.oi, d.perf],
    name: d.name,
    itemStyle: { color: d.color },
    symbolSize: Math.max(8, Math.min(30, Math.sqrt(d.flops / 100))),
    _extra: d,
  });
});

const series = [
  { name: 'Cube Ceiling', type: 'line', data: cubeRoof, lineStyle: { color: '#E8D0A9', width: 2 },
    symbol: 'none', z: 1 },
  { name: 'Vector Ceiling', type: 'line', data: vecRoof, lineStyle: { color: '#B8D4E3', width: 2, type: 'dashed' },
    symbol: 'none', z: 1 },
];

Object.entries(units).forEach(([unit, pts]) => {
  series.push({
    name: unit, type: 'scatter', data: pts, z: 2,
  });
});

// Ridge point markers
series.push({
  name: 'Ridge', type: 'scatter', symbol: 'diamond', symbolSize: 12,
  data: [
    { value: [cubeRidge, cubePeak], itemStyle: { color: '#E8D0A9' } },
    { value: [vecRidge, vecPeak], itemStyle: { color: '#B8D4E3' } },
  ],
  z: 3,
});

chart.setOption({
  title: { text: 'Roofline Analysis', left: 'center', textStyle: { color: '#e2e8f0' } },
  tooltip: {
    trigger: 'item',
    formatter: p => {
      if (p.seriesType === 'line') return '';
      const d = p.data._extra || {};
      if (!d.name) return '';
      const bn = d.bottleneck==='compute' ? '计算受限 (OI≥ridge)' : '访存受限 (OI<ridge)';
      return `<b>${d.name}</b> [${d.unit}]<br/>`
        + `<b>OI</b> = FLOPS÷Bytes = ${d.flops}÷${d.bytes} = <b>${d.oi}</b><br/>`
        + `Ridge = peak÷BW = ${d.peak}÷${bw} = ${d.ridge}<br/>`
        + `<b>${bn}</b> (利用率 ${(d.ratio*100).toFixed(0)}%)<br/>`
        + `<hr style="border-color:#444;margin:4px 0"/>`
        + `Perf: ${d.perf} FLOP/cy<br/>`
        + `Compute: ${d.comp_cy} cy | DMA: ${d.dma_cy} cy<br/>`
        + `Total: ${d.cycles} cy = max(compute, dma)`;
    }
  },
  legend: { top: 30, textStyle: { color: '#94a3b8' } },
  xAxis: { type: 'log', name: 'Operational Intensity (FLOP/Byte)', min: oiMin, max: oiMax,
    nameTextStyle: { color: '#94a3b8' }, axisLabel: { color: '#94a3b8' } },
  yAxis: { type: 'log', name: 'Performance (FLOP/cycle)', min: 0.1,
    nameTextStyle: { color: '#94a3b8' }, axisLabel: { color: '#94a3b8' } },
  series: series,
});
window.addEventListener('resize', () => chart.resize());
</script>
</body></html>
"""
