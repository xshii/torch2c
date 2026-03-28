"""unified_viz — 统一面板：5 图联动，共享时间轴 + node_id hover 同步高亮。

布局：
  左侧 3 行（共享 X=cycles）：pipeline 甘特图 / 内存生命周期 / TID 调度
  右侧 2 行：roofline 散点图 / fusion 拓扑图

联动：
  - 左侧 3 图 X 轴缩放/平移同步（echarts.connect）
  - hover 任意节点 → 所有图同步高亮同一 node_id
"""

from __future__ import annotations

import json
import os

from torch2c.common import Graph, get_logger
from torch2c.viz._utils import CU_COLOR, STORAGE_COLOR, ensure_viz_dir, human_size, shape_str
from torch2c.common.sizing import calc_padded_size, get_dim_align

logger = get_logger("viz.unified")


def emit_unified_html(
    graph: Graph,
    output_dir: str,
    cube_size: int,
    hw_config: dict | None = None,
    dma_plans: list | None = None,
    title: str | None = None,
) -> str:
    """生成统一面板 HTML，返回文件路径。"""
    viz_dir = ensure_viz_dir(output_dir)
    path = os.path.join(viz_dir, "unified.html")

    data = {
        "pipeline": _build_pipeline_data(graph, cube_size, hw_config, dma_plans),
        "memory": _build_memory_data(graph, cube_size, hw_config, dma_plans),
        "tid": _build_tid_data(graph),
        "roofline": _build_roofline_data(graph, hw_config),
        "fusion": _build_fusion_data(graph),
        "title": title or "torch2c Unified View",
    }

    html = _TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Unified HTML: %s", path)
    return path


# ── 数据构建 ──


def _build_pipeline_data(graph, cube_size, hw_config, dma_plans):
    """甘特图数据：每个 op 的 (start_cycle, duration, unit, node_id)。"""
    compute = (hw_config or {}).get("compute", {})
    cube_rate = compute.get("cube_ops_per_cycle", 4096)
    vec_rate = compute.get("vector_ops_per_cycle", 128)
    dma_bw = compute.get("dma_bytes_per_cycle", 256)

    plan_map = {p.node_id: p for p in (dma_plans or graph.dma_plans or [])}
    items = []
    t = 0  # current cycle

    def _dma_item(nid, instr, op_type, prefix=""):
        nonlocal t
        dma_cy = max(1, instr.size_bytes // dma_bw) if dma_bw else 1
        convert = f"{instr.src_format}→{instr.dst_format}" if instr.src_format != instr.dst_format else instr.src_format
        item = {
            "nid": nid, "start": t, "dur": dma_cy, "unit": "dma",
            "label": f"{prefix}{instr.op} {instr.tensor_id}",
            "type": op_type,
            "fmt": convert,
            "dtype": instr.dtype,
            "size": instr.size_bytes,
            "src_fmt": instr.src_format,
            "dst_fmt": instr.dst_format,
        }
        t += dma_cy
        return item

    # ── Bulk load DMA（如果有）──
    bulk_load = plan_map.get("__bulk_load__")
    if bulk_load:
        for ld in bulk_load.loads:
            items.append(_dma_item("__bulk_load__", ld, "dma_load", "bulk "))

    # ── 计算节点（含 per-op DMA）──
    for nid in (graph.execution_order or []):
        node = graph.nodes.get(nid)
        if not node:
            continue
        cu = (node.compute_unit or "vector").lower()
        rf = node.params.get("_roofline", {})
        cycles = rf.get("node_cycles", 0)
        if cycles == 0:
            flops = rf.get("flops", 0)
            if cu == "cube" and cube_rate:
                cycles = max(1, flops // cube_rate)
            elif vec_rate:
                cycles = max(1, flops // vec_rate) if flops else 1
            else:
                cycles = 1

        # Per-op DMA loads
        plan = plan_map.get(nid)
        if plan:
            for ld in plan.loads:
                items.append(_dma_item(nid, ld, "dma_load"))

        # Compute
        rf_info = {}
        if rf:
            rf_info = {"flops": rf.get("flops", 0), "bottleneck": rf.get("bottleneck", ""),
                       "oi": rf.get("oi", 0)}
        items.append({"nid": nid, "start": t, "dur": cycles, "unit": cu,
                      "label": f"{nid} ({node.npu_op or '?'})", "type": "compute", **rf_info})
        t += cycles

        # Per-op DMA stores
        if plan:
            for st in plan.stores:
                items.append(_dma_item(nid, st, "dma_store"))

    # ── Bulk store DMA（如果有）──
    bulk_store = plan_map.get("__bulk_store__")
    if bulk_store:
        for st in bulk_store.stores:
            items.append(_dma_item("__bulk_store__", st, "dma_store", "bulk "))

    return items


def _build_memory_data(graph, cube_size, hw_config, dma_plans):
    """内存块数据：每个 tensor 的 (hbm_offset, size, first_cycle, last_cycle)。"""
    # 简化：用 execution_order index 作为 cycle 近似
    order_map = {nid: i for i, nid in enumerate(graph.execution_order or [])}
    items = []
    for tid, t in graph.tensors.items():
        if t.hbm_offset is None:
            continue
        size = t.hbm_size or 0
        first = order_map.get(t.producer_node_id, 0) if t.producer_node_id else 0
        consumers = [order_map.get(c, 0) for c in t.consumer_node_ids if c in order_map]
        last = max(consumers) if consumers else first
        if t.is_model_output:
            last = len(order_map)
        items.append({
            "tid": tid, "offset": t.hbm_offset, "size": size,
            "first": first, "last": last,
            "storage": t.storage or "hbm",
            "nid": t.producer_node_id or "",
            "shape": shape_str(t.shape) if t.shape else "?",
            "dtype": t.dtype,
        })
    return items


def _build_tid_data(graph):
    """TID 数据：同 tid_viz。"""
    plan_map = {p.node_id: p for p in (graph.dma_plans or [])}

    # 检测主路径
    try:
        from torch2c.d_emission.tid_assign.tid_assign import _find_critical_path, _classify_branches
        main_path = _find_critical_path(graph)
        branch_map = _classify_branches(graph, main_path)
    except Exception:
        main_path = set(graph.execution_order or [])
        branch_map = {nid: 0 for nid in (graph.execution_order or [])}

    items = []
    for nid in (graph.execution_order or []):
        node = graph.nodes.get(nid)
        if not node:
            continue
        plan = plan_map.get(nid)
        is_main = nid in main_path

        if plan:
            for ld in plan.loads:
                items.append({"tid": ld.task_id, "unit": "dma", "nid": nid,
                              "label": f"load {ld.tensor_id}", "main": is_main, "type": "dma_load"})
        items.append({"tid": node.task_id, "unit": (node.compute_unit or "vector").lower(),
                      "nid": nid, "label": f"{nid} ({node.npu_op or '?'})", "main": is_main, "type": "compute"})
        if plan:
            for st in plan.stores:
                items.append({"tid": st.task_id, "unit": "dma", "nid": nid,
                              "label": f"store {st.tensor_id}", "main": is_main, "type": "dma_store"})

    # bulk
    for key in ("__bulk_load__", "__bulk_store__"):
        plan = plan_map.get(key)
        if not plan:
            continue
        for instr in plan.loads + plan.stores:
            items.append({"tid": instr.task_id, "unit": "dma", "nid": key,
                          "label": f"{instr.op} {instr.tensor_id}", "main": True, "type": f"dma_{instr.op}"})

    items.sort(key=lambda x: x["tid"])
    return items


def _build_roofline_data(graph, hw_config):
    """Roofline 散点数据。"""
    compute = (hw_config or {}).get("compute", {})
    cube_peak = compute.get("cube_ops_per_cycle", 4096)
    vec_peak = compute.get("vector_ops_per_cycle", 128)
    dma_bw = compute.get("dma_bytes_per_cycle", 256)
    bw = dma_bw * 2

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
        nc = rf.get("node_cycles", 1)
        perf = flops / nc if nc > 0 else 0
        cu = (node.compute_unit or "vector").lower()
        points.append({
            "nid": nid, "oi": oi, "perf": round(perf, 2), "flops": flops,
            "unit": cu, "bottleneck": rf.get("bottleneck", "?"),
            "color": CU_COLOR.get(cu, "#999"),
        })
    return {"points": points, "cube_peak": cube_peak, "vec_peak": vec_peak, "bw": bw}


def _build_fusion_data(graph):
    """Fusion 拓扑数据。"""
    group_colors = [
        "#ef4444", "#f97316", "#eab308", "#22c55e", "#06b6d4",
        "#3b82f6", "#8b5cf6", "#ec4899", "#14b8a6", "#f59e0b",
    ]
    group_ids = {}
    next_color = [0]

    nodes = []
    for nid in (graph.execution_order or list(graph.nodes)):
        node = graph.nodes.get(nid)
        if not node:
            continue
        fg = node.params.get("_fusion_group", "")
        if fg and fg not in group_ids:
            group_ids[fg] = next_color[0] % len(group_colors)
            next_color[0] += 1
        color = group_colors[group_ids[fg]] if fg else "#64748b"
        nodes.append({"nid": nid, "op": node.npu_op or node.op_type,
                      "group": fg, "color": color,
                      "unit": (node.compute_unit or "vector").lower()})

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
                    edges.append({"source": nid, "target": cid,
                                  "local": t.storage in ("local", "pipe")})
    return {"nodes": nodes, "edges": edges}


# ── HTML 模板 ──

_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Unified View</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
:root { --bg: #0f172a; --border: #334155; --text: #e2e8f0; --dim: #94a3b8; --accent: #38bdf8; }
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: var(--bg); color: var(--text); font-family: -apple-system, sans-serif;
  height: 100vh; display: grid;
  grid-template-columns: 1fr 360px;
  grid-template-rows: 1fr 1fr 1fr;
  gap: 2px;
}
.panel { background: #1e293b; border-radius: 4px; position: relative; overflow: hidden; }
.panel-title { position: absolute; top: 4px; left: 8px; font-size: 11px; color: var(--dim);
  z-index: 2; pointer-events: none; }
#pipelinePanel { grid-row: 1; grid-column: 1; }
#memoryPanel   { grid-row: 2; grid-column: 1; }
#tidPanel      { grid-row: 3; grid-column: 1; }
#rooflinePanel { grid-row: 1 / 3; grid-column: 2; }
#fusionPanel   { grid-row: 3; grid-column: 2; }
.chart { width: 100%; height: 100%; }
.highlight-bar { position: fixed; top: 0; width: 2px; height: 100vh; background: var(--accent);
  pointer-events: none; z-index: 999; display: none; }
</style>
</head><body>

<div class="panel" id="pipelinePanel"><span class="panel-title">Pipeline Schedule</span><div class="chart" id="c1"></div></div>
<div class="panel" id="memoryPanel"><span class="panel-title">Memory Lifetime</span><div class="chart" id="c2"></div></div>
<div class="panel" id="tidPanel"><span class="panel-title">TID Schedule</span><div class="chart" id="c3"></div></div>
<div class="panel" id="rooflinePanel"><span class="panel-title">Roofline</span><div class="chart" id="c4"></div></div>
<div class="panel" id="fusionPanel"><span class="panel-title">Fusion Groups</span><div class="chart" id="c5"></div></div>

<script>
const D = __DATA__;
const cuColor = { cube:'#E8D0A9', vector:'#B8D4E3', idma:'#B8E3D4', dma:'#E3B8B8' };
const typeColor = { dma_load:'#4ade80', dma_store:'#f87171' };
const lanes = ['cube','vector','idma','dma'];

// ── Charts ──
const c1 = echarts.init(document.getElementById('c1'), 'dark');
const c2 = echarts.init(document.getElementById('c2'), 'dark');
const c3 = echarts.init(document.getElementById('c3'), 'dark');
const c4 = echarts.init(document.getElementById('c4'), 'dark');
const c5 = echarts.init(document.getElementById('c5'), 'dark');

// ── 1. Pipeline Gantt ──
const pData = D.pipeline.map(d => ({
  value: [d.start, lanes.indexOf(d.unit), d.start + d.dur, d.dur],
  itemStyle: { color: typeColor[d.type] || cuColor[d.unit] || '#999' },
  _nid: d.nid, _label: d.label,
  _fmt: d.fmt || '', _dtype: d.dtype || '', _size: d.size || 0,
  _flops: d.flops || 0, _oi: d.oi || 0, _bn: d.bottleneck || '',
}));
c1.setOption({
  grid: { left: 60, right: 20, top: 24, bottom: 24 },
  xAxis: { type: 'value', name: 'cycles', nameTextStyle:{color:'#64748b',fontSize:10}, axisLabel:{color:'#64748b',fontSize:10} },
  yAxis: { type: 'category', data: lanes, axisLabel:{color:'#94a3b8'} },
  tooltip: { formatter: p => { const d=p.data; if(!d) return '';
    let s = `<b>${d._nid}</b><br/>${d._label}<br/>cycle ${d.value[0]}~${d.value[2]}`;
    if(d._fmt) s += `<br/>format: ${d._fmt}`;
    if(d._dtype) s += ` | dtype: ${d._dtype}`;
    if(d._size) s += ` | ${(d._size/1024).toFixed(1)}KB`;
    if(d._flops) s += `<br/>FLOPS: ${d._flops} | OI: ${d._oi} | ${d._bn}`;
    return s; } },
  series: [{ type:'custom', renderItem:(params,api)=>{
    const s=api.value(0), lane=api.value(1), e=api.value(2);
    const start=api.coord([s,lane]), end=api.coord([e,lane]);
    const h = api.size([0,1])[1]*0.6;
    return {type:'rect',shape:{x:start[0],y:start[1]-h/2,width:Math.max(end[0]-start[0],2),height:h},
      style:api.style()};
  }, encode:{x:[0,2],y:1}, data:pData, z:2 }],
});

// ── 2. Memory Lifetime ──
const mData = D.memory.map(d => ({
  value: [d.first, d.offset, d.last, d.offset + d.size],
  itemStyle: { color: d.storage==='local'?'#4169E1':d.storage==='pipe'?'#2E8B57':'#6b7280', opacity:0.7 },
  _nid: d.nid, _tid: d.tid, _shape: d.shape,
}));
const mMaxY = Math.max(...D.memory.map(d=>d.offset+d.size), 1);
c2.setOption({
  grid: { left: 80, right: 20, top: 24, bottom: 24 },
  xAxis: { type: 'value', name: 'op index', nameTextStyle:{color:'#64748b',fontSize:10}, axisLabel:{color:'#64748b',fontSize:10} },
  yAxis: { type: 'value', name: 'HBM offset', max: mMaxY, nameTextStyle:{color:'#64748b',fontSize:10}, axisLabel:{color:'#64748b',fontSize:10} },
  tooltip: { formatter: p => { const d=p.data; return d?`<b>${d._tid}</b> (${d._shape})<br/>HBM ${d.value[1]}~${d.value[3]}<br/>op ${d.value[0]}~${d.value[2]}`:'' } },
  series: [{ type:'custom', renderItem:(params,api)=>{
    const x0=api.coord([api.value(0),api.value(1)]), x1=api.coord([api.value(2),api.value(3)]);
    return {type:'rect',shape:{x:x0[0],y:x1[1],width:Math.max(x1[0]-x0[0],2),height:Math.max(x0[1]-x1[1],2)},
      style:api.style()};
  }, encode:{x:[0,2],y:[1,3]}, data:mData, z:2 }],
});

// ── 3. TID Schedule ──
const tData = D.tid.map(d => ({
  value: [d.tid, lanes.indexOf(d.unit)],
  symbolSize: d.main ? [18,26] : [12,18],
  symbol: 'roundRect',
  itemStyle: { color: typeColor[d.type] || cuColor[d.unit]||'#999', opacity: d.main?1:0.4,
    borderColor: d.main?'#fff':'transparent', borderWidth: d.main?1:0 },
  _nid: d.nid, _label: d.label,
}));
c3.setOption({
  grid: { left: 60, right: 20, top: 24, bottom: 24 },
  xAxis: { type: 'value', name: 'TID', nameTextStyle:{color:'#64748b',fontSize:10}, axisLabel:{color:'#64748b',fontSize:10} },
  yAxis: { type: 'category', data: lanes, axisLabel:{color:'#94a3b8'} },
  tooltip: { formatter: p => { const d=p.data; return d?`<b>TID ${d.value[0]}</b><br/>${d._nid}<br/>${d._label}`:'' } },
  series: [{ type:'scatter', data:tData, z:2 }],
});

// ── 4. Roofline ──
const R = D.roofline;
const oiMin=0.01, oiMax=1000;
function roof(peak) { const p=[]; for(let oi=oiMin;oi<=oiMax;oi*=1.1) p.push([oi,Math.min(oi*R.bw,peak)]); return p; }
const rPts = R.points.map(d => ({
  value:[d.oi,d.perf], symbolSize:Math.max(6,Math.min(24,Math.sqrt(d.flops/100))),
  itemStyle:{color:d.color}, _nid:d.nid, _bn:d.bottleneck,
}));
c4.setOption({
  grid: { left: 50, right: 20, top: 24, bottom: 40 },
  xAxis: { type:'log', name:'OI (FLOP/B)', min:oiMin, max:oiMax, nameTextStyle:{color:'#64748b',fontSize:10}, axisLabel:{color:'#64748b',fontSize:10} },
  yAxis: { type:'log', name:'FLOP/cy', min:0.1, nameTextStyle:{color:'#64748b',fontSize:10}, axisLabel:{color:'#64748b',fontSize:10} },
  tooltip: { formatter: p => { const d=p.data; return d._nid?`<b>${d._nid}</b><br/>OI=${d.value[0]}<br/>${d._bn}`:'' } },
  series: [
    { type:'line', data:roof(R.cube_peak), lineStyle:{color:'#E8D0A9',width:2}, symbol:'none', z:1 },
    { type:'line', data:roof(R.vec_peak), lineStyle:{color:'#B8D4E3',width:2,type:'dashed'}, symbol:'none', z:1 },
    { type:'scatter', data:rPts, z:2 },
  ],
});

// ── 5. Fusion ──
const F = D.fusion;
const fNodes = F.nodes.map(d => ({
  id:d.nid, name:d.nid+'\n'+d.op, symbolSize:d.group?28:18,
  itemStyle:{color:d.color}, label:{show:true,fontSize:8,color:'#e2e8f0'},
  _nid:d.nid,
}));
const fEdges = F.edges.map(d => ({
  source:d.source, target:d.target,
  lineStyle:{color:d.local?'#4ade80':'#475569', type:d.local?'dashed':'solid', width:d.local?2:1},
}));
c5.setOption({
  series:[{ type:'graph', layout:'force', data:fNodes, links:fEdges, roam:true, draggable:true,
    force:{repulsion:120,edgeLength:[40,120],gravity:0.15},
    edgeSymbol:['none','arrow'], edgeSymbolSize:6,
    emphasis:{focus:'adjacency'}, z:2 }],
});

// ── Hover 联动 ──
let highlightedNid = null;

function highlightAll(nid) {
  if (nid === highlightedNid) return;
  highlightedNid = nid;
  // Pipeline
  pData.forEach((d,i) => { c1.dispatchAction({type: d._nid===nid?'highlight':'downplay', seriesIndex:0, dataIndex:i}); });
  // TID
  tData.forEach((d,i) => { c3.dispatchAction({type: d._nid===nid?'highlight':'downplay', seriesIndex:0, dataIndex:i}); });
  // Memory
  mData.forEach((d,i) => { c2.dispatchAction({type: d._nid===nid?'highlight':'downplay', seriesIndex:0, dataIndex:i}); });
  // Roofline
  rPts.forEach((d,i) => { c4.dispatchAction({type: d._nid===nid?'highlight':'downplay', seriesIndex:2, dataIndex:i}); });
  // Fusion
  const fi = fNodes.findIndex(d => d._nid===nid);
  if (fi >= 0) c5.dispatchAction({type:'highlight', seriesIndex:0, dataIndex:fi});
}

[c1,c2,c3,c4,c5].forEach(chart => {
  chart.on('mouseover', p => {
    const nid = p.data?._nid;
    if (nid) highlightAll(nid);
  });
});

// ── Resize ──
window.addEventListener('resize', () => { [c1,c2,c3,c4,c5].forEach(c=>c.resize()); });
</script>
</body></html>
"""
