"""unified_viz — 统一面板：5 图联动，共享时间轴 + node_id hover 同步高亮。

布局：
  左侧 3 行（共享 X=cycles）：pipeline 甘特图 / 内存生命周期 / TID 调度
  右侧 2 行：roofline 散点图 / fusion 拓扑图

联动：
  - 左侧 3 图 X 轴缩放/平移同步（echarts.connect）
  - hover 任意节点 → 所有图同步高亮同一 node_id

数据来源：复用各独立 viz 模块的 build_*_data 函数，保证同源。
"""

from __future__ import annotations

import json
import os

from torch2c.common import Graph, get_logger
from torch2c.viz._utils import ensure_viz_dir
from torch2c.viz.graph_viz import build_pipeline_data
from torch2c.viz.lifetime_viz import build_memory_data
from torch2c.viz.roofline_viz import build_roofline_data
from torch2c.viz.fusion_viz import build_fusion_data
from torch2c.viz.tid_viz import build_tid_data

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
        "pipeline": build_pipeline_data(graph, cube_size, hw_config, dma_plans),
        "memory": build_memory_data(graph, cube_size, hw_config, dma_plans),
        "tid": build_tid_data(graph, hw_config, dma_plans),
        "roofline": build_roofline_data(graph, hw_config),
        "fusion": build_fusion_data(graph),
        "title": title or "torch2c Unified View",
    }

    html = _TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Unified HTML: %s", path)
    return path


# ── HTML 模板 ──

_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Unified View</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/xlsx@0.18.5/dist/xlsx.full.min.js"></script>
<style>
:root {
  --bg: #0f172a; --surface: #1e293b; --surface2: #263348;
  --border: #334155; --text: #e2e8f0; --dim: #94a3b8;
  --accent: #38bdf8; --accent-dim: #0c4a6e;
  --green: #4ade80; --red: #f87171;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: var(--bg); color: var(--text);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  height: 100vh; display: flex; flex-direction: column; overflow: hidden;
}

/* ── Timeline stack (top 3 charts) ── */
.timeline-stack { flex: 1; display: flex; flex-direction: column; gap: 1px;
  min-height: 0; padding: 2px 2px 0; }
.timeline-panel { flex: 1; background: var(--surface); border-radius: 6px;
  position: relative; overflow: visible; min-height: 0; }
.timeline-panel .panel-label {
  position: absolute; top: 6px; left: 10px; font-size: 10px; font-weight: 600;
  color: var(--dim); letter-spacing: 0.5px; text-transform: uppercase;
  z-index: 2; pointer-events: none; opacity: 0.7;
}

/* ── Bottom tab section ── */
.tab-section { flex-shrink: 0; display: flex; flex-direction: column; }
.tabs {
  display: flex; align-items: center; gap: 0;
  background: var(--surface); border-top: 1px solid var(--border);
  padding: 0 4px;
}
.tab {
  padding: 7px 20px; font-size: 12px; font-weight: 500; color: var(--dim);
  cursor: pointer; border-bottom: 2px solid transparent;
  transition: color 0.15s, border-color 0.15s;
  user-select: none;
}
.tab:hover { color: var(--text); }
.tab.active { color: var(--accent); border-bottom-color: var(--accent); }
.tab-panel { display: none; height: 32vh; background: var(--surface); }
.tab-panel.active { display: block; }

.chart { width: 100%; height: 100%; }

</style>
</head><body>

<div class="timeline-stack">
  <div class="timeline-panel"><span class="panel-label">Pipeline Schedule</span><div class="chart" id="c1"></div></div>
  <div class="timeline-panel"><span class="panel-label">Memory Lifetime</span><div class="chart" id="c2"></div></div>
  <div class="timeline-panel"><span class="panel-label">TID Schedule</span><div class="chart" id="c3"></div></div>
</div>
<div class="tab-section">
  <div class="tabs">
    <div class="tab active" data-tab="rooflineTab">Roofline</div>
    <div class="tab" data-tab="fusionTab">Fusion</div>
  </div>
  <div class="tab-panel active" id="rooflineTab"><div class="chart" id="c4"></div></div>
  <div class="tab-panel" id="fusionTab"><div class="chart" id="c5"></div></div>
</div>

<script>
const D = __DATA__;
const cuColor = { cube:'#E8D0A9', vector:'#B8D4E3', idma:'#B8E3D4', dma:'#E3B8B8' };
const typeColor = { dma_load:'#4ade80', dma_store:'#f87171' };
const lanes = ['cube','vector','dma','idma'];
const tipStyle = { backgroundColor:'rgba(15,23,42,0.95)', borderColor:'#334155',
  appendToBody: true, confine: true,
  textStyle:{color:'#e2e8f0',fontSize:12}, extraCssText:'border-radius:6px;box-shadow:0 4px 12px rgba(0,0,0,0.4);max-width:360px;' };

// ── Charts ──
const c1 = echarts.init(document.getElementById('c1'), 'dark');
const c2 = echarts.init(document.getElementById('c2'), 'dark');
const c3 = echarts.init(document.getElementById('c3'), 'dark');
const c4 = echarts.init(document.getElementById('c4'), 'dark');
const c5 = echarts.init(document.getElementById('c5'), 'dark');

// 共享 dataZoom 配置（滚轮缩放 + 拖拽平移，Esc 还原）
const sharedZoom = [
  { type: 'inside', xAxisIndex: 0, filterMode: 'none' },
];
// 联动：任意一个图 zoom 时同步其他两个
const timelineCharts = [c1, c2, c3];
let _syncing = false;
timelineCharts.forEach(src => {
  src.on('datazoom', p => {
    if (_syncing) return;
    _syncing = true;
    const opt = src.getOption();
    const z = opt.dataZoom[0];
    timelineCharts.forEach(dst => {
      if (dst !== src) dst.dispatchAction({ type:'dataZoom', start:z.start, end:z.end });
    });
    _syncing = false;
  });
});
// Esc 还原全部
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    timelineCharts.forEach(c => c.dispatchAction({ type:'dataZoom', start:0, end:100 }));
  }
});

// ── 1. Pipeline Gantt (data from graph_viz.build_pipeline_data) ──
const pBars = D.pipeline.bars || [];
const pData = pBars.map(d => ({
  value: [d.s, d.lane, d.s + d.d, d.d],
  itemStyle: { color: d.color },
  _nid: d.nid, _label: d.op,
  _ins: d.ins || [], _outs: d.outs || [],
}));
c1.setOption({
  grid: { left: 60, right: 20, top: 24, bottom: 24 },
  dataZoom: sharedZoom,
  xAxis: { type: 'value', name: 'cycles', nameTextStyle:{color:'#64748b',fontSize:10}, axisLabel:{color:'#64748b',fontSize:10} },
  yAxis: { type: 'category', data: lanes, axisLabel:{color:'#94a3b8'} },
  tooltip: { ...tipStyle, formatter: p => { const d=p.data; if(!d) return '';
    let s = `<b style="color:#38bdf8">${d._nid}</b><br/>${d._label}<br/>`
      + `<span style="color:#94a3b8">cycle ${d.value[0]} → ${d.value[2]}  (${d.value[3]} cy)</span>`;
    if(d._ins.length) { s += '<br/><b>Inputs:</b>'; d._ins.forEach(t => { s += `<br/>&ensp;<span style="color:#94a3b8">${t.id}</span> ${t.desc}`; }); }
    if(d._outs.length) { s += '<br/><b>Outputs:</b>'; d._outs.forEach(t => { s += `<br/>&ensp;<span style="color:#94a3b8">${t.id}</span> ${t.desc}`; }); }
    return s; } },
  series: [{ type:'custom', renderItem:(params,api)=>{
    const s=api.value(0), lane=api.value(1), e=api.value(2);
    const start=api.coord([s,lane]), end=api.coord([e,lane]);
    const h = api.size([0,1])[1]*0.6;
    return {type:'rect',shape:{x:start[0],y:start[1]-h/2,width:Math.max(end[0]-start[0],2),height:h},
      style:api.style()};
  }, encode:{x:[0,2],y:1}, data:pData, z:2 }],
});

// ── 2. Memory Lifetime (data from lifetime_viz.build_memory_data) ──
const M = D.memory;
const mBlocks = M.blocks || [];
const mData = mBlocks.map(d => ({
  value: d.value,
  itemStyle: { color: d.color, opacity:0.7 },
  _nid: d.name, _label: d.label, _zone: d.value[5],
  _shape: d.shape, _dtype: d.dtype, _fmt: d.fmt,
}));
const mTotalY = M.total_y || 1;
const mHbmBase = M.hbm_base || 0;
const mL1Max = M.l1_max || 0;
c2.setOption({
  grid: { left: 80, right: 20, top: 24, bottom: 24 },
  dataZoom: sharedZoom,
  xAxis: { type: 'value', name: 'cycles', nameTextStyle:{color:'#64748b',fontSize:10}, axisLabel:{color:'#64748b',fontSize:10} },
  yAxis: { type: 'value', name: 'address', min:0, max:mTotalY, nameTextStyle:{color:'#64748b',fontSize:10},
    axisLabel:{color:'#64748b',fontSize:10, formatter:v=>{
      if(v>=mHbmBase) return 'HBM '+(((v-mHbmBase)/1024)|0)+'K';
      if(v<=mL1Max) return 'L1 '+((v/1024)|0)+'K';
      return '';
    }} },
  tooltip: { ...tipStyle, formatter: p => { const d=p.data; if(!d) return '';
    const v=d.value; const zone=v[5]||'';
    const zoneColor = zone==='L1' ? '#60a5fa' : '#94a3b8';
    const a0=zone==='HBM'?v[1]-mHbmBase:v[1]; const a1=zone==='HBM'?v[3]-mHbmBase:v[3];
    return `<b style="color:${zoneColor}">${d._label}</b> <span style="color:${zoneColor}">[${zone}]</span>`
      + `<br/>${d._shape} ${d._dtype}${d._fmt&&d._fmt!=='nd'?' <b>'+d._fmt+'</b>':''}`
      + `<br/><span style="color:#94a3b8">addr:</span> ${(a0/1024)|0}K → ${(a1/1024)|0}K`
      + `<br/><span style="color:#94a3b8">cycle:</span> ${v[0]} → ${v[2]}`
      + `<br/><span style="color:#94a3b8">size:</span> ${(v[4]/1024).toFixed(1)}K`;
  } },
  series: [{ type:'custom', renderItem:(params,api)=>{
    const x0=api.value(0),y0=api.value(1),x1=api.value(2),y1=api.value(3);
    const p0=api.coord([x0,y1]), p1=api.coord([x1,y0]);
    return {type:'rect',shape:{x:p0[0],y:p0[1],width:Math.max(p1[0]-p0[0],2),height:Math.max(p1[1]-p0[1],2)},
      style:{fill:api.visual('color'),stroke:'#fff',lineWidth:0.3}};
  }, encode:{x:[0,2],y:[1,3]}, data:mData, z:2 }],
});

// ── 3. TID Schedule (cycle-aligned, data from tid_viz.build_tid_data) ──
const tData = D.tid.filter(d => d.start !== undefined).map(d => ({
  value: [d.start, lanes.indexOf(d.unit), d.end, d.end - d.start],
  itemStyle: { color: typeColor[d.type] || cuColor[d.unit]||'#999',
    opacity: d.main?1:0.4, borderColor: d.main?'#fff':'transparent', borderWidth: d.main?1:0 },
  _nid: d.nid, _label: d.label, _tid: d.tid,
}));
c3.setOption({
  grid: { left: 60, right: 20, top: 24, bottom: 24 },
  dataZoom: sharedZoom,
  xAxis: { type: 'value', name: 'cycles', nameTextStyle:{color:'#64748b',fontSize:10}, axisLabel:{color:'#64748b',fontSize:10} },
  yAxis: { type: 'category', data: lanes, axisLabel:{color:'#94a3b8'} },
  tooltip: { ...tipStyle, formatter: p => { const d=p.data; if(!d) return '';
    const raw = D.tid.find(t => t.tid===d._tid);
    let s = `<b style="color:#38bdf8">TID ${d._tid}</b> — ${d._label}`
      + `<br/><span style="color:#94a3b8">${d._nid}</span>`
      + `<br/><span style="color:#94a3b8">cycle:</span> ${d.value[0]} → ${d.value[2]} (${d.value[3]} cy)`;
    if (raw && raw.fmt) s += `<br/><span style="color:#94a3b8">format:</span> ${raw.fmt} | ${raw.dtype} | ${((raw.size||0)/1024).toFixed(1)}K | ${raw.shape}`;
    if (raw && raw.ins && raw.ins.length) {
      s += '<br/><b>Inputs:</b>';
      raw.ins.forEach(t => { s += `<br/>&ensp;<span style="color:#94a3b8">${t.id}</span> ${t.shape} ${t.dtype}${t.fmt?' <b>'+t.fmt+'</b>':''}` });
    }
    if (raw && raw.outs && raw.outs.length) {
      s += '<br/><b>Outputs:</b>';
      raw.outs.forEach(t => { s += `<br/>&ensp;<span style="color:#94a3b8">${t.id}</span> ${t.shape} ${t.dtype}` });
    }
    return s; } },
  series: [{ type:'custom', renderItem:(params,api)=>{
    const s=api.value(0), lane=api.value(1), e=api.value(2);
    const start=api.coord([s,lane]), end=api.coord([e,lane]);
    const h = api.size([0,1])[1]*0.6;
    return {type:'rect',shape:{x:start[0],y:start[1]-h/2,width:Math.max(end[0]-start[0],2),height:h},
      style:api.style()};
  }, encode:{x:[0,2],y:1}, data:tData, z:2 }],
});

// ── 4. Roofline (data from roofline_viz.build_roofline_data) ──
const R = D.roofline;
const oiMin=0.01, oiMax=1000;
function roof(peak) { const p=[]; for(let oi=oiMin;oi<=oiMax;oi*=1.1) p.push([oi,Math.min(oi*R.bw,peak)]); return p; }
const rPts = R.points.map(d => ({
  value:[d.oi,d.perf], symbolSize:Math.max(6,Math.min(24,Math.sqrt(d.flops/100))),
  itemStyle:{color:d.color}, _nid:d.nid, _d:d,
}));
c4.setOption({
  grid: { left: 50, right: 20, top: 24, bottom: 40 },
  xAxis: { type:'log', name:'OI (FLOP/B)', min:oiMin, max:oiMax, nameTextStyle:{color:'#64748b',fontSize:10}, axisLabel:{color:'#64748b',fontSize:10} },
  yAxis: { type:'log', name:'FLOP/cy', min:0.1, nameTextStyle:{color:'#64748b',fontSize:10}, axisLabel:{color:'#64748b',fontSize:10} },
  tooltip: { ...tipStyle, formatter: p => { const r=p.data?._d; if(!r) return '';
    const isComp = r.bottleneck==='compute';
    const bnColor = isComp ? '#fbbf24' : '#60a5fa';
    const bnText = isComp ? '计算受限 (OI ≥ ridge)' : '访存受限 (OI < ridge)';
    return `<b style="color:#38bdf8">${r.name}</b> <span style="color:#94a3b8">[${r.unit}]</span>`
      + `<br/><span style="color:#94a3b8">OI</span> = ${r.flops} ÷ ${r.bytes} = <b>${r.oi}</b>`
      + `<br/><span style="color:#94a3b8">Ridge</span> = ${r.peak} ÷ ${R.bw} = ${r.ridge}`
      + `<br/><b style="color:${bnColor}">${bnText}</b>`
      + `<br/><span style="color:#94a3b8">利用率</span> ${(r.ratio*100).toFixed(0)}%`
      + `<br/><span style="color:#94a3b8">Compute:</span> ${r.comp_cy} cy &ensp;<span style="color:#94a3b8">DMA:</span> ${r.dma_cy} cy &ensp;<span style="color:#94a3b8">Total:</span> ${r.cycles} cy`; } },
  series: [
    { type:'line', data:roof(R.cube_peak), lineStyle:{color:'#E8D0A9',width:2}, symbol:'none', z:1 },
    { type:'line', data:roof(R.vec_peak), lineStyle:{color:'#B8D4E3',width:2,type:'dashed'}, symbol:'none', z:1 },
    { type:'scatter', data:rPts, z:2 },
  ],
});

// ── 5. Fusion (data from fusion_viz.build_fusion_data, echarts-ready) ──
const F = D.fusion;
const fNodes = F.nodes;
const fEdges = F.edges;
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
  pData.forEach((d,i) => { c1.dispatchAction({type: d._nid===nid?'highlight':'downplay', seriesIndex:0, dataIndex:i}); });
  tData.forEach((d,i) => { c3.dispatchAction({type: d._nid===nid?'highlight':'downplay', seriesIndex:0, dataIndex:i}); });
  mData.forEach((d,i) => { c2.dispatchAction({type: d._nid===nid?'highlight':'downplay', seriesIndex:0, dataIndex:i}); });
  rPts.forEach((d,i) => { c4.dispatchAction({type: d._nid===nid?'highlight':'downplay', seriesIndex:2, dataIndex:i}); });
  const fi = fNodes.findIndex(d => (d._nid||d.id)===nid);
  if (fi >= 0) c5.dispatchAction({type:'highlight', seriesIndex:0, dataIndex:fi});
}

[c1,c2,c3,c4,c5].forEach(chart => {
  chart.on('mouseover', p => {
    const nid = p.data?._nid;
    if (nid) highlightAll(nid);
  });
});

// ── 点击：三图同时显示同一 nid 的 tooltip ──
let pinnedNid = null;

function pinTooltips(nid) {
  if (pinnedNid === nid) { unpinTooltips(); return; }
  pinnedNid = nid;

  // Pipeline
  const pi = pData.findIndex(d => d._nid===nid);
  if (pi >= 0) c1.dispatchAction({type:'showTip', seriesIndex:0, dataIndex:pi});

  // TID
  const ti = tData.findIndex(d => d._nid===nid);
  if (ti >= 0) c3.dispatchAction({type:'showTip', seriesIndex:0, dataIndex:ti});

  // Memory — 从 pipeline bar 的 ins/outs 或 DMA nid 提取 tensor ID，匹配 memory 块
  const tensorIds = new Set();
  const pBar = pi >= 0 ? pData[pi] : null;
  if (pBar) {
    (pBar._ins||[]).forEach(t => tensorIds.add(t.id));
    (pBar._outs||[]).forEach(t => tensorIds.add(t.id));
  }
  // DMA nid 格式: __bulk_load_t_0 / __dma_load_node_0_t_2
  const dmaMatch = nid.match(/_(t_\d+)$/);
  if (dmaMatch) tensorIds.add(dmaMatch[1]);

  if (tensorIds.size) {
    mData.forEach((d,i) => {
      if (tensorIds.has(d._nid)) c2.dispatchAction({type:'showTip', seriesIndex:0, dataIndex:i});
    });
  }

  // Roofline — node_id 直接匹配
  const ri = rPts.findIndex(d => d._nid===nid);
  if (ri >= 0) c4.dispatchAction({type:'showTip', seriesIndex:2, dataIndex:ri});

  highlightAll(nid);
}

function unpinTooltips() {
  pinnedNid = null;
  [c1,c2,c3,c4].forEach(c => c.dispatchAction({type:'hideTip'}));
}

timelineCharts.forEach(chart => {
  chart.on('click', p => {
    const nid = p.data?._nid;
    if (nid) pinTooltips(nid);
  });
});
// Esc 也取消 pin
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') unpinTooltips();
});

// ── Tab switching (Roofline / Fusion only) ──
const tabCharts = {rooflineTab:c4, fusionTab:c5};
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    const panel = document.getElementById(tab.dataset.tab);
    panel.classList.add('active');
    const chart = tabCharts[tab.dataset.tab];
    if (chart) setTimeout(() => chart.resize(), 10);
  });
});

// ── Resize ──
window.addEventListener('resize', () => { [c1,c2,c3,c4,c5].forEach(c=>c.resize()); });

// ── Excel Export ──
function exportXlsx() {
  if (typeof XLSX === 'undefined') { alert('SheetJS not loaded'); return; }

  const wb = XLSX.utils.book_new();

  // Sheet 1: Pipeline Schedule
  const pRows = (D.pipeline.bars||[]).map(b => ({
    nid: b.nid, op: b.op, lane: lanes[b.lane]||'?',
    start_cycle: b.s, end_cycle: b.e, duration: b.d, tid: b.tid,
    inputs: (b.ins||[]).map(t => t.id + ' ' + t.desc).join('; '),
    outputs: (b.outs||[]).map(t => t.id + ' ' + t.desc).join('; '),
  }));
  XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(pRows), 'Pipeline');

  // Sheet 2: Memory Lifetime
  const mRows = (D.memory.blocks||[]).map(b => {
    const v = b.value; const zone = v[5]||'';
    const base = zone==='HBM' ? (D.memory.hbm_base||0) : 0;
    return {
      tensor: b.name, zone: zone, shape: b.shape, dtype: b.dtype, format: b.fmt,
      addr_start: v[1]-base, addr_end: v[3]-base, size_bytes: v[4],
      start_cycle: v[0], end_cycle: v[2],
    };
  });
  XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(mRows), 'Memory');

  // Sheet 3: TID Schedule
  const tRows = D.tid.map(d => ({
    tid: d.tid, nid: d.nid, unit: d.unit, label: d.label,
    type: d.type, main_path: d.main,
    start_cycle: d.start, end_cycle: d.end, duration: d.dur,
    dep_cube: d.deps?.cube||0, dep_vector: d.deps?.vector||0,
    dep_dma: d.deps?.dma||0, dep_idma: d.deps?.idma||0,
  }));
  XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(tRows), 'TID');

  // Sheet 4: Roofline
  const rRows = (D.roofline.points||[]).map(d => ({
    nid: d.nid, name: d.name, unit: d.unit,
    oi: d.oi, perf_flop_per_cy: d.perf, flops: d.flops,
    bottleneck: d.bottleneck, cycles: d.cycles,
  }));
  const rMeta = [
    {param:'cube_peak', value:D.roofline.cube_peak},
    {param:'vec_peak', value:D.roofline.vec_peak},
    {param:'bandwidth', value:D.roofline.bw},
  ];
  const rSheet = XLSX.utils.json_to_sheet(rMeta);
  XLSX.utils.sheet_add_json(rSheet, rRows, {origin:'A5'});
  XLSX.utils.book_append_sheet(wb, rSheet, 'Roofline');

  // Sheet 5: Fusion
  const fRows = (D.fusion.nodes||[]).map(d => ({
    nid: d.id||d._nid, unit: d._unit, group: d._group||'', role: d._role||'',
  }));
  XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(fRows), 'Fusion');

  XLSX.writeFile(wb, 'torch2c_schedule.xlsx');
}
</script>
</body></html>
"""
