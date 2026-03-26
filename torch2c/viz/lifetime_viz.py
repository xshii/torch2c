"""lifetime_viz — 统一 HBM+L1 内存生命周期图 + DMA 搬移箭头（ECharts HTML）。

单坐标系：L1 在下，HBM 在上，DMA load/store 用箭头连接。
横轴为 DMA 感知的预估时间（cycles），纵轴为统一地址偏移。
绑定到 memory_planner（⑧）之后自动生成。
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass

from torch2c.common import Graph, get_logger
from torch2c.common import calc_padded_size
from torch2c.viz._utils import STORAGE_COLOR, ensure_viz_dir, human_size, shape_str
from torch2c.viz.cost_model import estimate_all

logger = get_logger(__name__)


@dataclass
class MemBlock:
    """内存块。"""

    tid: str
    offset: int
    size: int
    start_cycle: int
    end_cycle: int
    storage: str
    is_weight: bool
    is_input: bool
    is_output: bool
    shape: str = ""
    dtype: str = ""
    fmt: str = ""


# ── 时间线计算 ────────────────────────────────────────────


def _compute_timing(
    graph: Graph, hw_config: dict | None, dma_plans: list | None,
) -> tuple[dict[str, tuple[int, int]], int]:
    """使用 graph_viz._schedule_ops 计算含 DMA 的完整 cycle 时间线。

    Returns:
        (op_timing: {nid: (start_cycle, end_cycle)}, total_cycles)
    """
    from torch2c.viz.graph_viz import _schedule_ops

    costs = estimate_all(graph, hw_config)
    ops = _schedule_ops(graph, costs, dma_plans)
    timing = {op.nid: (op.start, op.end) for op in ops}
    total = max((op.end for op in ops), default=0)
    return timing, total


def _tensor_involved_ops(
    graph: Graph, dma_plans: list | None,
) -> dict[str, list[str]]:
    """构建 tensor_id → 涉及的算子 nid 列表（含 DMA 虚拟节点）。"""
    result: dict[str, list[str]] = defaultdict(list)

    # DMA 算子
    if dma_plans:
        for dp in dma_plans:
            for inst in dp.loads:
                if dp.node_id == "__bulk_load__":
                    nid = "__bulk_load__"
                else:
                    nid = f"__dma_load_{dp.node_id}"
                result[inst.tensor_id].append(nid)
            for inst in dp.stores:
                if dp.node_id == "__bulk_store__":
                    nid = "__bulk_store__"
                else:
                    nid = f"__dma_store_{dp.node_id}"
                result[inst.tensor_id].append(nid)

    # 计算算子
    for nid, node in graph.nodes.items():
        for tid in node.inputs:
            result[tid].append(nid)
        for tid in node.outputs:
            result[tid].append(nid)
        for _, tid in node.absorbed_inputs.items():
            result[tid].append(nid)

    return dict(result)


# ── 块构建 ─────────────────────────────────────────────


def _is_bulk_mode(dma_plans: list | None) -> bool:
    """检测是否为 bulk DMA 模式。"""
    if not dma_plans:
        return False
    return any(dp.node_id == "__bulk_load__" for dp in dma_plans)


def _is_eviction_mode(dma_plans: list | None) -> bool:
    """检测是否为 per-op eviction 模式（每个 DmaPlan 有独立 l1_layout）。"""
    if not dma_plans:
        return False
    return any(dp.l1_layout for dp in dma_plans
               if dp.node_id not in ("__bulk_load__", "__bulk_store__"))


def _build_blocks(
    graph: Graph, cube_size: int,
    op_timing: dict[str, tuple[int, int]], total_cycles: int,
    tensor_ops: dict[str, list[str]],
    mode: str,
    bulk: bool = False,
) -> list[MemBlock]:
    """构建某个内存空间的 DMA 感知块列表。

    mode: "hbm" | "l1"
    bulk: True 时 L1 中的 tensor 驻留到 total_cycles（bulk DMA 不释放 L1）。
    """
    blocks: list[MemBlock] = []
    for tid, t in graph.tensors.items():
        # 按 mode 过滤 + 获取 offset/size
        if mode == "l1":
            if t.l1_offset is None:
                continue
            offset = t.l1_offset
            size = calc_padded_size(t.shape, t.dtype, t.format, (cube_size, cube_size))
        elif mode == "hbm":
            if t.storage != "hbm":
                continue
            if t.hbm_offset is None:
                continue
            offset = t.hbm_offset
            size = t.hbm_size
        else:
            continue

        if offset is None or size is None:
            continue

        # 从涉及的算子中获取 cycle 范围
        ops = tensor_ops.get(tid, [])
        starts = [op_timing[nid][0] for nid in ops if nid in op_timing]
        ends = [op_timing[nid][1] for nid in ops if nid in op_timing]

        if not starts:
            continue

        start_c = min(starts)
        end_c = max(ends)

        # Bulk 模式下 L1 空间不释放，所有 tensor 驻留到结束
        if bulk and mode == "l1":
            end_c = total_cycles

        blocks.append(MemBlock(
            tid=tid, offset=offset, size=size,
            start_cycle=start_c, end_cycle=end_c,
            storage=t.storage,
            is_weight=t.is_weight, is_input=t.is_model_input,
            is_output=t.is_model_output,
            shape=shape_str(t.shape), dtype=t.dtype or "?",
            fmt=t.format or "nd",
        ))

    blocks.sort(key=lambda b: b.offset)
    return blocks


def _build_eviction_l1_blocks(
    graph: Graph, cube_size: int,
    op_timing: dict[str, tuple[int, int]],
    dma_plans: list,
) -> list[MemBlock]:
    """Eviction 模式：按 per-op l1_layout 生成 L1 块，每个算子独立时间段。

    对于 tiled 算子，展开为 N 个 per-tile 块以体现分片执行。
    """
    blocks: list[MemBlock] = []
    for dp in dma_plans:
        if dp.node_id in ("__bulk_load__", "__bulk_store__"):
            continue
        if not dp.l1_layout:
            continue
        nid = dp.node_id
        if nid not in op_timing:
            continue

        tile_info = dp.tile_info
        num_tiles = tile_info.get("num_tiles", 1) if tile_info else 1

        if num_tiles > 1:
            # ── tiled：为每个 tile 生成独立的 L1 块 ──
            for ti in range(num_tiles):
                load_nid = f"__dma_load_{nid}_t{ti}"
                store_nid = f"__dma_store_{nid}_t{ti}"
                comp_nid = f"{nid}_t{ti}"
                # 从 per-tile ScheduledOp 获取时间范围
                tile_start = None
                tile_end = None
                for k in (load_nid, comp_nid, store_nid):
                    if k in op_timing:
                        s, e = op_timing[k]
                        tile_start = min(tile_start, s) if tile_start is not None else s
                        tile_end = max(tile_end, e) if tile_end is not None else e
                if tile_start is None:
                    continue

                for tid, l1_off in dp.l1_layout.items():
                    t = graph.tensors.get(tid)
                    if not t:
                        continue
                    if tile_info and tid in tile_info.get("tiled_tensors", {}):
                        dim_idx = tile_info["tiled_tensors"][tid]
                        tiled_shape = list(t.shape)
                        tiled_shape[dim_idx] = tile_info["tile_size"]
                        size = calc_padded_size(tiled_shape, t.dtype, t.format, (cube_size, cube_size))
                        shape_label = shape_str(tiled_shape)
                    else:
                        size = calc_padded_size(t.shape, t.dtype, t.format, (cube_size, cube_size))
                        shape_label = shape_str(t.shape)
                    blocks.append(MemBlock(
                        tid=f"{tid}@{nid}_t{ti}", offset=l1_off, size=size,
                        start_cycle=tile_start, end_cycle=tile_end,
                        storage=t.storage,
                        is_weight=t.is_weight, is_input=t.is_model_input,
                        is_output=t.is_model_output,
                        shape=shape_label, dtype=t.dtype or "?",
                        fmt=t.format or "nd",
                    ))
        else:
            # ── 非 tiled：整个算子一个块 ──
            start_c, end_c = op_timing[nid]
            dma_load_nid = f"__dma_load_{nid}"
            dma_store_nid = f"__dma_store_{nid}"
            if dma_load_nid in op_timing:
                start_c = min(start_c, op_timing[dma_load_nid][0])
            if dma_store_nid in op_timing:
                end_c = max(end_c, op_timing[dma_store_nid][1])

            for tid, l1_off in dp.l1_layout.items():
                t = graph.tensors.get(tid)
                if not t:
                    continue
                size = calc_padded_size(t.shape, t.dtype, t.format, (cube_size, cube_size))
                blocks.append(MemBlock(
                    tid=f"{tid}@{nid}", offset=l1_off, size=size,
                    start_cycle=start_c, end_cycle=end_c,
                    storage=t.storage,
                    is_weight=t.is_weight, is_input=t.is_model_input,
                    is_output=t.is_model_output,
                    shape=shape_str(t.shape), dtype=t.dtype or "?",
                    fmt=t.format or "nd",
                ))
    blocks.sort(key=lambda b: (b.start_cycle, b.offset))
    return blocks


# ── DMA 箭头数据 ──────────────────────────────────────────


def _arrow_cycle(
    op_timing: dict[str, tuple[int, int]], dma_nid: str, nid: str, *, start: bool,
) -> int | None:
    """获取 DMA 箭头的 cycle 位置。"""
    if dma_nid in op_timing:
        t = op_timing[dma_nid]
        return (t[0] + t[1]) // 2
    if nid in op_timing:
        return op_timing[nid][0] if start else op_timing[nid][1]
    return None


def _build_dma_arrows(
    dma_plans: list | None,
    op_timing: dict[str, tuple[int, int]],
    hbm_base: float,
) -> list[list]:
    """构建 DMA 箭头数据: [cycle, from_y, to_y, direction(0=load,1=store), tid, size]。"""
    if not dma_plans:
        return []
    arrows: list[list] = []
    for dp in dma_plans:
        nid = dp.node_id
        for inst in dp.loads:
            dma_nid = nid if nid == "__bulk_load__" else f"__dma_load_{nid}"
            cycle = _arrow_cycle(op_timing, dma_nid, nid, start=True)
            if cycle is None:
                continue
            hbm_y = hbm_base + inst.hbm_offset + inst.size_bytes / 2
            l1_y = inst.l1_offset + inst.size_bytes / 2
            arrows.append([cycle, hbm_y, l1_y, 0, inst.tensor_id,
                           inst.size_bytes])
        for inst in dp.stores:
            dma_nid = nid if nid == "__bulk_store__" else f"__dma_store_{nid}"
            cycle = _arrow_cycle(op_timing, dma_nid, nid, start=False)
            if cycle is None:
                continue
            l1_y = inst.l1_offset + inst.size_bytes / 2
            hbm_y = hbm_base + inst.hbm_offset + inst.size_bytes / 2
            arrows.append([cycle, l1_y, hbm_y, 1, inst.tensor_id,
                           inst.size_bytes])
    return arrows


# ── 统一块数据 ────────────────────────────────────────────


def _unified_block_dict(b: MemBlock, y_offset: float, zone: str) -> dict:
    """将 MemBlock 转换为统一坐标空间的 dict。"""
    tags = []
    if b.is_weight:
        tags.append("W")
    if b.is_input:
        tags.append("I")
    if b.is_output:
        tags.append("O")
    tag_str = f" [{','.join(tags)}]" if tags else ""
    y0 = y_offset + b.offset
    color = "#4169E1" if zone == "L1" else STORAGE_COLOR.get(b.storage, "#999")
    return {
        "name": b.tid, "zone": zone,
        "value": [b.start_cycle, y0, b.end_cycle,
                  y0 + b.size, b.size, zone],
        "label": f"{b.tid} {human_size(b.size)}{tag_str}",
        "color": color,
        "shape": b.shape, "dtype": b.dtype, "fmt": b.fmt,
    }


# ── 渲染 ──────────────────────────────────────────────


def render_lifetime(
    graph: Graph, cube_size: int,
    hw_config: dict | None = None,
    dma_plans: list | None = None,
    title: str | None = None,
) -> str:
    """生成统一 HBM+L1 内存生命周期图 + DMA 搬移箭头。"""
    op_timing, total_cycles = _compute_timing(graph, hw_config, dma_plans)
    tensor_ops = _tensor_involved_ops(graph, dma_plans)
    bulk = _is_bulk_mode(dma_plans)
    eviction = _is_eviction_mode(dma_plans)

    hbm_blocks = _build_blocks(graph, cube_size, op_timing, total_cycles,
                               tensor_ops, "hbm")
    if eviction:
        l1_blocks = _build_eviction_l1_blocks(
            graph, cube_size, op_timing, dma_plans)
    else:
        l1_blocks = _build_blocks(graph, cube_size, op_timing, total_cycles,
                                  tensor_ops, "l1", bulk=bulk)

    hbm_max = max((b.offset + b.size for b in hbm_blocks), default=0)
    l1_max = max((b.offset + b.size for b in l1_blocks), default=0)
    gap = max(l1_max, hbm_max) * 0.12
    hbm_base = l1_max + gap
    total_y = hbm_base + hbm_max

    blocks_data = [_unified_block_dict(b, 0, "L1") for b in l1_blocks]
    blocks_data += [_unified_block_dict(b, hbm_base, "HBM") for b in hbm_blocks]
    arrows_data = _build_dma_arrows(dma_plans, op_timing, hbm_base)

    return _render_html(
        json.dumps(blocks_data, ensure_ascii=False),
        json.dumps(arrows_data, ensure_ascii=False),
        total_cycles, l1_max, hbm_max, hbm_base, total_y, gap, title,
    )


def _render_html(
    blocks_json: str, arrows_json: str,
    total_cycles: int, l1_max: int, hbm_max: int,
    hbm_base: float, total_y: float, gap: float, title: str | None,
) -> str:
    """生成统一可视化 HTML 模板。"""
    page_title = f"Unified Memory — {title}" if title else "Unified Memory Lifetime"
    chart_title = page_title
    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>{page_title}</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
  body {{ margin:0; padding:16px; font-family:sans-serif; background:#fafafa; }}
  #chart {{ width:100%; height:92vh; }}
  .legend {{ position:absolute; top:12px; right:60px; font-size:12px;
             background:rgba(255,255,255,0.9); padding:6px 12px; border-radius:4px;
             box-shadow:0 1px 4px rgba(0,0,0,0.15); z-index:10; }}
  .legend span {{ margin-right:12px; }}
  .legend i {{ display:inline-block; width:12px; height:12px; margin-right:4px;
               vertical-align:middle; border-radius:2px; }}
</style>
</head><body>
<div class="legend">
  <span><i style="background:#4169E1"></i>L1</span>
  <span><i style="background:#999"></i>HBM</span>
  <span><i style="background:rgba(65,105,225,0.7)"></i>Load (HBM&rarr;L1)</span>
  <span><i style="background:rgba(225,65,65,0.7)"></i>Store (L1&rarr;HBM)</span>
</div>
<div id="chart"></div>
<script>
var chart = echarts.init(document.getElementById('chart'));
var blocksData = {blocks_json};
var arrowsData = {arrows_json};
var maxTime = {total_cycles};
var l1Max = {l1_max};
var hbmMax = {hbm_max};
var hbmBase = {hbm_base};
var totalY = {total_y};

function fmtAddr(v) {{ return (v/1024).toFixed(0) + 'K'; }}

function renderBlock(params, api) {{
    var x0 = api.value(0), y0 = api.value(1);
    var x1 = api.value(2), y1 = api.value(3);
    var p0 = api.coord([x0, y1]);
    var p1 = api.coord([x1, y0]);
    var rect = echarts.graphic.clipRectByRect(
        {{ x: p0[0], y: p0[1], width: Math.max(p1[0]-p0[0], 2), height: Math.max(p1[1]-p0[1], 2) }},
        {{ x: params.coordSys.x, y: params.coordSys.y,
           width: params.coordSys.width, height: params.coordSys.height }}
    );
    return rect && {{
        type: 'rect',
        shape: {{ x: rect.x, y: rect.y, width: rect.width, height: rect.height, r: 1 }},
        style: {{ fill: api.visual('color'), stroke: '#fff', lineWidth: 0.5 }}
    }};
}}

function renderArrow(params, api) {{
    var cycle = api.value(0);
    var fromY = api.value(1);
    var toY = api.value(2);
    var dir = api.value(3);
    var from = api.coord([cycle, fromY]);
    var to = api.coord([cycle, toY]);
    var color = dir === 0 ? 'rgba(65,105,225,0.7)' : 'rgba(225,65,65,0.7)';
    var dx = to[0] - from[0], dy = to[1] - from[1];
    var len = Math.sqrt(dx*dx + dy*dy);
    if (len < 2) return null;
    var nx = dx/len, ny = dy/len;
    var sz = 6;
    return {{
        type: 'group',
        children: [
            {{ type: 'line',
               shape: {{ x1: from[0], y1: from[1], x2: to[0], y2: to[1] }},
               style: {{ stroke: color, lineWidth: 1.5 }} }},
            {{ type: 'polygon',
               shape: {{ points: [
                   [to[0], to[1]],
                   [to[0] - sz*nx + sz*0.4*ny, to[1] - sz*ny - sz*0.4*nx],
                   [to[0] - sz*nx - sz*0.4*ny, to[1] - sz*ny + sz*0.4*nx]
               ] }},
               style: {{ fill: color }} }}
        ]
    }};
}}

function fmtBlockTooltip(params) {{
    if (!params.value) return '';
    var v = params.value; var d = params.data;
    var fmtStr = d.fmt && d.fmt !== 'nd' ? ' ' + d.fmt : '';
    var zone = v[5] || '';
    var addrBase = zone === 'HBM' ? v[1] - hbmBase : v[1];
    var addrEnd = zone === 'HBM' ? v[3] - hbmBase : v[3];
    return '<b>' + d.label + '</b> [' + zone + ']<br/>'
        + '<span style="color:#666">' + d.shape + ' ' + d.dtype + fmtStr + '</span><br/>'
        + 'Address: ' + fmtAddr(addrBase) + ' &rarr; ' + fmtAddr(addrEnd) + '<br/>'
        + 'Time: ' + v[0] + ' &rarr; ' + v[2] + ' cycles<br/>'
        + 'Size: ' + (v[4]/1024).toFixed(1) + 'K';
}}

function fmtArrowTooltip(params) {{
    if (!params.value) return '';
    var v = params.value;
    var dir = v[3] === 0 ? 'Load (HBM&rarr;L1)' : 'Store (L1&rarr;HBM)';
    var tid = v[4] || '';
    return '<b>DMA ' + dir + '</b><br/>'
        + 'Tensor: ' + tid + '<br/>'
        + 'Size: ' + (v[5]/1024).toFixed(1) + 'K<br/>'
        + 'Cycle: ' + v[0];
}}

var option = {{
    title: {{ text: '{chart_title}', left: 'center', top: 0,
              subtext: blocksData.length + ' tensors, ' + arrowsData.length + ' DMA transfers' }},
    tooltip: {{ formatter: function(p) {{
        return p.seriesIndex === 0 ? fmtBlockTooltip(p) : fmtArrowTooltip(p);
    }} }},
    grid: {{ left: 90, right: 40, top: 60, bottom: 60 }},
    xAxis: {{ type: 'value', name: 'Cycles', min: 0, max: maxTime,
              splitLine: {{ lineStyle: {{ color: '#eee' }} }} }},
    yAxis: {{ type: 'value', name: 'Address', min: 0, max: totalY,
              axisLabel: {{ formatter: function(v) {{
                  if (v >= hbmBase) return 'HBM ' + fmtAddr(v - hbmBase);
                  if (v <= l1Max) return 'L1 ' + fmtAddr(v);
                  return '';
              }} }},
              splitLine: {{ show: false }} }},
    dataZoom: [
        {{ type: 'slider', xAxisIndex: 0, bottom: 5, start: 0, end: 100 }},
        {{ type: 'inside', xAxisIndex: 0 }},
        {{ type: 'slider', yAxisIndex: 0, right: 5, start: 0, end: 100, orient: 'vertical' }},
        {{ type: 'inside', yAxisIndex: 0, orient: 'vertical' }}
    ],
    series: [
        {{
            type: 'custom', name: 'Memory Blocks',
            renderItem: renderBlock,
            encode: {{ x: [0,2], y: [1,3] }},
            data: blocksData.map(function(d) {{
                return {{ name: d.name, value: d.value, label: d.label,
                          itemStyle: {{ color: d.color }},
                          shape: d.shape, dtype: d.dtype, fmt: d.fmt }};
            }}),
            label: {{
                show: true, position: 'inside', fontSize: 8, color: '#fff',
                overflow: 'truncate',
                formatter: function(p) {{ return p.data.label; }}
            }}
        }},
        {{
            type: 'custom', name: 'DMA Transfers',
            renderItem: renderArrow,
            encode: {{ x: 0, y: [1,2] }},
            data: arrowsData.map(function(d) {{
                return {{ value: d }};
            }}),
            z: 10
        }}
    ],
    graphic: [
        {{ type: 'text', left: 10, top: '30%',
           style: {{ text: 'HBM', fill: '#999', fontSize: 14, fontWeight: 'bold',
                     textAlign: 'center' }},
           rotation: -Math.PI/2 }},
        {{ type: 'text', left: 10, top: '70%',
           style: {{ text: 'L1', fill: '#4169E1', fontSize: 14, fontWeight: 'bold',
                     textAlign: 'center' }},
           rotation: -Math.PI/2 }}
    ]
}};

chart.setOption(option);

/* 添加区域分隔标记线 */
chart.setOption({{
    series: [{{
        markLine: {{
            silent: true, symbol: 'none',
            lineStyle: {{ color: '#aaa', type: 'dashed', width: 1.5 }},
            data: [{{ yAxis: hbmBase - {gap} * 0.5,
                      label: {{ show: true, formatter: '── L1 \\u2191 │ \\u2193 HBM ──',
                                position: 'insideEndTop', color: '#888', fontSize: 11 }} }}]
        }}
    }}]
}});

window.addEventListener('resize', function() {{ chart.resize(); }});
</script>
</body></html>"""


# ── 对外接口 ──────────────────────────────────────────────


def emit_lifetime_html(graph: Graph, output_dir: str, cube_size: int,
                       hw_config: dict | None = None,
                       dma_plans: list | None = None,
                       title: str | None = None) -> str:
    """生成统一 HBM+L1 + DMA 箭头图到 output_dir/viz/lifetime.html。"""
    viz_dir = ensure_viz_dir(output_dir)
    path = os.path.join(viz_dir, "lifetime.html")
    html = render_lifetime(graph, cube_size, hw_config=hw_config,
                           dma_plans=dma_plans, title=title)
    with open(path, "w") as f:
        f.write(html)
    logger.info("统一内存生命周期 HTML 已写入: %s", path)
    return path
