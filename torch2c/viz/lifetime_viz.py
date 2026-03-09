"""lifetime_viz — 内存生命周期图（ECharts 交互式 HTML）。

三段独立地址空间上下排列：HBM / L1(local)。
横轴为 DMA 感知的预估时间（cycles），纵轴为各段内部地址偏移。
绑定到 memory_planner（⑧）之后自动生成。
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass

from torch2c.common import Graph, get_logger
from torch2c.memory_planner._utils import calc_padded_size
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
            size = calc_padded_size(t.shape, t.dtype, t.format, cube_size)
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


def _blocks_to_json(blocks: list[MemBlock]) -> list[dict]:
    """将块列表序列化为 ECharts data。"""
    result = []
    for b in blocks:
        tags = []
        if b.is_weight:
            tags.append("W")
        if b.is_input:
            tags.append("I")
        if b.is_output:
            tags.append("O")
        tag_str = f" [{','.join(tags)}]" if tags else ""
        result.append({
            "name": b.tid,
            "value": [b.start_cycle, b.offset, b.end_cycle,
                       b.offset + b.size, b.size, b.storage],
            "label": f"{b.tid} {human_size(b.size)}{tag_str}",
            "color": STORAGE_COLOR.get(b.storage, "#999"),
            "shape": b.shape, "dtype": b.dtype, "fmt": b.fmt,
        })
    return result


# ── 渲染 ──────────────────────────────────────────────


def render_lifetime(graph: Graph, cube_size: int,
                    hw_config: dict | None = None,
                    dma_plans: list | None = None,
                    title: str | None = None) -> str:
    """生成 HBM + L1 双段内存生命周期图 HTML。"""
    op_timing, total_cycles = _compute_timing(graph, hw_config, dma_plans)
    tensor_ops = _tensor_involved_ops(graph, dma_plans)
    bulk = _is_bulk_mode(dma_plans)

    hbm_blocks = _build_blocks(graph, cube_size, op_timing, total_cycles,
                               tensor_ops, "hbm")
    l1_blocks = _build_blocks(graph, cube_size, op_timing, total_cycles,
                              tensor_ops, "l1", bulk=bulk)

    hbm_data = _blocks_to_json(hbm_blocks)
    l1_data = _blocks_to_json(l1_blocks)

    hbm_max = max((b.offset + b.size for b in hbm_blocks), default=0)
    l1_max = max((b.offset + b.size for b in l1_blocks), default=0)

    hbm_json = json.dumps(hbm_data, ensure_ascii=False)
    l1_json = json.dumps(l1_data, ensure_ascii=False)

    page_title = f"Memory Lifetime — {title}" if title else "Memory Lifetime"
    hbm_title = f"HBM Memory Lifetime — {title}" if title else "HBM Memory Lifetime"
    l1_title = f"L1 (Local) Memory Lifetime — {title}" if title else "L1 (Local) Memory Lifetime"

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>{page_title}</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
  body {{ margin:0; padding:16px; font-family:sans-serif; background:#fafafa; }}
  #chart {{ width:100%; height:92vh; }}
</style>
</head><body>
<div id="chart"></div>
<script>
var chart = echarts.init(document.getElementById('chart'));
var hbmData = {hbm_json};
var l1Data = {l1_json};
var maxTime = {total_cycles};
var hbmMax = {hbm_max};
var l1Max = {l1_max};

function makeRenderItem(gridIdx) {{
    return function(params, api) {{
        var x0 = api.value(0);
        var y0 = api.value(1);
        var x1 = api.value(2);
        var y1 = api.value(3);

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
    }};
}}

function fmtAddr(v) {{ return (v/1024).toFixed(0) + 'K'; }}
function fmtTooltip(params) {{
    if (!params.value) return '';
    var v = params.value;
    var d = params.data;
    var fmtStr = d.fmt && d.fmt !== 'nd' ? ' ' + d.fmt : '';
    return '<b>' + d.label + '</b><br/>'
        + '<span style="color:#666">' + d.shape + ' ' + d.dtype + fmtStr + '</span><br/>'
        + 'Address: ' + fmtAddr(v[1]) + ' → ' + fmtAddr(v[3]) + '<br/>'
        + 'Time: ' + v[0] + ' → ' + v[2] + ' cycles<br/>'
        + 'Size: ' + (v[4]/1024).toFixed(1) + 'K<br/>'
        + 'Storage: ' + v[5];
}}

function makeSeries(data, xIdx, yIdx) {{
    return {{
        type: 'custom',
        renderItem: makeRenderItem(0),
        encode: {{ x: [0,2], y: [1,3] }},
        xAxisIndex: xIdx,
        yAxisIndex: yIdx,
        data: data.map(function(d) {{
            return {{
                name: d.name, value: d.value, label: d.label,
                itemStyle: {{ color: d.color }}
            }};
        }}),
        label: {{
            show: true, position: 'inside', fontSize: 8, color: '#fff',
            overflow: 'truncate',
            formatter: function(p) {{ return p.data.label; }}
        }}
    }};
}}

var option = {{
    title: [
        {{ text: '{hbm_title}', left: 'center', top: 0,
           subtext: hbmData.length + ' tensors, ' + fmtAddr(hbmMax) + ' used' }},
        {{ text: '{l1_title}', left: 'center', top: '52%',
           subtext: l1Data.length + ' tensors, ' + fmtAddr(l1Max) + ' used' }}
    ],
    tooltip: {{ formatter: fmtTooltip }},
    grid: [
        {{ left: 80, right: 40, top: 50, height: '38%' }},
        {{ left: 80, right: 40, top: '58%', height: '34%' }}
    ],
    xAxis: [
        {{ type: 'value', name: 'Cycles', min: 0, max: maxTime, gridIndex: 0,
           splitLine: {{ lineStyle: {{ color: '#eee' }} }} }},
        {{ type: 'value', name: 'Cycles', min: 0, max: maxTime, gridIndex: 1,
           splitLine: {{ lineStyle: {{ color: '#eee' }} }} }}
    ],
    yAxis: [
        {{ type: 'value', name: 'HBM (bytes)', min: 0, max: hbmMax || 1, gridIndex: 0,
           axisLabel: {{ formatter: fmtAddr }} }},
        {{ type: 'value', name: 'L1 (bytes)', min: 0, max: l1Max || 1, gridIndex: 1,
           axisLabel: {{ formatter: fmtAddr }} }}
    ],
    dataZoom: [
        {{ type: 'slider', xAxisIndex: [0,1], bottom: 5, start: 0, end: 100 }},
        {{ type: 'inside', xAxisIndex: [0,1] }},
        {{ type: 'slider', yAxisIndex: 0, right: 5, start: 0, end: 100, orient: 'vertical',
           top: 50, height: '38%' }},
        {{ type: 'slider', yAxisIndex: 1, right: 5, start: 0, end: 100, orient: 'vertical',
           top: '58%', height: '34%' }}
    ],
    series: [
        makeSeries(hbmData, 0, 0),
        makeSeries(l1Data, 1, 1)
    ]
}};

chart.setOption(option);
window.addEventListener('resize', function() {{ chart.resize(); }});
</script>
</body></html>"""


# ── 对外接口 ──────────────────────────────────────────────


def emit_lifetime_html(graph: Graph, output_dir: str, cube_size: int,
                       hw_config: dict | None = None,
                       dma_plans: list | None = None,
                       title: str | None = None) -> str:
    """生成内存生命周期图到 output_dir/viz/lifetime.html，返回文件路径。"""
    viz_dir = ensure_viz_dir(output_dir)
    path = os.path.join(viz_dir, "lifetime.html")
    html = render_lifetime(graph, cube_size, hw_config=hw_config,
                           dma_plans=dma_plans, title=title)
    with open(path, "w") as f:
        f.write(html)
    logger.info("内存生命周期 HTML 已写入: %s", path)
    return path
