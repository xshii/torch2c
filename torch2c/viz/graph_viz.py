"""graph_viz — Pipeline Schedule 甘特图（MindSpore 风格，ECharts 交互式 HTML）。

横轴时间（cycles），纵轴 4 条泳道（cube / vector / dma / idma）。
每个算子是一个水平色块，DMA 搬运显示在 DMA 泳道，依赖用箭头连线。
绑定到 memory_planner（⑧）之后自动生成。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from torch2c.common import Graph, get_logger
from torch2c.viz._utils import CU_COLOR, ensure_viz_dir, shape_str
from torch2c.viz.cost_model import estimate_all

logger = get_logger(__name__)

_LANES = ["cube", "vector", "dma", "idma"]
_LANE_INDEX = {name: i for i, name in enumerate(_LANES)}

# DMA 搬运的估算代价（cycles per KB）
_DMA_CYCLES_PER_KB = 2


@dataclass
class ScheduledOp:
    """调度后的算子。"""

    nid: str
    op: str
    lane: str
    lane_idx: int
    start: int
    end: int
    duration: int
    tid: int
    deps: list[str] = field(default_factory=list)
    is_summary: bool = False  # tiled op 的 summary 条，仅供 lifetime_viz 时间查找


def _estimate_dma_cost(size_bytes: int) -> int:
    """估算 DMA 搬运代价。"""
    return max(1, (size_bytes // 1024) * _DMA_CYCLES_PER_KB)


def _schedule_single_op(
    scheduled: list, lane_end: dict, op_end: dict,
    node, nid: str, cu: str, dp, costs: dict,
    bulk_load,
) -> None:
    """调度单个非 tiled 算子（load → compute → store）。"""
    load_nid = None
    if dp and dp.loads:
        load_nid = f"__dma_load_{nid}"
        total_size = sum(inst.size_bytes for inst in dp.loads)
        dur = _estimate_dma_cost(total_size)
        dep_ready = max((op_end.get(d, 0) for d in node.dependencies), default=0)
        start = max(lane_end["dma"], dep_ready)
        end = start + dur
        lane_end["dma"] = end
        op_end[load_nid] = end
        scheduled.append(ScheduledOp(
            nid=load_nid, op=f"dma_load ({len(dp.loads)})",
            lane="dma", lane_idx=_LANE_INDEX["dma"],
            start=start, end=end, duration=dur, tid=0,
            deps=list(node.dependencies),
        ))

    duration = costs.get(nid, 1)
    dep_ready = max((op_end.get(d, 0) for d in node.dependencies), default=0)
    if bulk_load and "__bulk_load__" in op_end:
        dep_ready = max(dep_ready, op_end["__bulk_load__"])
    if load_nid and load_nid in op_end:
        dep_ready = max(dep_ready, op_end[load_nid])
    start = max(lane_end[cu], dep_ready)
    end = start + duration
    lane_end[cu] = end
    op_end[nid] = end

    compute_deps = list(node.dependencies)
    if load_nid:
        compute_deps.append(load_nid)
    elif bulk_load:
        compute_deps.append("__bulk_load__")
    scheduled.append(ScheduledOp(
        nid=nid, op=node.npu_op or node.op_type,
        lane=cu, lane_idx=_LANE_INDEX[cu],
        start=start, end=end, duration=duration,
        tid=node.task_id, deps=compute_deps,
    ))

    if dp and dp.stores:
        store_nid = f"__dma_store_{nid}"
        total_size = sum(inst.size_bytes for inst in dp.stores)
        dur = _estimate_dma_cost(total_size)
        start = max(lane_end["dma"], op_end[nid])
        end = start + dur
        lane_end["dma"] = end
        op_end[store_nid] = end
        scheduled.append(ScheduledOp(
            nid=store_nid, op=f"dma_store ({len(dp.stores)})",
            lane="dma", lane_idx=_LANE_INDEX["dma"],
            start=start, end=end, duration=dur, tid=0,
            deps=[nid],
        ))


def _schedule_tiled_op(
    scheduled: list, lane_end: dict, op_end: dict,
    node, nid: str, cu: str, dp, costs: dict,
    num_tiles: int, bulk_load,
) -> None:
    """调度 tiled 算子：每个 tile 独立 load → compute → store。

    num_buffers > 1 时使用流水线依赖：load_t{i+1} 不等 store_t{i}，
    而是等 load_t{i}（DMA lane 串行）+ 缓冲释放（store_t{i-num_buffers+1}）。
    """
    tile_info = dp.tile_info if dp else None
    num_buffers = tile_info.get("num_buffers", 1) if tile_info else 1
    pipelined = num_buffers > 1

    per_tile_cost = max(1, costs.get(nid, 1) // num_tiles)
    load_size = sum(inst.size_bytes for inst in dp.loads) if dp and dp.loads else 0
    store_size = sum(inst.size_bytes for inst in dp.stores) if dp and dp.stores else 0

    dep_ready = max((op_end.get(d, 0) for d in node.dependencies), default=0)
    if bulk_load and "__bulk_load__" in op_end:
        dep_ready = max(dep_ready, op_end["__bulk_load__"])

    prev_store_nid = None
    # 跟踪整个 tiled op 的时间范围
    first_load_start = None
    first_comp_start = None
    last_store_end = None
    last_comp_end = None
    # 流水线用：记录每个 tile 的 store nid，用于缓冲释放依赖
    store_nids: list[str] = []

    for ti in range(num_tiles):
        if pipelined:
            # 流水线依赖：load_t{i} 只等 DMA lane + 缓冲释放
            # 缓冲释放 = store_t{i - num_buffers + 1} 完成
            buf_release_idx = ti - num_buffers + 1
            if buf_release_idx >= 0 and buf_release_idx < len(store_nids):
                tile_dep = op_end.get(store_nids[buf_release_idx], dep_ready)
            else:
                tile_dep = dep_ready
        else:
            # 单缓冲：load_t{i} 等 store_t{i-1}
            tile_dep = dep_ready if ti == 0 else op_end.get(prev_store_nid, dep_ready)

        # load
        load_nid = f"__dma_load_{nid}_t{ti}"
        if load_size > 0:
            dur = _estimate_dma_cost(load_size)
            start = max(lane_end["dma"], tile_dep)
            end = start + dur
            lane_end["dma"] = end
            op_end[load_nid] = end
            if pipelined:
                ldeps = list(node.dependencies) if ti == 0 else [f"__dma_load_{nid}_t{ti-1}"]
            else:
                ldeps = list(node.dependencies) if ti == 0 else [prev_store_nid]
            scheduled.append(ScheduledOp(
                nid=load_nid, op=f"tile{ti}_load",
                lane="dma", lane_idx=_LANE_INDEX["dma"],
                start=start, end=end, duration=dur, tid=0, deps=ldeps,
            ))
            if first_load_start is None:
                first_load_start = start

        # compute
        comp_nid = f"{nid}_t{ti}" if num_tiles > 1 else nid
        comp_dep = op_end.get(load_nid, tile_dep) if load_size > 0 else tile_dep
        start = max(lane_end[cu], comp_dep)
        end = start + per_tile_cost
        lane_end[cu] = end
        op_end[comp_nid] = end
        cdeps = [load_nid] if load_size > 0 else []
        op_label = f"{node.npu_op or node.op_type} [t{ti}/{num_tiles}]"
        scheduled.append(ScheduledOp(
            nid=comp_nid, op=op_label,
            lane=cu, lane_idx=_LANE_INDEX[cu],
            start=start, end=end, duration=per_tile_cost,
            tid=node.task_id, deps=cdeps,
        ))
        if first_comp_start is None:
            first_comp_start = start
        last_comp_end = end

        # store
        store_nid = f"__dma_store_{nid}_t{ti}"
        if store_size > 0:
            dur = _estimate_dma_cost(store_size)
            start = max(lane_end["dma"], op_end[comp_nid])
            end = start + dur
            lane_end["dma"] = end
            op_end[store_nid] = end
            scheduled.append(ScheduledOp(
                nid=store_nid, op=f"tile{ti}_store",
                lane="dma", lane_idx=_LANE_INDEX["dma"],
                start=start, end=end, duration=dur, tid=0, deps=[comp_nid],
            ))
            prev_store_nid = store_nid
            last_store_end = end
            store_nids.append(store_nid)
        else:
            prev_store_nid = comp_nid
            store_nids.append(comp_nid)

    # ── 添加 summary ScheduledOp 供 lifetime_viz 查找时间 ──
    # 计算 summary：base nid 覆盖整个 tiled op 时间范围
    op_start = first_load_start if first_load_start is not None else (first_comp_start or 0)
    op_final = last_store_end if last_store_end is not None else (last_comp_end or 0)
    op_end[nid] = op_final

    # DMA load summary（从第一个 tile load 开始到第一个 tile load 结束）
    if first_load_start is not None:
        first_load_end = op_end.get(f"__dma_load_{nid}_t0", first_load_start)
        summary_load_nid = f"__dma_load_{nid}"
        op_end[summary_load_nid] = first_load_end
        scheduled.append(ScheduledOp(
            nid=summary_load_nid, op=f"dma_load (×{num_tiles})",
            lane="dma", lane_idx=_LANE_INDEX["dma"],
            start=first_load_start, end=first_load_end,
            duration=first_load_end - first_load_start, tid=0,
            is_summary=True,
        ))

    # DMA store summary（最后一个 tile store）
    if last_store_end is not None:
        last_tile = num_tiles - 1
        last_store_nid = f"__dma_store_{nid}_t{last_tile}"
        last_store_start = op_end.get(last_store_nid, last_store_end) - (last_store_end - op_end.get(last_store_nid, last_store_end))
        summary_store_nid = f"__dma_store_{nid}"
        # 找最后 tile store 的 start time
        for sop in reversed(scheduled):
            if sop.nid == last_store_nid:
                last_store_start = sop.start
                break
        op_end[summary_store_nid] = last_store_end
        scheduled.append(ScheduledOp(
            nid=summary_store_nid, op=f"dma_store (×{num_tiles})",
            lane="dma", lane_idx=_LANE_INDEX["dma"],
            start=last_store_start, end=last_store_end,
            duration=last_store_end - last_store_start, tid=0,
            is_summary=True,
        ))

    # Compute summary（覆盖所有 tile 的计算范围）
    scheduled.append(ScheduledOp(
        nid=nid, op=f"{node.npu_op or node.op_type} (×{num_tiles})",
        lane=cu, lane_idx=_LANE_INDEX[cu],
        start=first_comp_start or 0, end=last_comp_end or 0,
        duration=(last_comp_end or 0) - (first_comp_start or 0), tid=node.task_id,
        is_summary=True,
    ))


def _schedule_ops(
    graph: Graph, costs: dict[str, int], dma_plans: list | None = None,
) -> list[ScheduledOp]:
    """贪心流水线调度，含 DMA 搬运。"""
    lane_end = {lane: 0 for lane in _LANES}
    op_end: dict[str, int] = {}
    scheduled: list[ScheduledOp] = []

    # 构建 node_id → DmaPlan 索引
    dma_map: dict[str, object] = {}
    if dma_plans:
        for dp in dma_plans:
            dma_map[dp.node_id] = dp

    # bulk DMA 在最前面
    bulk_load = dma_map.pop("__bulk_load__", None)
    bulk_store = dma_map.pop("__bulk_store__", None)

    # ── bulk load ──
    if bulk_load and bulk_load.loads:
        total_size = sum(inst.size_bytes for inst in bulk_load.loads)
        dur = _estimate_dma_cost(total_size)
        start = lane_end["dma"]
        end = start + dur
        lane_end["dma"] = end
        op_end["__bulk_load__"] = end
        tids = [inst.tensor_id for inst in bulk_load.loads]
        scheduled.append(ScheduledOp(
            nid="__bulk_load__", op=f"bulk_load ({len(bulk_load.loads)} tensors)",
            lane="dma", lane_idx=_LANE_INDEX["dma"],
            start=start, end=end, duration=dur, tid=0,
        ))

    for nid in graph.execution_order:
        node = graph.nodes[nid]
        cu = (node.compute_unit or "vector").lower()
        if cu not in _LANE_INDEX:
            cu = "vector"

        dp = dma_map.get(nid)
        tile_info = dp.tile_info if dp else None
        num_tiles = tile_info.get("num_tiles", 1) if tile_info else 1

        if num_tiles > 1:
            # ── tiled op: 重复 load → compute → store × num_tiles ──
            _schedule_tiled_op(
                scheduled, lane_end, op_end, node, nid, cu, dp, costs,
                num_tiles, bulk_load,
            )
        else:
            # ── 非 tiled: 单次 load → compute → store ──
            _schedule_single_op(
                scheduled, lane_end, op_end, node, nid, cu, dp, costs,
                bulk_load,
            )

    # ── bulk store ──
    if bulk_store and bulk_store.stores:
        total_size = sum(inst.size_bytes for inst in bulk_store.stores)
        dur = _estimate_dma_cost(total_size)
        # bulk store 需要等所有计算完成
        all_done = max(op_end.values()) if op_end else 0
        start = max(lane_end["dma"], all_done)
        end = start + dur
        lane_end["dma"] = end
        op_end["__bulk_store__"] = end
        # 依赖最后一个计算节点
        last_compute = graph.execution_order[-1] if graph.execution_order else None
        scheduled.append(ScheduledOp(
            nid="__bulk_store__", op=f"bulk_store ({len(bulk_store.stores)} tensors)",
            lane="dma", lane_idx=_LANE_INDEX["dma"],
            start=start, end=end, duration=dur, tid=0,
            deps=[last_compute] if last_compute else [],
        ))

    return scheduled


def render_schedule(
    graph: Graph, cube_size: int,
    hw_config: dict | None = None,
    dma_plans: list | None = None,
) -> str:
    """生成 MindSpore 风格 Pipeline Schedule HTML。"""
    costs = estimate_all(graph, hw_config)
    ops = _schedule_ops(graph, costs, dma_plans)
    if not ops:
        return "<html><body>No ops</body></html>"

    max_time = max(o.end for o in ops)
    # op_map: 非 summary 优先，summary 仅在无 per-tile 条目时补位
    op_map: dict[str, ScheduledOp] = {}
    for o in ops:
        if o.is_summary:
            op_map.setdefault(o.nid, o)  # 仅在 nid 不存在时填充
        else:
            op_map[o.nid] = o

    def _tensor_desc(tid: str) -> str:
        t = graph.tensors.get(tid)
        if not t:
            return tid
        parts = [shape_str(t.shape), t.dtype or "?"]
        if t.format and t.format != "nd":
            parts.append(t.format)
        return " ".join(parts)

    bars = []
    for o in ops:
        if o.is_summary:
            continue  # summary 条仅供 lifetime_viz，不在甘特图中显示
        node = graph.nodes.get(o.nid)
        if node:
            ins = [{"id": tid, "desc": _tensor_desc(tid)} for tid in node.inputs]
            outs = [{"id": tid, "desc": _tensor_desc(tid)} for tid in node.outputs]
        else:
            ins, outs = [], []
        bars.append({
            "s": o.start, "e": o.end, "d": o.duration,
            "lane": o.lane_idx, "op": o.op, "tid": o.tid, "nid": o.nid,
            "color": CU_COLOR.get(o.lane, "#999"),
            "ins": ins, "outs": outs,
        })

    # 依赖线（跳过 summary → summary，per-tile 箭头独立展示）
    deps = []
    for o in ops:
        if o.is_summary:
            continue  # summary 条不生成出向箭头
        for dep_nid in o.deps:
            dep = op_map.get(dep_nid)
            if dep:
                deps.append({
                    "x1": dep.end, "y1": dep.lane_idx,
                    "x2": o.start, "y2": o.lane_idx,
                })

    bars_json = json.dumps(bars, ensure_ascii=False)
    deps_json = json.dumps(deps)
    lanes_json = json.dumps([l.upper() for l in _LANES])
    colors_json = json.dumps([CU_COLOR.get(l, "#999") for l in _LANES])

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>NPU Pipeline Schedule</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
  body {{ margin:0; padding:16px; font-family:sans-serif; background:#fafafa; }}
  #chart {{ width:100%; height:90vh; }}
</style>
</head><body>
<div id="chart"></div>
<script>
var chart = echarts.init(document.getElementById('chart'));
var bars = {bars_json};
var deps = {deps_json};
var lanes = {lanes_json};
var laneColors = {colors_json};
var maxTime = {max_time};
var laneH = 60;   // 每条泳道像素高度
var laneGap = 20;  // 泳道间距
var nLanes = lanes.length;

function renderItem(params, api) {{
    var s = api.value(0);  // start
    var lane = api.value(1); // lane_idx
    var dur = api.value(2); // duration

    var x0 = api.coord([s, 0])[0];
    var x1 = api.coord([s + dur, 0])[0];
    var barW = Math.max(x1 - x0, 2);

    var chartH = api.getHeight();
    var gridTop = 70;
    var gridBot = 50;
    var usable = chartH - gridTop - gridBot;
    var totalH = nLanes * laneH + (nLanes - 1) * laneGap;
    var yOff = gridTop + (usable - totalH) / 2;

    var y = yOff + lane * (laneH + laneGap);

    var rect = echarts.graphic.clipRectByRect(
        {{ x: x0, y: y + 4, width: barW, height: laneH - 8 }},
        {{ x: params.coordSys.x, y: params.coordSys.y,
           width: params.coordSys.width, height: params.coordSys.height }}
    );

    return rect && {{
        type: 'rect',
        shape: {{ x: rect.x, y: rect.y, width: rect.width, height: rect.height, r: 3 }},
        style: Object.assign({{}}, api.style(), {{ stroke: '#333', lineWidth: 0.5 }}),
        textContent: {{
            style: {{
                text: barW > 30 ? api.value(3) : '',
                fill: '#333',
                fontSize: 9,
                overflow: 'truncate',
                width: barW - 4
            }}
        }},
        textConfig: {{ position: 'inside' }}
    }};
}}

// 泳道背景 + 标签
var bgGraphics = [];
(function() {{
    var gridTop = 70, gridBot = 50;
    function calcY(chartH) {{
        var usable = chartH - gridTop - gridBot;
        var totalH = nLanes * laneH + (nLanes - 1) * laneGap;
        return gridTop + (usable - totalH) / 2;
    }}

    for (var i = 0; i < nLanes; i++) {{
        // 泳道背景条
        bgGraphics.push({{
            type: 'rect', z: -10,
            shape: {{ x: 0, y: 0, width: 0, height: laneH }},
            style: {{ fill: laneColors[i] + '15' }},
            _lane: i, _type: 'bg'
        }});
        // 泳道标签
        bgGraphics.push({{
            type: 'text', z: 10,
            style: {{
                text: lanes[i], x: 10, y: 0,
                fontSize: 13, fontWeight: 'bold',
                fill: laneColors[i]
            }},
            _lane: i, _type: 'label'
        }});
    }}
}})();

var option = {{
    title: {{
        text: 'NPU Pipeline Schedule',
        subtext: bars.length + ' ops, ~' + maxTime + ' cycles',
        left: 'center'
    }},
    tooltip: {{
        formatter: function(p) {{
            if (!p.value) return '';
            var b = p.data._bar;
            var s = '<b>' + p.value[3] + '</b> <span style="color:#888">(' + b.nid + ')</span><br/>'
                + 'Lane: ' + lanes[p.value[1]] + '&ensp;TID: ' + p.value[4] + '<br/>'
                + 'Time: ' + p.value[0] + ' → ' + (p.value[0]+p.value[2]) + ' (' + p.value[2] + ' cycles)<br/>';
            if (b.ins && b.ins.length) {{
                s += '<br/><b>Inputs:</b><br/>';
                b.ins.forEach(function(t) {{ s += '&ensp;' + t.id + ' <span style="color:#666">' + t.desc + '</span><br/>'; }});
            }}
            if (b.outs && b.outs.length) {{
                s += '<b>Outputs:</b><br/>';
                b.outs.forEach(function(t) {{ s += '&ensp;' + t.id + ' <span style="color:#666">' + t.desc + '</span><br/>'; }});
            }}
            return s;
        }}
    }},
    grid: {{ left: 80, right: 30, top: 70, bottom: 50 }},
    xAxis: {{
        type: 'value', name: 'Cycles', min: 0, max: maxTime,
        splitLine: {{ lineStyle: {{ color: '#eee' }} }}
    }},
    yAxis: {{ show: false, min: 0, max: 1 }},
    dataZoom: [
        {{ type: 'slider', xAxisIndex: 0, bottom: 10, start: 0, end: 100 }},
        {{ type: 'inside', xAxisIndex: 0 }}
    ],
    graphic: bgGraphics,
    series: [{{
        type: 'custom',
        renderItem: renderItem,
        encode: {{ x: [0], tooltip: [0,1,2,3,4] }},
        data: bars.map(function(b) {{
            return {{
                value: [b.s, b.lane, b.d, b.op, b.tid],
                itemStyle: {{ color: b.color }},
                _bar: b
            }};
        }})
    }}]
}};

chart.setOption(option);

// 定位泳道背景和标签（需要在渲染后获取坐标系尺寸）
function updateLayout() {{
    var chartW = chart.getWidth();
    var chartH = chart.getHeight();
    var gridTop = 70, gridBot = 50;
    var usable = chartH - gridTop - gridBot;
    var totalH = nLanes * laneH + (nLanes - 1) * laneGap;
    var yOff = gridTop + (usable - totalH) / 2;

    var newGraphic = [];
    for (var i = 0; i < bgGraphics.length; i++) {{
        var g = bgGraphics[i];
        var lane = g._lane;
        var y = yOff + lane * (laneH + laneGap);
        if (g._type === 'bg') {{
            newGraphic.push({{
                type: 'rect', z: -10,
                shape: {{ x: 80, y: y, width: chartW - 110, height: laneH }},
                style: {{ fill: laneColors[lane] + '18' }}
            }});
        }} else {{
            newGraphic.push({{
                type: 'text', z: 10,
                style: {{
                    text: lanes[lane], x: 8, y: y + laneH/2 - 8,
                    fontSize: 13, fontWeight: 'bold', fill: laneColors[lane]
                }}
            }});
        }}
    }}

    // 依赖连线
    deps.forEach(function(d) {{
        var px1 = chart.convertToPixel('grid', [d.x1, 0]);
        var px2 = chart.convertToPixel('grid', [d.x2, 0]);
        if (!px1 || !px2) return;
        var y1 = yOff + d.y1 * (laneH + laneGap) + laneH / 2;
        var y2 = yOff + d.y2 * (laneH + laneGap) + laneH / 2;
        newGraphic.push({{
            type: 'line', z: 5,
            shape: {{ x1: px1[0], y1: y1, x2: px2[0], y2: y2 }},
            style: {{ stroke: 'rgba(180,0,0,0.25)', lineWidth: 1 }}
        }});
        // 小箭头
        var angle = Math.atan2(y2 - y1, px2[0] - px1[0]);
        var sz = 4;
        newGraphic.push({{
            type: 'polygon', z: 5,
            shape: {{ points: [
                [px2[0], y2],
                [px2[0] - sz*Math.cos(angle-0.5), y2 - sz*Math.sin(angle-0.5)],
                [px2[0] - sz*Math.cos(angle+0.5), y2 - sz*Math.sin(angle+0.5)]
            ] }},
            style: {{ fill: 'rgba(180,0,0,0.25)' }}
        }});
    }});

    chart.setOption({{ graphic: newGraphic }});
}}

setTimeout(updateLayout, 300);
chart.on('dataZoom', function() {{ setTimeout(updateLayout, 100); }});
window.addEventListener('resize', function() {{ chart.resize(); setTimeout(updateLayout, 100); }});
</script>
</body></html>"""


# ── 对外接口 ──────────────────────────────────────────────


def emit_graph_html(graph: Graph, output_dir: str, cube_size: int,
                    hw_config: dict | None = None,
                    dma_plans: list | None = None) -> str:
    """生成 Pipeline Schedule HTML，返回文件路径。"""
    viz_dir = ensure_viz_dir(output_dir)
    path = os.path.join(viz_dir, "pipeline.html")
    html = render_schedule(graph, cube_size, hw_config, dma_plans)
    with open(path, "w") as f:
        f.write(html)
    logger.info("Pipeline Schedule HTML 已写入: %s", path)
    return path
