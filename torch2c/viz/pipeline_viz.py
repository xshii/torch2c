"""pipeline_viz — 编译器 Pass 流水线可视化。

从 pipeline.get_pass_topology() 自动生成 pass 拓扑，
无需手动维护 pass 列表。新增/删除 pass 时自动适配。
"""

from __future__ import annotations

import json
import os
from typing import Any

from torch2c.common import get_logger
from torch2c.viz._utils import ensure_viz_dir

logger = get_logger(__name__)

_PHASE_COLORS = {
    "a_capture": ("#dbeafe", "#93c5fd"),
    "b_lowering": ("#dcfce7", "#86efac"),
    "c_backend": ("#ffedd5", "#fdba74"),
    "d_emission": ("#f3e8ff", "#c4b5fd"),
}


# ── 从 pipeline config 获取 pass 拓扑 ────────────────────


def _get_topology() -> dict:
    from torch2c.integration.pipeline import get_pass_topology
    return get_pass_topology()


# ── 布局 ─────────────────────────────────────────────────


def _compute_layout(topo: dict) -> dict[str, Any]:
    """计算 SVG 坐标。自动从 topo 生成，不硬编码 pass 列表。"""
    passes = topo["passes"]
    phases = topo["phases"]

    required = [p for p in passes if not p["optional"]]
    optional = [p for p in passes if p["optional"]]

    box_w, box_h = 105, 48
    gap_x = 32
    main_y = 160
    margin_left = 40

    # Required pass 位置
    req_nodes: list[dict] = []
    x = margin_left
    name_to_idx: dict[str, int] = {}
    for i, p in enumerate(required):
        req_nodes.append({
            "name": p["name"], "number": p["number"], "phase": p["phase"],
            "x": x, "y": main_y, "w": box_w, "h": box_h,
        })
        name_to_idx[p["name"]] = i
        x += box_w + gap_x
    total_w = x + margin_left

    # Arrows
    arrows: list[dict] = []
    for i in range(len(req_nodes) - 1):
        a, b = req_nodes[i], req_nodes[i + 1]
        arrows.append({
            "x1": a["x"] + box_w, "y1": main_y + box_h // 2,
            "x2": b["x"], "y2": main_y + box_h // 2,
        })

    # Optional passes — find insertion point by adjacent required passes
    arc_y_base = main_y - 60
    opt_nodes: list[dict] = []
    arcs: list[dict] = []

    # Group optionals by their position between required passes
    # An optional pass at position i in the full list sits between required[j] and required[j+1]
    req_positions = [(p["name"], i) for i, p in enumerate(passes) if not p["optional"]]
    opt_groups: dict[tuple[int, int], list[dict]] = {}
    for p in optional:
        pos = next(i for i, pp in enumerate(passes) if pp["name"] == p["name"])
        # Find surrounding required passes
        before = max((ri for _, ri in req_positions if ri < pos), default=0)
        after = min((ri for _, ri in req_positions if ri > pos), default=len(passes) - 1)
        before_name = passes[before]["name"]
        after_name = passes[after]["name"]
        bi = name_to_idx.get(before_name, 0)
        ai = name_to_idx.get(after_name, len(req_nodes) - 1)
        opt_groups.setdefault((bi, ai), []).append(p)

    for (bi, ai), group in opt_groups.items():
        src, dst = req_nodes[bi], req_nodes[ai]
        n = len(group)
        span = dst["x"] - (src["x"] + box_w)
        opt_w = min(85, max(55, (span - (n + 1) * 10) // n))
        opt_h = 36
        total_opt_w = n * opt_w + (n - 1) * 12
        start_x = src["x"] + box_w + (span - total_opt_w) // 2
        arc_y = arc_y_base - 10

        for j, p in enumerate(group):
            opt_nodes.append({
                "name": p["name"], "number": p["number"], "phase": p["phase"],
                "x": start_x + j * (opt_w + 12), "y": arc_y, "w": opt_w, "h": opt_h,
            })

        sx = src["x"] + box_w // 2
        dx = dst["x"] + box_w // 2
        arcs.append({"sx": sx, "sy": main_y, "dx": dx, "dy": main_y,
                      "ctrl_y": arc_y - 30})

    # Phase regions
    phase_regions: list[dict] = []
    phase_map: dict[str, list[int]] = {}
    for i, rn in enumerate(req_nodes):
        phase_map.setdefault(rn["phase"], []).append(i)
    for ph in phases:
        pid = ph["id"]
        if pid not in phase_map:
            continue
        indices = phase_map[pid]
        first, last = req_nodes[indices[0]], req_nodes[indices[-1]]
        px = first["x"] - 15
        pw = (last["x"] + box_w) - first["x"] + 30
        fill, border = _PHASE_COLORS.get(pid, ("#f8fafc", "#cbd5e1"))
        phase_regions.append({
            "label": ph["label"], "fill": fill, "border": border,
            "x": px, "y": arc_y_base - 50, "w": pw, "h": box_h + 130,
        })

    return {
        "required": req_nodes, "optional": opt_nodes,
        "phases": phase_regions, "arrows": arrows, "arcs": arcs,
        "width": total_w, "height": main_y + box_h + 60,
    }


# ── Debug 数据 ────────────────────────────────────────────


def _load_debug_snapshots(debug_dir: str) -> dict[str, dict]:
    snapshots: dict[str, dict] = {}
    if not os.path.isdir(debug_dir):
        return snapshots
    for fname in sorted(os.listdir(debug_dir)):
        if not fname.endswith(".json") or fname.endswith("_diff.json"):
            continue
        base = fname[:-5]
        parts = base.split("_", 2)
        pass_name = parts[2] if len(parts) >= 3 else base
        try:
            with open(os.path.join(debug_dir, fname), encoding="utf-8") as f:
                snapshots[pass_name] = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return snapshots


def _build_pass_data(
    topo: dict,
    pass_timing: dict | None,
    debug_dir: str | None,
) -> str:
    from torch2c.common.graph_ir import graph_diff

    snapshots = _load_debug_snapshots(debug_dir) if debug_dir else {}
    all_names = [p["name"] for p in topo["passes"]]
    timing = pass_timing or {}

    data: dict[str, dict] = {}
    for p in topo["passes"]:
        name = p["name"]
        entry: dict[str, Any] = {
            "name": name, "number": p["number"],
            "input": p.get("input", ""),
            "output": p.get("output", ""),
            "desc": p.get("desc", ""),
        }

        # Timing / enabled
        t = timing.get(name, {})
        entry["enabled"] = t.get("enabled", True)
        if "duration_ms" in t:
            entry["duration_ms"] = t["duration_ms"]

        # Graph snapshot
        snap = snapshots.get(name)
        if snap:
            entry["after"] = {
                "nodes": len(snap.get("nodes", {})),
                "tensors": len(snap.get("tensors", {})),
            }
            idx = all_names.index(name) if name in all_names else -1
            prev_snap = None
            if idx > 0:
                for pi in range(idx - 1, -1, -1):
                    if all_names[pi] in snapshots:
                        prev_snap = snapshots[all_names[pi]]
                        break
            if prev_snap:
                entry["before"] = {
                    "nodes": len(prev_snap.get("nodes", {})),
                    "tensors": len(prev_snap.get("tensors", {})),
                }
                diff = graph_diff(prev_snap, snap)
                entry["diff"] = {k: len(v) if isinstance(v, (list, dict)) else v
                                 for k, v in diff.items()}
        data[name] = entry

    return json.dumps(data, ensure_ascii=False)


# ── HTML ──────────────────────────────────────────────────


def _render_html(layout: dict, pass_data_json: str, snap_json: str = "{}") -> str:
    svg_w, svg_h = layout["width"], layout["height"]
    svg_parts: list[str] = []

    for p in layout["phases"]:
        svg_parts.append(
            f'<rect x="{p["x"]}" y="{p["y"]}" width="{p["w"]}" height="{p["h"]}" '
            f'rx="8" fill="{p["fill"]}" stroke="{p["border"]}" stroke-width="1.5"/>')
        svg_parts.append(
            f'<text x="{p["x"]+p["w"]//2}" y="{p["y"]+16}" text-anchor="middle" '
            f'font-size="14" font-weight="700" fill="{p["border"]}" opacity="0.8">'
            f'{p["label"]}</text>')

    for arc in layout["arcs"]:
        svg_parts.append(
            f'<path d="M {arc["sx"]},{arc["sy"]} C {arc["sx"]},{arc["ctrl_y"]} '
            f'{arc["dx"]},{arc["ctrl_y"]} {arc["dx"]},{arc["dy"]}" fill="none" '
            f'stroke="#9ca3af" stroke-width="1.5" stroke-dasharray="6,3" opacity="0.5"/>')

    for a in layout["arrows"]:
        svg_parts.append(
            f'<line x1="{a["x1"]}" y1="{a["y1"]}" x2="{a["x2"]}" y2="{a["y2"]}" '
            f'stroke="#4b5563" stroke-width="2" marker-end="url(#arrow)"/>')

    for rn in layout["required"]:
        fill, border = _PHASE_COLORS.get(rn["phase"], ("#f8fafc", "#cbd5e1"))
        svg_parts.append(
            f'<g class="pn" data-name="{rn["name"]}" cursor="pointer" '
            f'onclick="sel(\'{rn["name"]}\')">'
            f'<rect x="{rn["x"]}" y="{rn["y"]}" width="{rn["w"]}" height="{rn["h"]}" '
            f'rx="6" fill="white" stroke="{border}" stroke-width="2.5"/>'
            f'<text x="{rn["x"]+rn["w"]//2}" y="{rn["y"]+20}" text-anchor="middle" '
            f'font-size="12" font-weight="700" fill="#1f2937">{rn["name"]}</text>'
            f'<text x="{rn["x"]+rn["w"]//2}" y="{rn["y"]+36}" text-anchor="middle" '
            f'font-size="9" fill="#6b7280" class="timing-text"></text>'
            f'<circle class="sd" cx="{rn["x"]+rn["w"]-10}" cy="{rn["y"]+10}" r="4" fill="#22c55e"/>'
            f'</g>')

    for on in layout["optional"]:
        svg_parts.append(
            f'<g class="pn opt" data-name="{on["name"]}" cursor="pointer" '
            f'onclick="sel(\'{on["name"]}\')">'
            f'<rect x="{on["x"]}" y="{on["y"]}" width="{on["w"]}" height="{on["h"]}" '
            f'rx="14" fill="white" stroke="#9ca3af" stroke-width="1.5" stroke-dasharray="5,3"/>'
            f'<text x="{on["x"]+on["w"]//2}" y="{on["y"]+15}" text-anchor="middle" '
            f'font-size="11" font-weight="600" fill="#374151">{on["name"]}</text>'
            f'<text x="{on["x"]+on["w"]//2}" y="{on["y"]+28}" text-anchor="middle" '
            f'font-size="8" fill="#9ca3af" class="timing-text"></text>'
            f'<circle class="sd" cx="{on["x"]+on["w"]-8}" cy="{on["y"]+8}" r="3.5" fill="#9ca3af"/>'
            f'</g>')

    svg = "\n".join(svg_parts)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>torch2c Pipeline</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f8fafc;color:#1e293b}}
.hd{{padding:14px 24px;background:#fff;border-bottom:1px solid #e2e8f0;display:flex;align-items:center;gap:16px}}
.hd h1{{font-size:18px;font-weight:700}}
.hd .sub{{font-size:13px;color:#64748b}}
.bar{{padding:6px 24px;background:#f1f5f9;border-bottom:1px solid #e2e8f0;display:flex;gap:12px;align-items:center;font-size:13px}}
.bar label{{cursor:pointer;display:flex;align-items:center;gap:4px}}
.sw{{overflow-x:auto;padding:20px}}
svg{{display:block;min-width:{svg_w}px}}
.pn rect{{transition:filter .15s}}.pn:hover rect{{filter:brightness(.93)}}
.pn.off rect{{opacity:.3}}.pn.off text{{opacity:.3}}
#dp{{background:#fff;border-top:1px solid #e2e8f0;padding:0;max-height:0;overflow:hidden;transition:max-height .3s,padding .3s}}
#dp.open{{max-height:200px;padding:12px 24px}}
.pb{{display:flex;align-items:flex-start;gap:14px;font-size:12px;flex-wrap:wrap}}
.pb .nm{{font-weight:700;font-size:15px;min-width:140px}}
.pb .ms{{color:#3b82f6;font-weight:600}}
.pb .st{{color:#64748b}}.pb .st b{{color:#1e293b}}
.pb .a{{color:#16a34a;font-weight:600}}.pb .r{{color:#dc2626;font-weight:600}}.pb .c{{color:#d97706;font-weight:600}}
.pb .io{{display:flex;gap:20px;margin-top:4px;font-size:11px}}
.pb .io-label{{font-weight:700;color:#64748b;min-width:40px}}
.pb .io-val{{color:#1e293b}}
.pb .desc{{margin-top:6px;font-size:12px;color:#475569;line-height:1.6;max-width:900px}}
</style></head><body>
<div class="hd"><h1>torch2c Pipeline</h1><span class="sub">Click pass to inspect · Timing shown on nodes</span></div>
<div class="bar"><label><input type="checkbox" id="so" checked onchange="tog(this.checked)"> Optional passes</label></div>
<div class="sw">
<svg viewBox="0 0 {svg_w} {svg_h}" width="{svg_w}" height="{svg_h}">
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#4b5563"/></marker></defs>
{svg}
</svg></div>
<div id="dp"></div>
<div id="tip" style="display:none;position:fixed;z-index:100;background:#1e293b;color:#f1f5f9;border-radius:8px;padding:10px 14px;font-size:11px;max-width:420px;line-height:1.5;pointer-events:none;box-shadow:0 4px 16px rgba(0,0,0,.3);white-space:pre-wrap;font-family:monospace"></div>
<div id="gs" style="display:none;border-top:2px solid #e2e8f0;position:relative;">
  <div id="gh" style="padding:8px 24px;background:#f1f5f9;display:flex;align-items:center;gap:12px;font-size:13px;">
    <button onclick="navP(-1)" style="padding:3px 10px;border:1px solid #cbd5e1;border-radius:4px;background:#fff;cursor:pointer">&larr;</button>
    <b id="gt"></b>
    <button onclick="navP(1)" style="padding:3px 10px;border:1px solid #cbd5e1;border-radius:4px;background:#fff;cursor:pointer">&rarr;</button>
    <span style="color:#94a3b8;font-size:11px">&larr;&rarr; 方向键切换 · Esc 关闭</span>
    <button onclick="closeG()" style="margin-left:auto;padding:3px 10px;border:1px solid #cbd5e1;border-radius:4px;background:#fff;cursor:pointer">Close</button>
  </div>
  <div id="gp" style="overflow:auto;padding:12px;max-height:70vh"></div>
  <div id="gl" style="position:absolute;top:44px;right:12px;background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:8px 12px;font-size:11px;box-shadow:0 2px 8px rgba(0,0,0,.08)">
    <b>Diff</b><br>
    <span style="color:#16a34a">&#9632;</span> Added
    <span style="color:#eab308;margin-left:6px">&#9632;</span> Changed
    <span style="color:#ef4444;margin-left:6px">&#9633;</span> Removed<br>
    <b style="margin-top:4px;display:inline-block">Click</b><br>
    <span style="color:#3b82f6">&#9135;</span> Upstream
    <span style="color:#f97316;margin-left:6px">&#9135;</span> Downstream
  </div>
</div>
<script>
var D=__PASS_DATA__,S=__SNAP_DATA__,act=null;
// Show timing on nodes
document.querySelectorAll('.pn').forEach(function(g){{
  var d=D[g.dataset.name];if(!d)return;
  var tt=g.querySelector('.timing-text');if(!tt)return;
  if(d.enabled===false){{g.classList.add('off');var dot=g.querySelector('.sd');if(dot)dot.setAttribute('fill','#d1d5db');tt.textContent='disabled';return}}
  if(d.duration_ms!==undefined)tt.textContent=d.duration_ms+'ms';
}});
// 默认选中第一个 pass
setTimeout(function(){{var first=document.querySelector('.pn');if(first)sel(first.dataset.name)}},0);
// Pass order for keyboard nav
var PO=Object.keys(S);
var ci=-1;
function navP(d){{if(ci<0)return;var x=ci+d;if(x<0)x=PO.length-1;if(x>=PO.length)x=0;sel(PO[x])}}
function closeG(){{document.getElementById('gs').style.display='none';ci=-1;hl(null);act=null;document.getElementById('dp').classList.remove('open')}}
document.addEventListener('keydown',function(e){{
  if(ci<0)return;
  if(e.key==='ArrowLeft'){{e.preventDefault();navP(-1)}}
  else if(e.key==='ArrowRight'){{e.preventDefault();navP(1)}}
  else if(e.key==='Escape'){{e.preventDefault();closeG()}}
  else if(e.key==='ArrowDown'||e.key==='ArrowUp'){{
    e.preventDefault();
    var gp=document.getElementById('gp');if(!gp)return;
    var all=Array.from(gp.querySelectorAll('.gn'));if(!all.length)return;
    var cur=gp.querySelector('.gn[data-sel="1"]');
    var idx=cur?all.indexOf(cur):-1;
    var next=e.key==='ArrowDown'?idx+1:idx-1;
    if(next<0)next=all.length-1;if(next>=all.length)next=0;
    all[next].click();
    all[next].scrollIntoView({{behavior:'smooth',block:'center'}});
  }}
}});

function sel(n){{
  var p=document.getElementById('dp');
  if(act===n){{closeG();return}}
  act=n;hl(n);ci=PO.indexOf(n);
  if(S[n])showG(n);
  var d=D[n]||{{}},h='<div class="pb"><div>';
  h+='<div class="nm">'+n+(d.duration_ms!==undefined?' <span class="ms">'+d.duration_ms+'ms</span>':'')+'</div>';
  if(d.after){{
    h+='<span class="st"><b>'+d.after.nodes+'</b> nodes / <b>'+d.after.tensors+'</b> tensors</span>';
  }}
  if(d.diff){{
    var df=d.diff,pp=[];
    if(df.nodes_added)pp.push('<span class="a">+'+df.nodes_added+'</span>');
    if(df.nodes_removed)pp.push('<span class="r">-'+df.nodes_removed+'</span>');
    if(df.nodes_changed)pp.push('<span class="c">~'+df.nodes_changed+'</span>');
    if(pp.length)h+=' <span class="st">nodes '+pp.join(' ')+'</span>';
    var tp=[];
    if(df.tensors_added)tp.push('<span class="a">+'+df.tensors_added+'</span>');
    if(df.tensors_removed)tp.push('<span class="r">-'+df.tensors_removed+'</span>');
    if(df.tensors_changed)tp.push('<span class="c">~'+df.tensors_changed+'</span>');
    if(tp.length)h+=' <span class="st">tensors '+tp.join(' ')+'</span>';
  }}
  // Input/Output
  if(d.input||d.output){{
    h+='<div class="io">';
    if(d.input)h+='<div><span class="io-label">Input:</span> <span class="io-val">'+d.input+'</span></div>';
    if(d.output)h+='<div><span class="io-label">Output:</span> <span class="io-val">'+d.output+'</span></div>';
    h+='</div>';
  }}
  // Description
  if(d.desc)h+='<div class="desc">'+d.desc+'</div>';
  h+='</div></div>';p.innerHTML=h;p.classList.add('open');
}}
function hl(n){{document.querySelectorAll('.pn').forEach(function(g){{g.querySelector('rect').style.filter=g.dataset.name===n?'drop-shadow(0 0 6px rgba(59,130,246,.5))':''}})}}
function tog(s){{document.querySelectorAll('.pn.opt').forEach(function(g){{g.style.display=s?'':'none'}});document.querySelectorAll('path[stroke-dasharray]').forEach(function(p){{p.style.display=s?'':'none'}})}}

function showG(name){{
  document.getElementById('gt').textContent=name+' ('+(ci+1)+'/'+PO.length+')';
  document.getElementById('gs').style.display='block';
  renderG(S[name],document.getElementById('gp'));
  document.getElementById('gs').scrollIntoView({{behavior:'smooth'}});
}}

function renderG(G,pane){{
  var c=document.createElement('div');c.style.position='relative';
  var nodes=G.nodes.filter(function(n){{return n.status!=='removed'}});
  var nW=160,nH=48,gY=20,hH=32,mL=20,mT=hH+8;
  var nm={{}};nodes.forEach(function(n){{nm[n.id]=n}});
  // Parent map for topo Y
  var par={{}};nodes.forEach(function(n){{par[n.id]=[]}});
  G.edges.forEach(function(e){{if(par[e.to]&&nm[e.from])par[e.to].push(e.from)}});
  // Layout: 4 CU columns, topo Y
  var lanes=['cube','vector','idma','dma'];
  var lC={{cube:'#dbeafe',vector:'#dcfce7',idma:'#fef9c3',dma:'#f3e8ff'}};
  var lW=nW+30,lX={{}},lY={{}};
  lanes.forEach(function(l,i){{lX[l]=mL+i*lW;lY[l]=mT}});
  var pos={{}},nh={{}};
  nodes.forEach(function(n){{nh[n.id]=nH}});
  nodes.forEach(function(n){{
    var cu=(n.compute_unit||'vector').toLowerCase();if(lanes.indexOf(cu)<0)cu='vector';
    var minY=lY[cu];
    (par[n.id]||[]).forEach(function(p){{var pp=pos[p];if(pp){{var b=pp.y+nH+gY;if(b>minY)minY=b}}}});
    pos[n.id]={{x:lX[cu]+(lW-nW)/2,y:minY}};lY[cu]=minY+nH+gY;
  }});
  var maxY=0;lanes.forEach(function(l){{if(lY[l]>maxY)maxY=lY[l]}});
  var tW=mL+lanes.length*lW+20,tH=maxY+40;
  c.style.width=tW+'px';c.style.height=tH+'px';
  // Lane bg+header
  lanes.forEach(function(l,i){{
    var bg=document.createElement('div');
    bg.style.cssText='position:absolute;left:'+lX[l]+'px;top:0;width:'+lW+'px;height:'+tH+'px;background:'+lC[l]+';border-right:1px solid #e2e8f0;pointer-events:none;opacity:.4';
    c.appendChild(bg);
    var hd=document.createElement('div');
    hd.style.cssText='position:absolute;left:'+lX[l]+'px;top:0;width:'+lW+'px;height:'+hH+'px;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:800;color:#374151;text-transform:uppercase;letter-spacing:1px;border-bottom:2px solid #94a3b8;background:rgba(255,255,255,.7)';
    hd.textContent=l;c.appendChild(hd);
  }});
  // SVG edges
  var svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
  svg.style.cssText='position:absolute;left:0;top:0;width:'+tW+'px;height:'+tH+'px;pointer-events:none;overflow:visible';
  var df=document.createElementNS('http://www.w3.org/2000/svg','defs');
  df.innerHTML='<marker id="ga" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#94a3b8"/></marker>';
  svg.appendChild(df);
  G.edges.forEach(function(e){{
    var f=pos[e.from],t=pos[e.to];if(!f||!t)return;
    var x1=f.x+nW/2,y1=f.y+nH,x2=t.x+nW/2,y2=t.y;
    var p=document.createElementNS('http://www.w3.org/2000/svg','path');
    var my=(y1+y2)/2;
    p.setAttribute('d','M '+x1+','+y1+' C '+x1+','+my+' '+x2+','+my+' '+x2+','+y2);
    p.setAttribute('fill','none');p.setAttribute('stroke','#cbd5e1');
    p.setAttribute('stroke-width','1.5');p.setAttribute('marker-end','url(#ga)');
    p.setAttribute('data-from',e.from);p.setAttribute('data-to',e.to);
    p.dataset.oc='#cbd5e1';svg.appendChild(p);
  }});
  c.appendChild(svg);
  // Parent/child maps for highlight
  var gP={{}},gC={{}};
  G.nodes.forEach(function(n){{gP[n.id]=[];gC[n.id]=[]}});
  G.edges.forEach(function(e){{if(gC[e.from])gC[e.from].push(e.to);if(gP[e.to])gP[e.to].push(e.from)}});
  var selN=null;
  function clrH(){{
    c.querySelectorAll('.gn').forEach(function(el){{el.style.opacity='1';el.style.boxShadow=''}});
    svg.querySelectorAll('path[data-from]').forEach(function(p){{p.classList.remove('ehu','ehd');p.style.opacity='1';p.style.strokeWidth='1.5';p.style.stroke=p.dataset.oc||'#cbd5e1'}});
  }}
  // Nodes
  nodes.forEach(function(n){{
    var p=pos[n.id];if(!p)return;
    var d=document.createElement('div');d.className='gn';d.dataset.nid=n.id;
    var st=n.status;
    var bdr='1.5px solid #cbd5e1',bg='#fff',col='#334155';
    if(st==='changed'){{bg='#fef9c3';bdr='1.5px solid #eab308';col='#713f12'}}
    else if(st==='added'){{bg='#dcfce7';bdr='1.5px solid #22c55e';col='#14532d'}}
    d.style.cssText='position:absolute;left:'+p.x+'px;top:'+p.y+'px;width:'+nW+'px;height:'+nH+'px;border-radius:10px;cursor:pointer;font-family:monospace;text-align:center;border:'+bdr+';background:'+bg+';color:'+col+';transition:box-shadow .15s;overflow:hidden';
    if(st==='added')d.style.boxShadow='0 0 0 3px #bbf7d0';
    var op=n.npu_op||n.op_type||'';
    if(op.startsWith('aten.'))op=op.slice(5).split('.')[0];
    // Build tooltip HTML: tensor IDs + opt_log
    var schedInfo=(n.schedule_order!=null?' #'+n.schedule_order:'')+(n.task_id?' task='+n.task_id:'');
    var tipHtml='<div style="font-weight:700;font-size:13px;margin-bottom:6px">'+n.id+' <span style="opacity:.6;font-weight:400">'+op+'</span>'+(schedInfo?'<span style="color:#a78bfa;margin-left:8px;font-size:11px">'+schedInfo+'</span>':'')+'</div>';
    function fmtTid(tid){{
      var t=G.tensors[tid];if(!t)return tid;
      var s=tid+' '+JSON.stringify(t.shape)+' '+t.dtype;
      if(t.is_weight)s+=' [W]';
      return s;
    }}
    if(n.inputs&&n.inputs.length)tipHtml+='<div style="color:#93c5fd">in: '+n.inputs.map(fmtTid).join('<br>&nbsp;&nbsp;&nbsp;&nbsp;')+'</div>';
    if(n.outputs&&n.outputs.length)tipHtml+='<div style="color:#86efac">out: '+n.outputs.map(fmtTid).join('<br>&nbsp;&nbsp;&nbsp;&nbsp;')+'</div>';
    if(n.opt_log&&n.opt_log.length){{
      tipHtml+='<div style="border-top:1px solid #334155;margin-top:6px;padding-top:6px">';
      n.opt_log.forEach(function(l){{
        tipHtml+='<div style="margin:3px 0"><span style="color:#fbbf24;font-weight:600">['+l.action+']</span> '+l.detail+'</div>';
      }});
      tipHtml+='</div>';
    }}
    d.addEventListener('mouseenter',function(ev){{
      var t=document.getElementById('tip');t.innerHTML=tipHtml;t.style.display='block';
      t.style.left=Math.min(ev.clientX+12,window.innerWidth-440)+'px';
      t.style.top=Math.min(ev.clientY+12,window.innerHeight-200)+'px';
    }});
    d.addEventListener('mousemove',function(ev){{
      var t=document.getElementById('tip');
      t.style.left=Math.min(ev.clientX+12,window.innerWidth-440)+'px';
      t.style.top=Math.min(ev.clientY+12,window.innerHeight-200)+'px';
    }});
    d.addEventListener('mouseleave',function(){{document.getElementById('tip').style.display='none'}});
    var tid_str=(n.task_id)?'T'+n.task_id:'';
    var sched_str=(n.schedule_order!=null)?'#'+n.schedule_order:'';
    var meta=[tid_str,sched_str].filter(Boolean).join(' ');
    d.innerHTML='<div style="padding:3px 8px;font-size:10px;display:flex;justify-content:space-between;border-bottom:1px solid rgba(0,0,0,.06);color:#64748b"><span>'+n.id+(meta?' <span style="color:#a78bfa">'+meta+'</span>':'')+'</span><span>'+(n.compute_unit||'').toUpperCase()+'</span></div><div style="padding:4px 8px;font-size:13px;font-weight:700">'+op+'</div>';
    d.onclick=function(){{
      var was=selN===n.id;
      c.querySelectorAll('.gn').forEach(function(el){{el.dataset.sel='0'}});
      if(was){{selN=null;clrH();return}}
      selN=n.id;d.dataset.sel='1';
      var dP={{}},dC={{}};
      (gP[n.id]||[]).forEach(function(x){{dP[x]=1}});(gC[n.id]||[]).forEach(function(x){{dC[x]=1}});
      c.querySelectorAll('.gn').forEach(function(el){{var id=el.dataset.nid;el.style.opacity=(id===n.id||dP[id]||dC[id])?'1':'0.12'}});
      d.style.boxShadow='0 0 0 3px #3b82f6';
      svg.querySelectorAll('path[data-from]').forEach(function(p){{
        var f=p.getAttribute('data-from'),t=p.getAttribute('data-to');
        p.classList.remove('ehu','ehd');
        if(f===n.id&&dC[t]){{p.style.opacity='1';p.style.strokeWidth='3';p.style.stroke='#f97316';p.classList.add('ehd')}}
        else if(t===n.id&&dP[f]){{p.style.opacity='1';p.style.strokeWidth='3';p.style.stroke='#3b82f6';p.classList.add('ehu')}}
        else{{p.style.opacity='.06';p.style.strokeWidth='1'}}
      }});
    }};
    c.appendChild(d);
  }});
  pane.innerHTML='';pane.appendChild(c);
}}
</script>
<style>
@keyframes fd{{from{{stroke-dashoffset:16}}to{{stroke-dashoffset:0}}}}
@keyframes fu{{from{{stroke-dashoffset:0}}to{{stroke-dashoffset:16}}}}
.ehd{{stroke-dasharray:8,8!important;animation:fd .6s linear infinite}}
.ehu{{stroke-dasharray:8,8!important;animation:fu .6s linear infinite}}
</style></body></html>""".replace("__PASS_DATA__", pass_data_json).replace("__SNAP_DATA__", snap_json)


# ── 对外接口 ──────────────────────────────────────────────


def _build_all_snapshots(topo: dict, debug_dir: str | None) -> str:
    """构建所有 pass 的图数据 JSON，供内嵌甬道图渲染。"""
    from torch2c.viz.pass_detail_viz import build_graph_data

    if not debug_dir:
        return "{}"
    snapshots = _load_debug_snapshots(debug_dir)
    if not snapshots:
        return "{}"

    all_names = [p["name"] for p in topo["passes"]]
    result: dict = {}
    for name, snap in snapshots.items():
        idx = all_names.index(name) if name in all_names else -1
        prev = None
        if idx > 0:
            for pi in range(idx - 1, -1, -1):
                if all_names[pi] in snapshots:
                    prev = snapshots[all_names[pi]]
                    break
        result[name] = build_graph_data(snap, prev)
    return json.dumps(result, ensure_ascii=False)


def emit_pipeline_html(
    output_dir: str,
    pass_timing: dict | None = None,
    debug_dir: str | None = None,
) -> str:
    """生成 pipeline 可视化 HTML，返回文件路径。"""
    topo = _get_topology()
    layout = _compute_layout(topo)
    data_json = _build_pass_data(topo, pass_timing, debug_dir)
    snap_json = _build_all_snapshots(topo, debug_dir)
    html = _render_html(layout, data_json, snap_json)

    viz_dir = ensure_viz_dir(output_dir)
    path = os.path.join(viz_dir, "pipeline.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Pipeline HTML: %s", path)
    return path
