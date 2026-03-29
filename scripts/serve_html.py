"""Flask 服务器 — 浏览编译产出的 HTML 可视化文件。

用法：
    python scripts/serve_html.py [--ip 0.0.0.0] [--port 5050] [--output-root output]

功能：
    - 左侧栏：按模型分组 + 文件修改日期
    - 右侧：2 宫格 / 4 宫格布局
    - 拖拽文件到宫格渲染
"""

import argparse
import json
import os
import signal
import socket
import sys
from glob import glob
from pathlib import Path

from flask import Flask, abort, request, send_file

ROOT = Path(__file__).resolve().parent.parent


def _find_free_port(ip: str, start: int, max_tries: int = 20) -> int:
    """从 start 开始找一个可用端口。"""
    for i in range(max_tries):
        port = start + i
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((ip, port))
                return port
            except OSError:
                continue
    print(f"端口 {start}~{start + max_tries - 1} 全被占用")
    sys.exit(1)


def _kill_existing(ip: str, port: int) -> None:
    """如果目标端口已被占用，尝试杀掉占用进程。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((ip, port))
            return
        except OSError:
            pass

    import subprocess
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
        pids = result.stdout.strip().split()
        for pid in pids:
            if pid.isdigit():
                print(f"  杀掉端口 {port} 上的进程 PID={pid}")
                os.kill(int(pid), signal.SIGTERM)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


# ── 上传安全检查 ──

# 允许的 import 白名单（模块前缀）
_ALLOWED_IMPORTS = frozenset([
    "torch", "math", "typing", "dataclasses", "collections",
    "functools", "itertools", "enum", "abc",
])

# 禁止的函数 / 属性调用
_BANNED_CALLS = frozenset([
    "exec", "eval", "compile", "__import__", "globals", "locals",
    "getattr", "setattr", "delattr", "breakpoint",
    "exit", "quit",
])

# 禁止的模块级属性访问
_BANNED_ATTRS = frozenset([
    "system", "popen", "exec", "spawn", "remove", "unlink", "rmdir",
    "rmtree", "makedirs", "rename", "kill", "Popen", "call", "run",
])


def _check_model_safety(source: str) -> list[str]:
    """AST 静态安全检查。返回违规列表，空 = 通过。"""
    import ast
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]

    violations = []

    for node in ast.walk(tree):
        # 检查 import
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name.split(".")[0]
                if mod not in _ALLOWED_IMPORTS:
                    violations.append(
                        f"Line {node.lineno}: import '{alias.name}' not allowed "
                        f"(only {', '.join(sorted(_ALLOWED_IMPORTS))})")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mod = node.module.split(".")[0]
                if mod not in _ALLOWED_IMPORTS:
                    violations.append(
                        f"Line {node.lineno}: from '{node.module}' import not allowed")

        # 检查危险函数调用
        elif isinstance(node, ast.Call):
            fname = ""
            if isinstance(node.func, ast.Name):
                fname = node.func.id
            elif isinstance(node.func, ast.Attribute):
                fname = node.func.attr
            if fname in _BANNED_CALLS:
                violations.append(
                    f"Line {node.lineno}: call to '{fname}()' not allowed")

        # 检查危险属性访问（os.system 等）
        elif isinstance(node, ast.Attribute):
            if node.attr in _BANNED_ATTRS:
                violations.append(
                    f"Line {node.lineno}: access to '.{node.attr}' not allowed")

        # 检查 open()
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "open":
                violations.append(
                    f"Line {node.lineno}: open() not allowed")

    # 必须有 model 和 dummy_input
    top_names = {
        n.targets[0].id for n in tree.body
        if isinstance(n, ast.Assign) and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
    }
    if "model" not in top_names:
        violations.append("Missing top-level 'model = ...' assignment")
    if "dummy_input" not in top_names:
        violations.append("Missing top-level 'dummy_input = ...' assignment")

    return violations[:10]  # 最多报 10 条


def create_app(output_root: str) -> Flask:
    app = Flask(__name__)
    output_dir = Path(output_root).resolve()

    @app.route("/api/files")
    def api_files():
        """返回 JSON 文件列表，含修改时间。"""
        htmls = sorted(glob(str(output_dir / "**" / "*.html"), recursive=True))
        result: dict[str, list] = {}
        for path in htmls:
            rel = os.path.relpath(path, output_dir)
            parts = rel.split(os.sep)
            group = parts[0] if len(parts) > 1 else "(root)"
            name = "/".join(parts[1:]) if len(parts) > 1 else parts[0]
            mtime = os.path.getmtime(path)
            result.setdefault(group, []).append({
                "name": name,
                "path": rel,
                "mtime": mtime,
            })
        return json.dumps(result, ensure_ascii=False)

    @app.route("/")
    def index():
        return _INDEX_HTML

    @app.route("/view/<path:rel_path>")
    def view(rel_path: str):
        full = (output_dir / rel_path).resolve()
        if not str(full).startswith(str(output_dir)) or not full.is_file():
            abort(404)
        return send_file(full)

    @app.route("/api/demos")
    def api_demos():
        """返回可用 demo 列表。"""
        demo_dirs = [
            ROOT / "torch2c" / "a_capture" / "graph_capture" / "demo",
            ROOT / "torch2c" / "integration" / "demo",
        ]
        demos = []
        for d in demo_dirs:
            for f in sorted(d.glob("demo_*.py")):
                demos.append({
                    "name": f.stem.replace("demo_", "").replace("_", " ").title(),
                    "file": str(f.relative_to(ROOT)),
                    "stem": f.stem,
                })
            # also include files with model/dummy_input exports
            for f in sorted(d.glob("*.py")):
                if f.stem.startswith("demo_") or f.stem.startswith("_"):
                    continue
                # check if it has 'model' and 'dummy_input'
                text = f.read_text(errors="ignore")
                if "model =" in text and "dummy_input =" in text:
                    demos.append({
                        "name": f.stem.replace("_", " ").title(),
                        "file": str(f.relative_to(ROOT)),
                        "stem": f.stem,
                    })
        return json.dumps(demos, ensure_ascii=False)

    @app.route("/api/run-demo", methods=["POST"])
    def run_demo():
        """运行指定 demo，返回编译结果。"""
        import subprocess
        data = request.get_json()
        demo_file = data.get("file", "")
        mode = data.get("mode", "minimal")
        full_path = (ROOT / demo_file).resolve()
        if not str(full_path).startswith(str(ROOT)) or not full_path.is_file():
            return json.dumps({"ok": False, "error": "file not found"}), 404

        cmd = [
            str(ROOT / ".venv" / "bin" / "python"),
            str(ROOT / "scripts" / "compile_model.py"),
            str(full_path), "--mode", mode, "--no-open", "--no-debug",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120, cwd=str(ROOT),
            )
            return json.dumps({
                "ok": result.returncode == 0,
                "stdout": result.stdout[-2000:] if result.stdout else "",
                "stderr": result.stderr[-2000:] if result.stderr else "",
            })
        except subprocess.TimeoutExpired:
            return json.dumps({"ok": False, "error": "timeout"})

    @app.route("/api/download-demo")
    def download_demo():
        """下载 demo 源文件。"""
        demo_file = request.args.get("file", "")
        full_path = (ROOT / demo_file).resolve()
        if not str(full_path).startswith(str(ROOT)) or not full_path.is_file():
            abort(404)
        return send_file(full_path, as_attachment=True)

    @app.route("/api/upload-model", methods=["POST"])
    def upload_model():
        """上传自定义模型文件，安全检查后编译。"""
        import subprocess
        import tempfile
        f = request.files.get("file")
        if not f or not f.filename.endswith(".py"):
            return json.dumps({"ok": False, "error": "need .py file"})
        source = f.read().decode("utf-8", errors="replace")
        # ── 安全检查 ──
        violations = _check_model_safety(source)
        if violations:
            return json.dumps({"ok": False,
                               "error": "Safety check failed:\n" + "\n".join(violations)})
        # 保存到临时文件
        tmp = tempfile.NamedTemporaryFile(
            suffix=".py", delete=False, dir=str(ROOT / "output"), mode="w")
        tmp.write(source)
        tmp.close()
        cmd = [
            str(ROOT / ".venv" / "bin" / "python"),
            str(ROOT / "scripts" / "compile_model.py"),
            tmp.name, "--mode", "full", "--no-open", "--no-debug",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120, cwd=str(ROOT))
            os.unlink(tmp.name)
            return json.dumps({
                "ok": result.returncode == 0,
                "stdout": result.stdout[-2000:] if result.stdout else "",
                "stderr": result.stderr[-2000:] if result.stderr else "",
            })
        except subprocess.TimeoutExpired:
            os.unlink(tmp.name)
            return json.dumps({"ok": False, "error": "compile timeout (120s)"})

    @app.route("/api/shutdown", methods=["POST"])
    def shutdown():
        func = request.environ.get("werkzeug.server.shutdown")
        if func:
            func()
        else:
            os.kill(os.getpid(), signal.SIGTERM)
        return "OK"

    return app


_INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>torch2c Viewer</title>
<style>
:root {
  --sidebar-w: 280px;
  --bg: #0f172a;
  --bg-sidebar: #1e293b;
  --bg-cell: #1e293b;
  --border: #334155;
  --text: #e2e8f0;
  --text-dim: #94a3b8;
  --accent: #38bdf8;
  --accent-hover: #7dd3fc;
  --drop-highlight: rgba(56, 189, 248, 0.15);
  --drop-border: #38bdf8;
  --group-bg: #0f172a;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: var(--bg); color: var(--text);
  height: 100vh; overflow: hidden;
  display: flex;
}

/* ── Sidebar ── */
.sidebar {
  width: var(--sidebar-w); min-width: 180px;
  background: var(--bg-sidebar);
  border-right: 1px solid var(--border);
  display: flex; flex-direction: column;
  height: 100vh; flex-shrink: 0;
}
.resize-handle {
  width: 5px; cursor: col-resize;
  background: transparent; flex-shrink: 0;
  transition: background 0.15s;
}
.resize-handle:hover, .resize-handle.active {
  background: var(--accent);
}
.sidebar-header {
  padding: 12px 16px; border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
}
.sidebar-header h1 {
  font-size: 15px; font-weight: 600;
  letter-spacing: 0.5px;
}
.sidebar-header .shutdown-btn {
  width: 24px; height: 24px; border: 1px solid #f87171; background: transparent;
  color: #f87171; border-radius: 4px; cursor: pointer; font-size: 12px;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
  transition: all 0.15s;
}
.sidebar-header .shutdown-btn:hover { background: #f87171; color: #fff; }
.sidebar-header .subtitle { font-size: 11px; color: var(--text-dim); margin-top: 4px; }
.sidebar-search {
  padding: 8px 12px;
  border-bottom: 1px solid var(--border);
}
.sidebar-search input {
  width: 100%; padding: 6px 10px;
  background: var(--bg); border: 1px solid var(--border);
  border-radius: 6px; color: var(--text); font-size: 12px;
  outline: none;
}
.sidebar-search input:focus { border-color: var(--accent); }
.sidebar-search input::placeholder { color: var(--text-dim); }
.file-list {
  flex: 1; overflow-y: auto;
  padding: 8px 0;
}
.file-list::-webkit-scrollbar { width: 4px; }
.file-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
.group-header {
  padding: 8px 14px 4px;
  font-size: 11px; font-weight: 600; text-transform: uppercase;
  color: var(--accent); letter-spacing: 0.5px;
  background: var(--group-bg);
  position: sticky; top: 0; z-index: 1;
}
.file-item {
  padding: 5px 14px 5px 20px;
  font-size: 12px; cursor: grab;
  display: flex; justify-content: space-between; align-items: center;
  transition: background 0.15s;
  user-select: none;
}
.file-item:hover { background: rgba(56, 189, 248, 0.08); }
.file-item .name {
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  flex: 1; margin-right: 8px;
}
.file-item .date { font-size: 10px; color: var(--text-dim); white-space: nowrap; }

/* ── Main ── */
.main {
  flex: 1; display: flex; flex-direction: column;
  overflow: hidden;
}
.toolbar {
  padding: 8px 16px;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 8px;
  background: var(--bg-sidebar);
}
.toolbar label { font-size: 12px; color: var(--text-dim); }
.layout-btn {
  padding: 4px 12px; border: 1px solid var(--border);
  background: transparent; color: var(--text); border-radius: 4px;
  font-size: 12px; cursor: pointer; transition: all 0.15s;
}
.layout-btn:hover { border-color: var(--accent); color: var(--accent); }
.layout-btn.active { background: var(--accent); color: var(--bg); border-color: var(--accent); font-weight: 600; }
.grid-container {
  flex: 1; display: flex; flex-direction: column;
  padding: 2px; overflow: hidden;
}
.grid-row { display: flex; flex: 1; min-height: 0; }
.grid-col-resize {
  width: 5px; cursor: col-resize; background: transparent;
  flex-shrink: 0; transition: background 0.15s;
}
.grid-row-resize {
  height: 5px; cursor: row-resize; background: transparent;
  flex-shrink: 0; transition: background 0.15s;
}
.grid-col-resize:hover, .grid-col-resize.active,
.grid-row-resize:hover, .grid-row-resize.active { background: var(--accent); }
.drag-overlay {
  position: fixed; top: 0; left: 0; right: 0; bottom: 0;
  z-index: 9999; cursor: inherit;
}

/* ── Cell ── */
.cell {
  background: var(--bg-cell); border: 2px dashed var(--border);
  border-radius: 6px; position: relative;
  display: flex; align-items: center; justify-content: center;
  overflow: hidden; transition: border-color 0.2s, background 0.2s;
}
.cell.drag-over {
  border-color: var(--drop-border); background: var(--drop-highlight);
}
.cell .placeholder {
  color: var(--text-dim); font-size: 13px;
  pointer-events: none; text-align: center; line-height: 1.6;
}
.cell iframe {
  width: 100%; height: 100%; border: none;
}
/* 底部悬浮栏：鼠标接近底部 40px 区域才显示 */
.cell .cell-hover-zone {
  position: absolute; bottom: 0; left: 0; right: 0; height: 40px; z-index: 2;
}
.cell .cell-bar {
  position: absolute; bottom: 0; left: 0; right: 0;
  height: 30px;
  background: linear-gradient(0deg, rgba(15,23,42,0.92) 60%, transparent 100%);
  display: flex; align-items: center;
  padding: 0 10px; opacity: 0; transition: opacity 0.2s;
  pointer-events: none; z-index: 3;
}
.cell .cell-hover-zone:hover ~ .cell-bar,
.cell .cell-bar:hover { opacity: 1; pointer-events: auto; }
.cell-bar .cell-title { font-size: 11px; color: var(--text-dim); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex:1; }
.cell-bar .cell-btn {
  width: 22px; height: 22px; border: none;
  border-radius: 3px; cursor: pointer; font-size: 11px;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0; margin-left: 6px; transition: background 0.15s;
}
.cell-bar .cell-export { background: rgba(74,222,128,0.5); color: #fff; font-size: 12px; }
.cell-bar .cell-export:hover { background: #4ade80; }
.cell-bar .cell-fullscreen { background: rgba(56,189,248,0.5); color: #fff; }
.cell-bar .cell-fullscreen:hover { background: #38bdf8; }
.cell-bar .cell-close { background: rgba(239,68,68,0.7); color: #fff; font-size: 13px; }
.cell-bar .cell-close:hover { background: #ef4444; }
.refresh-btn {
  padding: 4px 12px; border: 1px solid var(--border);
  background: transparent; color: var(--text); border-radius: 4px;
  font-size: 12px; cursor: pointer; transition: all 0.15s;
  margin-left: auto;
}
.refresh-btn:hover { border-color: var(--accent); color: var(--accent); }
.refresh-btn.spinning { animation: spin 0.6s linear; }
@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
.file-item.new-file { background: rgba(34,197,94,0.12); }
.file-item.new-file .name { color: #4ade80; }

/* ── Demo Section ── */
.demo-section {
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}
.demo-header {
  padding: 10px 14px 6px;
  font-size: 12px; font-weight: 700; color: var(--accent);
  letter-spacing: 0.3px;
  background: linear-gradient(135deg, rgba(56,189,248,0.08) 0%, transparent 100%);
}
.demo-list { padding: 4px 0; }
.demo-item {
  padding: 5px 14px 5px 16px; font-size: 12px;
  display: flex; align-items: center; gap: 6px;
  transition: background 0.15s;
}
.demo-item:hover { background: rgba(56,189,248,0.08); }
.demo-badge {
  font-size: 9px; font-weight: 600; padding: 1px 5px; border-radius: 3px;
  flex-shrink: 0; text-transform: uppercase; letter-spacing: 0.3px;
}
.demo-badge.easy { background: rgba(74,222,128,0.15); color: #4ade80; }
.demo-badge.medium { background: rgba(251,191,36,0.15); color: #fbbf24; }
.demo-badge.hard { background: rgba(248,113,113,0.15); color: #f87171; }
.demo-item .demo-name { flex: 1; font-size: 12px; }
.demo-item .demo-dl {
  padding: 2px 6px; border: 1px solid var(--border); background: transparent;
  color: var(--text-dim); border-radius: 3px; font-size: 9px; cursor: pointer;
  transition: all 0.15s; white-space: nowrap;
}
.demo-item .demo-dl:hover { border-color: var(--accent); color: var(--accent); }
.demo-item .demo-run {
  padding: 2px 8px; border: 1px solid var(--border); background: transparent;
  color: var(--text-dim); border-radius: 3px; font-size: 10px; cursor: pointer;
  transition: all 0.15s; white-space: nowrap;
}
.demo-item .demo-run:hover { border-color: #4ade80; color: #4ade80; }
.demo-item .demo-run.running { color: #fbbf24; border-color: #fbbf24; pointer-events: none; }
.demo-item .demo-run.done { color: #4ade80; border-color: #4ade80; }
.demo-item .demo-run.fail { color: #f87171; border-color: #f87171; }
/* ── Upload ── */
.upload-area {
  margin: 4px 14px 8px; padding: 10px;
  border: 1px dashed var(--border); border-radius: 6px;
  text-align: center; font-size: 11px; color: var(--text-dim);
  cursor: pointer; transition: all 0.15s;
}
.upload-area:hover { border-color: var(--accent); color: var(--accent); }
.upload-area.dragging { border-color: var(--accent); background: rgba(56,189,248,0.08); }
.upload-area input { display: none; }
</style>
</head>
<body>

<!-- Sidebar -->
<div class="sidebar">
  <div class="sidebar-header">
    <div><h1>torch2c Viewer</h1><div class="subtitle" id="fileCount"></div></div>
    <button class="shutdown-btn" id="shutdownBtn" title="Shutdown server">&#x23FB;</button>
  </div>
  <div class="demo-section">
    <div class="demo-header">Quick Demos</div>
    <div class="demo-list" id="demoList"></div>
    <div class="upload-area" id="uploadArea">
      Drop .py model here or <u>click to upload</u>
      <input type="file" id="uploadInput" accept=".py">
    </div>
  </div>
  <div class="sidebar-search">
    <input type="text" id="search" placeholder="Search files...">
  </div>
  <div class="file-list" id="fileList"></div>
</div>

<!-- Resize Handle -->
<div class="resize-handle" id="resizeHandle"></div>

<!-- Main -->
<div class="main">
  <div class="toolbar">
    <label>Layout:</label>
    <button class="layout-btn active" data-layout="1">1</button>
    <button class="layout-btn" data-layout="2">2</button>
    <button class="layout-btn" data-layout="4">4</button>
    <button class="refresh-btn" id="refreshBtn" title="Refresh file list">&#x21bb; Refresh</button>
  </div>
  <div class="grid-container" id="grid" style="position:relative"></div>
</div>

<script>
const grid = document.getElementById('grid');
const fileList = document.getElementById('fileList');
const searchInput = document.getElementById('search');
let layout = 1;
let allFiles = {};
let knownPaths = new Set();

// ── Drag overlay (prevents iframes from stealing mouse events) ──
let _overlay = null;
function addOverlay(cursor) {
  if (_overlay) return;
  _overlay = document.createElement('div');
  _overlay.className = 'drag-overlay';
  _overlay.style.cursor = cursor || 'col-resize';
  document.body.appendChild(_overlay);
}
function removeOverlay() {
  if (_overlay) { _overlay.remove(); _overlay = null; }
}

// ── Layout ──
let colRatio = 0.5, rowRatio = 0.5;
function setLayout(n) {
  layout = n;
  colRatio = 0.5; rowRatio = 0.5;
  document.querySelectorAll('.layout-btn').forEach(b => b.classList.toggle('active', +b.dataset.layout === n));
  buildCells();
}
document.querySelectorAll('.layout-btn').forEach(b => b.addEventListener('click', () => setLayout(+b.dataset.layout)));

function makeCell(idx, existing) {
  const cell = document.createElement('div');
  cell.className = 'cell';
  cell.style.flex = '1';
  cell.dataset.idx = idx;
  if (existing[idx] && existing[idx].src) {
    loadInCell(cell, existing[idx].src, existing[idx].title);
  } else {
    cell.innerHTML = '<div class="placeholder">Drop HTML here</div>';
  }
  // drag 事件监听在 cell 上，dragover 时显示遮罩防止 iframe 吞事件
  cell.addEventListener('dragenter', e => {
    e.preventDefault(); cell.classList.add('drag-over');
    let mask = cell.querySelector('.drag-mask');
    if (!mask) {
      mask = document.createElement('div');
      mask.className = 'drag-mask';
      mask.style.cssText = 'position:absolute;inset:0;z-index:10;background:rgba(56,189,248,0.08);';
      mask.addEventListener('dragover', ev => ev.preventDefault());
      mask.addEventListener('dragleave', () => { cell.classList.remove('drag-over'); mask.remove(); });
      mask.addEventListener('drop', ev => {
        ev.preventDefault(); cell.classList.remove('drag-over'); mask.remove();
        const path = ev.dataTransfer.getData('text/plain');
        const name = ev.dataTransfer.getData('text/name');
        if (path) loadInCell(cell, '/view/' + path, name || path);
      });
      cell.appendChild(mask);
    }
  });
  cell.addEventListener('dragover', e => e.preventDefault());
  cell.addEventListener('drop', e => {
    e.preventDefault(); cell.classList.remove('drag-over');
    const mask = cell.querySelector('.drag-mask'); if (mask) mask.remove();
    const path = e.dataTransfer.getData('text/plain');
    const name = e.dataTransfer.getData('text/name');
    if (path) loadInCell(cell, '/view/' + path, name || path);
  });
  return cell;
}

function makeColResize(row) {
  const h = document.createElement('div');
  h.className = 'grid-col-resize';
  let dragging = false;
  h.addEventListener('mousedown', e => {
    e.preventDefault(); dragging = true; h.classList.add('active');
    document.body.style.cursor = 'col-resize'; document.body.style.userSelect = 'none';
    addOverlay('col-resize');
  });
  document.addEventListener('mousemove', e => {
    if (!dragging) return;
    const rect = row.getBoundingClientRect();
    colRatio = Math.max(0.15, Math.min(0.85, (e.clientX - rect.left) / rect.width));
    applySizes();
  });
  document.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false; h.classList.remove('active');
    document.body.style.cursor = ''; document.body.style.userSelect = '';
    removeOverlay();
  });
  return h;
}

function makeRowResize() {
  const h = document.createElement('div');
  h.className = 'grid-row-resize';
  let dragging = false;
  h.addEventListener('mousedown', e => {
    e.preventDefault(); dragging = true; h.classList.add('active');
    document.body.style.cursor = 'row-resize'; document.body.style.userSelect = 'none';
    addOverlay('row-resize');
  });
  document.addEventListener('mousemove', e => {
    if (!dragging) return;
    const rect = grid.getBoundingClientRect();
    rowRatio = Math.max(0.15, Math.min(0.85, (e.clientY - rect.top) / rect.height));
    applySizes();
  });
  document.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false; h.classList.remove('active');
    document.body.style.cursor = ''; document.body.style.userSelect = '';
    removeOverlay();
  });
  return h;
}

function applySizes() {
  const rows = grid.querySelectorAll('.grid-row');
  const cells = grid.querySelectorAll('.cell');
  if (layout === 2) {
    const rows = grid.querySelectorAll('.grid-row');
    if (rows[0]) rows[0].style.flex = rowRatio;
    if (rows[1]) rows[1].style.flex = 1 - rowRatio;
  } else if (layout === 4) {
    if (rows[0]) rows[0].style.flex = rowRatio;
    if (rows[1]) rows[1].style.flex = 1 - rowRatio;
    const topCells = rows[0]?.querySelectorAll('.cell');
    const botCells = rows[1]?.querySelectorAll('.cell');
    if (topCells?.[0]) topCells[0].style.flex = colRatio;
    if (topCells?.[1]) topCells[1].style.flex = 1 - colRatio;
    if (botCells?.[0]) botCells[0].style.flex = colRatio;
    if (botCells?.[1]) botCells[1].style.flex = 1 - colRatio;
  }
}

function buildCells() {
  const existing = [...grid.querySelectorAll('.cell')].map(c => ({
    src: c.querySelector('iframe')?.src || '',
    title: c.querySelector('.cell-title')?.textContent || '',
  }));
  grid.innerHTML = '';

  if (layout === 1) {
    const row = document.createElement('div');
    row.className = 'grid-row';
    row.appendChild(makeCell(0, existing));
    grid.appendChild(row);
  } else if (layout === 2) {
    const row1 = document.createElement('div');
    row1.className = 'grid-row';
    row1.appendChild(makeCell(0, existing));
    grid.appendChild(row1);
    grid.appendChild(makeRowResize());
    const row2 = document.createElement('div');
    row2.className = 'grid-row';
    row2.appendChild(makeCell(1, existing));
    grid.appendChild(row2);
  } else if (layout === 4) {
    const row1 = document.createElement('div');
    row1.className = 'grid-row';
    row1.appendChild(makeCell(0, existing));
    row1.appendChild(makeColResize(row1));
    row1.appendChild(makeCell(1, existing));
    grid.appendChild(row1);
    grid.appendChild(makeRowResize());
    const row2 = document.createElement('div');
    row2.className = 'grid-row';
    row2.appendChild(makeCell(2, existing));
    row2.appendChild(makeColResize(row2));
    row2.appendChild(makeCell(3, existing));
    grid.appendChild(row2);
  }
  applySizes();
}

function loadInCell(cell, url, title) {
  cell.innerHTML = '';
  const bar = document.createElement('div');
  bar.className = 'cell-bar';
  const t = document.createElement('span');
  t.className = 'cell-title';
  t.textContent = title;
  const exportBtn = document.createElement('button');
  exportBtn.className = 'cell-btn cell-export';
  exportBtn.innerHTML = '&#x2913;';
  exportBtn.title = 'Export Excel';
  const fsBtn = document.createElement('button');
  fsBtn.className = 'cell-btn cell-fullscreen';
  fsBtn.innerHTML = '&#x26F6;';
  fsBtn.title = 'Fullscreen';
  const closeBtn = document.createElement('button');
  closeBtn.className = 'cell-btn cell-close';
  closeBtn.innerHTML = '&times;';
  closeBtn.onclick = () => {
    cell.innerHTML = '<div class="placeholder">Drop HTML here</div>';
  };
  bar.appendChild(t);
  bar.appendChild(exportBtn);
  bar.appendChild(fsBtn);
  bar.appendChild(closeBtn);
  // 底部悬浮触发区
  const hoverZone = document.createElement('div');
  hoverZone.className = 'cell-hover-zone';
  cell.appendChild(hoverZone);
  cell.appendChild(bar);
  const iframe = document.createElement('iframe');
  iframe.src = url;
  iframe.setAttribute('allowfullscreen', '');
  cell.appendChild(iframe);
  fsBtn.onclick = () => {
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      iframe.requestFullscreen().catch(() => {
        cell.requestFullscreen().catch(() => {});
      });
    }
  };
  exportBtn.onclick = () => {
    try {
      const fn = iframe.contentWindow.exportXlsx;
      if (fn) { fn(); } else { alert('This page does not support Excel export'); }
    } catch(e) { alert('Export not available for this page'); }
  };
}

// ── Sidebar ──
function formatDate(ts) {
  const d = new Date(ts * 1000);
  const pad = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

let newPaths = new Set();

function renderFileList(filter) {
  fileList.innerHTML = '';
  let total = 0;
  const q = (filter || '').toLowerCase();
  const groups = Object.keys(allFiles).sort((a, b) => {
    const ma = Math.max(...allFiles[a].map(f => f.mtime));
    const mb = Math.max(...allFiles[b].map(f => f.mtime));
    return mb - ma;
  });
  const MAX_EXPANDED = 2;
  groups.forEach((group, gi) => {
    const items = allFiles[group].filter(f => !q || f.name.toLowerCase().includes(q) || group.toLowerCase().includes(q));
    if (!items.length) return;
    total += items.length;
    items.sort((a, b) => b.mtime - a.mtime);

    const collapsed = !q && gi >= MAX_EXPANDED;
    const container = document.createElement('div');

    const header = document.createElement('div');
    header.className = 'group-header';
    header.style.cssText = 'cursor:pointer;display:flex;justify-content:space-between;align-items:center;';
    header.innerHTML = `<span>${group}</span><span style="font-size:10px;opacity:0.6">${collapsed ? '+ '+items.length : '−'}</span>`;
    const body = document.createElement('div');
    body.style.display = collapsed ? 'none' : 'block';

    header.onclick = () => {
      const show = body.style.display === 'none';
      body.style.display = show ? 'block' : 'none';
      header.querySelector('span:last-child').textContent = show ? '−' : '+ ' + items.length;
    };

    for (const f of items) {
      const div = document.createElement('div');
      const isNew = newPaths.has(f.path);
      div.className = 'file-item' + (isNew ? ' new-file' : '');
      div.draggable = true;
      const newBadge = isNew ? '<span style="color:#4ade80;font-size:10px;margin-right:4px">NEW</span>' : '';
      div.innerHTML = `${newBadge}<span class="name" title="${f.path}">${f.name}</span><span class="date">${formatDate(f.mtime)}</span>`;
      div.addEventListener('dragstart', e => {
        e.dataTransfer.setData('text/plain', f.path);
        e.dataTransfer.setData('text/name', group + ' / ' + f.name);
        e.dataTransfer.effectAllowed = 'copy';
      });
      div.addEventListener('click', () => {
        const cells = grid.querySelectorAll('.cell');
        let target = cells[0];
        for (const c of cells) { if (!c.querySelector('iframe')) { target = c; break; } }
        loadInCell(target, '/view/' + f.path, group + ' / ' + f.name);
      });
      body.appendChild(div);
    }
    container.appendChild(header);
    container.appendChild(body);
    fileList.appendChild(container);
  });
  document.getElementById('fileCount').textContent = total + ' files';
}

searchInput.addEventListener('input', () => renderFileList(searchInput.value));

// ── Refresh ──
function collectPaths(data) {
  const s = new Set();
  for (const g of Object.values(data)) for (const f of g) s.add(f.path);
  return s;
}

function refreshFiles() {
  const btn = document.getElementById('refreshBtn');
  btn.classList.add('spinning');
  fetch('/api/files')
    .then(r => r.json())
    .then(data => {
      // detect new files
      const incoming = collectPaths(data);
      newPaths = new Set();
      for (const p of incoming) { if (!knownPaths.has(p)) newPaths.add(p); }
      knownPaths = incoming;
      allFiles = data;
      renderFileList(searchInput.value);
      btn.classList.remove('spinning');
      if (newPaths.size > 0) {
        btn.textContent = `\u21bb ${newPaths.size} new`;
        setTimeout(() => { btn.textContent = '\u21bb Refresh'; }, 3000);
      }
    });
}

document.getElementById('refreshBtn').addEventListener('click', refreshFiles);
// ── Sidebar Resize ──
const sidebar = document.querySelector('.sidebar');
const handle = document.getElementById('resizeHandle');
let dragging = false;
handle.addEventListener('mousedown', e => {
  e.preventDefault();
  dragging = true;
  handle.classList.add('active');
  document.body.style.cursor = 'col-resize';
  document.body.style.userSelect = 'none';
  addOverlay('col-resize');
});
document.addEventListener('mousemove', e => {
  if (!dragging) return;
  const w = Math.max(180, Math.min(e.clientX, window.innerWidth - 200));
  sidebar.style.width = w + 'px';
});
document.addEventListener('mouseup', () => {
  if (!dragging) return;
  dragging = false;
  handle.classList.remove('active');
  document.body.style.cursor = '';
  document.body.style.userSelect = '';
  removeOverlay();
});

document.getElementById('shutdownBtn').addEventListener('click', () => {
  if (confirm('Shutdown server?')) {
    fetch('/api/shutdown', { method: 'POST' }).then(() => {
      document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;color:#94a3b8;font-size:18px">Server stopped.</div>';
    });
  }
});

// ── Demo Section ──
const demoList = document.getElementById('demoList');

// 难度排序表（stem → 排序权重 + 难度标签）
const DEMO_DIFFICULTY = {
  'demo_matmul': [1, 'easy'],
  'demo_matmul_tiling': [2, 'easy'],
  'demo_embedding': [3, 'easy'],
  'demo_residual': [4, 'medium'],
  'demo_mlp': [5, 'medium'],
  'demo_mha_tiling': [6, 'medium'],
  'demo_multi_batch': [7, 'medium'],
  'demo_decoder': [8, 'hard'],
  'demo_gpt_block': [9, 'hard'],
};

const PINNED_DEMOS = ['demo_matmul', 'demo_matmul_tiling', 'demo_mlp'];

function renderDemos(demos) {
  demos.sort((a, b) => {
    const wa = (DEMO_DIFFICULTY[a.stem] || [50, 'medium'])[0];
    const wb = (DEMO_DIFFICULTY[b.stem] || [50, 'medium'])[0];
    return wa - wb;
  });
  demoList.innerHTML = '';
  const pinned = demos.filter(d => PINNED_DEMOS.includes(d.stem));
  const rest = demos.filter(d => !PINNED_DEMOS.includes(d.stem));
  // 按 PINNED_DEMOS 顺序排列置顶项
  pinned.sort((a, b) => PINNED_DEMOS.indexOf(a.stem) - PINNED_DEMOS.indexOf(b.stem));

  function addItem(d) {
    const diff = (DEMO_DIFFICULTY[d.stem] || [50, 'medium'])[1];
    const div = document.createElement('div');
    div.className = 'demo-item';
    div.innerHTML = `<span class="demo-badge ${diff}">${diff}</span>`
      + `<span class="demo-name">${d.name}</span>`;
    const dlBtn = document.createElement('button');
    dlBtn.className = 'demo-dl';
    dlBtn.textContent = '.py';
    dlBtn.title = 'Download demo source';
    dlBtn.onclick = (e) => { e.stopPropagation(); window.open('/api/download-demo?file=' + encodeURIComponent(d.file)); };
    const runBtn = document.createElement('button');
    runBtn.className = 'demo-run';
    runBtn.textContent = 'Run';
    runBtn.onclick = (e) => { e.stopPropagation(); runDemo(d, runBtn); };
    div.appendChild(dlBtn);
    div.appendChild(runBtn);
    demoList.appendChild(div);
  }

  pinned.forEach(addItem);

  if (rest.length) {
    const moreWrap = document.createElement('div');
    const toggle = document.createElement('div');
    toggle.style.cssText = 'padding:4px 16px;font-size:11px;color:var(--text-dim);cursor:pointer;user-select:none;transition:color 0.15s;';
    toggle.textContent = `+ ${rest.length} more demos...`;
    toggle.onmouseenter = () => { toggle.style.color = 'var(--accent)'; };
    toggle.onmouseleave = () => { toggle.style.color = 'var(--text-dim)'; };
    const restContainer = document.createElement('div');
    restContainer.style.display = 'none';
    rest.forEach(d => {
      const diff = (DEMO_DIFFICULTY[d.stem] || [50, 'medium'])[1];
      const div = document.createElement('div');
      div.className = 'demo-item';
      div.innerHTML = `<span class="demo-badge ${diff}">${diff}</span><span class="demo-name">${d.name}</span>`;
      const dlBtn = document.createElement('button');
      dlBtn.className = 'demo-dl'; dlBtn.textContent = '.py'; dlBtn.title = 'Download demo source';
      dlBtn.onclick = (e) => { e.stopPropagation(); window.open('/api/download-demo?file=' + encodeURIComponent(d.file)); };
      const runBtn = document.createElement('button');
      runBtn.className = 'demo-run'; runBtn.textContent = 'Run';
      runBtn.onclick = (e) => { e.stopPropagation(); runDemo(d, runBtn); };
      div.appendChild(dlBtn); div.appendChild(runBtn);
      restContainer.appendChild(div);
    });
    let expanded = false;
    toggle.onclick = () => {
      expanded = !expanded;
      restContainer.style.display = expanded ? 'block' : 'none';
      toggle.textContent = expanded ? `- collapse` : `+ ${rest.length} more demos...`;
    };
    moreWrap.appendChild(toggle);
    moreWrap.appendChild(restContainer);
    demoList.appendChild(moreWrap);
  }
}

// ── Upload ──
const uploadArea = document.getElementById('uploadArea');
const uploadInput = document.getElementById('uploadInput');
uploadArea.addEventListener('click', () => uploadInput.click());
uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragging'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragging'));
uploadArea.addEventListener('drop', e => {
  e.preventDefault(); uploadArea.classList.remove('dragging');
  const file = e.dataTransfer.files[0];
  if (file && file.name.endsWith('.py')) uploadFile(file);
});
uploadInput.addEventListener('change', () => {
  if (uploadInput.files[0]) uploadFile(uploadInput.files[0]);
});
function uploadFile(file) {
  uploadArea.textContent = 'Uploading ' + file.name + '...';
  const formData = new FormData();
  formData.append('file', file);
  fetch('/api/upload-model', { method: 'POST', body: formData })
    .then(r => r.json())
    .then(data => {
      if (data.ok) {
        uploadArea.innerHTML = '<span style="color:#4ade80">Compiled! Refreshing...</span>';
        refreshFiles();
        setTimeout(() => {
          uploadArea.innerHTML = 'Drop .py model here or <u>click to upload</u><input type="file" id="uploadInput" accept=".py">';
        }, 3000);
      } else {
        uploadArea.innerHTML = '<span style="color:#f87171">Failed: ' + (data.error||'unknown') + '</span>';
        setTimeout(() => {
          uploadArea.innerHTML = 'Drop .py model here or <u>click to upload</u><input type="file" id="uploadInput" accept=".py">';
        }, 4000);
      }
    });
}

function runDemo(demo, btn) {
  btn.className = 'demo-run running';
  btn.textContent = 'Running...';
  fetch('/api/run-demo', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ file: demo.file, mode: 'both' }),
  })
  .then(r => r.json())
  .then(data => {
    if (data.ok) {
      btn.className = 'demo-run done';
      btn.textContent = 'Done';
      // refresh file list to pick up new outputs
      refreshFiles();
    } else {
      btn.className = 'demo-run fail';
      btn.textContent = 'Failed';
      console.error(data.stderr || data.error);
    }
    setTimeout(() => { btn.className = 'demo-run'; btn.textContent = 'Run'; }, 4000);
  })
  .catch(() => {
    btn.className = 'demo-run fail';
    btn.textContent = 'Error';
    setTimeout(() => { btn.className = 'demo-run'; btn.textContent = 'Run'; }, 4000);
  });
}

// ── Init ──
fetch('/api/demos').then(r => r.json()).then(renderDemos);
fetch('/api/files')
  .then(r => r.json())
  .then(data => {
    allFiles = data;
    knownPaths = collectPaths(data);
    renderFileList();
    buildCells();
    // 自动打开第一个 unified.html
    for (const group of Object.keys(data).sort((a,b) => {
      const ma = Math.max(...data[a].map(f=>f.mtime));
      const mb = Math.max(...data[b].map(f=>f.mtime));
      return mb - ma;
    })) {
      const uf = data[group].find(f => f.name.includes('unified.html'));
      if (uf) {
        const cell = grid.querySelector('.cell');
        if (cell) loadInCell(cell, '/view/' + uf.path, group + ' / ' + uf.name);
        break;
      }
    }
  });
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Flask HTML 查看器")
    parser.add_argument("--ip", default="0.0.0.0", help="绑定 IP（默认 0.0.0.0）")
    parser.add_argument("--port", type=int, default=5050, help="端口（默认 5050）")
    parser.add_argument("--output-root", default=str(ROOT / "output"),
                        help="output 根目录（默认 output/）")
    args = parser.parse_args()

    _kill_existing(args.ip, args.port)
    port = _find_free_port(args.ip, args.port)

    app = create_app(args.output_root)
    print(f"\n  torch2c HTML Viewer: http://{args.ip}:{port}")
    print(f"  output 目录: {args.output_root}\n")
    app.run(host=args.ip, port=port, debug=False)


if __name__ == "__main__":
    main()
