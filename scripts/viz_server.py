"""Flask 可视化服务器 — 远端访问编译产物。

用法:
  python scripts/viz_server.py              # 默认 0.0.0.0:8080，服务 output/compile_viz/
  python scripts/viz_server.py --port 9090  # 自定义端口
  python scripts/viz_server.py --compile    # 先编译再启动
"""

from __future__ import annotations

import argparse
import os

from flask import Flask, send_from_directory

app = Flask(__name__)
SERVE_DIR = ""


@app.route("/")
def index():
    # 优先 viz/pipeline.html，否则列出 debug 快照
    viz_index = os.path.join(SERVE_DIR, "viz", "pipeline.html")
    if os.path.isfile(viz_index):
        return send_from_directory(os.path.join(SERVE_DIR, "viz"), "pipeline.html")
    # Fallback: 列出 debug 目录
    debug_dir = os.path.join(SERVE_DIR, "debug")
    if os.path.isdir(debug_dir):
        files = sorted(os.listdir(debug_dir))
        links = "".join(f'<li><a href="debug/{f}">{f}</a></li>' for f in files if f.endswith(".json"))
        return f"<h2>Debug Snapshots</h2><ul>{links}</ul>"
    return "No viz or debug output found. Run with --compile first."


@app.route("/<path:filename>")
def serve_file(filename):
    return send_from_directory(SERVE_DIR, filename)


def main():
    global SERVE_DIR

    parser = argparse.ArgumentParser(description="torch2c 可视化服务器")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--output-dir", default="output/compile_viz")
    parser.add_argument("--compile", action="store_true",
                        help="启动前先编译 DemoEncoder")
    args = parser.parse_args()

    if args.compile:
        print("=== 编译 DemoEncoder ===")
        import torch
        from torch2c.a_capture.graph_capture.demo.demo_model import DemoEncoder
        from torch2c.common.paths import INTEGRATION_CONFIG_DIR
        from torch2c.integration.pipeline import compile
        import json
        from torch2c.viz.pipeline_viz import emit_pipeline_html

        model = DemoEncoder()
        x = torch.randn(1, 32, 64)
        mask = torch.zeros(1, 1, 32, 32)
        compile(
            model, x,
            config_dir=str(INTEGRATION_CONFIG_DIR),
            output_dir=args.output_dir,
            mask=mask,
            debug_dump=True,
        )
        debug_dir = os.path.join(args.output_dir, "debug")
        tp = os.path.join(debug_dir, "pass_timing.json")
        timing = json.load(open(tp)) if os.path.isfile(tp) else None
        emit_pipeline_html(args.output_dir, pass_timing=timing,
                           debug_dir=debug_dir)
        print("=== 编译完成 ===\n")

    SERVE_DIR = os.path.abspath(args.output_dir)
    if not os.path.isdir(SERVE_DIR):
        print(f"错误: {SERVE_DIR} 不存在，请先运行 --compile")
        return

    print(f"服务器启动:")
    print(f"  http://{args.host}:{args.port}/")
    print(f"  服务目录: {SERVE_DIR}")
    print(f"  Ctrl+C 退出\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
