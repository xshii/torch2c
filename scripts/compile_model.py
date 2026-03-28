"""编译当前模型文件 — 支持 minimal（无优化）和 full（全优化）两种模式。

用法：
    python scripts/compile_model.py <model_file.py> [--mode minimal|full]

模型文件约定：
    - 导出 `model`（nn.Module 实例）
    - 导出 `dummy_input`（torch.Tensor）
    - 可选导出 `mask`（torch.Tensor）

示例模型文件：
    import torch
    import torch.nn as nn

    class MyModel(nn.Module):
        def forward(self, x):
            return x @ x.transpose(-1, -2)

    model = MyModel()
    dummy_input = torch.randn(1, 32, 64)
"""

import argparse
import glob
import importlib.util
import os
import platform
import subprocess
import sys
import time

import torch


def _open_html(output_dir: str) -> None:
    """查找并打开 output_dir 下的所有 HTML 文件。"""
    htmls = sorted(glob.glob(os.path.join(output_dir, "**", "*.html"), recursive=True))
    if not htmls:
        return
    opener = "open" if platform.system() == "Darwin" else "xdg-open"
    for html in htmls:
        print(f"  HTML: {html}")
        try:
            subprocess.Popen([opener, html], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            pass


def _load_model_module(path: str):
    """动态加载模型文件，返回 module。"""
    spec = importlib.util.spec_from_file_location("_user_model", path)
    if spec is None or spec.loader is None:
        print(f"无法加载: {path}")
        sys.exit(1)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(description="编译模型为 C 工程")
    parser.add_argument("model_file", help="模型 Python 文件路径")
    parser.add_argument("--mode", choices=["minimal", "full", "both"],
                        default="both", help="编译模式（默认 both）")
    parser.add_argument("--output", default=None, help="输出目录（默认 output/<ModelName>）")
    parser.add_argument("--no-debug", action="store_true", help="关闭 debug_dump")
    parser.add_argument("--no-open", action="store_true", help="不自动打开 HTML")
    parser.add_argument("--serve", action="store_true", help="编译后启动 Flask 查看器")
    parser.add_argument("--ip", default="0.0.0.0", help="Flask 绑定 IP（默认 0.0.0.0）")
    parser.add_argument("--port", type=int, default=5050, help="Flask 端口（默认 5050）")
    args = parser.parse_args()

    # 加载模型文件
    mod = _load_model_module(args.model_file)
    model = getattr(mod, "model", None)
    dummy_input = getattr(mod, "dummy_input", None)
    mask = getattr(mod, "mask", None)

    if model is None or dummy_input is None:
        print("模型文件必须导出 `model` 和 `dummy_input`")
        sys.exit(1)

    from torch2c.common.paths import INTEGRATION_CONFIG_DIR
    from torch2c.integration.pipeline import compile

    model_name = type(model).__name__
    config_dir = str(INTEGRATION_CONFIG_DIR)
    timestamp = time.strftime("%Y%m%d_%H%M")

    modes = [args.mode] if args.mode != "both" else ["minimal", "full"]

    for mode in modes:
        base = args.output or f"output/{model_name}"
        output_dir = f"{base}_{mode}_{timestamp}"

        toggles = "minimal" if mode == "minimal" else None

        print(f"\n{'='*60}")
        print(f"  编译模式: {mode.upper()}")
        print(f"  模型: {model_name}")
        print(f"  输出: {output_dir}")
        print(f"{'='*60}\n")

        t0 = time.perf_counter()
        compile(
            model, dummy_input,
            config_dir=config_dir,
            output_dir=output_dir,
            mask=mask,
            pass_toggles=toggles,
            debug_dump=not args.no_debug,
        )
        elapsed = time.perf_counter() - t0

        # 生成 pass 流水线可视化
        if not args.no_debug:
            from torch2c.viz.pipeline_viz import emit_pipeline_html
            debug_dir = os.path.join(output_dir, "debug")
            timing_path = os.path.join(debug_dir, "pass_timing.json")
            import json
            timing = json.load(open(timing_path)) if os.path.isfile(timing_path) else None
            emit_pipeline_html(output_dir, pass_timing=timing, debug_dir=debug_dir)

        print(f"\n  完成: {output_dir} ({elapsed:.1f}s)")

        # 列出生成的 C 文件
        src_dir = os.path.join(output_dir, "src")
        if os.path.isdir(src_dir):
            files = sorted(os.listdir(src_dir))
            print(f"  C 文件: {', '.join(files)}")

        # 查找并打开 HTML
        if not args.no_open:
            _open_html(output_dir)

    if len(modes) == 2:
        print(f"\n对比两个版本:")
        base = args.output or f"output/{model_name}"
        print(f"  diff {base}_minimal_{timestamp}/src/ {base}_full_{timestamp}/src/")

    if args.serve:
        from serve_html import _kill_existing, _find_free_port, create_app

        _kill_existing(args.ip, args.port)
        port = _find_free_port(args.ip, args.port)
        app = create_app("output")
        print(f"\n  Flask 查看器: http://{args.ip}:{port}")
        app.run(host=args.ip, port=port, debug=False)


if __name__ == "__main__":
    main()
