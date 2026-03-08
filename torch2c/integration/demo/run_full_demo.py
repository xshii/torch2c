"""端到端 demo：编译 3 层 3 头 Encoder 模型并生成完整 C 工程。

包含两种精度模式：
- mixed：混合精度（linear1/norm1 fp32，其余 fp16）
- fp16：全 fp16

用法：
    python -m torch2c.integration.demo.run_full_demo            # 两种都跑
    python -m torch2c.integration.demo.run_full_demo --mixed    # 只跑混合精度
    python -m torch2c.integration.demo.run_full_demo --fp16     # 只跑全 fp16
"""

from __future__ import annotations

import os
import sys

from torch2c.common import DEFAULT_OUTPUT_DIR, setup_logging
from torch2c.integration.demo._runner import PRECISION_LABEL, compile_and_validate


def main() -> None:
    setup_logging("INFO")
    base = str(DEFAULT_OUTPUT_DIR)

    args = sys.argv[1:]
    modes: list[str] = []
    if "--mixed" in args:
        modes.append("mixed")
    if "--fp16" in args:
        modes.append("fp16")
    if not modes:
        modes = ["mixed", "fp16"]

    results: list[dict] = []
    for mode in modes:
        out_dir = os.path.join(base, mode) if len(modes) > 1 else base
        label = PRECISION_LABEL[mode]
        print(f"\n{'=' * 60}")
        print(f"  {label} ({mode})")
        print(f"{'=' * 60}")

        r = compile_and_validate(mode, out_dir)
        results.append(r)

        print(f"编译完成：{r['output_dir']}")
        status = "通过" if r["passed"] else "失败"
        print(f"  [{label}] C golden 比对{status}!")

    # 汇总
    print(f"\n{'=' * 60}")
    print("  汇总")
    print(f"{'=' * 60}")
    for r in results:
        label = PRECISION_LABEL[r["precision"]]
        status = "PASS ✓" if r["passed"] else "FAIL ✗"
        print(f"  {label:10s}  {status}  max_abs={r['max_abs']:.6f}  cosine={r['cosine']:.6f}")

    if not all(r["passed"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
