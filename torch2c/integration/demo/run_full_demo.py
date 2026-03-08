"""端到端 demo：编译 2 层 Encoder 模型并生成完整 C 工程。

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

import torch

from torch2c.common import DEFAULT_OUTPUT_DIR, INTEGRATION_CONFIG_DIR, setup_logging
from torch2c.integration.demo.encoder_model import EncoderModel
from torch2c.integration.pipeline import compile

_CONFIG_DIR = str(INTEGRATION_CONFIG_DIR)


def _run_one(precision: str, output_dir: str) -> bool:
    """编译并验证单个精度模式，返回是否通过。"""
    label = "混合精度" if precision == "mixed" else "全 FP16"
    print(f"\n{'=' * 60}")
    print(f"  {label} ({precision})")
    print(f"{'=' * 60}")

    model = EncoderModel(d_model=256, dim_ff=512, num_layers=2, precision=precision)
    model.eval()

    batch, seq, d_model = 1, 32, 256
    dummy_input = torch.randn(batch, seq, d_model)
    mask = torch.zeros(batch, seq, seq)

    out = compile(
        model=model,
        dummy_input=dummy_input,
        config_dir=_CONFIG_DIR,
        output_dir=output_dir,
        mask=mask,
    )
    print(f"编译完成：{os.path.abspath(out)}")

    from torch2c.integration.demo.validate_output import validate
    validate(out)

    from torch2c.integration.demo.validate_c_output import validate_c
    result = validate_c(out)
    if result["passed"]:
        print(f"  [{label}] C golden 比对通过!")
    else:
        print(f"  [{label}] C golden 比对失败! (exit {result['returncode']})")
        print(result["stdout"])
    return result["passed"]


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

    results: dict[str, bool] = {}
    for mode in modes:
        out_dir = os.path.join(base, mode) if len(modes) > 1 else base
        results[mode] = _run_one(mode, out_dir)

    # 汇总
    print(f"\n{'=' * 60}")
    print("  汇总")
    print(f"{'=' * 60}")
    for mode, passed in results.items():
        label = "混合精度" if mode == "mixed" else "全 FP16"
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {label:10s}  {status}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
