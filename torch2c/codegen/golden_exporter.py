"""golden_exporter — 将 PyTorch 输入输出导出为二进制文件 + 描述文件。"""

from __future__ import annotations

import os

import numpy as np

from torch2c.common import get_logger

logger = get_logger("codegen.golden_exporter")


def _write_desc(path: str, shape: list[int], dtype: str, fmt: str, total_bytes: int) -> None:
    with open(path, "w") as f:
        f.write(f"shape: {','.join(str(s) for s in shape)}\n")
        f.write(f"dtype: {dtype}\n")
        f.write(f"format: {fmt}\n")
        f.write("byte_order: little_endian\n")
        f.write(f"total_bytes: {total_bytes}\n")


def export_golden(
    inputs: list[np.ndarray],
    outputs: list[np.ndarray],
    output_dir: str,
    dtype: str = "fp16",
    fmt: str = "nd",
) -> None:
    """导出 golden 数据。"""
    logger.info("golden_exporter: 导出 golden 到 %s", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for i, arr in enumerate(inputs):
        bin_path = os.path.join(output_dir, f"input_{i}.bin")
        desc_path = os.path.join(output_dir, f"input_{i}.desc")
        arr.tofile(bin_path)
        _write_desc(desc_path, list(arr.shape), dtype, fmt, arr.nbytes)

    for i, arr in enumerate(outputs):
        bin_path = os.path.join(output_dir, f"output_{i}.bin")
        desc_path = os.path.join(output_dir, f"output_{i}.desc")
        arr.tofile(bin_path)
        _write_desc(desc_path, list(arr.shape), dtype, fmt, arr.nbytes)

    logger.info("golden_exporter: 完成，%d 输入 + %d 输出", len(inputs), len(outputs))
