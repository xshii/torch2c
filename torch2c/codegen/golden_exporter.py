"""golden_exporter — 将 PyTorch 输入输出导出为二进制文件或 C 静态数组。"""

from __future__ import annotations

import os

import numpy as np

from torch2c.common import get_logger

logger = get_logger("codegen.golden_exporter")

_HEX_PER_LINE = 16


def _write_desc(path: str, shape: list[int], dtype: str, fmt: str, total_bytes: int) -> None:
    with open(path, "w") as f:
        f.write(f"shape: {','.join(str(s) for s in shape)}\n")
        f.write(f"dtype: {dtype}\n")
        f.write(f"format: {fmt}\n")
        f.write("byte_order: little_endian\n")
        f.write(f"total_bytes: {total_bytes}\n")


def _array_to_hex(arr: np.ndarray) -> str:
    """将 numpy 数组转为 C 十六进制字节初始化列表。"""
    raw = arr.tobytes()
    parts = [f"0x{b:02x}" for b in raw]
    lines = []
    for i in range(0, len(parts), _HEX_PER_LINE):
        lines.append("    " + ", ".join(parts[i : i + _HEX_PER_LINE]))
    return ",\n".join(lines)


# ---- 文件模式（现有） ----


def export_golden(
    inputs: list[np.ndarray],
    outputs: list[np.ndarray],
    output_dir: str,
    dtype: str | list[str] = "fp16",
    fmt: str = "nd",
    output_dtype: list[str] | None = None,
) -> None:
    """导出 golden 数据为二进制文件 + 描述文件。

    Args:
        dtype: 输入 dtype。字符串时所有输入共享；列表时 per-tensor。
        output_dtype: 输出 dtype 列表。None 时使用 dtype 值。
    """
    logger.info("golden_exporter: 导出 golden 到 %s", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 标准化 dtype
    if isinstance(dtype, str):
        in_dtypes = [dtype] * len(inputs)
    else:
        in_dtypes = dtype
    out_dtypes = output_dtype if output_dtype is not None else (
        [dtype] * len(outputs) if isinstance(dtype, str) else dtype[:len(outputs)]
    )

    for i, arr in enumerate(inputs):
        bin_path = os.path.join(output_dir, f"input_{i}.bin")
        desc_path = os.path.join(output_dir, f"input_{i}.desc")
        arr.tofile(bin_path)
        dt = in_dtypes[i] if i < len(in_dtypes) else "fp16"
        _write_desc(desc_path, list(arr.shape), dt, fmt, arr.nbytes)

    for i, arr in enumerate(outputs):
        bin_path = os.path.join(output_dir, f"output_{i}.bin")
        desc_path = os.path.join(output_dir, f"output_{i}.desc")
        arr.tofile(bin_path)
        dt = out_dtypes[i] if i < len(out_dtypes) else "fp16"
        _write_desc(desc_path, list(arr.shape), dt, fmt, arr.nbytes)

    logger.info("golden_exporter: 完成，%d 输入 + %d 输出", len(inputs), len(outputs))


# ---- 静态模式（嵌入 C 代码） ----


def emit_golden_h(
    inputs: list[np.ndarray],
    outputs: list[np.ndarray],
    input_offsets: list[int],
    output_offsets: list[int],
) -> str:
    """生成 model_golden.h — golden 数据作为静态数组 + load/get 函数。"""
    lines = [
        "#ifndef MODEL_GOLDEN_H",
        "#define MODEL_GOLDEN_H",
        "",
        "#include <stddef.h>",
        "#include <string.h>",
        "",
    ]

    # 输入数据数组
    for i, arr in enumerate(inputs):
        nbytes = arr.nbytes
        lines.append(f"/* input_{i}: shape={list(arr.shape)}, {nbytes} bytes */")
        lines.append(f"static const unsigned char golden_input_{i}_data[{nbytes}] = {{")
        lines.append(_array_to_hex(arr))
        lines.append("};")
        lines.append("")

    # 输出数据数组
    for i, arr in enumerate(outputs):
        nbytes = arr.nbytes
        lines.append(f"/* output_{i}: shape={list(arr.shape)}, {nbytes} bytes */")
        lines.append(f"static const unsigned char golden_output_{i}_data[{nbytes}] = {{")
        lines.append(_array_to_hex(arr))
        lines.append("};")
        lines.append("")

    # load_golden_inputs: 将输入数据 memcpy 到 HBM
    lines.append("static inline void load_golden_inputs(unsigned char* hbm) {")
    for i, arr in enumerate(inputs):
        lines.append(
            f"    memcpy(hbm + {input_offsets[i]}, golden_input_{i}_data, "
            f"sizeof(golden_input_{i}_data));"
        )
    lines.append("}")
    lines.append("")

    # golden_output 指针/大小查询
    lines.append("static inline const unsigned char* golden_output_ptr(int idx) {")
    for i in range(len(outputs)):
        cond = "if" if i == 0 else "else if"
        lines.append(f"    {cond} (idx == {i}) return golden_output_{i}_data;")
    lines.append("    return (const unsigned char*)0;")
    lines.append("}")
    lines.append("")

    lines.append("static inline size_t golden_output_size(int idx) {")
    for i in range(len(outputs)):
        cond = "if" if i == 0 else "else if"
        lines.append(f"    {cond} (idx == {i}) return sizeof(golden_output_{i}_data);")
    lines.append("    return 0;")
    lines.append("}")
    lines.append("")

    lines += ["#endif", ""]
    return "\n".join(lines)


def export_golden_static(
    inputs: list[np.ndarray],
    outputs: list[np.ndarray],
    input_offsets: list[int],
    output_offsets: list[int],
    output_path: str,
) -> None:
    """导出 golden 数据为 C 静态数组头文件。"""
    logger.info("golden_exporter: 导出静态 golden 到 %s", output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    content = emit_golden_h(inputs, outputs, input_offsets, output_offsets)
    with open(output_path, "w") as f:
        f.write(content)
    logger.info("golden_exporter: 已生成 %s", output_path)
