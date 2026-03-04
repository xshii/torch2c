"""weight_exporter — 将权重导出为 C 静态数组（model_weights.h）。"""

from __future__ import annotations

import os

import numpy as np

from npu_compiler.common import CodegenError, get_logger

logger = get_logger("codegen.weight_exporter")

_DTYPE_NP = {"fp16": np.float16, "fp32": np.float32, "int8": np.int8}

_HEX_BYTES_PER_LINE = 16


def _array_to_hex(arr: np.ndarray) -> str:
    """将 numpy 数组转为 C 十六进制字节初始化列表。"""
    raw = arr.tobytes()
    parts = [f"0x{b:02x}" for b in raw]
    lines = []
    for i in range(0, len(parts), _HEX_BYTES_PER_LINE):
        lines.append("    " + ", ".join(parts[i : i + _HEX_BYTES_PER_LINE]))
    return ",\n".join(lines)


def emit_weights_h(
    weights: dict[str, np.ndarray], offsets: dict[str, int] | list[int] | None = None
) -> str:
    """生成 model_weights.h 内容。

    Args:
        weights: {state_dict_key: numpy_array} 映射。
        offsets: 权重名→HBM 偏移的字典，或与 weights 顺序一致的列表。
    """
    lines = [
        "#ifndef MODEL_WEIGHTS_H",
        "#define MODEL_WEIGHTS_H",
        "",
        "#include <stddef.h>",
        "#include <string.h>",
        "",
    ]

    entries: list[tuple[str, str, int | None]] = []  # (safe_name, tid, offset)
    for tid, arr in weights.items():
        safe = tid.replace(".", "_").replace("-", "_")
        nbytes = arr.nbytes
        hex_data = _array_to_hex(arr)
        lines.append(f"/* {tid}: shape={list(arr.shape)}, dtype={arr.dtype}, {nbytes} bytes */")
        lines.append(f"static const unsigned char {safe}_data[{nbytes}] = {{")
        lines.append(hex_data)
        lines.append("};")
        lines.append("")
        # 解析偏移
        if isinstance(offsets, dict):
            offset = offsets.get(tid)
        elif isinstance(offsets, list) and len(offsets) > len(entries):
            offset = offsets[len(entries)]
        else:
            offset = None
        entries.append((safe, tid, offset))

    lines.append("static inline void load_weights(unsigned char* hbm) {")
    if all(e[2] is not None for e in entries):
        for safe, _tid, offset in entries:
            lines.append(f"    memcpy(hbm + {offset}, {safe}_data, sizeof({safe}_data));")
    else:
        lines.append("    (void)hbm;")
    lines.append("}")
    lines += ["", "#endif", ""]
    return "\n".join(lines)


def export_weights(
    state_dict: dict,
    output_path: str,
    dtype: str = "fp16",
    offsets: dict[str, int] | list[int] | None = None,
) -> None:
    """将 PyTorch state_dict 导出为 model_weights.h。"""
    logger.info("weight_exporter: 导出权重到 %s", output_path)
    np_dtype = _DTYPE_NP.get(dtype)
    if np_dtype is None:
        raise CodegenError(f"不支持的 dtype: {dtype}")

    weights = {}
    for name, tensor in state_dict.items():
        arr = tensor.detach().cpu().numpy().astype(np_dtype)
        weights[name] = arr

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(emit_weights_h(weights, offsets))
    logger.info("weight_exporter: 已生成 %s", output_path)
