"""weight_exporter — 将权重导出为 C 静态数组（model_weights.h）。"""

from __future__ import annotations

import os

import numpy as np

from npu_compiler.common import CodegenError, get_logger

logger = get_logger("codegen.weight_exporter")

_DTYPE_NP = {"fp16": np.float16, "fp32": np.float32, "int8": np.int8}


def _array_to_hex(arr: np.ndarray) -> str:
    """将 numpy 数组转为 C 十六进制字节初始化列表。"""
    raw = arr.tobytes()
    parts = [f"0x{b:02x}" for b in raw]
    lines = []
    for i in range(0, len(parts), 16):
        lines.append("    " + ", ".join(parts[i:i + 16]))
    return ",\n".join(lines)


def emit_weights_h(weights: dict[str, np.ndarray]) -> str:
    """生成 model_weights.h 内容。

    Args:
        weights: {tensor_id: numpy_array} 映射。
    """
    lines = ["#ifndef MODEL_WEIGHTS_H", "#define MODEL_WEIGHTS_H", "",
             "#include <stddef.h>", ""]

    for tid, arr in weights.items():
        safe = tid.replace(".", "_").replace("-", "_")
        nbytes = arr.nbytes
        hex_data = _array_to_hex(arr)
        lines.append(f"/* {tid}: shape={list(arr.shape)}, "
                     f"dtype={arr.dtype}, {nbytes} bytes */")
        lines.append(f"static const unsigned char {safe}_data[{nbytes}] = {{")
        lines.append(hex_data)
        lines.append("};")
        lines.append("")

    lines.append("static inline void load_weights(unsigned char* hbm) {")
    lines.append("    (void)hbm; /* weights loaded via offsets */")
    lines.append("}")
    lines += ["", "#endif", ""]
    return "\n".join(lines)


def export_weights(state_dict: dict, output_path: str,
                   dtype: str = "fp16") -> None:
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
        f.write(emit_weights_h(weights))
    logger.info("weight_exporter: 已生成 %s", output_path)
