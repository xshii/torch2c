"""统一 dtype 元信息：字节数 + C 枚举名。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DtypeInfo:
    """单个 dtype 的元信息。"""
    bytes: int
    c_enum: str


DTYPE_INFO: dict[str, DtypeInfo] = {
    "fp16": DtypeInfo(bytes=2, c_enum="NPU_DTYPE_FP16"),
    "fp32": DtypeInfo(bytes=4, c_enum="NPU_DTYPE_FP32"),
    "bf16": DtypeInfo(bytes=2, c_enum="NPU_DTYPE_BF16"),
    "int8": DtypeInfo(bytes=1, c_enum="NPU_DTYPE_INT8"),
    "int32": DtypeInfo(bytes=4, c_enum="NPU_DTYPE_INT32"),
}


def dtype_bytes(dtype: str) -> int:
    """返回 dtype 的字节数，默认 2 (fp16)。"""
    info = DTYPE_INFO.get(dtype)
    return info.bytes if info else 2


def dtype_c_enum(dtype: str) -> str:
    """返回 dtype 对应的 C 枚举名。"""
    info = DTYPE_INFO.get(dtype)
    return info.c_enum if info else "NPU_DTYPE_FP16"
