"""codegen 内部共享工具函数。"""

from __future__ import annotations

import os
from pathlib import Path

from npu_compiler.common import get_logger, load_config

logger = get_logger("codegen._helpers")

# ---- NPU C 枚举映射 ----

DTYPE_MAP = {
    "fp16": "NPU_DTYPE_FP16", "fp32": "NPU_DTYPE_FP32",
    "bf16": "NPU_DTYPE_BF16", "int8": "NPU_DTYPE_INT8",
    "int32": "NPU_DTYPE_INT32",
}

FORMAT_MAP = {
    "nd": "NPU_FORMAT_ND", "nz": "NPU_FORMAT_NZ",
    "nc1hwc0": "NPU_FORMAT_NC1HWC0",
}

# param type → C type 映射
PARAM_TYPE_C = {
    "addr": "void*", "int": "int", "float": "float",
    "enum": "int", "int_array": "const int*",
}

# ---- 配置加载 ----

_DEFAULT_CONFIG_DIR = Path(__file__).parent / "config"


def resolve_config_dir(config_dir: str | None) -> str:
    """解析 config 目录路径，默认使用模块内 config/。"""
    return config_dir if config_dir else str(_DEFAULT_CONFIG_DIR)


def load_signatures(config_dir: str | None = None) -> dict:
    """加载 c_api_signatures.yaml。"""
    d = resolve_config_dir(config_dir)
    return load_config(
        os.path.join(d, "c_api_signatures.yaml"),
        required_keys=["compute_ops"],
    )


# ---- 文件输出 ----

def write_file(path: str, content: str) -> None:
    """写入文件并记录日志。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    logger.info("已生成 %s", path)


def write_files(directory: str, files: list[tuple[str, str]]) -> None:
    """批量写入文件到指定目录。"""
    os.makedirs(directory, exist_ok=True)
    for name, content in files:
        write_file(os.path.join(directory, name), content)


# ---- C 代码生成辅助 ----

def c_header_guard(guard_name: str, body: str) -> str:
    """生成带 include guard 的 C 头文件。"""
    return f"#ifndef {guard_name}\n#define {guard_name}\n\n{body}\n#endif\n"
