"""codegen 内部共享工具函数。"""

from __future__ import annotations

import os
from pathlib import Path

from torch2c.common import DTYPE_INFO, INTEGRATION_CONFIG_DIR, get_logger, load_config

logger = get_logger("codegen._helpers")

# ---- 模板加载 ----

_TMPL_DIR = Path(__file__).parent / "templates"


def load_template(name: str) -> str:
    """从 codegen/templates/ 加载模板文件内容。"""
    with open(_TMPL_DIR / name, "r", encoding="utf-8") as f:
        return f.read()


# ---- NPU C 枚举映射 ----

DTYPE_C_ENUM_MAP = {k: v.c_enum for k, v in DTYPE_INFO.items()}

FORMAT_MAP = {
    "nd": "NPU_FORMAT_ND",
    "nz": "NPU_FORMAT_NZ",
    "nc1hwc0": "NPU_FORMAT_NC1HWC0",
}

# param type → C type 映射
PARAM_TYPE_C = {
    "addr": "void*",
    "int": "int",
    "float": "float",
    "enum": "int",
    "int_array": "const int*",
    "tensor_desc": "npu_tensor_t",
    "dtype_enum": "npu_dtype_t",
    "format_enum": "npu_format_t",
}

# ---- 配置加载 ----

_DEFAULT_CONFIG_DIR = INTEGRATION_CONFIG_DIR


def load_signatures(config_dir: str | None = None) -> dict:
    """加载 c_api_signatures.yaml。"""
    d = config_dir if config_dir else str(_DEFAULT_CONFIG_DIR)
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


# ---- 签名查找 ----

_OP_SECTIONS = ("compute_ops", "dma_ops", "idma_ops")


def find_op_sig(signatures: dict, npu_op: str) -> dict | None:
    """跨所有 section 查找算子签名。"""
    for section in _OP_SECTIONS:
        sig = signatures.get(section, {}).get(npu_op)
        if sig is not None:
            return sig
    return None


# ---- compute op 声明生成 ----


def gen_compute_decl(name: str, sig: dict, *, include_optional: bool = False) -> str:
    """从 YAML 签名生成单个 C 函数声明（自动注入 TidInfo 为第一参数）。"""
    params = sig.get("params", [])
    if include_optional:
        params = params + sig.get("optional_params", [])
    args_parts = ["TidInfo tid"]
    args_parts.extend(
        f"{PARAM_TYPE_C.get(p['type'], 'int')} {p['name']}" for p in params
    )
    return f"void {name}({', '.join(args_parts)});"


def gen_compute_declarations(signatures: dict) -> str:
    """从 c_api_signatures.yaml 生成所有 op 的 C 函数声明。

    按计算单元前缀分组输出：cube_ / vector_ / scalar_ / dma_ / idma_。
    """
    groups: dict[str, list[str]] = {
        "cube": [], "vector": [], "scalar": [], "dma": [], "idma": [],
    }
    for section in ["compute_ops", "dma_ops", "idma_ops"]:
        ops = signatures.get(section, {})
        for name, sig in ops.items():
            prefix = name.split("_", 1)[0]
            groups.setdefault(prefix, []).append(gen_compute_decl(name, sig))

    lines: list[str] = []
    for unit, label in [("cube", "cube compute ops"), ("vector", "vector compute ops"),
                        ("scalar", "scalar compute ops"), ("dma", "dma ops"),
                        ("idma", "idma ops")]:
        decls = groups.get(unit, [])
        if decls:
            lines.append(f"/* ---- {label} ---- */")
            lines.extend(decls)
            lines.append("")
    return "\n".join(lines)
