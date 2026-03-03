"""mock_emitter — 根据 c_api_signatures.yaml 生成 npu_mock.h。"""

from __future__ import annotations

import os

from npu_compiler.common import get_logger

from .._helpers import PARAM_TYPE_C, load_signatures, write_file

logger = get_logger("codegen.mock_emitter")


def _gen_func_decl(name: str, sig: dict) -> str:
    all_params = sig.get("params", []) + sig.get("optional_params", [])
    if not all_params:
        return f"void {name}(void);"
    args = ", ".join(
        f"{PARAM_TYPE_C.get(p['type'], 'int')} {p['name']}"
        for p in all_params
    )
    return f"void {name}({args});"


def emit_mock_h(signatures: dict) -> str:
    """生成 npu_mock.h 内容。"""
    lines = [
        "/* 自动生成 — 仅用于编译验证，不含实现 */",
        "#ifndef NPU_MOCK_H",
        "#define NPU_MOCK_H",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
        "typedef enum { NPU_DTYPE_FP16=0, NPU_DTYPE_FP32, NPU_DTYPE_BF16, "
        "NPU_DTYPE_INT8, NPU_DTYPE_INT32 } npu_dtype_t;",
        "typedef enum { NPU_FORMAT_ND=0, NPU_FORMAT_NZ, "
        "NPU_FORMAT_NC1HWC0 } npu_format_t;",
        "",
    ]
    for section in ["compute_ops", "dma_ops", "sync_ops"]:
        ops = signatures.get(section, {})
        if ops:
            lines.append(f"/* {section} */")
            for name, sig in ops.items():
                lines.append(_gen_func_decl(name, sig))
            lines.append("")

    lines += [
        "#ifdef NPU_DEBUG_DUMP",
        '#include <stdio.h>',
        '#define NPU_LOG(fmt, ...) printf("[NPU] " fmt "\\n", ##__VA_ARGS__)',
        "#else",
        "#define NPU_LOG(fmt, ...) ((void)0)",
        "#endif",
        "",
        "#endif",
        "",
    ]
    return "\n".join(lines)


def run(output_dir: str, config_dir: str | None = None) -> None:
    """生成 npu_mock.h 到 output_dir。"""
    logger.info("mock_emitter: 开始生成 npu_mock.h")
    sigs = load_signatures(config_dir)
    write_file(os.path.join(output_dir, "npu_mock.h"), emit_mock_h(sigs))
